import ctypes

import comfy_aimdo.control as control

assert control.init("cuda")

import torch

from comfy_aimdo.torch import get_tensor_from_raw_ptr


M = 1024 * 1024
CUDA_SUCCESS = 0


control.init_device(torch.cuda.current_device())

cuda = ctypes.CDLL("libcuda.so.1")
cuda.cuMemAllocAsync.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_void_p]
cuda.cuMemAllocAsync.restype = ctypes.c_int
cuda.cuMemFreeAsync.argtypes = [ctypes.c_uint64, ctypes.c_void_p]
cuda.cuMemFreeAsync.restype = ctypes.c_int


def cuda_check(status):
    if status != CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed with status {status}")


def alloc(size, stream):
    ptr = ctypes.c_uint64()
    cuda_check(cuda.cuMemAllocAsync(ctypes.byref(ptr), size, stream.cuda_stream))
    return ptr.value


def free(ptr, stream):
    cuda_check(cuda.cuMemFreeAsync(ptr, stream.cuda_stream))


def tensor(ptr, size):
    return get_tensor_from_raw_ptr(ptr, size, torch.device("cuda"))


def test_alias_and_replay(stream, size):
    first_iteration = None
    usage_before = control.get_total_vram_usage()

    control.push_record(stream)
    for iteration in range(5):
        control.iterate()
        torch.cuda.nvtx.range_push(f"alias-iteration-{iteration}")

        first = alloc(size, stream)
        first_tensor = tensor(first, size)
        first_tensor.fill_(17 + iteration)
        del first_tensor
        free(first, stream)

        second = alloc(size, stream)
        second_tensor = tensor(second, size)
        assert torch.all(second_tensor[:4096] == 17 + iteration).item()
        assert torch.all(second_tensor[-4096:] == 17 + iteration).item()
        second_tensor.fill_(33 + iteration)
        del second_tensor
        free(second, stream)

        pointers = (first, second)
        if first_iteration is None:
            first_iteration = pointers
            assert first != second
        else:
            assert pointers == first_iteration

        torch.cuda.nvtx.range_pop()
    control.pop()

    assert control.get_total_vram_usage() == usage_before
    print(f"alias/replay {size // M} MiB: {first_iteration[0]:#x}, {first_iteration[1]:#x}")


def test_nested(stream):
    recorded_pointer = None

    control.push_record(stream)
    for outer in range(3):
        control.iterate()
        torch.cuda.nvtx.range_push(f"outer-iteration-{outer}")
        control.push_record(stream)
        for inner in range(4):
            control.iterate()
            ptr = alloc(M, stream)
            value = tensor(ptr, M)
            value.fill_(outer * 4 + inner)
            del value
            free(ptr, stream)
            if recorded_pointer is None:
                recorded_pointer = ptr
            else:
                assert ptr == recorded_pointer
        control.pop()
        torch.cuda.nvtx.range_pop()
    control.pop()
    print(f"nested replay: {recorded_pointer:#x}")


def test_other_stream_passthrough(stream, other):
    control.push_record(stream)
    control.iterate()
    ptr = alloc(M, other)
    value = tensor(ptr, M)
    with torch.cuda.stream(other):
        value.fill_(91)
    del value
    free(ptr, other)
    other.synchronize()
    control.pop()
    print("other-stream passthrough: ok")


def test_external_free_passthrough(stream):
    ptr = alloc(M, stream)
    control.push_record(stream)
    control.iterate()
    free(ptr, stream)
    control.pop()
    print("external free passthrough: ok")


stream = torch.cuda.Stream()
other = torch.cuda.Stream()
with torch.cuda.stream(stream):
    test_alias_and_replay(stream, M)
    test_alias_and_replay(stream, 3 * M)
    test_nested(stream)
    test_other_stream_passthrough(stream, other)
    test_external_free_passthrough(stream)

control.deinit()
print("CUDA loop-record tests passed")
