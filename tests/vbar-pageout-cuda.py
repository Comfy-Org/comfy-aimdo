"""Run on a disposable CUDA host with the matching native Aimdo build."""
import ctypes
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")

import torch
import comfy_aimdo.control as aimdo

assert aimdo.init("cuda") and aimdo.init_device(0)
from comfy_aimdo.model_vbar import ModelVBAR, vbar_signature_compare
from comfy_aimdo.torch import get_tensor_from_raw_ptr
from comfy_aimdo.vram_buffer import VRAMBuffer

PAGE = 32 * 1024 ** 2
device = torch.device("cuda", 0)
size = PAGE + 123

for pinned in (False, True):
    calls = []
    buffer = torch.full((size + 22,), 177, dtype=torch.uint8, pin_memory=pinned)
    saved = False

    def pageout(offset, length):
        global saved
        calls.append((offset, length))
        if not saved:
            value = get_tensor_from_raw_ptr(vbar.base_addr, size, device)
            buffer[11:-11].copy_(value)
            saved = True

    vbar = ModelVBAR(size, 0, pageout_callback=pageout)
    signature = vbar.fault(vbar.base_addr, size)
    assert signature is not None
    value = get_tensor_from_raw_ptr(vbar.base_addr, size, device)
    value[:PAGE].fill_(31)
    value[PAGE:].fill_(79)
    vbar.set_watermark(0)
    assert not calls  # Live consumers still pin the pages.
    del value
    vbar.unpin(vbar.base_addr, size)
    assert calls == [(0, PAGE), (PAGE, 123)]
    assert bool((buffer[:11] == 177).all() & (buffer[-11:] == 177).all())
    assert bool((buffer[11:11 + PAGE] == 31).all() & (buffer[11 + PAGE:-11] == 79).all())

    # Fault only allocates: the owner supplies the contents and detects new pages.
    vbar.set_watermark(size)
    restored = vbar.fault(vbar.base_addr, size)
    assert restored is not None and not vbar_signature_compare(signature, restored)
    value = get_tensor_from_raw_ptr(vbar.base_addr, size, device)
    value.copy_(buffer[11:-11])
    assert bool((value[:PAGE] == 31).all() & (value[PAGE:] == 79).all())
    del value
    vbar.unpin(vbar.base_addr, size)
    vbar.set_watermark(0)
    assert vbar.loaded_size() == 0
    vbar.set_watermark(size)
    assert vbar.fault(vbar.base_addr, size) is not None
    calls.clear()
    vbar.close()
    assert not calls  # Destruction never requests backing.
    print("PAGEOUT_PASS pinned=", pinned, flush=True)

cuda = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
cuda.cuMemAlloc_v2.restype = ctypes.c_int
cuda.cuMemAllocAsync.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_void_p]
cuda.cuMemAllocAsync.restype = ctypes.c_int
stream = torch.cuda.current_stream()
headroom = aimdo.get_simple_vram_headroom()

for operation in ("watermark", "free_memory", "unpin", "fault", "grow", "malloc", "malloc_async", "graph_small", "graph_large"):
    failed = True

    def pageout(offset, length):
        if failed:
            raise OSError("injected writeback failure")

    vbar = ModelVBAR(2 * PAGE, 0, pageout_callback=pageout)
    assert vbar.fault(vbar.base_addr, 2 * PAGE) is not None
    value = get_tensor_from_raw_ptr(vbar.base_addr, 2 * PAGE, device)
    value.fill_(79)
    torch.cuda.synchronize()
    if operation == "unpin":
        vbar.set_watermark(0)
    else:
        vbar.unpin(vbar.base_addr, 2 * PAGE)
    graph = target = None
    if operation in ("watermark", "free_memory", "unpin"):
        action = {
            "watermark": lambda: vbar.set_watermark(0),
            "free_memory": lambda: vbar.free_memory(2 * PAGE),
            "unpin": lambda: vbar.unpin(vbar.base_addr, 2 * PAGE),
        }[operation]
    else:
        # Force eviction without actually filling the GPU. Failure may originate
        # in a different VBAR from the one whose fault requested the reclamation.
        aimdo.set_simple_vram_headroom(1 << 60)
        if operation == "fault":
            target = ModelVBAR(PAGE, 0)
            action = lambda: target.fault(target.base_addr, PAGE)
        elif operation == "grow":
            target = VRAMBuffer(PAGE, 0)
            action = lambda: target.get(PAGE)
        else:
            if operation.startswith("graph_"):
                graph = aimdo.record(stream)
            pointer = ctypes.c_uint64(123)
            if operation == "malloc":
                result = cuda.cuMemAlloc_v2(ctypes.byref(pointer), 1024)
            else:
                amount = PAGE if operation == "graph_large" else 1024
                result = cuda.cuMemAllocAsync(ctypes.byref(pointer), amount, stream.cuda_stream)
            assert result == 999 and pointer.value == 0, (operation, result, pointer.value)
            action = None
    if action is not None:
        try:
            action()
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"{operation}: writeback error was lost")
    aimdo.set_simple_vram_headroom(headroom)
    if graph is not None:
        graph.abort()
    del graph, target
    assert vbar.get_residency() == [1, 1], operation  # Retained, with no leaked consumer pins.
    assert bool((value == 79).all()), operation
    del value

    # The client repairs its callback. No Aimdo error latch to clear, and a
    # subsequent fault must neither fail nor consume an error from an earlier call.
    failed = False
    vbar.prioritize()
    assert vbar.fault(vbar.base_addr, 2 * PAGE) is not None
    vbar.unpin(vbar.base_addr, 2 * PAGE)
    assert vbar.free_memory(2 * PAGE) == 2 * PAGE
    assert vbar.loaded_size() == 0
    vbar.close()
    print("PAGEOUT_FAILURE_RECOVERED", operation, flush=True)

del vbar, buffer
torch.cuda.synchronize()
aimdo.deinit()
