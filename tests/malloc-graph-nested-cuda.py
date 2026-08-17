import gc

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024


assert aimdo.init("cuda")
assert torch.cuda.get_allocator_backend() == "cudaMallocAsync"
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())


def inner():
    graph.push("inner")
    first = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
    second = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
    pointers = (first.data_ptr(), second.data_ptr())
    del first
    third = torch.empty(16 * M, dtype=torch.uint8, device="cuda")
    pointers += (third.data_ptr(),)
    del second
    del third
    graph.pop()
    return pointers


def outer():
    first = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
    second = torch.empty(16 * M, dtype=torch.uint8, device="cuda")
    pointers = (first.data_ptr(), second.data_ptr())
    del second

    inner_pointers = inner()
    assert inner() == inner_pointers
    assert inner() == inner_pointers

    last = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
    last_pointer = last.data_ptr()
    del first
    del last
    return pointers, inner_pointers, last_pointer


pointers = outer()
graph.pop()

graph.replay()
assert outer() == pointers
graph.pop()

del graph
gc.collect()
aimdo.deinit()
print("Nested CUDA malloc graph test passed")
