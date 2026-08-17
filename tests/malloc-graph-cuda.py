import gc

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024


assert aimdo.init("cuda")
assert torch.cuda.get_allocator_backend() == "cudaMallocAsync"
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())

a = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
b = torch.empty(16 * M, dtype=torch.uint8, device="cuda")
a_ptr = a.data_ptr()
b_ptr = b.data_ptr()
del a
del b
c = torch.empty(24 * M, dtype=torch.uint8, device="cuda")
assert c.data_ptr() == a_ptr
c_ptr = c.data_ptr()
del c

assert graph.peak_used == 24 * M
assert graph.virtual_bytes == 24 * M
assert graph.physical_bytes == 24 * M
graph.pop()

outside = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
del outside

graph.replay()
a = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
b = torch.empty(16 * M, dtype=torch.uint8, device="cuda")
assert (a.data_ptr(), b.data_ptr()) == (a_ptr, b_ptr)
del a
del b
c = torch.empty(24 * M, dtype=torch.uint8, device="cuda")
assert c.data_ptr() == c_ptr
del c
graph.pop()

del graph
gc.collect()

aimdo.deinit()
print("CUDA malloc graph test passed")
