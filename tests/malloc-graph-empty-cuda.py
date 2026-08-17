import gc

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024


assert aimdo.init("cuda")
assert torch.cuda.get_allocator_backend() == "cudaMallocAsync"
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
small = torch.empty(1 * M, dtype=torch.uint8, device="cuda")
del small
empty = torch.empty(0, device="cuda")
assert empty.data_ptr() == 0
del empty
graph.pop()

assert graph.virtual_bytes == 0
assert graph.physical_bytes == 0

graph.replay()
small = torch.empty(2 * M, dtype=torch.uint8, device="cuda")
del small
empty = torch.empty(0, device="cuda")
assert empty.data_ptr() == 0
del empty
graph.pop()

del graph
gc.collect()
aimdo.deinit()
print("Empty CUDA malloc graph test passed")
