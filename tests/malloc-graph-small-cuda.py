import gc

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
first = torch.empty(1 * M, dtype=torch.uint8, device="cuda")
second = torch.empty(2 * M, dtype=torch.uint8, device="cuda")
pointers = (first.data_ptr(), second.data_ptr())
del first
replacement = torch.empty(M // 2, dtype=torch.uint8, device="cuda")
replacement_pointer = replacement.data_ptr()
assert replacement_pointer == pointers[0]
del second
del replacement
graph.pop()

assert graph.virtual_bytes == 8 * M
assert graph.physical_bytes == 8 * M

graph.push()
first = torch.empty(1 * M, dtype=torch.uint8, device="cuda")
second = torch.empty(2 * M, dtype=torch.uint8, device="cuda")
assert (first.data_ptr(), second.data_ptr()) == pointers
del first
replacement = torch.empty(M // 2, dtype=torch.uint8, device="cuda")
assert replacement.data_ptr() == replacement_pointer
del second
del replacement
graph.pop()

del graph
gc.collect()
aimdo.deinit()
print("Small CUDA malloc graph test passed")
