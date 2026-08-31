import gc

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
first = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
hole = torch.empty(16 * M, dtype=torch.uint8, device="cuda")
last = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
pointers = (first.data_ptr(), hole.data_ptr(), last.data_ptr())
del hole
extended = torch.empty(24 * M, dtype=torch.uint8, device="cuda")
extended_pointer = extended.data_ptr()
del first
del last
del extended
graph.pop()

assert graph.peak_used == 40 * M
assert graph.virtual_bytes == 56 * M
assert graph.physical_bytes == 40 * M

graph.push()
first = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
hole = torch.empty(16 * M, dtype=torch.uint8, device="cuda")
last = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
assert (first.data_ptr(), hole.data_ptr(), last.data_ptr()) == pointers
del hole
extended = torch.empty(24 * M, dtype=torch.uint8, device="cuda")
assert extended.data_ptr() == extended_pointer
del first
del last
del extended
graph.pop()

del graph
gc.collect()
aimdo.deinit()
print("Fragmented CUDA malloc graph test passed")
