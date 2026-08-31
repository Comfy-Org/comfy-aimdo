import gc

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
live = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
freed = torch.empty(16 * M, dtype=torch.uint8, device="cuda")
freed_pointer = freed.data_ptr()
del freed
alias = torch.empty(24 * M, dtype=torch.uint8, device="cuda")
assert alias.data_ptr() != freed_pointer
pointers = (live.data_ptr(), alias.data_ptr())
del alias
del live
spanning = torch.empty(48 * M, dtype=torch.uint8, device="cuda")
spanning[8 * M:16 * M].fill_(1)
spanning[24 * M:32 * M].fill_(2)
assert spanning[8 * M:16 * M].min().item() == 1
assert spanning[24 * M:32 * M].min().item() == 2
spanning_pointer = spanning.data_ptr()
del spanning
graph.pop()

assert graph.peak_used == 56 * M
assert graph.virtual_bytes == 104 * M
assert graph.physical_bytes == 56 * M

graph.push()
live = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
freed = torch.empty(16 * M, dtype=torch.uint8, device="cuda")
del freed
alias = torch.empty(24 * M, dtype=torch.uint8, device="cuda")
assert (live.data_ptr(), alias.data_ptr()) == pointers
del alias
del live
spanning = torch.empty(48 * M, dtype=torch.uint8, device="cuda")
assert spanning.data_ptr() == spanning_pointer
del spanning
graph.pop()

del graph
gc.collect()
aimdo.deinit()
print("Aliased physical-page CUDA malloc graph test passed")
