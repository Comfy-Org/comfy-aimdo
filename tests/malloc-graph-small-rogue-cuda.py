import gc

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
first = torch.empty(M, dtype=torch.uint8, device="cuda")
second = torch.empty(2 * M, dtype=torch.uint8, device="cuda")
pointers = (first.data_ptr(), second.data_ptr())
first.fill_(41)
second.fill_(42)
graph.pop()
assert graph.rogue_count == 2

graph.push()
replacement_first = torch.empty(M, dtype=torch.uint8, device="cuda")
replacement_second = torch.empty(2 * M, dtype=torch.uint8, device="cuda")
assert replacement_first.data_ptr() not in pointers
assert replacement_second.data_ptr() not in pointers
del replacement_first, replacement_second
graph.pop()
assert graph.rogue_count == 2

del graph
gc.collect()
assert first[0].item() == 41
assert second[0].item() == 42
del first
gc.collect()
assert second[-1].item() == 42
del second
gc.collect()
aimdo.deinit()
