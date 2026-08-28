import gc

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
rogue = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
rogue_pointer = rogue.data_ptr()
tail = torch.empty(16 * M, dtype=torch.uint8, device="cuda")
tail_pointer = tail.data_ptr()
del tail
assert not graph.pop()

graph.push()
tail = torch.empty(16 * M, dtype=torch.uint8, device="cuda")
assert tail.data_ptr() == tail_pointer
assert tail.data_ptr() != rogue_pointer
del tail
assert not graph.pop()

del graph
gc.collect()
del rogue
gc.collect()
aimdo.deinit()
