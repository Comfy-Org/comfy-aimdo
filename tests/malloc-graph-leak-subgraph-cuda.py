import gc

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
graph.push("inner")
value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
pointer = value.data_ptr()
value.fill_(31)
graph.pop()

replacement = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
assert replacement.data_ptr() != pointer
del replacement
graph.pop()

graph.push()
graph.push("inner")
graph.pop()
replacement = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
assert replacement.data_ptr() != pointer
del replacement
graph.pop()
assert value[0].item() == 31

del graph
gc.collect()
assert value[-1].item() == 31
del value
gc.collect()
aimdo.deinit()
