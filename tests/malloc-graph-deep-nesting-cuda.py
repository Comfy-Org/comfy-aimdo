import gc

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
graph.push("outer")
graph.push("inner")
value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
pointer = value.data_ptr()
del value
graph.pop()
graph.pop()
graph.pop()

graph.push()
graph.push("outer")
graph.push("inner")
value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
assert value.data_ptr() == pointer
del value
graph.pop()
graph.pop()
graph.pop()

del graph
gc.collect()
aimdo.deinit()
print("Deeply nested CUDA malloc graph test passed")
