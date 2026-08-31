import gc

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
first = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
graph.pause()
temporary = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
del temporary
graph.resume()
second = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
del second, first
assert not graph.pop()
assert graph.physical_bytes == 16 * M

graph.push()
first = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
second = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
del second, first
assert not graph.pop()

del graph
gc.collect()
aimdo.deinit()
print("CUDA malloc graph pause test passed")
