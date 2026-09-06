import gc

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")
torch.cuda.synchronize()
baseline = aimdo.get_total_vram_usage()

graph = aimdo.record(torch.cuda.current_stream())
value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
del value
graph.pop()
assert aimdo.get_total_vram_usage() == baseline + 8 * M

del graph
gc.collect()
assert aimdo.get_total_vram_usage() == baseline

aimdo.deinit()
print("CUDA malloc graph destructor test passed")
