import gc

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
other = torch.cuda.Stream()
with torch.cuda.stream(other):
    value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
    del value
graph.pop()
other.synchronize()

assert graph.virtual_bytes == 0
assert graph.physical_bytes == 0
graph.replay()
graph.pop()

del graph
gc.collect()
aimdo.deinit()
print("Off-stream CUDA allocation exclusion test passed")
