import gc

import comfy_aimdo.control as aimdo
import torch


assert aimdo.init("cuda")
assert torch.cuda.get_allocator_backend() == "cudaMallocAsync"
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
graph.push("empty")
graph.pop()
graph.push("empty")
graph.pop()
graph.pop()

graph.replay()
graph.push("empty")
graph.pop()
graph.push("empty")
graph.pop()
graph.pop()

del graph
gc.collect()
aimdo.deinit()
print("Empty CUDA malloc subgraph test passed")
