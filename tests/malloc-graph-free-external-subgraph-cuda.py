import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
graph = aimdo.record(torch.cuda.current_stream())
graph.push("inner")
del value
assert not graph.pop()
assert not graph.pop()

graph.push()
graph.push("inner")
assert not graph.pop()
assert not graph.pop()
print("CUDA malloc graph external free subgraph test passed")
