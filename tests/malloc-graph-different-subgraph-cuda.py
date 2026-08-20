import comfy_aimdo.control as aimdo
import torch


assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
graph.push("first")
graph.pop()
graph.pop()

for name in ("second", "first", "second"):
    graph.push()
    graph.push(name)
    graph.pop()
    graph.pop()

print("Different subgraph branch test passed")
