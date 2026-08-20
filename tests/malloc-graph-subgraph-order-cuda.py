import comfy_aimdo.control as aimdo
import torch


assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
graph.push("first")
graph.pop()
graph.push("second")
graph.pop()
graph.pop()

graph.push()
graph.push("second")
graph.pop()
graph.push("first")
graph.pop()
graph.pop()

print("Reordered subgraph branch test passed")
