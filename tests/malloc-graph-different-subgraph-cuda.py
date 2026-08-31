import comfy_aimdo.control as aimdo
import torch


assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
graph.push("first")
assert not graph.pop()
assert not graph.pop()

for name, broken in (("second", True), ("first", False), ("second", False)):
    graph.push()
    graph.push(name)
    assert not graph.pop()
    assert graph.pop() == broken

print("Different subgraph branch test passed")
