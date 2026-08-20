import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
assert not graph.iterate("block")
value = torch.empty(1 * M, dtype=torch.uint8, device="cuda")
del value
assert not graph.iterate()
assert not graph.pop()

graph.push()
assert not graph.iterate("block")
value = torch.empty(2 * M, dtype=torch.uint8, device="cuda")
del value
assert graph.iterate()
assert not graph.pop()

print("CUDA malloc graph iterate break test passed")
