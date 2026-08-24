import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
for _ in range(2):
    value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
    del value
assert not graph.pop()

graph.push()
value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
del value
assert graph.pop()

graph.push()
for _ in range(2):
    value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
    del value
assert not graph.pop()

print("Missing allocation branch test passed")
