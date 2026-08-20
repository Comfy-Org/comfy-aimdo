import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
first = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
second = torch.empty(16 * M, dtype=torch.uint8, device="cuda")
del first
del second
graph.pop()

graph.replay()
first = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
second = torch.empty(16 * M, dtype=torch.uint8, device="cuda")
del second
del first
graph.pop()

print("Reordered free branch test passed")
