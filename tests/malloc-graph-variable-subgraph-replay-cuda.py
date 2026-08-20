import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())


def iteration():
    graph.push("inner")
    value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
    del value
    graph.pop()


iteration()
iteration()
graph.pop()

for count in (0, 1, 3):
    graph.push()
    for _ in range(count):
        iteration()
    graph.pop()

print("Variable subgraph replay test passed")
