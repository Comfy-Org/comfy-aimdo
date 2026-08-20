import os

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream(), assert_graph_breaks=True)
value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
del value
graph.pop()

graph.push()
try:
    torch.empty(16 * M, dtype=torch.uint8, device="cuda")
except RuntimeError as error:
    print(f"Assert graph break: {error}", flush=True)
    os._exit(0)
raise AssertionError("assert_graph_breaks did not fail the allocation")
