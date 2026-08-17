import os

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024
ERROR = "aimdo memory compile error"

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
graph.push("inner")
value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
del value
graph.pop()

value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
del value

try:
    graph.push("inner")
except RuntimeError as error:
    assert ERROR in str(error)
    print(f"Allocation before subgraph replay: {error}", flush=True)
    os._exit(0)
raise AssertionError(f"an allocation before subgraph replay did not raise {ERROR}")
