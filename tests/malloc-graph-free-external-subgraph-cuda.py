import os

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024
ERROR = "aimdo memory compile error"

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
graph = aimdo.record(torch.cuda.current_stream())
graph.push("inner")
del value

try:
    graph.pop()
except RuntimeError as error:
    assert ERROR in str(error)
    print(f"Free external allocation in subgraph: {error}", flush=True)
    os._exit(0)
raise AssertionError(f"freeing an external allocation in a subgraph did not raise {ERROR}")
