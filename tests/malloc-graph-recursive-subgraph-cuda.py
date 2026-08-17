import os

import comfy_aimdo.control as aimdo
import torch


ERROR = "aimdo memory compile error"

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
graph.push("inner")
try:
    graph.push("inner")
except RuntimeError as error:
    assert ERROR in str(error)
    print(f"Recursive subgraph: {error}", flush=True)
    os._exit(0)
raise AssertionError(f"recursively pushing a subgraph did not raise {ERROR}")
