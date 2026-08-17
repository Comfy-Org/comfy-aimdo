import os

import comfy_aimdo.control as aimdo
import torch


ERROR = "aimdo memory compile error"

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
graph.push("first")
graph.pop()
graph.pop()

graph.replay()
try:
    graph.push("second")
except RuntimeError as error:
    assert ERROR in str(error)
    print(f"Different subgraph: {error}", flush=True)
    os._exit(0)
raise AssertionError(f"changing a subgraph did not raise {ERROR}")
