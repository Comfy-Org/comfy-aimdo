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
graph.push("first")
graph.pop()
try:
    graph.push("second")
except RuntimeError as error:
    assert ERROR in str(error)
    print(f"Extra subgraph: {error}", flush=True)
    os._exit(0)
raise AssertionError(f"an extra subgraph did not raise {ERROR}")
