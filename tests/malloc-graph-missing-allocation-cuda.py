import os

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024
ERROR = "aimdo memory compile error"

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
first = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
del first
second = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
del second
graph.pop()

graph.replay()
first = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
del first

try:
    graph.pop()
except RuntimeError as error:
    assert ERROR in str(error)
    print(f"Missing allocation: {error}", flush=True)
    os._exit(0)
raise AssertionError(f"a missing allocation did not raise {ERROR}")
