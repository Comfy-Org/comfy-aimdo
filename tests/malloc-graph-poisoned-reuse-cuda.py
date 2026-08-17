import os

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024
ERROR = "aimdo memory compile error"

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
del value
graph.pop()

graph.replay()
try:
    graph.pop()
except RuntimeError as error:
    assert ERROR in str(error)
else:
    raise AssertionError(f"a missing allocation did not raise {ERROR}")

try:
    graph.replay()
except RuntimeError as error:
    assert ERROR in str(error)
    print(f"Poisoned graph reuse: {error}", flush=True)
    os._exit(0)
raise AssertionError(f"reusing a poisoned graph did not raise {ERROR}")
