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
    torch.empty(16 * M, dtype=torch.uint8, device="cuda")
except RuntimeError:
    pass
else:
    raise AssertionError("changing an allocation size did not fail")

try:
    graph.pop()
except RuntimeError as error:
    assert ERROR in str(error)
    print(f"Changed allocation size: {error}", flush=True)
    os._exit(0)
raise AssertionError(f"changing an allocation size did not raise {ERROR}")
