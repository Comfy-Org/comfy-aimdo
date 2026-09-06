import gc
import threading

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024
ERROR = "aimdo memory compile error"

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
graph.push("inner")
freed = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
survivor = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
survivor.fill_(29)
graph.pop()
holder = [freed]
del freed
thread = threading.Thread(target=lambda: holder.pop())
thread.start()
thread.join()

try:
    graph.pop()
except RuntimeError as error:
    assert ERROR in str(error)
else:
    raise AssertionError(f"freeing a candidate rogue from an outer scope did not raise {ERROR}")

del graph
gc.collect()
assert survivor[0].item() == 29
del survivor
gc.collect()
aimdo.deinit()
