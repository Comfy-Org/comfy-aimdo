import gc
import threading

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
pointer = value.data_ptr()
value.fill_(17)
graph.pop()

graph.push()
replacement = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
assert replacement.data_ptr() != pointer
replacement.fill_(23)
del replacement
graph.pop()
assert value[0].item() == 17

del graph
gc.collect()
assert value[-1].item() == 17

holder = [value]
del value
thread = threading.Thread(target=lambda: holder.pop())
thread.start()
thread.join()
gc.collect()
aimdo.deinit()
