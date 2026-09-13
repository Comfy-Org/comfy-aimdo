import gc
import threading

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")
torch.cuda.synchronize()
baseline = aimdo.get_total_vram_usage()

graph = aimdo.record(torch.cuda.current_stream(), assert_graph_breaks=True)
root = torch.empty(M, dtype=torch.uint8, device="cuda")
root.fill_(17)
graph.push("block")
nested = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
nested.fill_(23)
graph.abort()
graph.abort()
assert graph.rogue_count == 2
del graph
gc.collect()

replacement = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
replacement.fill_(31)
assert root[0].item() == 17
assert nested[-1].item() == 23
del replacement

holder = [nested]
del nested
thread = threading.Thread(target=lambda: holder.pop())
thread.start()
thread.join()
del root
gc.collect()

graph = aimdo.record(torch.cuda.current_stream())
graph.push("block")
value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
pointer = value.data_ptr()
del value
assert not graph.pop()
assert not graph.pop()

graph.push()
graph.push("block")
value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
assert value.data_ptr() == pointer
value.fill_(41)
del graph
gc.collect()

replacement = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
replacement.fill_(47)
assert value[0].item() == 41
del replacement, value
gc.collect()
torch.cuda.synchronize()
assert aimdo.get_total_vram_usage() == baseline

aimdo.deinit()
print("CUDA malloc graph abort test passed")
