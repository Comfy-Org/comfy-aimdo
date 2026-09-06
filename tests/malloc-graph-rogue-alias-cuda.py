import gc

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
for size in (8, 16):
    graph.push("large")
    value = torch.empty(size * M, dtype=torch.uint8, device="cuda")
    del value
    graph.pop()

graph.push("large")
large_rogue = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
large_rogue.fill_(17)
graph.pop()
graph.push("large")
value = torch.empty(16 * M, dtype=torch.uint8, device="cuda")
value.fill_(23)
del value
graph.pop()
assert large_rogue[0].item() == 17

for size in (M, 2 * M):
    graph.push("small")
    value = torch.empty(size, dtype=torch.uint8, device="cuda")
    del value
    graph.pop()

graph.push("small")
small_rogue = torch.empty(M, dtype=torch.uint8, device="cuda")
small_pointer = small_rogue.data_ptr()
small_rogue.fill_(31)
graph.pop()
graph.push("small")
value = torch.empty(2 * M, dtype=torch.uint8, device="cuda")
assert value.data_ptr() // (8 * M) != small_pointer // (8 * M)
value.fill_(47)
del value
graph.pop()
assert small_rogue[0].item() == 31

graph.pop()
del graph
gc.collect()
assert large_rogue[-1].item() == 17
assert small_rogue[-1].item() == 31
del large_rogue, small_rogue
gc.collect()
aimdo.deinit()
