import gc

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

first_graph = aimdo.record(torch.cuda.current_stream())
first = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
first_pointer = first.data_ptr()
del first
first_graph.pop()

second_graph = aimdo.record(torch.cuda.current_stream())
second = torch.empty(16 * M, dtype=torch.uint8, device="cuda")
second_pointer = second.data_ptr()
del second
second_graph.pop()

first_graph.push()
first = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
assert first.data_ptr() == first_pointer
del first
first_graph.pop()

second_graph.push()
second = torch.empty(16 * M, dtype=torch.uint8, device="cuda")
assert second.data_ptr() == second_pointer
del second
second_graph.pop()

del first_graph
del second_graph
gc.collect()
aimdo.deinit()
print("Multiple CUDA malloc graphs test passed")
