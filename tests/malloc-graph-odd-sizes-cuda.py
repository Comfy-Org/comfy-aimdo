import gc

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
first = torch.empty(8 * M + 1, dtype=torch.uint8, device="cuda")
first_pointer = first.data_ptr()
del first
second = torch.empty(24 * M + 1, dtype=torch.uint8, device="cuda")
second_pointer = second.data_ptr()
del second
graph.pop()

assert graph.peak_used == 32 * M
assert graph.virtual_bytes == 48 * M
assert graph.physical_bytes == 32 * M

graph.replay()
first = torch.empty(8 * M + 1, dtype=torch.uint8, device="cuda")
assert first.data_ptr() == first_pointer
del first
second = torch.empty(24 * M + 1, dtype=torch.uint8, device="cuda")
assert second.data_ptr() == second_pointer
del second
graph.pop()

del graph
gc.collect()
aimdo.deinit()
print("Odd-sized CUDA malloc graph test passed")
