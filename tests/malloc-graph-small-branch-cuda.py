import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
first = torch.empty(1 * M, dtype=torch.uint8, device="cuda")
first_pointer = first.data_ptr()
del first
graph.pop()

graph.push()
first = torch.empty(1 * M, dtype=torch.uint8, device="cuda")
second = torch.empty(2 * M, dtype=torch.uint8, device="cuda")
assert first.data_ptr() == first_pointer
assert second.data_ptr() != first_pointer
del second, first
graph.pop()

graph.push()
first = torch.empty(1 * M, dtype=torch.uint8, device="cuda")
assert first.data_ptr() == first_pointer
del first
graph.pop()

print("Small allocation branch test passed")
