import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
first_pointer = value.data_ptr()
del value
graph.pop()

graph.push()
first = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
second = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
assert first.data_ptr() == first_pointer
assert second.data_ptr() != first_pointer
del first, second
graph.pop()

graph.push()
value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
assert value.data_ptr() == first_pointer
del value
graph.pop()

print("Extra allocation branch test passed")
