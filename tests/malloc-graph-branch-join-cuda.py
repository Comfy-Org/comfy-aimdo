import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
outer = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
outer_pointer = outer.data_ptr()
graph.push("inner")
inner = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
inner_pointer = inner.data_ptr()
del inner
assert not graph.pop()
tail = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
tail_pointer = tail.data_ptr()
del tail, outer
assert not graph.pop()

graph.push()
outer = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
assert outer.data_ptr() == outer_pointer
graph.push("inner")
inner = torch.empty(16 * M, dtype=torch.uint8, device="cuda")
assert inner.data_ptr() != outer_pointer
del inner
assert graph.pop()
tail = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
assert tail.data_ptr() == tail_pointer
del tail, outer
assert not graph.pop()

graph.push()
outer = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
graph.push("inner")
inner = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
assert inner.data_ptr() == inner_pointer
del inner
assert not graph.pop()
tail = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
assert tail.data_ptr() == tail_pointer
del tail, outer
assert not graph.pop()

print("CUDA malloc graph branch join test passed")
