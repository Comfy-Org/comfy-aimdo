import gc

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
assert graph.push("wrapper")
outer_pointer = None
inner_pointer = None
for outer in range(2):
    assert graph.iterate("outer") == (outer == 0)
    outer_value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
    outer_pointer = outer_pointer or outer_value.data_ptr()
    assert outer_value.data_ptr() == outer_pointer

    for inner in range(2):
        assert graph.iterate("inner") == (outer == 0 and inner == 0)
        inner_value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
        inner_pointer = inner_pointer or inner_value.data_ptr()
        assert inner_value.data_ptr() == inner_pointer
        del inner_value
    graph.iterate()
    del outer_value
graph.iterate()
graph.pop()
graph.pop()

graph.push()
assert not graph.push("wrapper")
for _ in range(2):
    assert not graph.iterate("outer")
    outer_value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
    assert outer_value.data_ptr() == outer_pointer

    for _ in range(3):
        assert not graph.iterate("inner")
        inner_value = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
        assert inner_value.data_ptr() == inner_pointer
        del inner_value
    graph.iterate()
    del outer_value
graph.iterate()
graph.pop()
graph.pop()

del graph
gc.collect()
aimdo.deinit()
print("Nested CUDA malloc graph iterate test passed")
