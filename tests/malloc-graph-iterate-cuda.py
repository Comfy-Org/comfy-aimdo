import gc

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
pointer = None
for _ in range(3):
    assert not graph.iterate("block")
    value = torch.empty(1 * M, dtype=torch.uint8, device="cuda")
    if pointer is None:
        pointer = value.data_ptr()
    else:
        assert value.data_ptr() == pointer
    del value
assert not graph.iterate()
assert not graph.pop()

graph.push()
for _ in range(2):
    assert not graph.iterate("block")
    value = torch.empty(1 * M, dtype=torch.uint8, device="cuda")
    assert value.data_ptr() == pointer
    del value
assert not graph.iterate()
assert not graph.pop()

del graph
gc.collect()
aimdo.deinit()
print("CUDA malloc graph iterate test passed")
