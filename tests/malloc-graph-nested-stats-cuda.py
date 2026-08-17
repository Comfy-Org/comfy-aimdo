import gc

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024

assert aimdo.init("cuda")
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device="cuda")

graph = aimdo.record(torch.cuda.current_stream())
outer = torch.empty(8 * M, dtype=torch.uint8, device="cuda")


def inner():
    graph.push("inner")
    first = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
    second = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
    del first
    del second
    graph.pop()


inner()
inner()
del outer
graph.pop()

assert graph.peak_used == 24 * M
assert graph.virtual_bytes == 24 * M
assert graph.physical_bytes == 24 * M

graph.replay()
outer = torch.empty(8 * M, dtype=torch.uint8, device="cuda")
inner()
inner()
del outer
graph.pop()

del graph
gc.collect()
aimdo.deinit()
print("Nested CUDA malloc graph statistics test passed")
