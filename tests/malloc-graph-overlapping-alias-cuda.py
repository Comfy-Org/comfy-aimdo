# Regression: an alias collision must not skip a usable overlapping window.
# No model needed; at most 80 MiB of graph pages on the failing implementation.
import os
os.environ['PYTORCH_ALLOC_CONF'] = 'backend:cudaMallocAsync'
import json
import torch
import comfy_aimdo.control as aimdo

M = 1024 * 1024
assert aimdo.init('cuda')
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device='cuda')
graph = aimdo.record(torch.cuda.current_stream())
a = torch.empty(16 * M, dtype=torch.uint8, device='cuda')
b = torch.empty(8 * M, dtype=torch.uint8, device='cuda')
del a
c = torch.empty(24 * M, dtype=torch.uint8, device='cuda')
del b, c
before = graph.virtual_bytes
assert before == 48 * M, f'Expected the six-page alias layout, got {before} bytes.'
d = torch.empty(32 * M, dtype=torch.uint8, device='cuda')
for page in range(4):
    d[page * 8 * M:(page + 1) * 8 * M].fill_(37 + page)
after = graph.virtual_bytes
physical = graph.physical_bytes
for page in range(4):
    assert d[page * 8 * M].item() == 37 + page
    assert d[(page + 1) * 8 * M - 1].item() == 37 + page
print(json.dumps({'beforeVirtualBytes': before, 'afterVirtualBytes': after,
                  'physicalBytes': physical, 'reused': before == after}))
assert before == after, 'A valid overlapping range was skipped.'
assert physical == 32 * M, f'Expected 32 MiB of physical pages, got {physical} bytes.'
del d
graph.pop()
del graph
aimdo.deinit()
