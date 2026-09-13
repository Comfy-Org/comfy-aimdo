# A simple_vram_headroom set after init_devices() steers the next VBAR fault.
# The model is a quarter of VRAM; the runtime headroom leaves room for half.
import gc
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")

import comfy_aimdo.control as aimdo
import torch


M = 1024 * 1024
PAGE = 32 * M
CHUNK = 128 * M
DEFAULT_HEADROOM = 256 * M

assert aimdo.init("cuda")
import comfy_aimdo.torch  # noqa: E402
from comfy_aimdo.model_vbar import ModelVBAR, vbar_fault, vbar_unpin  # noqa: E402

device = torch.cuda.current_device()
assert aimdo.init_device(device)
torch.empty(1, device="cuda")
cuda_device = torch.device("cuda", device)

free, capacity = torch.cuda.mem_get_info(device)
model = capacity // 4
model -= model % CHUNK
chunks = model // CHUNK

vbar = ModelVBAR(model * 10, device)
vbar.prioritize()
allocs = [vbar.alloc(CHUNK) for _ in range(chunks)]


def forward(fill=None):
    faulted = 0
    for alloc in allocs:
        if vbar_fault(alloc) is None:
            continue
        if fill is not None:
            comfy_aimdo.torch.aimdo_to_tensor(alloc, cuda_device).fill_(fill)
        vbar_unpin(alloc)
        faulted += 1
    torch.cuda.synchronize()
    return faulted


forward(1)
assert vbar.loaded_size() == model

free_now, _ = torch.cuda.mem_get_info(device)
headroom = free_now + aimdo.get_total_vram_usage() - model // 2
budget = capacity - headroom
aimdo.set_simple_vram_headroom(headroom)
assert aimdo.get_simple_vram_headroom() == headroom

faulted = forward()
pressed = vbar.loaded_size()
assert faulted < chunks
assert pressed <= budget + PAGE
assert pressed < model * 0.9

spill = torch.empty(min(model // 4, budget // 2), dtype=torch.uint8, device=cuda_device)
torch.cuda.synchronize()
forward()
assert vbar.loaded_size() <= pressed - spill.numel() + 2 * PAGE
del spill
torch.cuda.empty_cache()

aimdo.set_simple_vram_headroom(DEFAULT_HEADROOM)
vbar.prioritize()
forward()
assert vbar.loaded_size() == model

del allocs
del vbar
gc.collect()
torch.cuda.synchronize()
aimdo.deinit()
print("simple_vram_headroom runtime test passed")
