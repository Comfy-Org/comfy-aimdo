# A second init_devices() must be refused, not reinstall the CUDA hooks over
# the live ones and crash the next allocation.
import comfy_aimdo.control as aimdo
import torch


assert aimdo.init("cuda")
device = torch.cuda.current_device()
assert aimdo.init_device(device)
assert not aimdo.init_device(device)

x = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
x.fill_(1)
torch.cuda.synchronize()
assert x[-1].item() == 1
assert aimdo.get_total_vram_usage() > 0
del x
aimdo.deinit()
print("init_devices twice test passed")
