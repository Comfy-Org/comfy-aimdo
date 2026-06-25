"""Intel XPU (Level Zero) backend smoke test.

Self-contained: uses only comfy-aimdo + a torch XPU build (no ComfyUI, no model
files). It exercises the parts of the XPU backend that matter:

  * backend auto-detection + DLL/so load (control.init)
  * the zeMemAllocDevice/zeMemFree detour + VRAM accounting
  * VBAR reserve / fault / map under real VRAM pressure (offloading)
  * the host->device copy path (ze_cuMemcpyHtoDAsync) + DLPack interop
  * data integrity of faulted-in weights

Run on a machine with a real Intel XPU and a torch '+xpu' build:

    python examples/xpu_smoke.py

Exits 0 on success, non-zero (with an assertion) on failure.
"""
import gc
import math
import sys

# VERY IMPORTANT: control.init() must run before torch (or anything importing
# torch) is imported.
import comfy_aimdo.control

assert comfy_aimdo.control.detect_vendor() == "xpu", (
    "this smoke test requires a torch '+xpu' build; "
    f"detected backend = {comfy_aimdo.control.detect_vendor()!r}"
)
assert comfy_aimdo.control.init(), "comfy_aimdo.control.init() failed (DLL/so load)"
comfy_aimdo.control.set_log_info()

import torch
import comfy_aimdo.torch
from comfy_aimdo.model_vbar import ModelVBAR, vbar_fault, vbar_unpin, vbar_signature_compare

assert torch.xpu.is_available(), "torch.xpu.is_available() is False"

DEV_INDEX = torch.xpu.current_device()
DEV = f"xpu:{DEV_INDEX}"
comfy_aimdo.control.init_device(DEV_INDEX)

M = 1024 ** 2
signatures = {}


def run_layer(input_tensor, weight, cpu_source):
    """Fault a weight in; if resident, copy + integrity-check it; else fall back
    to a plain host->device copy. Returns (output, was_offloaded)."""
    signature = vbar_fault(weight)
    if signature is not None:
        weight_tensor = (
            comfy_aimdo.torch.aimdo_to_tensor(weight, torch.device(DEV))
            .view(dtype=input_tensor.dtype)
            .view(input_tensor.shape)
        )
        if not vbar_signature_compare(signature, signatures.get(weight, None)):
            weight_tensor.copy_(cpu_source)
        # Integrity: the resident weight must match what we copied in.
        assert torch.equal(weight_tensor.to("cpu"), cpu_source), (
            "faulted-in weight does not match source (H2D copy corruption)"
        )
        signatures[weight] = signature
        output = input_tensor + weight_tensor
        vbar_unpin(weight)
        return output, False

    # Offloaded: aimdo declined the fault under pressure -> host fallback.
    return input_tensor + cpu_source.to(DEV), True


def run_model(weights, cpu_weight, iterations=3):
    offloaded_total = 0
    x = torch.zeros(cpu_weight.shape, device=DEV, dtype=cpu_weight.dtype)
    for i in range(iterations):
        for layer_weight in weights:
            x, was_offloaded = run_layer(x, layer_weight, cpu_weight)
            offloaded_total += int(was_offloaded)
    gc.collect()
    torch.xpu.empty_cache()
    return offloaded_total


gpu_size = torch.xpu.get_device_properties(DEV_INDEX).total_memory
dtype = torch.float16

# Build a model whose weights sum to ~1.5x VRAM so the offloader MUST evict.
num_layers = 30
scale_factor = max(1, gpu_size * 3 // (2 * num_layers * M * dtype.itemsize))
vbar = ModelVBAR(gpu_size * 5, device=DEV_INDEX)  # VBAR may exceed VRAM
shape = (1024, 1024, scale_factor)
weights = [vbar.alloc(math.prod(shape) * dtype.itemsize) for _ in range(num_layers)]
cpu_weight = torch.ones(shape, dtype=dtype)

model_bytes = num_layers * math.prod(shape) * dtype.itemsize
print(f"[smoke] device={DEV} vram={gpu_size / M:.0f}M "
      f"model={model_bytes / M:.0f}M ({model_bytes / gpu_size:.2f}x vram)")

offloaded = run_model(weights, cpu_weight)
comfy_aimdo.control.analyze()

# Under 1.5x-VRAM pressure at least one fault must have been declined; otherwise
# the offloader / accounting detour is not actually engaging.
assert offloaded > 0, (
    "no weights were offloaded under 1.5x-VRAM pressure - the zeMemAllocDevice "
    "accounting detour or VBAR eviction is not working"
)

print(f"[smoke] PASS: {offloaded} fault(s) offloaded under pressure, "
      f"all resident weights passed integrity checks")
sys.exit(0)
