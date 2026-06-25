"""Intel XPU end-to-end aimdo offload test through ComfyUI (Z-Image Turbo).

Unlike examples/xpu_smoke.py (self-contained, synthetic weights), this runs a
real text-to-image generation through ComfyUI on a model that does NOT fit in
VRAM, so it can only succeed if aimdo's DynamicVRAM offloading is working.

Why this workload: Z-Image Turbo is a ~11.7 GB diffusion model and its Qwen3-4B
text encoder is ~7.7 GB. On a typical Intel Arc GPU (~10-12 GB) the weights far
exceed VRAM. WITHOUT aimdo, the driver spills the overflow into "shared" system
VRAM, which makes each step crawl and generation time balloon to effectively
infinite. WITH aimdo, weights stay host-resident and are faulted into a bounded
VRAM working set, keeping VRAM under the hardware limit and generation fast.

The graph mirrors the stock ComfyUI "Text to Image (Z-Image-Turbo)" template:

    UNETLoader -> ModelSamplingAuraFlow(shift=3)
    CLIPLoader(qwen_3_4b, type=lumina2) -> CLIPTextEncode (+ ConditioningZeroOut)
    VAELoader(ae) ; SD3 16-ch latent (1024x1024)
    KSampler(steps=8, cfg=1.0, res_multistep, simple) -> VAEDecode -> save

Requirements:
  * a machine with a real Intel XPU and a torch '+xpu' build
  * a ComfyUI checkout importable on PYTHONPATH
  * the Z-Image Turbo model files available to ComfyUI (see env vars below)

Environment variables (all optional):
  AIMDO_TEST_MODELS_ROOT  if set, its diffusion_models/ text_encoders/ vae/
                          subfolders are registered with ComfyUI. Otherwise the
                          models must already be in ComfyUI's configured paths.
  AIMDO_TEST_UNET         diffusion model filename (default z_image_turbo_bf16.safetensors)
  AIMDO_TEST_CLIP         text encoder filename   (default qwen_3_4b.safetensors)
  AIMDO_TEST_VAE          vae filename            (default ae.safetensors)
  AIMDO_TEST_OUT          output PNG path         (default xpu_comfyui_offload_out.png)

Run:

    PYTHONPATH=<comfy-aimdo>:<ComfyUI> python examples/xpu_comfyui_offload.py

Exits 0 on success, non-zero (with an assertion) on failure.
"""
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO)

UNET = os.environ.get("AIMDO_TEST_UNET", "z_image_turbo_bf16.safetensors")
CLIP = os.environ.get("AIMDO_TEST_CLIP", "qwen_3_4b.safetensors")
VAE = os.environ.get("AIMDO_TEST_VAE", "ae.safetensors")
OUT = os.environ.get("AIMDO_TEST_OUT", "xpu_comfyui_offload_out.png")
MODELS_ROOT = os.environ.get("AIMDO_TEST_MODELS_ROOT")

# ---- 0. load the aimdo backend FIRST (as ComfyUI's main.py does) -------------
# comfy.model_management imports comfy_aimdo.host_buffer at module load and binds
# lib = control.lib once, so control.init() MUST run before any comfy import.
import comfy_aimdo.control

assert comfy_aimdo.control.detect_vendor() == "xpu", (
    "this test requires a torch '+xpu' build; "
    f"detected backend = {comfy_aimdo.control.detect_vendor()!r}"
)
assert comfy_aimdo.control.init(), "comfy_aimdo.control.init() failed (DLL/so load)"

# ---- 1. confirm ComfyUI's main.py gate would enable aimdo on this machine ----
import comfy.model_management as mm
from comfy.cli_args import enables_dynamic_vram

gate = (enables_dynamic_vram()
        and (mm.is_nvidia() or mm.is_intel_xpu())
        and not mm.is_wsl())
print(f"[gate] enables_dynamic_vram={enables_dynamic_vram()} is_intel_xpu={mm.is_intel_xpu()} "
      f"is_wsl={mm.is_wsl()} -> aimdo gate={gate}")
assert gate, "ComfyUI's main.py gate would NOT enable aimdo on this machine"

import torch

dev = mm.get_torch_device()
vram_mb = torch.xpu.get_device_properties(dev.index).total_memory >> 20
print(f"[hw] {torch.xpu.get_device_properties(dev.index).name}  VRAM total = {vram_mb} MB")

# ---- 2. replicate main.py's aimdo enablement block --------------------------
assert comfy_aimdo.control.init_device(dev.index), "comfy_aimdo.control.init_device failed"
comfy_aimdo.control.set_log_warning()

import comfy.model_patcher
import comfy.memory_management

comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
comfy.memory_management.aimdo_enabled = True
print("[enable] DynamicVRAM (aimdo) enabled; CoreModelPatcher = ModelPatcherDynamic")

# ---- 3. build the Z-Image Turbo graph through the node API -------------------
import folder_paths

if MODELS_ROOT:
    for sub in ("diffusion_models", "text_encoders", "vae"):
        folder_paths.add_model_folder_path(sub, os.path.join(MODELS_ROOT, sub))

import nodes
import comfy_extras.nodes_model_advanced as nma
import comfy_aimdo.model_vbar

model = nodes.UNETLoader().load_unet(UNET, "default")[0]
print(f"[load] unet loaded; patcher type = {type(model).__name__}")
assert isinstance(model, comfy.model_patcher.ModelPatcherDynamic), (
    "model is not a ModelPatcherDynamic - aimdo path not active"
)
model = nma.ModelSamplingAuraFlow().patch_aura(model, 3.0)[0]

clip = nodes.CLIPLoader().load_clip(CLIP, "lumina2", "default")[0]
vae = nodes.VAELoader().load_vae(VAE)[0]

pos = nodes.CLIPTextEncode().encode(
    clip,
    "Latina female with thick wavy hair, harbor boats and pastel houses behind. "
    "Breezy seaside light, warm tones, cinematic close-up.")[0]
neg = nodes.ConditioningZeroOut().zero_out(pos)[0]

# SD3/Lumina2-style empty latent (16 channels, 8x spatial downscale).
latent = {"samples": torch.zeros(
    [1, 16, 1024 // 8, 1024 // 8],
    device=mm.intermediate_device(), dtype=mm.intermediate_dtype())}

print("[sample] starting KSampler (8 steps, res_multistep/simple, cfg=1.0) ...")
t0 = time.time()
samples = nodes.KSampler().sample(
    model, 0, 8, 1.0, "res_multistep", "simple", pos, neg, latent, 1.0)[0]
dt = time.time() - t0

vbar_mb = comfy_aimdo.model_vbar.vbars_analyze() >> 20
total_mb = comfy_aimdo.control.get_total_vram_usage() >> 20
print(f"[sample] done in {dt:.1f}s")
print(f"[aimdo] VBAR-resident VRAM = {vbar_mb} MB | detour-accounted VRAM = {total_mb} MB")

# aimdo must keep the VRAM working set under the hardware limit even though the
# diffusion model alone (~11.7 GB) is larger than VRAM. If accounting reported
# more than VRAM, weights were not actually being offloaded.
assert total_mb <= vram_mb, (
    f"detour-accounted VRAM ({total_mb} MB) exceeds hardware VRAM ({vram_mb} MB) - "
    "weights were not offloaded"
)

images = nodes.VAEDecode().decode(vae, samples)[0]
print(f"[decode] image tensor: {tuple(images.shape)} {images.dtype} {images.device}")

import numpy as np
from PIL import Image

arr = (images[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
Image.fromarray(arr).save(OUT)
var = float(arr.astype(np.float32).var())
print(f"[save] {OUT} pixel-variance={var:.1f}")
assert var > 50.0, "image looks empty/uniform - generation likely failed"

print(f"[PASS] Z-Image Turbo generated a valid 1024x1024 image on a {vram_mb} MB "
      f"Intel XPU via aimdo offloading (VRAM held to {total_mb} MB)")
sys.exit(0)
