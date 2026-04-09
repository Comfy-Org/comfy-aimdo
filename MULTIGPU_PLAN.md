# comfy-aimdo Multi-GPU Support Plan

## Problem
aimdo's VRAM management assumes a single GPU. With 2+ GPUs, `budget_deficit()` uses global `total_vram_usage` (sum of ALL GPUs) against one GPU's `vram_capacity`, creating a phantom deficit that triggers constant unnecessary eviction. Result: 2-GPU is **slower** than 1-GPU (79s vs 56s), while no-aimdo 2-GPU runs at 37s.

## Benchmark Baseline (2×RTX 4090, Qwen-Image 38GB, CFG=7, 20 steps)
| Config | Time | vs 1-GPU | Status |
|--------|------|----------|--------|
| 1-GPU + aimdo | 56.39s | 1.00× | ✅ stable |
| 2-GPU + aimdo (original) | — | — | 💥 segfault |
| 2-GPU + aimdo (mutex only) | 95.39s | 0.59× | ✅ stable |
| 2-GPU + aimdo (dev-aware eviction) | 78.85s | 0.72× | ✅ stable |
| 2-GPU no aimdo | 37.32s | 1.51× | ❌ CUDA errors after ~2 runs |
| **2-GPU + aimdo (Phases 1–5)** | **41.29s** | **1.37×** | ✅ stable (7/7 runs) |

## Branch
`multigpu-thread-safety` — contains mutex + device-aware eviction (current stable baseline).

---

## Phase 1: Per-Device State Object
**Status:** ✅ Done

Create `AimdoDeviceState` struct with per-device fields currently stored as globals:
```c
typedef struct {
    bool inited;
    uint64_t vram_capacity;
    CUcontext ctx;
    uint64_t usage_last_check;
    ssize_t deficit_sync;
    uint64_t last_check_tick;
    const char *prevailing_method;
} AimdoDeviceState;

extern AimdoDeviceState g_dev[AIMDO_MAX_DEVICES];
```

Changes:
- **`control.c`**: Define `g_dev[]`. Split `init()` into global-once + per-device init. Add `ensure_device_init(device)` for lazy init.
- **`plat.h`**: Declare the struct and extern. Keep `total_vram_usage` for diagnostics only.
- **`model-vbar.c`**: Call `ensure_device_init(mv->device)` in `vbar_allocate()`.
- **`vbar_allocate()`**: Use `g_dev[device].vram_capacity` instead of global `vram_capacity` for VBAR sizing.

Notes:
- ComfyUI currently calls `init_device(device_0_index)` once. We don't need to change the Python API — lazy init handles other devices.
- Windows: store WDDM adapter/node per device (future — only if testing on multi-GPU Windows).

---

## Phase 2: Context Safety
**Status:** ✅ Done

Add save/restore context helpers:
```c
bool with_device_ctx(int device, CUcontext *prev);
void restore_ctx(CUcontext prev);
```

Wrap all context-sensitive CUDA calls:
- `cuMemGetInfo` in `poll_budget_deficit` / `cuda_budget_deficit`
- `cuCtxSynchronize` in `vbars_free_locked_dev`, `vbars_free_for_vbar`, `vbar_fault`, `vbar_unpin`, `vbar_free`, `vbar_free_memory`
- Fix Linux `ensure_ctx()` to not override pytorch's context when it's already set for device 1

Key principle: **never leave a different context active than what the caller had**.

---

## Phase 3: Per-Device Hybrid Budget
**Status:** ✅ Done

Replace global `budget_deficit()` in `vbar_fault` with per-device version using `AimdoDeviceState`:
```c
size_t budget_deficit_dev(size_t size, int device) {
    AimdoDeviceState *s = &g_dev[device];
    uint64_t usage = dev_vram_usage[device];
    poll_budget_deficit_dev(device);  // cuMemGetInfo with correct ctx
    ssize_t simple = (ssize_t)(usage + HEADROOM + size) - (ssize_t)s->vram_capacity;
    ssize_t delta = s->deficit_sync + (ssize_t)usage - (ssize_t)s->usage_last_check + size;
    return (size_t)MAX(MAX(simple, delta), 0);
}
```

**Critical**: Must keep the `cuMemGetInfo` backstop (hybrid approach). Pure accounting OOM'd in testing because `cudaFreeAsync` decrements counters before memory is actually reusable.

Also make `poll_budget_deficit_dev(device)` — calls `cuMemGetInfo` with the correct per-device context (depends on Phase 2).

---

## Phase 4: Allocator Hooks Device-Aware
**Status:** ✅ Done

In `aimdo_cuda_malloc` / `aimdo_cuda_malloc_async`:
- Determine device via `current_cuda_device()`
- Call `ensure_device_init(dev)`
- Use `vbars_free_dev(budget_deficit_dev(size, dev), dev)` instead of global
- OOM retry path: also evict from same device only

Expose `vbars_free_dev()` as the device-filtered wrapper (we already have `vbars_free_locked_dev` internally).

---

## Phase 5: Counter Thread Safety
**Status:** ✅ Done

`dev_vram_add/sub` run under different locks or no lock. Fix with atomics:
```c
static inline void dev_vram_add(int device, size_t size) {
    __atomic_add_fetch(&total_vram_usage, size, __ATOMIC_RELAXED);
    if (device >= 0 && device < AIMDO_MAX_DEVICES)
        __atomic_add_fetch(&dev_vram_usage[device], size, __ATOMIC_RELAXED);
}
```
Windows equivalent: `InterlockedExchangeAdd64`.

---

## Phase 6 (Future): Per-Device Locking
**Status:** ✅ Done (partial) — moved budget_deficit_dev outside vbar_lock, removed inner per-page deficit check. Result: no measurable improvement (41.47s vs 41.29s). Lock contention is not the bottleneck — the ~4s gap to no-aimdo is inherent VMM overhead (cuMemCreate/Map/SetAccess per page).

The global `vbar_lock` serializes both GPUs. `cuCtxSynchronize()` is called while holding it, blocking the other GPU's `vbar_fault`.

Options:
- Per-device lock for device-local operations
- Global lock only for list mutations (insert/remove)
- Avoid `cuCtxSynchronize` on hot path if possible

---

## Implementation Order
Phase 1 → 2 → 3 → 4 → 5 → **benchmark** → decide Phase 6

## Expected Outcome
Each GPU sees its own ~20GB usage vs 24GB capacity. Phantom deficit eliminated. Should approach the no-aimdo 37s target while remaining crash-free.

## Key Learnings from Earlier Attempts
1. **Can't remove `budget_deficit` pre-check entirely** — aimdo consuming too much VRAM causes `cudaErrorLaunchFailure` crash in pytorch's async allocator.
2. **Pure per-device accounting (without cuMemGetInfo backstop) causes OOM** — `cudaFreeAsync` decrements counters before memory is actually freed, so accounting under-reports real usage.
3. **No-aimdo 2-GPU is unstable** — `cudaErrorInvalidValue` after ~2 runs, even with async offload disabled. Aimdo provides stability that pytorch's default offloading doesn't.
