#include "plat.h"
#include "aimdo-time.h"

uint64_t vram_capacity;
uint64_t total_vram_usage;
uint64_t total_vram_last_check;
uint64_t dev_vram_usage[AIMDO_MAX_DEVICES];
ssize_t deficit_sync;
const char *prevailing_deficit_method;
CUcontext aimdo_cuda_ctx;

/* Phase 1: per-device state */
AimdoDeviceState g_dev[AIMDO_MAX_DEVICES];

bool cuda_budget_deficit() {
    uint64_t now = GET_TICK();
    static uint64_t last_check = 0;
    size_t free_vram = 0;
    size_t total_vram = 0;

    if (now - last_check < 2000) {
        return true;
    }
    last_check = now;
    total_vram_last_check = total_vram_usage;
    if (!CHECK_CU(cuMemGetInfo(&free_vram, &total_vram))) {
        return false;
    }
    deficit_sync = (ssize_t)VRAM_HEADROOM - (ssize_t)free_vram;
    prevailing_deficit_method = "cuMemGetInfo";
    return true;
}

/* Phase 3: per-device cuMemGetInfo with context switching (Phase 2) */
bool poll_budget_deficit_dev(int device) {
    if (device < 0 || device >= AIMDO_MAX_DEVICES || !g_dev[device].inited)
        return false;

    AimdoDeviceState *s = &g_dev[device];
    uint64_t now = GET_TICK();

    if (now - s->last_check_tick < 2000) {
        return true;
    }

    CUcontext prev;
    if (!with_device_ctx(device, &prev))
        return false;

    size_t free_vram = 0, total_vram = 0;
    bool ok = CHECK_CU(cuMemGetInfo(&free_vram, &total_vram));

    restore_ctx(prev);

    if (!ok)
        return false;

    s->last_check_tick = now;
    s->usage_last_check = dev_vram_usage[device];
    s->deficit_sync = (ssize_t)VRAM_HEADROOM - (ssize_t)free_vram;
    s->prevailing_method = "cuMemGetInfo (per-dev)";
    return true;
}

/* Phase 1: lazy per-device init */
void ensure_device_init(int device) {
    if (device < 0 || device >= AIMDO_MAX_DEVICES)
        return;

    AimdoDeviceState *s = &g_dev[device];
    if (s->inited)
        return;

    CUdevice dev;
    if (!CHECK_CU(cuDeviceGet(&dev, device)))
        return;

    uint64_t cap = 0;
    if (!CHECK_CU(cuDeviceTotalMem(&cap, dev)))
        return;

    CUcontext ctx = NULL;
    if (!CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev)))
        return;

    s->vram_capacity = cap;
    s->ctx = ctx;
    s->prevailing_method = "none";
    s->inited = true;

    char dev_name[256];
    if (!CHECK_CU(cuDeviceGetName(dev_name, sizeof(dev_name), dev)))
        sprintf(dev_name, "<unknown>");

    log(INFO, "comfy-aimdo device %d init: %s (VRAM: %zu MB)\n",
        device, dev_name, (size_t)(cap / (1024 * 1024)));
}

SHARED_EXPORT
void aimdo_analyze() {
    size_t free_bytes = 0, total_bytes = 0;

    log(DEBUG, "--- VRAM Stats ---\n");

    CHECK_CU(cuMemGetInfo(&free_bytes, &total_bytes));
    log(DEBUG, "  Aimdo Recorded Usage:  %7zu MB\n", total_vram_usage / M);
    log(DEBUG, "  Cuda:  %7zu MB / %7zu MB Free\n", free_bytes / M, total_bytes / M);

    for (int i = 0; i < AIMDO_MAX_DEVICES; i++) {
        if (dev_vram_usage[i])
            log(DEBUG, "  Device %d Usage:  %7zu MB (cap %zu MB)\n",
                i, (size_t)(dev_vram_usage[i] / M),
                g_dev[i].inited ? (size_t)(g_dev[i].vram_capacity / M) : 0);
    }

    vbars_analyze(true);
    allocations_analyze();
}

SHARED_EXPORT
uint64_t get_total_vram_usage() {
    return total_vram_usage;
}

SHARED_EXPORT
bool init(int cuda_device_id) {
    CUdevice dev;
    char dev_name[256];

    log_reset_shots();

    if (!CHECK_CU(cuDeviceGet(&dev, cuda_device_id)) ||
        !CHECK_CU(cuDeviceTotalMem(&vram_capacity, dev)) ||
        !CHECK_CU(cuDevicePrimaryCtxRetain(&aimdo_cuda_ctx, dev)) ||
        !CHECK_CU(cuCtxSetCurrent(aimdo_cuda_ctx)) ||
        !plat_init(dev)) {
        return false;
    }

    if (!CHECK_CU(cuDeviceGetName(dev_name, sizeof(dev_name), dev))) {
        sprintf(dev_name, "<unknown>");
    }

    /* Also populate g_dev for the primary device */
    ensure_device_init(cuda_device_id);

    log(INFO, "comfy-aimdo inited for GPU: %s (VRAM: %zu MB)\n",
        dev_name, (size_t)(vram_capacity / (1024 * 1024)));
    return true;
}

SHARED_EXPORT
void cleanup() {
    plat_cleanup();
}
