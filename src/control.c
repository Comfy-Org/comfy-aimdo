#include "plat.h"
#include "aimdo-time.h"

uint64_t vram_capacity;
uint64_t total_vram_usage;
uint64_t total_vram_last_check;
ssize_t deficit_sync;
const char *prevailing_deficit_method;

int64_t total_pin_usage;

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
    deficit_sync = (ssize_t)VRAM_HEADROOM - free_vram;
    prevailing_deficit_method = "cuMemGetInfo";
    return true;
}

SHARED_EXPORT
void aimdo_analyze() {
    size_t free_bytes = 0, total_bytes = 0;

    log(DEBUG, "--- VRAM Stats ---\n");

    CHECK_CU(cuMemGetInfo(&free_bytes, &total_bytes));
    log(DEBUG, "  Aimdo Recorded Usage:  %7zu MB\n", total_vram_usage / M);
    log(DEBUG, "  Cuda:  %7zu MB / %7zu MB Free\n", free_bytes / M, total_bytes / M);

    vbars_analyze();
    allocations_analyze();
}

SHARED_EXPORT
uint64_t get_total_vram_usage() {
    return total_vram_usage;
}

SHARED_EXPORT
void notify_pin(int64_t size) {
    total_pin_usage += size;
    log(VERBOSE, "%s: +%zuk -> %zu\n", __func__, size / K, total_pin_usage / M);
}

SHARED_EXPORT
bool init(int cuda_device_id) {
    CUdevice dev;
    char dev_name[256];

    total_pin_usage = 0;
    log_reset_shots();

    if (!CHECK_CU(cuDeviceGet(&dev, cuda_device_id)) ||
        !CHECK_CU(cuDeviceTotalMem(&vram_capacity, dev)) ||
        !plat_init(dev)) {
        return false;
    }

    if (!CHECK_CU(cuDeviceGetName(dev_name, sizeof(dev_name), dev))) {
        sprintf(dev_name, "<unknown>");
    }

    log(INFO, "comfy-aimdo inited for GPU: %s (VRAM: %zu MB)\n",
        dev_name, (size_t)(vram_capacity / (1024 * 1024)));
    return true;
}

SHARED_EXPORT
void cleanup() {
    plat_cleanup();
}
