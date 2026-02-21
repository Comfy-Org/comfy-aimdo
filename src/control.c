#include "plat.h"

uint64_t vram_capacity;
uint64_t total_vram_usage;

#define VRAM_HEADROOM (256 * 1024 * 1024)

size_t cuda_budget_deficit(int device, size_t bytes) {
    size_t free_vram = 0, total_vram = 0;

    ssize_t deficit = (ssize_t)(total_vram_usage + bytes + VRAM_HEADROOM) - vram_capacity;

    if (CHECK_CU(cuMemGetInfo(&free_vram, &total_vram))) {
        ssize_t deficit_cuda = (ssize_t)(VRAM_HEADROOM + bytes) - free_vram;

        if (deficit_cuda > deficit) {
            deficit = deficit_cuda;
        }
    }

    if (deficit > 0) {
        log(DEBUG, "Imminent VRAM OOM detected. Deficit: %zd MB\n", deficit / (1024 * 1024));
    }

    return (deficit > 0) ? (size_t)deficit : 0;
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
bool init(int cuda_device_id) {
    CUdevice dev;
    char dev_name[256];

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
