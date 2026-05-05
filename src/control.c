#include "plat.h"
#include "aimdo-time.h"
#include "xfer-file.h"

_Thread_local AimdoContext *g_devctx;

static AimdoContext *g_all_devctxs;
static size_t g_all_devctx_count;

SHARED_EXPORT
void *get_devctx(int device_id) {
    for (size_t i = 0; i < g_all_devctx_count; i++) {
        if (g_all_devctxs[i]._device_id == device_id) {
            return &g_all_devctxs[i];
        }
    }

    return NULL;
}

bool set_devctx_for_device(int device_id) {
    for (size_t i = 0; i < g_all_devctx_count; i++) {
        if (g_all_devctxs[i]._device_id == device_id) {
            set_devctx(&g_all_devctxs[i]);
            return true;
        }
    }
    set_devctx(NULL);
    return false;
}

bool set_devctx_for_current_cuda_device(void) {
    CUdevice device;

    if (!CHECK_CU(cuCtxGetDevice(&device))) {
        set_devctx(NULL);
        return false;
    }

    return set_devctx_for_device((int)device);
}

SHARED_EXPORT
bool plat_init() {
    log_reset_shots();
    if (!aimdo_cuda_runtime_init()) {
        return false;
    }
    if (!xfer_file_init()) {
        aimdo_cuda_runtime_cleanup();
        return false;
    }
    if (aimdo_setup_hooks()) {
        return true;
    }

    xfer_file_cleanup();
    aimdo_cuda_runtime_cleanup();
    return false;
}

SHARED_EXPORT
void plat_cleanup() {
    xfer_file_cleanup();
    aimdo_teardown_hooks();
    aimdo_cuda_runtime_cleanup();
}

bool cuda_budget_deficit(const char **prevailing_deficit_method) {
    uint64_t now = GET_TICK();
    size_t free_vram = 0;
    size_t total_vram = 0;

    if (now - control_timestamp_last_check < 2000) {
        return true;
    }
    control_timestamp_last_check = now;
    total_vram_last_check = total_vram_usage;
    if (!CHECK_CU(cuMemGetInfo(&free_vram, &total_vram))) {
        return false;
    }
    deficit_sync = (ssize_t)VRAM_HEADROOM - (ssize_t)free_vram;
    log(DEBUG,
        "%s: cuMemGetInfo poll free=%zu MB total=%zu MB deficit_sync=%zd MB recorded=%zu MB\n",
        __func__, free_vram / M, total_vram / M, deficit_sync / (ssize_t)M, total_vram_usage / M);
    *prevailing_deficit_method = "cuMemGetInfo";
    log(DEBUG, "%s: prevailing method %s\n", __func__, *prevailing_deficit_method);
    return true;
}

SHARED_EXPORT
void aimdo_analyze(void *devctx) {
    size_t free_bytes = 0, total_bytes = 0;

    set_devctx((AimdoContext *)devctx);

    log(DEBUG, "--- VRAM Stats ---\n");

    CHECK_CU(cuMemGetInfo(&free_bytes, &total_bytes));
    log(DEBUG, "  Aimdo Recorded Usage:  %7zu MB\n", total_vram_usage / M);
    log(DEBUG, "  Cuda:  %7zu MB / %7zu MB Free\n", free_bytes / M, total_bytes / M);

    vbars_analyze(devctx, true);
    allocations_analyze();
}

SHARED_EXPORT
uint64_t get_total_vram_usage(void *devctx) {
    set_devctx((AimdoContext *)devctx);
    return total_vram_usage;
}

SHARED_EXPORT
void cleanup(void) {
    for (size_t i = 0; i < g_all_devctx_count; i++) {
        set_devctx(&g_all_devctxs[i]);
        aimdo_wddm_cleanup();
        allocations_cleanup();

        free(highest_priority_p); /* FIXME: move the model_vbar. */
    }

    free(g_all_devctxs);
    g_all_devctxs = NULL;
    g_all_devctx_count = 0;
    set_devctx(NULL);
}

SHARED_EXPORT
bool init(const int *cuda_device_ids, size_t num_devices) {
    size_t i;

    if (g_all_devctxs ||
        !(g_all_devctxs = calloc(num_devices, sizeof(*g_all_devctxs)))) {
        return false;
    }
    g_all_devctx_count = num_devices;

    for (i = 0; i < num_devices; i++) {
        CUdevice dev;
        char dev_name[256];
        AimdoContext *devctx = &g_all_devctxs[i];

        devctx->_device_id = cuda_device_ids[i];
        set_devctx(devctx);

        if (!allocations_init() ||
            !CHECK_CU(cuDeviceGet(&dev, cuda_device_ids[i])) ||
            !CHECK_CU(cuDeviceTotalMem(&vram_capacity, dev)) ||
            !aimdo_wddm_init(dev)) {
            goto fail;
        }

        if (!CHECK_CU(cuDeviceGetName(dev_name, sizeof(dev_name), dev))) {
            sprintf(dev_name, "<unknown>");
        }

        log(INFO, "comfy-aimdo inited for GPU: %s (VRAM: %zu MB)\n",
            dev_name, (size_t)(vram_capacity / (1024 * 1024)));
    }

    set_devctx(NULL);
    return true;

fail:
    cleanup();
    return false;
}
