#include "vrambuf.h"

#define VRAM_CHUNK_SIZE      (16ULL * 1024 * 1024)

SHARED_EXPORT
void *vrambuf_create(int device, size_t max_size) {
    VramBuffer *buf;

    buf = (VramBuffer *)calloc(1, sizeof(*buf) + sizeof(CUmemGenericAllocationHandle) * max_size / VRAM_CHUNK_SIZE);
    if (!buf) {
        return NULL;
    }
    buf->device = device;
    buf->max_size = max_size;

    if (!CHECK_CU(cuMemAddressReserve(&buf->base_ptr, max_size, 0, 0, 0))) {
        free(buf);
        return NULL;
    }

    return (void *)buf;
}

SHARED_EXPORT
bool vrambuf_grow(void *arg, size_t required_size) {
    VramBuffer *buf = (VramBuffer *)arg;
    size_t grow_to;
    CUmemGenericAllocationHandle handle;
    CUresult err;

    if (!buf) {
        return false;
    }
    if (required_size > buf->max_size) {
        return false;
    }
    if (required_size <= buf->allocated) {
        return true;
    }

    grow_to = (required_size + VRAM_CHUNK_SIZE - 1) & ~(VRAM_CHUNK_SIZE - 1);
    if (grow_to > buf->max_size) {
        grow_to = buf->max_size;
    }

    vbars_free(wddm_budget_deficit(buf->device, grow_to - buf->allocated));
    while (buf->allocated < grow_to) {
        size_t to_allocate = grow_to - buf->allocated;
        if (to_allocate > VRAM_CHUNK_SIZE) {
            to_allocate = VRAM_CHUNK_SIZE;
        }
        if ((err = three_stooges(buf->base_ptr + buf->allocated, to_allocate, buf->device, &handle)) != CUDA_SUCCESS) {
            if (err != CUDA_ERROR_OUT_OF_MEMORY) {
                log(ERROR, "VRAM Allocation failed (non OOM)\n");
                return false;
            }
            log(DEBUG, "Pytorch allocator attempt exceeds available VRAM ...\n");
            vbars_free(VRAM_CHUNK_SIZE);
            if ((err = three_stooges(buf->base_ptr + buf->allocated, to_allocate, buf->device, &handle)) != CUDA_SUCCESS) {
                bool is_oom = err == CUDA_ERROR_OUT_OF_MEMORY;
                log(is_oom ? INFO : ERROR, "VRAM Allocation failed (%s)\n", is_oom ? "OOM" : "error");
                return false;
            }
        }

        buf->handles[buf->handle_count++] = handle;
        log(VVERBOSE, "%s: allocate vrambuf chunk @%llx, size %zuk\n", buf->base_ptr + buf->allocated, to_allocate / K);
        buf->allocated += to_allocate;
    }

    return true;
}

SHARED_EXPORT
CUdeviceptr vrambuf_get(void *arg) {
    VramBuffer *buf = (VramBuffer *)arg;

    if (!buf) {
        return 0;
    }
    return buf->base_ptr;
}

SHARED_EXPORT
void vrambuf_destroy(void *arg) {
    VramBuffer *buf = (VramBuffer *)arg;
    size_t i;

    if (!buf) {
        return;
    }

    if (buf->allocated > 0) {
        CHECK_CU(cuMemUnmap(buf->base_ptr, buf->allocated));
    }

    for (i = 0; i < buf->handle_count; i++) {
        CHECK_CU(cuMemRelease(buf->handles[i]));
    }

    CHECK_CU(cuMemAddressFree(buf->base_ptr, buf->max_size));
    total_vram_usage -= buf->allocated;
    free(buf);
}
