#include <cuda.h>
#include <stdbool.h>
#include <stdlib.h>
#include "plat.h"

#define MAX_VRAM_RESERVATION (8ULL * 1024 * 1024 * 1024)
#define VRAM_CHUNK_SIZE      (16ULL * 1024 * 1024)
#define MAX_HANDLES          (MAX_VRAM_RESERVATION / VRAM_CHUNK_SIZE)

typedef struct {
    CUdeviceptr base_ptr;
    size_t allocated;
    size_t handle_count;
    CUmemGenericAllocationHandle handles[MAX_HANDLES];
} VramBuffer;

SHARED_EXPORT
void *vrambuf_create() {
    VramBuffer *buf;
    
    buf = (VramBuffer *)calloc(1, sizeof(*buf));
    if (!buf) {
        return NULL;
    }

    if (!CHECK_CU(cuMemAddressReserve(&buf->base_ptr, MAX_VRAM_RESERVATION, 0, 0, 0))) {
        free(buf);
        return NULL;
    }
    
    return (void *)buf;
}

SHARED_EXPORT
bool vrambuf_grow(void *arg, size_t required_size) {
    VramBuffer *buf = (VramBuffer *)arg;
    size_t grow_to;
    size_t grow_size;
    CUmemGenericAllocationHandle handle;
    CUresult res;

    if (!buf) {
        return false;
    }
    if (required_size > MAX_VRAM_RESERVATION) {
        return false;
    }
    if (required_size <= buf->allocated) {
        return true;
    }

    grow_to = (required_size + VRAM_CHUNK_SIZE - 1) & ~(VRAM_CHUNK_SIZE - 1);
    grow_size = grow_to - buf->allocated;

    res = three_stooges(buf->base_ptr + buf->allocated, grow_size, 0, &handle);
    if (res == CUDA_SUCCESS) {
        buf->handles[buf->handle_count++] = handle;
        buf->allocated = grow_to;
        return true;
    }

    return false;
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

    CHECK_CU(cuMemAddressFree(buf->base_ptr, MAX_VRAM_RESERVATION));
    free(buf);
}
