#include "plat.h"
#include "hostbuf-plat.h"
#include "hostbuf-prewarm.h"
#include "xfer-file.h"

#define HOSTBUF_RESERVE_SIZE (128ULL * G)

typedef struct HostBuffer {
    void *base_address;
    uint64_t size;
    uint64_t committed_size;
    uint64_t reserved_size;
    uint64_t last_chunk_size;
    uint64_t prewarm;
} HostBuffer;

static bool hostbuf_grow(HostBuffer *hostbuf, uint64_t size) {
    size_t page_size = hostbuf_page_size();
    uint64_t target_committed = ALIGN_UP(size + hostbuf->prewarm, page_size);

    size_t tail_size;
    size_t prewarm_start;

    if (size <= hostbuf->size) {
        return true;
    }

    if (target_committed < hostbuf->committed_size + hostbuf->prewarm / 4) {
        /* Not worth it. Limit the prewarm thread launching and avoid syncing
         * previous large prewarm launches for smaller allocations.
         */
        target_committed = hostbuf->committed_size;
    }

    if (!hostbuf->base_address) {
        hostbuf->reserved_size = HOSTBUF_RESERVE_SIZE;
        hostbuf->base_address = hostbuf_reserve_address_space((size_t)hostbuf->reserved_size);
        if (!hostbuf->base_address) {
            return false;
        }
    }
    if (target_committed > hostbuf->reserved_size) {
        log(ERROR, "%s: requested %llu bytes beyond reserved host buffer %llu\n", __func__,
            (ull)target_committed, (ull)hostbuf->reserved_size);
        return false;
    }
    tail_size = (size_t)(target_committed - hostbuf->committed_size);
    prewarm_start = MAX(hostbuf->committed_size, size);

    if (tail_size &&
        !hostbuf_commit_address_space((char *)hostbuf->base_address + hostbuf->committed_size, tail_size)) {
        return false;
    }
    if (!CHECK_CU(cuMemHostRegister((char *)hostbuf->base_address + hostbuf->size,
                                    (size_t)(size - hostbuf->size), 0))) {
        goto fail_decommit;
    }
    if (tail_size && !hostbuf_prewarm_start((char *)hostbuf->base_address + prewarm_start,
                                            target_committed - prewarm_start)) {
        goto fail_unregister;
    }
    hostbuf->committed_size = target_committed;

    hostbuf->size = size;
    return true;

fail_unregister:
    CHECK_CU(cuMemHostUnregister((char *)hostbuf->base_address + hostbuf->size));
fail_decommit:
    if (tail_size) {
        hostbuf_decommit_address_space((char *)hostbuf->base_address + hostbuf->committed_size, tail_size);
    }
    return false;
}

static bool hostbuf_truncate_impl(HostBuffer *hostbuf, uint64_t size) {
    size_t page_size = hostbuf_page_size();
    uint64_t old_committed = hostbuf->committed_size;
    uint64_t new_committed = ALIGN_UP(size, page_size);

    if (size >= hostbuf->size ||
        !hostbuf_prewarm_start(NULL, 0) ||
        !CHECK_CU(cuMemHostUnregister((char *)hostbuf->base_address + size)) ||
        (new_committed < old_committed &&
        !hostbuf_decommit_address_space((char *)hostbuf->base_address + new_committed,
                                        (size_t)(old_committed - new_committed)))) {
        return false;
    }
    if (new_committed < old_committed) {
        old_committed = new_committed;
    }

    hostbuf->size = size;
    hostbuf->committed_size = old_committed;
    if (!size && hostbuf->base_address) {
        hostbuf_release_address_space(hostbuf->base_address, (size_t)hostbuf->reserved_size);
        hostbuf->base_address = NULL;
        hostbuf->committed_size = 0;
        hostbuf->reserved_size = 0;
    }
    return true;
}

SHARED_EXPORT
void *hostbuf_allocate(uint64_t prewarm) {
    HostBuffer *hostbuf = calloc(1, sizeof(*hostbuf));

    if (!hostbuf) {
        return NULL;
    }
    hostbuf->prewarm = prewarm;
    return hostbuf;
}

SHARED_EXPORT
void hostbuf_free(void *hostbuf_ptr) {
    HostBuffer *hostbuf = (HostBuffer *)hostbuf_ptr;

    if (!hostbuf) {
        return;
    }
    hostbuf_truncate_impl(hostbuf, 0);
    free(hostbuf);
}

SHARED_EXPORT
void *hostbuf_get_raw_address(void *hostbuf_ptr, uint64_t size, uint64_t offset) {
    HostBuffer *hostbuf = (HostBuffer *)hostbuf_ptr;
    void *ptr = NULL;

    if (!hostbuf || offset > UINT64_MAX - size) {
        return NULL;
    }
    if (hostbuf_grow(hostbuf, offset + size)) {
        ptr = (char *)hostbuf->base_address + offset;
    }
    return ptr;
}

SHARED_EXPORT
void *hostbuf_extend(void *hostbuf_ptr, uint64_t size, bool reallocate, int64_t *size_delta) {
    HostBuffer *hostbuf = (HostBuffer *)hostbuf_ptr;
    uint64_t old_size;
    uint64_t offset;

    if (!hostbuf) {
        return NULL;
    }
    old_size = hostbuf->size;
    *size_delta = 0;

    if (reallocate && hostbuf->last_chunk_size) {
        offset = hostbuf->size - hostbuf->last_chunk_size;
        if (!CHECK_CU(cuMemHostUnregister((char *)hostbuf->base_address + offset))) {
            return NULL;
        }
        hostbuf->size = offset;
    }

    offset = hostbuf->size;
    if (!hostbuf_grow(hostbuf, offset + size)) {
        *size_delta = (int64_t)(hostbuf->size - old_size);
        return NULL;
    }
    hostbuf->last_chunk_size = hostbuf->size - offset;
    *size_delta = (int64_t)(hostbuf->size - old_size);
    return (char *)hostbuf->base_address + offset;
}

SHARED_EXPORT
bool hostbuf_read_file_slice(void *hostbuf_ptr, uint64_t file_handle, uint64_t file_offset,
                             uint64_t size, uint64_t offset) {
    void *ptr = hostbuf_get_raw_address(hostbuf_ptr, size, offset);

    return size == 0 || (ptr && xfer_file_read(file_handle, file_offset, ptr, (size_t)size));
}

SHARED_EXPORT
bool hostbuf_truncate(void *hostbuf_ptr, uint64_t size) {
    HostBuffer *hostbuf = (HostBuffer *)hostbuf_ptr;

    if (!hostbuf) {
        return false;
    }
    return hostbuf_truncate_impl(hostbuf, size);
}
