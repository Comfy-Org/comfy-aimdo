#include "plat.h"
#include "xfer-file.h"
#include "hostbuf-file-reader.h"

#define HOSTBUF_FILE_READER_WINDOW (128ULL * 1024ULL * 1024ULL)
#define LEAD_IN_THRESHOLD (HOSTBUF_FILE_READER_WINDOW - 32ULL * 1024ULL * 1024ULL)

static bool hostbuf_file_reader_retire_active(void) {
    HostbufFileReaderSlot *slot;

    if (g_devctx->_hostbuf_file_reader_active < 0) {
        return true;
    }

    slot = &g_devctx->_hostbuf_file_reader_slots[g_devctx->_hostbuf_file_reader_active];
    return !slot->offset ||
           (!slot->event &&
           CHECK_CU(cuEventCreate(&slot->event, CU_EVENT_DISABLE_TIMING)) &&
           CHECK_CU(cuEventRecord(slot->event, (CUstream)slot->stream)));
}

static HostbufFileReaderSlot *hostbuf_file_reader_next(cudaStream_t stream) {
    HostbufFileReaderSlot *slot;

    g_devctx->_hostbuf_file_reader_active =
        (g_devctx->_hostbuf_file_reader_active + 1) % HOSTBUF_FILE_READER_SLOTS;
    slot = &g_devctx->_hostbuf_file_reader_slots[g_devctx->_hostbuf_file_reader_active];

    if (slot->buffer && slot->event) {
        if (!CHECK_CU(cuEventSynchronize(slot->event)) ||
            !CHECK_CU(cuEventDestroy(slot->event))) {
            return NULL;
        }
        slot->event = NULL;
    }
    xfer_copy_group_wait(&slot->copy_group);
    if (!slot->buffer &&
        !CHECK_CU(cuMemAllocHost((void **)&slot->buffer, HOSTBUF_FILE_READER_WINDOW))) {
        return NULL;
    }

    slot->offset = 0;
    slot->stream = (CUstream)stream;
    return slot;
}

static bool hostbuf_file_reader_read_impl(int device, uint64_t file_handle,
                                          uint64_t file_offset, uint64_t size,
                                          cudaStream_t stream, uint64_t device_ptr,
                                          void *cache_ptr, bool mark_cold) {
    if (size == 0) {
        return true;
    }
    if (!device_ptr || device < 0 || !set_devctx_for_device(device)) {
        return false;
    }

    while (size) {
        HostbufFileReaderSlot *slot = g_devctx->_hostbuf_file_reader_active < 0 ? NULL :
            &g_devctx->_hostbuf_file_reader_slots[g_devctx->_hostbuf_file_reader_active];
        size_t chunk;

        if (!slot || slot->stream != (CUstream)stream ||
            (slot->offset + size >= HOSTBUF_FILE_READER_WINDOW &&
             slot->offset >= LEAD_IN_THRESHOLD)) {
            if (!hostbuf_file_reader_retire_active() ||
                !(slot = hostbuf_file_reader_next(stream))) {
                return false;
            }
        }

        chunk = (size_t)MIN(size, HOSTBUF_FILE_READER_WINDOW - slot->offset);
        if (!xfer_file_read(file_handle, file_offset,
                            slot->buffer + slot->offset, chunk, mark_cold)) {
            return false;
        }
        if (!CHECK_CU(cuMemcpyHtoDAsync((CUdeviceptr)device_ptr,
                                        slot->buffer + slot->offset,
                                        chunk, (CUstream)stream))) {
            return false;
        }
        if (cache_ptr) {
            if (!slot->copy_group &&
                !(slot->copy_group = xfer_copy_group_create())) {
                return false;
            }
            if (!xfer_copy_group_add(slot->copy_group,
                                     slot->buffer + slot->offset,
                                     cache_ptr, chunk)) {
                return false;
            }
        }

        slot->offset += chunk;
        file_offset += chunk;
        device_ptr += chunk;
        if (cache_ptr) {
            cache_ptr = (char *)cache_ptr + chunk;
        }
        size -= chunk;
    }

    return true;
}

SHARED_EXPORT
bool hostbuf_file_reader_read(int device, uint64_t file_handle, uint64_t file_offset,
                              uint64_t size, cudaStream_t stream,
                              uint64_t device_ptr, bool mark_cold) {
    return hostbuf_file_reader_read_impl(device, file_handle, file_offset, size,
                                         stream, device_ptr, NULL, mark_cold);
}

bool hostbuf_file_reader_read_cached(int device, uint64_t file_handle,
                                     uint64_t file_offset, uint64_t size,
                                     cudaStream_t stream, uint64_t device_ptr,
                                     void *cache_ptr, bool mark_cold) {
    if (!cache_ptr) {
        return false;
    }
    return hostbuf_file_reader_read_impl(device, file_handle, file_offset, size,
                                         stream, device_ptr, cache_ptr, mark_cold);
}

bool hostbuf_file_reader_copy_cached(int device, const void *source, uint64_t size,
                                     cudaStream_t stream, uint64_t device_ptr) {
    if (!source || !device_ptr || device < 0 || !set_devctx_for_device(device)) {
        return false;
    }

    while (size) {
        HostbufFileReaderSlot *slot = g_devctx->_hostbuf_file_reader_active < 0 ? NULL :
            &g_devctx->_hostbuf_file_reader_slots[g_devctx->_hostbuf_file_reader_active];
        size_t chunk;

        if (!slot || slot->stream != (CUstream)stream ||
            (slot->offset + size >= HOSTBUF_FILE_READER_WINDOW &&
             slot->offset >= LEAD_IN_THRESHOLD)) {
            if (!hostbuf_file_reader_retire_active() ||
                !(slot = hostbuf_file_reader_next(stream))) {
                return false;
            }
        }

        chunk = (size_t)MIN(size, HOSTBUF_FILE_READER_WINDOW - slot->offset);
        if (chunk >= 8 * 1024 * 1024) {
            if (!slot->copy_group &&
                !(slot->copy_group = xfer_copy_group_create())) {
                return false;
            }
            if (!xfer_copy_group_add_parallel(slot->copy_group, source,
                                              slot->buffer + slot->offset,
                                              chunk)) {
                return false;
            }
            xfer_copy_group_sync(slot->copy_group);
        } else {
            memcpy(slot->buffer + slot->offset, source, chunk);
        }
        if (!CHECK_CU(cuMemcpyHtoDAsync((CUdeviceptr)device_ptr,
                                        slot->buffer + slot->offset,
                                        chunk, (CUstream)stream))) {
            return false;
        }

        slot->offset += chunk;
        source = (const char *)source + chunk;
        device_ptr += chunk;
        size -= chunk;
    }
    return true;
}

SHARED_EXPORT
void hostbuf_file_reader_cleanup(void) {
    if (!g_devctx) {
        return;
    }

    hostbuf_file_reader_retire_active();
    for (unsigned i = 0; i < HOSTBUF_FILE_READER_SLOTS; i++) {
        HostbufFileReaderSlot *slot = &g_devctx->_hostbuf_file_reader_slots[i];

        if (slot->buffer && slot->event) {
            CHECK_CU(cuEventSynchronize(slot->event));
            CHECK_CU(cuEventDestroy(slot->event));
        }
        xfer_copy_group_wait(&slot->copy_group);
        if (slot->buffer) {
            CHECK_CU(cuMemFreeHost(slot->buffer));
        }
    }
    memset(g_devctx->_hostbuf_file_reader_slots, 0,
           sizeof(g_devctx->_hostbuf_file_reader_slots));
    g_devctx->_hostbuf_file_reader_active = -1;
}
