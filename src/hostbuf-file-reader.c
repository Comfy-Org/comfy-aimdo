#include "plat.h"
#include "xfer-file.h"

#define HOSTBUF_FILE_READER_WINDOW (64ULL * 1024ULL * 1024ULL)
#define LEAD_IN_THRESHOLD (HOSTBUF_FILE_READER_WINDOW - 16ULL * 1024ULL * 1024ULL)

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

    if (!slot->buffer &&
        !CHECK_CU(cuMemAllocHost((void **)&slot->buffer, HOSTBUF_FILE_READER_WINDOW))) {
        return NULL;
    }

    slot->offset = 0;
    slot->stream = (CUstream)stream;
    return slot;
}

SHARED_EXPORT
bool hostbuf_file_reader_read(int device, uint64_t file_handle, uint64_t file_offset,
                              uint64_t size, cudaStream_t stream,
                              uint64_t device_ptr, bool mark_cold) {
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
        if (!xfer_file_read(file_handle, file_offset, slot->buffer + slot->offset,
                            chunk, mark_cold) ||
            !CHECK_CU(cuMemcpyHtoDAsync((CUdeviceptr)device_ptr,
                                        slot->buffer + slot->offset,
                                        chunk, (CUstream)stream))) {
            return false;
        }

        slot->offset += chunk;
        file_offset += chunk;
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
        if (slot->buffer) {
            CHECK_CU(cuMemFreeHost(slot->buffer));
        }
    }
    memset(g_devctx->_hostbuf_file_reader_slots, 0,
           sizeof(g_devctx->_hostbuf_file_reader_slots));
    g_devctx->_hostbuf_file_reader_active = -1;
}
