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
    if (!slot->offset) {
        return true;
    }
    if (slot->event) {
        log(AIMDO_LOG_ERROR, "%s: active slot %d already has a completion event\n", __func__,
            g_devctx->_hostbuf_file_reader_active);
        return false;
    }
    return CHECK_CU(cuEventCreate(&slot->event, CU_EVENT_DISABLE_TIMING)) &&
           CHECK_CU(cuEventRecord(slot->event, (CUstream)slot->stream));
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
        !CHECK_CU_OOM_ERROR(cuMemAllocHost((void **)&slot->buffer, HOSTBUF_FILE_READER_WINDOW))) {
        return NULL;
    }

    slot->offset = 0;
    slot->stream = (CUstream)stream;
    return slot;
}

static bool hostbuf_file_transfer(int device, uint64_t file_handle, uint64_t file_offset,
                                  uint64_t size, cudaStream_t stream,
                                  uint64_t device_ptr, bool mark_cold, bool write) {
    if (size == 0) {
        return true;
    }
    if (!device_ptr || device < 0 || !set_devctx_for_device(device)) {
        log(AIMDO_LOG_ERROR, "%s: input validation failed device_ptr=%p device=%d\n",
            __func__, (void *)(uintptr_t)device_ptr, device);
        return false;
    }

    while (size) {
        HostbufFileReaderSlot *slot = g_devctx->_hostbuf_file_reader_active < 0 ? NULL :
            &g_devctx->_hostbuf_file_reader_slots[g_devctx->_hostbuf_file_reader_active];

        if (!slot || slot->stream != (CUstream)stream ||
            (slot->offset + size >= HOSTBUF_FILE_READER_WINDOW &&
             slot->offset >= LEAD_IN_THRESHOLD)) {
            if (!hostbuf_file_reader_retire_active() ||
                !(slot = hostbuf_file_reader_next(stream))) {
                return false;
            }
        }

        size_t chunk = (size_t)MIN(size, HOSTBUF_FILE_READER_WINDOW - slot->offset);
        void *buffer = slot->buffer + slot->offset;
        slot->offset += chunk;

        if (!write && !xfer_file_read(file_handle, file_offset, buffer, chunk, mark_cold)) {
            log(AIMDO_LOG_ERROR, "%s: xfer_file_read failed handle=0x%llx offset=%llu size=%zu\n",
                __func__, (ull)file_handle, (ull)file_offset, chunk);
            return false;
        }

        CUresult copy_result = write ? cuMemcpyDtoHAsync(buffer, (CUdeviceptr)device_ptr, chunk, (CUstream)stream) :
                                      cuMemcpyHtoDAsync((CUdeviceptr)device_ptr, buffer, chunk, (CUstream)stream);
        if (!CHECK_CU(copy_result)) {
            log(AIMDO_LOG_ERROR, "%s: %s failed result=%d device_ptr=%p device=%d stream=%p size=%zu\n",
                __func__, write ? "cuMemcpyDtoHAsync" : "cuMemcpyHtoDAsync", (int)copy_result,
                (void *)(uintptr_t)device_ptr, device, (void *)stream, chunk);
            return false;
        }

        if (write) {
            CUresult sync_result = cuCtxSynchronize();
            if (!CHECK_CU(sync_result)) {
                log(AIMDO_LOG_ERROR, "%s: cuCtxSynchronize failed result=%d device_ptr=%p device=%d stream=%p size=%zu\n",
                    __func__, (int)sync_result, (void *)(uintptr_t)device_ptr, device, (void *)stream, chunk);
                return false;
            }
            if (!xfer_file_write(file_handle, file_offset, buffer, chunk)) {
                log(AIMDO_LOG_ERROR, "%s: xfer_file_write failed handle=0x%llx offset=%llu size=%zu\n",
                    __func__, (ull)file_handle, (ull)file_offset, chunk);
                return false;
            }
        }

        file_offset += chunk;
        device_ptr += chunk;
        size -= chunk;
    }

    return true;
}

SHARED_EXPORT
bool hostbuf_file_reader_read(int device, uint64_t file_handle, uint64_t file_offset,
                              uint64_t size, cudaStream_t stream,
                              uint64_t device_ptr, bool mark_cold) {
    return hostbuf_file_transfer(device, file_handle, file_offset, size, stream, device_ptr, mark_cold, false);
}

SHARED_EXPORT
bool hostbuf_file_writer_write(int device, uint64_t file_handle, uint64_t file_offset,
                               uint64_t size, cudaStream_t stream, uint64_t device_ptr) {
    return hostbuf_file_transfer(device, file_handle, file_offset, size, stream, device_ptr, false, true);
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
