#include "plat.h"
#include "xfer-file.h"

#define HOSTBUF_FILE_READER_WINDOW (64ULL * 1024ULL * 1024ULL)
#define LEAD_IN_THRESHOLD (HOSTBUF_FILE_READER_WINDOW - 16ULL * 1024ULL * 1024ULL)
#define HOSTBUF_FILE_READER_SLOTS 3

void *model_mmap_get(void *model_mmap_ptr);

typedef struct HostbufFileReaderSlot {
    uint8_t *buffer;
    uint64_t offset;
    cudaStream_t stream;
    CUevent event;
} HostbufFileReaderSlot;

static HostbufFileReaderSlot g_slots[HOSTBUF_FILE_READER_SLOTS];
static int g_active = -1;

static bool hostbuf_file_reader_retire_active(void) {
    HostbufFileReaderSlot *slot;

    if (g_active < 0) {
        return true;
    }

    slot = &g_slots[g_active];
    return !slot->offset ||
           (!slot->event &&
           CHECK_CU(cuEventCreate(&slot->event, CU_EVENT_DISABLE_TIMING)) &&
           CHECK_CU(cuEventRecord(slot->event, (CUstream)slot->stream)));
}

static HostbufFileReaderSlot *hostbuf_file_reader_next(cudaStream_t stream) {
    HostbufFileReaderSlot *slot;

    g_active = (g_active + 1) % HOSTBUF_FILE_READER_SLOTS;
    slot = &g_slots[g_active];

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
    slot->stream = stream;
    return slot;
}

SHARED_EXPORT
bool hostbuf_file_reader_read(int device, uint64_t source_ptr, uint64_t source_offset,
                              uint64_t size, cudaStream_t stream,
                              uint64_t device_ptr, bool mark_cold, int source_mode) {
    XferFileSource source = {
        .mode = source_mode,
        .prefetch = source_mode == XFER_FILE_SOURCE_MMAP,
    };

    if (size == 0) {
        return true;
    }
    if (!device_ptr || device < 0 || !set_devctx_for_device(device)) {
        return false;
    }
    if (source.mode == XFER_FILE_SOURCE_MMAP) {
        source.as.mmap = model_mmap_get((void *)(uintptr_t)source_ptr);
        if (!source.as.mmap) {
            return false;
        }
    } else {
        source.as.file_handle = source_ptr;
    }

    while (size) {
        HostbufFileReaderSlot *slot = g_active < 0 ? NULL : &g_slots[g_active];
        size_t chunk;

        if (!slot || slot->stream != stream ||
            (slot->offset + size >= HOSTBUF_FILE_READER_WINDOW &&
             slot->offset >= LEAD_IN_THRESHOLD)) {
            if (!hostbuf_file_reader_retire_active() ||
                !(slot = hostbuf_file_reader_next(stream))) {
                return false;
            }
        }

        chunk = (size_t)MIN(size, HOSTBUF_FILE_READER_WINDOW - slot->offset);
        if (!xfer_file_read(source, source_offset, slot->buffer + slot->offset,
                            chunk, mark_cold) ||
            !CHECK_CU(cuMemcpyHtoDAsync((CUdeviceptr)device_ptr,
                                        slot->buffer + slot->offset,
                                        chunk, (CUstream)stream))) {
            return false;
        }

        slot->offset += chunk;
        source_offset += chunk;
        device_ptr += chunk;
        size -= chunk;
    }

    return true;
}

SHARED_EXPORT
void hostbuf_file_reader_cleanup(void) {
    hostbuf_file_reader_retire_active();
    for (unsigned i = 0; i < HOSTBUF_FILE_READER_SLOTS; i++) {
        HostbufFileReaderSlot *slot = &g_slots[i];

        if (slot->buffer && slot->event) {
            CHECK_CU(cuEventSynchronize(slot->event));
            CHECK_CU(cuEventDestroy(slot->event));
        }
        if (slot->buffer) {
            CHECK_CU(cuMemFreeHost(slot->buffer));
        }
    }
    memset(g_slots, 0, sizeof(g_slots));
    g_active = -1;
}
