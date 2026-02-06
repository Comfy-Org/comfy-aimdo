#include "dynbuf.h"
#include "plat.h"
#include "plat-thread.h"
#include <cuda.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

typedef struct {
    DynamicBuffer staging_pin;
    CUdeviceptr dev_signal;
    uint32_t *host_signal;
    uint32_t counter;

    void *src;
    void *pin;
    volatile CUdeviceptr dest;
    size_t size;
    CUstream stream;

    volatile bool running;
    Mutex mutex;
    Thread thread;
} VBarSlot;

static THREAD_FUNC worker_proc(void *arg) {
    VBarSlot *slot = (VBarSlot *)arg;
    while (slot->running) {
        mutex_lock(slot->mutex);
        if (!slot->running) {
            break;
        }

        void *final_src = NULL;

        if (slot->src) {
            if (!slot->pin) {
                if (dynbuf_grow(&slot->staging_pin, slot->size)) {
                    slot->pin = slot->staging_pin.base_ptr;
                }
            }
            if (slot->pin) {
                memcpy(slot->pin, slot->src, slot->size);
            }
        }

        if (slot->pin) {
            final_src = slot->pin;
        }

        if (final_src) {
            slot->counter++;
            CHECK_CU(cuMemcpyHtoDAsync(slot->dest, (uintptr_t)final_src, slot->size, slot->stream));
            CHECK_CU(cuStreamWriteValue32(slot->stream, slot->dev_signal, slot->counter, CU_STREAM_WRITE_VALUE_DEFAULT));
            CHECK_CU(cuStreamFlush(slot->stream));
        }

        slot->dest = 0;
    }
    return 0;
}

SHARED_EXPORT
void *vbar_slot_create() {
    VBarSlot *slot = (VBarSlot *)calloc(1, sizeof(*slot));
    if (!slot) {
        return NULL;
    }

    if (!dynbuf_create(&slot->staging_pin) ||
        !CHECK_CU(cuMemHostAlloc((void **)&slot->host_signal, sizeof(*slot->host_signal), CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP))) {
        dynbuf_destroy(&slot->staging_pin);
        free(slot);
        return NULL;
    }

    CHECK_CU(cuMemHostGetDevicePointer(&slot->dev_signal, (uintptr_t)slot->host_signal, 0));

    slot->mutex = mutex_create();
    slot->running = true;
    mutex_lock(slot->mutex);

    if (!thread_create(&slot->thread, worker_proc, slot)) {
        CHECK_CU(cuMemHostFree(slot->host_signal));
        dynbuf_destroy(&slot->staging_pin);
        mutex_destroy(slot->mutex);
        free(slot);
        return NULL;
    }

    return slot;
}

/* Transfer Semantics:
 * If only pin is provided, the pinned data is transferred to the dest.
 * If src and pin are provided, the src is copied to the pin before being transferred from pin to dest.
 * If only src is provided, the data is copied to a staging buffer before being transferred to the dest.
 */
SHARED_EXPORT
void vbar_slot_transfer(VBarSlot *slot, void *src, void *pin, CUdeviceptr dest, size_t size, CUstream stream) {
    while (slot->dest != 0) {
        sleep_us(1);
    }

    slot->src = src;
    slot->pin = pin;
    slot->dest = dest;
    slot->size = size;
    slot->stream = stream;

    mutex_unlock(slot->mutex);
}

SHARED_EXPORT
void vbar_slot_wait(VBarSlot *slot, CUstream stream) {
    CHECK_CU(cuStreamWaitValue32(stream, slot->dev_signal, slot->counter, CU_STREAM_WAIT_VALUE_GEQ));
}

SHARED_EXPORT
void vbar_slot_destroy(VBarSlot *slot) {
    if (!slot) return;

    slot->running = false;
    mutex_unlock(slot->mutex);
    thread_join(slot->thread);

    dynbuf_destroy(&slot->staging_pin);
    CHECK_CU(cuMemHostFree(slot->host_signal));
    mutex_destroy(slot->mutex);
    free(slot);
}
