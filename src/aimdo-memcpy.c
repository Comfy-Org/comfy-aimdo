#include "plat.h"
#include "plat-thread.h"
#include <cuda.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

typedef struct TransferSegment {
    void *src;
    void *dest;
    size_t size;
#define TRANSFER_TYPE_MEMCPY    1
#define TRANSFER_TYPE_HTOD      2
#define TRANSFER_TYPE_EVENT     3
    int type;

    struct TransferSegment *next;
} TransferSegment;

typedef struct {
    TransferSegment *tasks;

    CUstream stream;

    CUdeviceptr dev_signal;
    uint32_t counter;

#define THREAD_STATE_NONE 0
#define THREAD_STATE_LIVE 1
#define THREAD_STATE_DEAD 2
    int thread_state;
    Mutex mutex;
    Thread thread;
} VBarSlot;

SHARED_EXPORT
void *vbar_slot_create(void *stream_ptr) {
    VBarSlot *slot = (VBarSlot *)calloc(1, sizeof(*slot));
    if (!slot) {
        goto fail;
    }

    slot->stream = (CUstream)stream_ptr;
    slot->mutex = mutex_create();

    if (!slot->mutex ||
        !CHECK_CU(cuMemAlloc(&slot->dev_signal, sizeof(uint32_t))) ||
        !CHECK_CU(cuMemsetD32(slot->dev_signal, 0, 1))) {
        goto fail;
    }

    return slot;
fail:
    if (slot->dev_signal) {
        cuMemFree(slot->dev_signal);
    }
    mutex_destroy(slot->mutex);
    free(slot);
    return NULL;
}

static THREAD_FUNC worker_proc(void *arg) {
    VBarSlot *slot = (VBarSlot *)arg;

    for (;;) {
        TransferSegment *t;

        mutex_lock(slot->mutex);
        if (!slot->tasks) {
            slot->thread_state = THREAD_STATE_DEAD;
            mutex_unlock(slot->mutex);
            return 0;
        }
        t = slot->tasks;
        slot->tasks = t->next;
        mutex_unlock(slot->mutex);

        if (t->type == TRANSFER_TYPE_MEMCPY) {
            log(VVERBOSE, "%s: memcpy %zu\n", __func__, t->size);
            memcpy(t->dest, t->src, t->size);
        } else if (t->type == TRANSFER_TYPE_HTOD) {
            log(VVERBOSE, "%s: htod %zu\n", __func__, t->size);
            CHECK_CU(cuMemcpyHtoDAsync((CUdeviceptr)t->dest, t->src, t->size, slot->stream));
        } else if (t->type == TRANSFER_TYPE_EVENT) {
            log(VVERBOSE, "%s: SEV@%llx -> %zu\n", __func__, slot->dev_signal, t->size);
            CHECK_CU(cuStreamWriteValue32(slot->stream, slot->dev_signal, t->size, CU_STREAM_WRITE_VALUE_DEFAULT));
        }

        free(t);
    }
    assert(false); /* unreachable */
    return 0;
}

SHARED_EXPORT
bool vbar_slot_transfer(void *s, void *src, void *dest, size_t size, int type) {
    VBarSlot *slot = (VBarSlot *)s;
    TransferSegment **i = &slot->tasks;
    TransferSegment *seg = calloc(1, sizeof(*seg));

    if (!seg) {
        return false;
    }

    *seg = (TransferSegment) {
        .src = src,
        .dest = dest,
        .size = size,
        .type = type
    };

    mutex_lock(slot->mutex);
    if (slot->thread_state == THREAD_STATE_DEAD) {
        thread_join(slot->thread);
        slot->thread_state = THREAD_STATE_NONE;
    }
    if (slot->thread_state == THREAD_STATE_NONE) {
        thread_create(&slot->thread, worker_proc, slot);
        slot->thread_state = THREAD_STATE_LIVE;
    }

    while (*i) {
        i = &(*i)->next;
    }
    *i = seg;
    mutex_unlock(slot->mutex);

    return true;
}

SHARED_EXPORT
void vbar_slot_wait(void *s, void *stream_ptr) {
    VBarSlot *slot = (VBarSlot *)s;

    slot->counter++;
    vbar_slot_transfer(s, NULL, NULL, slot->counter, TRANSFER_TYPE_EVENT);
    log(VVERBOSE, "%s: WFE@%llx -> %zu\n", __func__, slot->dev_signal, (size_t)slot->counter);
    CHECK_CU(cuStreamWaitValue32((CUstream)stream_ptr, slot->dev_signal, slot->counter, CU_STREAM_WAIT_VALUE_GEQ));
}

SHARED_EXPORT
void vbar_slot_destroy(void *s) {
    VBarSlot *slot = (VBarSlot *)s;
    if (!slot) {
        return;
    }

    if (slot->thread_state != THREAD_STATE_NONE) {
        thread_join(slot->thread);
    }

    for (TransferSegment *i = slot->tasks; i;) {
        TransferSegment *t = i;
        i = i->next;
        free(t);
    }

    cuMemFree(slot->dev_signal);
    mutex_destroy(slot->mutex);

    free(slot);
}
