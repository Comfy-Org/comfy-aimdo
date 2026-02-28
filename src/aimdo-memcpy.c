#include "plat.h"
#include "plat-thread.h"
#include <cuda.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define SYNC_IT 0

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
    CUcontext ctx;

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
        !CHECK_CU(cuStreamGetCtx(slot->stream, &slot->ctx)) ||
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

    log(VVERBOSE, "%s: Worker created\n", __func__);
    if (!CHECK_CU(cuCtxSetCurrent(slot->ctx))) {
        return 0;
    }

    for (;;) {
        TransferSegment *t;
        
        log(VVERBOSE, "%s: Iterating ...\n", __func__);
        mutex_lock(slot->mutex);
        log(VVERBOSE, "%s: Mutex Aquired\n", __func__);
        if (!slot->tasks) {
            slot->thread_state = THREAD_STATE_DEAD;
            log(VVERBOSE, "%s: Exiting\n", __func__);
            mutex_unlock(slot->mutex);
            return 0;
        }
        t = slot->tasks;
        slot->tasks = t->next;
        log(VVERBOSE, "%s: mutex released\n", __func__);
        mutex_unlock(slot->mutex);

        if (t->type == TRANSFER_TYPE_MEMCPY) {
            log(VVERBOSE, "%s: memcpy %p %p %zu\n", __func__, t->src, t->dest, t->size);
            memcpy(t->dest, t->src, t->size);
        } else if (t->type == TRANSFER_TYPE_HTOD) {
            log(VVERBOSE, "%s: htod %p %llx %zu\n", __func__, t->src, (CUdeviceptr)t->dest, t->size);
            CHECK_CU(cuMemcpyHtoDAsync((CUdeviceptr)t->dest, t->src, t->size, slot->stream));
#if SYNC_IT > 0
            CHECK_CU(cuStreamSynchronize(slot->stream));
#endif
        } else if (t->type == TRANSFER_TYPE_EVENT) {
#if SYNC_IT <= 1
            log(VVERBOSE, "%s: SEV@%llx -> %zu\n", __func__, slot->dev_signal, t->size);
            CHECK_CU(cuStreamWriteValue32(slot->stream, slot->dev_signal, t->size, CU_STREAM_WRITE_VALUE_DEFAULT));
#endif
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

    log(VVERBOSE, "%s: %p %p %zuk %d\n", __func__, src, dest, size / K, type);
    mutex_lock(slot->mutex);
    log(VVERBOSE, "%s: mutex aquired\n, __func__");
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
    log(VVERBOSE, "%s: mutex released\n", __func__);
    mutex_unlock(slot->mutex);

#if SYNC_IT > 0
    thread_join(slot->thread);
#endif
    return true;
}

SHARED_EXPORT
void vbar_slot_wait(void *s, void *stream_ptr) {
    VBarSlot *slot = (VBarSlot *)s;

    slot->counter++;
    vbar_slot_transfer(s, NULL, NULL, slot->counter, TRANSFER_TYPE_EVENT);
#if SYNC_IT <= 1
    log(VVERBOSE, "%s: WFE@%llx -> %zu\n", __func__, slot->dev_signal, (size_t)slot->counter);
    CHECK_CU(cuStreamWaitValue32((CUstream)stream_ptr, slot->dev_signal, slot->counter, CU_STREAM_WAIT_VALUE_GEQ));
#endif
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
