#pragma once

#include "thread-plat.h"

#include <stddef.h>
#include <string.h>

#define HOSTBUF_DECOMMIT_QUEUE_LIMIT (384ULL * 1024ULL * 1024ULL)

typedef struct HostbufDecommitState {
    Mutex mutex;
    CondVar done;
    size_t pending;
} HostbufDecommitState;

static inline void hostbuf_decommit_wait(HostbufDecommitState *state) {
    mutex_lock(state->mutex);
    while (state->pending) {
        condvar_wait(state->done, state->mutex);
    }
    mutex_unlock(state->mutex);
}

static inline bool hostbuf_decommit_state_init(HostbufDecommitState *state) {
    state->mutex = mutex_create();
    state->done = condvar_create();
    state->pending = 0;
    if (state->mutex && state->done) {
        return true;
    }
    condvar_destroy(state->done);
    mutex_destroy(state->mutex);
    memset(state, 0, sizeof(*state));
    return false;
}

static inline void hostbuf_decommit_state_destroy(HostbufDecommitState *state) {
    hostbuf_decommit_wait(state);
    condvar_destroy(state->done);
    mutex_destroy(state->mutex);
}

bool hostbuf_decommit_async(HostbufDecommitState *state, void *ptr, size_t size,
                            bool release_address_space);
