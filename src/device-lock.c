#include "plat.h"
#include "thread-plat.h"
#include "device-lock.h"

typedef struct DeviceGate {
    Mutex mutex;
    CondVar available;
    uint64_t owner;
    size_t operations;
} DeviceGate;

static _Thread_local uint64_t current_token;

uint64_t device_vram_usage(AimdoContext *context) {
    DeviceGate *gate = context->_device_gate;
    mutex_lock(gate->mutex);
    uint64_t bytes = context->_total_vram_usage;
    mutex_unlock(gate->mutex);
    return bytes;
}

void device_vram_adjust(AimdoContext *context, int64_t delta) {
    DeviceGate *gate = context->_device_gate;
    mutex_lock(gate->mutex);
    context->_total_vram_usage += delta;
    mutex_unlock(gate->mutex);
}

bool device_gate_init(AimdoContext *context) {
    DeviceGate *gate = calloc(1, sizeof(*gate));
    if (!gate) {
        return false;
    }
    gate->mutex = mutex_create();
    gate->available = condvar_create();
    context->_device_gate = gate;
    return gate->mutex && gate->available;
}

void device_gate_destroy(AimdoContext *context) {
    DeviceGate *gate = context->_device_gate;
    if (!gate) {
        return;
    }
    if (gate->available) {
        condvar_destroy(gate->available);
    }
    if (gate->mutex) {
        mutex_destroy(gate->mutex);
    }
    free(gate);
    context->_device_gate = NULL;
}

SHARED_EXPORT uint64_t device_lock_set_token(uint64_t token) {
    uint64_t previous = current_token;
    current_token = token;
    return previous;
}

SHARED_EXPORT bool device_lock_try(void *context, uint64_t token) {
    DeviceGate *gate = ((AimdoContext *)context)->_device_gate;
    bool acquired;
    if (!token) {
        return false;
    }
    mutex_lock(gate->mutex);
    acquired = !gate->owner && !gate->operations;
    if (acquired) {
        gate->owner = token;
    }
    mutex_unlock(gate->mutex);
    return acquired;
}

SHARED_EXPORT bool device_lock_release(void *context, uint64_t token) {
    DeviceGate *gate = ((AimdoContext *)context)->_device_gate;
    bool released;
    mutex_lock(gate->mutex);
    released = token && gate->owner == token && !gate->operations;
    if (released) {
        gate->owner = 0;
        condvar_broadcast(gate->available);
    }
    mutex_unlock(gate->mutex);
    return released;
}

SHARED_EXPORT void device_operation_begin(AimdoContext *context) {
    if (!context) {
        return;
    }
    DeviceGate *gate = context->_device_gate;
    mutex_lock(gate->mutex);
    while (gate->owner && gate->owner != current_token) {
        condvar_wait(gate->available, gate->mutex);
    }
    gate->operations++;
    mutex_unlock(gate->mutex);
}

SHARED_EXPORT void device_operation_end(AimdoContext *context) {
    if (!context) {
        return;
    }
    DeviceGate *gate = context->_device_gate;
    mutex_lock(gate->mutex);
    assert(gate->operations);
    gate->operations--;
    mutex_unlock(gate->mutex);
}
