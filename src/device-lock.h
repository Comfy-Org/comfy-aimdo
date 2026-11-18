#pragma once

bool device_gate_init(AimdoContext *context);
void device_gate_destroy(AimdoContext *context);
void device_operation_begin(AimdoContext *context);
void device_operation_end(AimdoContext *context);

/* Keep the gate around the whole operation, including early error returns. */
#define DEVICE_FUNCTION(result_type, name, parameters, context, arguments) \
    static result_type name##_unlocked parameters; \
    SHARED_EXPORT result_type name parameters { \
        AimdoContext *gate_context = (context); \
        device_operation_begin(gate_context); \
        result_type result = name##_unlocked arguments; \
        device_operation_end(gate_context); \
        return result; \
    } \
    static result_type name##_unlocked parameters

#define DEVICE_VOID_FUNCTION(name, parameters, context, arguments) \
    static void name##_unlocked parameters; \
    SHARED_EXPORT void name parameters { \
        AimdoContext *gate_context = (context); \
        device_operation_begin(gate_context); \
        name##_unlocked arguments; \
        device_operation_end(gate_context); \
    } \
    static void name##_unlocked parameters
