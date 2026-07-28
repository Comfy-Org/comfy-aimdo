#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "plat.h"

AimdoCudaDispatch g_cuda;
_Thread_local AimdoContext *g_devctx;
int log_level = __NONE__;
uint64_t log_shot_counter;
int64_t simple_vram_headroom = VRAM_HEADROOM;

static AimdoContext g_test_devctx = { ._device_id = 0 };
static CUdeviceptr g_next_va;
static CUmemGenericAllocationHandle g_next_handle;
static size_t g_reserve_calls;
static size_t g_create_calls;
static size_t g_map_calls;
static size_t g_access_calls;
static size_t g_unmap_calls;
static size_t g_release_calls;
static size_t g_address_free_calls;
static size_t g_sync_calls;
static CUresult g_sync_status;
static CUresult g_unmap_status;
static CUmemGenericAllocationHandle g_mapped_handles[32];
static CUcontext g_current_context = (CUcontext)(uintptr_t)0x1234;

bool set_devctx_for_current_cuda_device(void) {
    g_devctx = &g_test_devctx;
    return true;
}

bool set_devctx_for_device(int device_id) {
    if (device_id != 0) {
        g_devctx = NULL;
        return false;
    }
    g_devctx = &g_test_devctx;
    return true;
}

const char *get_level_str(int level) {
    (void)level;
    return "test";
}

void log_reset_shots(void) {}

static CUresult mock_get_error_string(CUresult error, const char **description) {
    (void)error;
    *description = "mock CUDA error";
    return CUDA_SUCCESS;
}

static CUresult mock_ctx_get_current(CUcontext *context) {
    *context = g_current_context;
    return CUDA_SUCCESS;
}

static CUresult mock_address_reserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                                     CUdeviceptr address, unsigned long long flags) {
    (void)alignment;
    (void)address;
    (void)flags;
    *ptr = g_next_va;
    g_next_va += size;
    g_reserve_calls++;
    return CUDA_SUCCESS;
}

static CUresult mock_address_free(CUdeviceptr ptr, size_t size) {
    (void)ptr;
    (void)size;
    g_address_free_calls++;
    return CUDA_SUCCESS;
}

static CUresult mock_mem_create(CUmemGenericAllocationHandle *handle, size_t size,
                                const CUmemAllocationProp *prop,
                                unsigned long long flags) {
    (void)size;
    (void)prop;
    (void)flags;
    *handle = g_next_handle++;
    g_create_calls++;
    return CUDA_SUCCESS;
}

static CUresult mock_mem_map(CUdeviceptr ptr, size_t size, size_t offset,
                             CUmemGenericAllocationHandle handle,
                             unsigned long long flags) {
    (void)ptr;
    (void)size;
    (void)offset;
    (void)flags;
    g_mapped_handles[g_map_calls++] = handle;
    return CUDA_SUCCESS;
}

static CUresult mock_set_access(CUdeviceptr ptr, size_t size,
                                const CUmemAccessDesc *desc, size_t count) {
    (void)ptr;
    (void)size;
    (void)desc;
    (void)count;
    g_access_calls++;
    return CUDA_SUCCESS;
}

static CUresult mock_unmap(CUdeviceptr ptr, size_t size) {
    (void)ptr;
    (void)size;
    g_unmap_calls++;
    return g_unmap_status;
}

static CUresult mock_release(CUmemGenericAllocationHandle handle) {
    (void)handle;
    g_release_calls++;
    return CUDA_SUCCESS;
}

static CUresult mock_stream_synchronize(CUstream stream) {
    (void)stream;
    g_sync_calls++;
    return g_sync_status;
}

#include "../src/loop-record.c"

static void reset_mocks(void) {
    memset(&g_test_devctx, 0, sizeof(g_test_devctx));
    g_test_devctx._device_id = 0;
    g_devctx = &g_test_devctx;
    g_next_va = 0x100000000ULL;
    g_next_handle = 1;
    g_reserve_calls = 0;
    g_create_calls = 0;
    g_map_calls = 0;
    g_access_calls = 0;
    g_unmap_calls = 0;
    g_release_calls = 0;
    g_address_free_calls = 0;
    g_sync_calls = 0;
    g_sync_status = CUDA_SUCCESS;
    g_unmap_status = CUDA_SUCCESS;
    memset(g_mapped_handles, 0, sizeof(g_mapped_handles));

    memset(&g_cuda, 0, sizeof(g_cuda));
    g_cuda.p_cuGetErrorString = mock_get_error_string;
    g_cuda.p_cuCtxGetCurrent = mock_ctx_get_current;
    g_cuda.p_cuStreamSynchronize = mock_stream_synchronize;
    g_cuda.p_cuMemAddressReserve = mock_address_reserve;
    g_cuda.p_cuMemAddressFree = mock_address_free;
    g_cuda.p_cuMemCreate = mock_mem_create;
    g_cuda.p_cuMemMap = mock_mem_map;
    g_cuda.p_cuMemSetAccess = mock_set_access;
    g_cuda.p_cuMemUnmap = mock_unmap;
    g_cuda.p_cuMemRelease = mock_release;
}

static CUdeviceptr recorded_malloc(size_t size, CUstream stream) {
    CUdeviceptr ptr = 0;
    CUresult status = CUDA_ERROR_INVALID_VALUE;

    assert(record_malloc_async(&ptr, size, stream, &status));
    assert(status == CUDA_SUCCESS);
    assert(ptr);
    return ptr;
}

static void recorded_free(CUdeviceptr ptr, CUstream stream) {
    CUresult status = CUDA_ERROR_INVALID_VALUE;

    assert(record_free(ptr, stream, true, &status));
    assert(status == CUDA_SUCCESS);
}

static void *pop_graph(int expected_status) {
    void *graph = NULL;

    assert(pop(&graph) == expected_status);
    return graph;
}

static void destroy_graph(void *graph) {
    assert(graph);
    assert(destroy_record(graph) == RECORD_OK);
}

static void test_record_replay_and_page_alias(void) {
    CUstream stream = (CUstream)(uintptr_t)0x10;
    CUdeviceptr first;
    CUdeviceptr second;
    size_t map_calls;

    reset_mocks();
    assert(push_record(stream, NULL) == RECORD_OK);
    assert(iterate() == RECORD_OK);
    first = recorded_malloc(M, stream);
    recorded_free(first, stream);
    second = recorded_malloc(M, stream);
    recorded_free(second, stream);

    assert(first != second);
    assert(g_reserve_calls == 2);
    assert(g_create_calls == 1);
    assert(g_map_calls == 2);
    assert(g_mapped_handles[0] == g_mapped_handles[1]);

    map_calls = g_map_calls;
    assert(iterate() == RECORD_OK);
    assert(recorded_malloc(M, stream) == first);
    recorded_free(first, stream);
    assert(recorded_malloc(M, stream) == second);
    recorded_free(second, stream);
    assert(g_map_calls == map_calls);
    assert(g_create_calls == 1);

    void *graph = pop_graph(RECORD_OK);
    assert(graph);
    assert(g_sync_calls == 0);
    assert(g_unmap_calls == 0);
    assert(g_release_calls == 0);

    assert(push_record(stream, graph) == RECORD_OK);
    assert(iterate() == RECORD_OK);
    assert(recorded_malloc(M, stream) == first);
    recorded_free(first, stream);
    assert(recorded_malloc(M, stream) == second);
    recorded_free(second, stream);
    assert(pop_graph(RECORD_OK) == graph);

    destroy_graph(graph);
    assert(g_sync_calls == 1);
    assert(g_unmap_calls == 2);
    assert(g_release_calls == 1);
    assert(g_address_free_calls == 2);
}

static void run_child(CUstream stream) {
    CUdeviceptr ptr;

    assert(push_record(stream, NULL) == RECORD_OK);
    for (int i = 0; i < 2; i++) {
        assert(iterate() == RECORD_OK);
        ptr = recorded_malloc(2 * M, stream);
        recorded_free(ptr, stream);
    }
    assert(pop_graph(RECORD_OK) == NULL);
}

static void test_nested_record_replay(void) {
    CUstream stream = (CUstream)(uintptr_t)0x20;
    size_t map_calls;

    reset_mocks();
    assert(push_record(stream, NULL) == RECORD_OK);
    assert(iterate() == RECORD_OK);
    run_child(stream);
    map_calls = g_map_calls;

    assert(iterate() == RECORD_OK);
    run_child(stream);
    assert(g_map_calls == map_calls);
    destroy_graph(pop_graph(RECORD_OK));
}

static void test_other_stream_is_not_recorded(void) {
    CUstream stream = (CUstream)(uintptr_t)0x30;
    CUstream other = (CUstream)(uintptr_t)0x31;
    CUdeviceptr ptr = 0;
    CUresult status = CUDA_SUCCESS;

    reset_mocks();
    assert(push_record(stream, NULL) == RECORD_OK);
    assert(iterate() == RECORD_OK);
    assert(!record_malloc_async(&ptr, M, other, &status));
    destroy_graph(pop_graph(RECORD_OK));
    assert(g_reserve_calls == 0);
}

static void test_mismatch_is_sticky(void) {
    CUstream stream = (CUstream)(uintptr_t)0x40;
    CUdeviceptr ptr;
    CUresult status;

    reset_mocks();
    assert(push_record(stream, NULL) == RECORD_OK);
    assert(iterate() == RECORD_OK);
    ptr = recorded_malloc(M, stream);
    recorded_free(ptr, stream);
    assert(iterate() == RECORD_OK);

    ptr = 1;
    status = CUDA_SUCCESS;
    assert(record_malloc_async(&ptr, 2 * M, stream, &status));
    assert(status == CUDA_ERROR_INVALID_VALUE);
    assert(ptr == 0);
    destroy_graph(pop_graph(RECORD_MISMATCH));
    assert(strstr(record_last_error(), "allocation size changed"));
}

static void test_compiled_pointer_cannot_leave_stream(void) {
    CUstream stream = (CUstream)(uintptr_t)0x50;
    CUstream other = (CUstream)(uintptr_t)0x51;
    CUdeviceptr ptr;
    CUresult status = CUDA_SUCCESS;

    reset_mocks();
    assert(push_record(stream, NULL) == RECORD_OK);
    assert(iterate() == RECORD_OK);
    ptr = recorded_malloc(M, stream);
    assert(record_free(ptr, other, true, &status));
    assert(status == CUDA_ERROR_INVALID_VALUE);
    destroy_graph(pop_graph(RECORD_MISMATCH));
    assert(strstr(record_last_error(), "outside its record stream"));
}

static void test_live_allocation_rejects_iteration(void) {
    CUstream stream = (CUstream)(uintptr_t)0x60;

    reset_mocks();
    assert(push_record(stream, NULL) == RECORD_OK);
    assert(iterate() == RECORD_OK);
    recorded_malloc(M, stream);
    assert(iterate() == RECORD_MISMATCH);
    destroy_graph(pop_graph(RECORD_MISMATCH));
    assert(strstr(record_last_error(), "allocations remain live"));
    assert(g_sync_calls == 1);
}

static void test_child_iteration_count_must_match(void) {
    CUstream stream = (CUstream)(uintptr_t)0x70;
    CUdeviceptr ptr;

    reset_mocks();
    assert(push_record(stream, NULL) == RECORD_OK);
    assert(iterate() == RECORD_OK);
    run_child(stream);
    assert(iterate() == RECORD_OK);

    assert(push_record(stream, NULL) == RECORD_OK);
    assert(iterate() == RECORD_OK);
    ptr = recorded_malloc(2 * M, stream);
    recorded_free(ptr, stream);
    assert(pop_graph(RECORD_MISMATCH) == NULL);
    assert(strstr(record_last_error(), "expected 2 iterations"));
    destroy_graph(pop_graph(RECORD_MISMATCH));
}

static void test_sync_failure_keeps_mappings_for_retry(void) {
    CUstream stream = (CUstream)(uintptr_t)0x80;
    CUdeviceptr ptr;

    reset_mocks();
    assert(push_record(stream, NULL) == RECORD_OK);
    assert(iterate() == RECORD_OK);
    ptr = recorded_malloc(M, stream);
    recorded_free(ptr, stream);

    void *graph = pop_graph(RECORD_OK);
    g_sync_status = CUDA_ERROR_INVALID_VALUE;
    assert(destroy_record(graph) == RECORD_CUDA_FAILURE);
    assert(g_unmap_calls == 0);
    assert(g_release_calls == 0);
    assert(strstr(record_last_error(), "destruction may be retried"));

    g_sync_status = CUDA_SUCCESS;
    assert(destroy_record(graph) == RECORD_OK);
    assert(g_unmap_calls == 1);
    assert(g_release_calls == 1);
}

static void test_teardown_failure_is_retryable(void) {
    CUstream stream = (CUstream)(uintptr_t)0x90;
    CUdeviceptr ptr;

    reset_mocks();
    assert(push_record(stream, NULL) == RECORD_OK);
    assert(iterate() == RECORD_OK);
    ptr = recorded_malloc(M, stream);
    recorded_free(ptr, stream);

    void *graph = pop_graph(RECORD_OK);
    g_unmap_status = CUDA_ERROR_INVALID_VALUE;
    assert(destroy_record(graph) == RECORD_CUDA_FAILURE);
    assert(g_release_calls == 0);
    assert(g_address_free_calls == 0);
    assert(strstr(record_last_error(), "cuMemUnmap failed"));

    g_unmap_status = CUDA_SUCCESS;
    assert(destroy_record(graph) == RECORD_OK);
    assert(g_release_calls == 1);
    assert(g_address_free_calls == 1);
}

static void test_context_change_does_not_teardown(void) {
    CUstream stream = (CUstream)(uintptr_t)0xa0;
    CUcontext original_context = g_current_context;
    CUdeviceptr ptr;

    reset_mocks();
    assert(push_record(stream, NULL) == RECORD_OK);
    assert(iterate() == RECORD_OK);
    ptr = recorded_malloc(M, stream);
    recorded_free(ptr, stream);

    g_current_context = (CUcontext)(uintptr_t)0x5678;
    assert(pop_graph(RECORD_UNSUPPORTED) == NULL);
    assert(g_sync_calls == 0);
    assert(g_unmap_calls == 0);

    g_current_context = original_context;
    destroy_graph(pop_graph(RECORD_OK));
}

int main(void) {
    test_record_replay_and_page_alias();
    test_nested_record_replay();
    test_other_stream_is_not_recorded();
    test_mismatch_is_sticky();
    test_compiled_pointer_cannot_leave_stream();
    test_live_allocation_rejects_iteration();
    test_child_iteration_count_must_match();
    test_sync_failure_keeps_mappings_for_retry();
    test_teardown_failure_is_retryable();
    test_context_change_does_not_teardown();
    puts("loop-record tests passed");
    return 0;
}
