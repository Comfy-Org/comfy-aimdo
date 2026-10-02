/* No GPU needed. Compile with function sections and discard unused sections:
 * Linux: gcc -O2 -ffunction-sections -fdata-sections -DAIMDO_CUDA -Isrc
 *        tests/integrated-ram-pressure.c -Wl,--gc-sections -o pressure-test
 * Windows: cl /O2 /Gy /DAIMDO_CUDA /Isrc /Isrc-win /FIcompiler.h
 *          tests/integrated-ram-pressure.c /link /OPT:REF dxgi.lib dxguid.lib
 */
#include "plat.h"
#include "aimdo-time.h"

static uint64_t test_tick = 10000;
static size_t available_ram;
static bool memory_query_ok = true;
static int memory_queries;
static int cuda_queries;
static int test_integrated;

#if defined(_WIN32) || defined(_WIN64)
static BOOL WINAPI test_memory_status(LPMEMORYSTATUSEX status) {
    assert(status->dwLength == sizeof(*status));
    memory_queries++;
    status->ullAvailPhys = available_ram;
    status->ullAvailPageFile = 128 * M; /* Physical RAM, not commit, is the policy. */
    return memory_query_ok;
}
#define GlobalMemoryStatusEx test_memory_status
#else
static FILE *test_meminfo(const char *path, const char *mode) {
    assert(strcmp(path, "/proc/meminfo") == 0);
    assert(strcmp(mode, "r") == 0);
    memory_queries++;
    if (!memory_query_ok) {
        return NULL;
    }
    FILE *file = tmpfile();
    assert(file);
    fprintf(file, "MemFree: 128 kB\nMemAvailable: %llu kB\n",
            (unsigned long long)(available_ram / K));
    rewind(file);
    return file;
}
#define fopen test_meminfo
#endif

#undef GET_TICK
#define GET_TICK() test_tick
#undef SHARED_EXPORT
#define SHARED_EXPORT
#include "../src/control.c"
#if defined(_WIN32) || defined(_WIN64)
#include "../src-win/shmem-detect.c"
#endif

AimdoCudaDispatch g_cuda;
int log_level = -1;
uint64_t log_shot_counter;

void aimdo_log(int level, const char *file, int line, const char *format, ...) {}

#if (defined(_WIN32) || defined(_WIN64)) && defined(AIMDO_CUDA)
bool aimdo_nvml_memory_info(void *handle, size_t *free_bytes, size_t *total_bytes) {
    assert(!"NVML must not be queried for integrated RAM pressure");
    return false;
}
#endif

static CUresult CUDAAPI test_attribute(int *value, CUdevice_attribute attribute, CUdevice device) {
    assert(device == 7);
#if defined(__HIP_PLATFORM_AMD__)
    assert((int)attribute == 16);
#else
    assert((int)attribute == 18);
#endif
    *value = test_integrated;
    return CUDA_SUCCESS;
}

static CUresult CUDAAPI test_cuda_memory(size_t *free_bytes, size_t *total_bytes) {
    cuda_queries++;
    *free_bytes = 512 * M;
    *total_bytes = 128ULL * G;
    return CUDA_SUCCESS;
}

int main(void) {
    AimdoContext context = {0};
    const char *method = "unknown";
    set_devctx(&context);
    g_cuda.p_cuDeviceGetAttribute = test_attribute;
    g_cuda.p_cuMemGetInfo = test_cuda_memory;

    assert(calculate_integrated_ram_headroom(16ULL * G) == 2ULL * G);
    assert(calculate_integrated_ram_headroom(64ULL * G) == 4ULL * G);
    assert(calculate_integrated_ram_headroom(256ULL * G) == 8ULL * G);
    assert(!is_integrated_cuda_device(7));
    test_integrated = 1;
    assert(is_integrated_cuda_device(7));

    integrated_device = true;
    integrated_ram_headroom = 8ULL * G;
    vram_capacity = 128ULL * G;
    total_vram_usage = 31ULL * G;
#if (defined(_WIN32) || defined(_WIN64)) && defined(AIMDO_CUDA)
    nvml_device = (void *)1; /* Integrated routing must take precedence. */
#endif
    available_ram = 12ULL * G;
    assert(poll_budget_deficit(&method));
    assert(deficit_sync == -(ssize_t)(4ULL * G));
    assert(strstr(method, "integrated RAM"));
    assert(memory_queries == 1 && cuda_queries == 0);

    /* Between polls, account for new allocations against the last RAM sample. */
    total_vram_usage += 3ULL * G;
    available_ram = 0; /* Must not be sampled before the 2-second interval. */
    assert(budget_deficit(2ULL * G) == (ssize_t)G);
    assert(memory_queries == 1);

    test_tick += 2000;
    available_ram = 7ULL * G;
    assert(poll_budget_deficit(&method));
    assert(deficit_sync == (ssize_t)G);
    assert(total_vram_last_check == total_vram_usage);
    assert(memory_queries == 2 && cuda_queries == 0);

    test_tick += 2000;
    available_ram = 8ULL * G;
    assert(budget_deficit(0) == 0);
    assert(budget_deficit(1) == 1);

    test_tick += 2000;
    memory_query_ok = false;
    assert(!poll_budget_deficit(&method));
    assert(deficit_sync == INTEGRATED_SIMPLE_ONLY_DEFICIT);
    assert(cuda_queries == 0);
    total_vram_usage = vram_capacity - VRAM_HEADROOM;
    assert(budget_deficit(1) == 1); /* The simple capacity limit still applies. */

    /* Discrete GPUs must retain their existing device-memory pressure path. */
    integrated_device = false;
#if (defined(_WIN32) || defined(_WIN64)) && defined(AIMDO_CUDA)
    nvml_device = NULL;
#endif
    total_vram_usage = 31ULL * G;
    test_tick += 2000;
    int prior_memory_queries = memory_queries;
    assert(poll_budget_deficit(&method));
    assert(memory_queries == prior_memory_queries && cuda_queries == 1);
#if defined(_WIN32) || defined(_WIN64)
    assert(deficit_sync == -(ssize_t)(416 * M));
    assert(strcmp(method, "cuMemGetInfo (Windows)") == 0);
#else
    assert(deficit_sync == -(ssize_t)(256 * M));
    assert(strcmp(method, "cuMemGetInfo") == 0);
#endif
    puts("integrated RAM pressure tests passed");
    return 0;
}
