#pragma once

#include "gpu_abi.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/types.h>
#endif

#define VMM_HASH_SIZE   (1 << 12)
#define SIZE_HASH_SIZE  1024
#define HOSTBUF_FILE_READER_SLOTS 3

typedef struct VramBuffer VramBuffer;
typedef struct SizeEntry SizeEntry;
typedef struct ModelVBAR ModelVBAR;

typedef struct HostbufFileReaderSlot {
    uint8_t *buffer;
    uint64_t offset;
    CUstream stream;
    CUevent event;
} HostbufFileReaderSlot;

typedef struct AimdoContext {
    int _device_id;
#if (defined(_WIN32) || defined(_WIN64)) && defined(AIMDO_CUDA)
    void *_nvml_device;
#endif

    uint64_t _vram_capacity;
    uint64_t _integrated_ram_headroom;
    uint64_t _extra_vram_headroom;
    uint64_t _malloc_async_clamp;
    uint64_t _total_vram_usage;
    uint64_t _total_vram_last_check;
    ssize_t _deficit_sync;
    uint64_t _control_timestamp_last_check;
    void *_highest_priority; /* ModelVBAR * */
    void *_lowest_priority; /* ModelVBAR * */
    bool _vbars_dirty;
    bool _allocations_dirty;
    bool _integrated_device;
    VramBuffer *_vmm_table[VMM_HASH_SIZE];
    SizeEntry *_size_table[SIZE_HASH_SIZE];
    void *_size_table_lock;
    void *_rogues;
    void *_rogue_candidates;
    HostbufFileReaderSlot _hostbuf_file_reader_slots[HOSTBUF_FILE_READER_SLOTS];
    int _hostbuf_file_reader_active;
#if defined(__HIP_PLATFORM_AMD__) && defined(_WIN32)
    VramBuffer *_va_pool;
    void *_va_pool_lock;
#endif
#if defined(_WIN32) || defined(_WIN64)
    void *_wddm_adapter; /* IDXGIAdapter3* */
    uint64_t _wddm_timestamp_last_check;
#endif
} AimdoContext;

extern _Thread_local AimdoContext *g_devctx;

static inline void set_devctx(AimdoContext *devctx) {
    g_devctx = devctx;
}

bool set_devctx_for_device(int device_id);
bool set_devctx_for_current_cuda_device(void);

#define vram_capacity               (g_devctx->_vram_capacity)
#if (defined(_WIN32) || defined(_WIN64)) && defined(AIMDO_CUDA)
#define nvml_device                 (g_devctx->_nvml_device)
#endif
#define integrated_ram_headroom     (g_devctx->_integrated_ram_headroom)
#define extra_vram_headroom         (g_devctx->_extra_vram_headroom)
#define malloc_async_clamp          (g_devctx->_malloc_async_clamp)
#define total_vram_usage            (g_devctx->_total_vram_usage)
#define total_vram_last_check       (g_devctx->_total_vram_last_check)
#define deficit_sync                (g_devctx->_deficit_sync)
#define highest_priority_p          (*(ModelVBAR **)&g_devctx->_highest_priority)
#define lowest_priority_p           (*(ModelVBAR **)&g_devctx->_lowest_priority)
#define vbars_dirty                 (g_devctx->_vbars_dirty)
#define allocations_dirty           (g_devctx->_allocations_dirty)
#define integrated_device           (g_devctx->_integrated_device)
#define vmm_table                   (g_devctx->_vmm_table)
#define size_table                  (g_devctx->_size_table)
#define size_table_lock             (g_devctx->_size_table_lock)
#define rogues                      (g_devctx->_rogues)
#define global_rogue_candidates     (g_devctx->_rogue_candidates)
#if defined(__HIP_PLATFORM_AMD__) && defined(_WIN32)
#define va_pool                     (g_devctx->_va_pool)
#define va_pool_lock                (g_devctx->_va_pool_lock)
#endif
#if defined(_WIN32) || defined(_WIN64)
#define g_wddm_adapter              (*(IDXGIAdapter3 **)&g_devctx->_wddm_adapter)
#define wddm_timestamp_last_check   (g_devctx->_wddm_timestamp_last_check)
#endif
#define control_timestamp_last_check (g_devctx->_control_timestamp_last_check)

#define highest_priority            (*highest_priority_p)
#define lowest_priority             (*lowest_priority_p)
