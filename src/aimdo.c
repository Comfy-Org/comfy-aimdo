#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stddef.h>

#include "plat.h"

#define MIN_ALLOC_SHIFT 21
#define MIN_ALLOC       (1 << MIN_ALLOC_SHIFT) /*2 MB as per Cuda page sizes and pytorch cacheing alloc */

#define VMM_HASH_SIZE   (1 << 12) 

typedef struct VMMEntry {
    void* ptr;
    CUmemGenericAllocationHandle handle;
    size_t size;
    struct VMMEntry* next;
} VMMEntry;

static VMMEntry* vmm_table[VMM_HASH_SIZE];

static inline unsigned int vmm_hash(void* ptr) {
    return ((uintptr_t)ptr >> MIN_ALLOC_SHIFT) % VMM_HASH_SIZE;
}

static int check_cu_impl(CUresult res, const char *label) {
    if (res != CUDA_SUCCESS && res != CUDA_ERROR_OUT_OF_MEMORY) {
        const char* desc;
        if (cuGetErrorString(res, &desc) != CUDA_SUCCESS) {
            desc = "<FATAL - CANNOT PARSE CUDA ERROR CODE>";

        }
        fprintf(stderr, "CUDA API FAILED : %s : %s\n", label, desc);
    }
    return (res == CUDA_SUCCESS);
}
#define CHECK_CU(x) check_cu_impl((x), #x)

SHARED_EXPORT
void *alloc_fn(size_t size, int device, cudaStream_t stream) {
    CUmemGenericAllocationHandle alloc_handle = 0;
    void *ptr = NULL;

    {
        CUresult err;
        CUmemAllocationProp prop = {
            .type = CU_MEM_ALLOCATION_TYPE_PINNED,
            .location.type = CU_MEM_LOCATION_TYPE_DEVICE,
            .location.id = device,
        };

        CUmemAccessDesc accessDesc = {
            .location.type = CU_MEM_LOCATION_TYPE_DEVICE,
            .location.id = device,
            .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };

        /* FIXME: failure unwind chain all of this */
        if (!CHECK_CU(err = cuMemAddressReserve((CUdeviceptr*)&ptr, size, 0, 0, 0)) || 
            !CHECK_CU(err = cuMemCreate(&alloc_handle, size, &prop, 0)) ||
            !CHECK_CU(err = cuMemMap((CUdeviceptr)ptr, size, 0, alloc_handle, 0)) ||
            !CHECK_CU(err = cuMemSetAccess((CUdeviceptr)ptr, size, &accessDesc, 1))) {
            if (err == CUDA_ERROR_OUT_OF_MEMORY) {
                fprintf(stderr, "DEBUG: OOMED\n");
            }
            return NULL;
        }
    }

    {
        unsigned int h = vmm_hash(ptr);
        VMMEntry* entry = malloc(sizeof(VMMEntry));

        if (entry == NULL) {
            fprintf(stderr, "FATAL: Host OOM\n");
            return NULL;
        }
        *entry = (VMMEntry) {
            .ptr = ptr,
            .handle = alloc_handle,
            .size = size,
            .next = vmm_table[h],
        };
        vmm_table[h] = entry;
    }

    printf("FK2 Custom Alloc: ptr=%p, size=%zu, device=%d phys(%llx)\n", ptr, size, device, (unsigned long long)alloc_handle);
    return ptr;
}

SHARED_EXPORT
void free_fn(void* ptr, size_t size, int device, cudaStream_t stream) {
    if (ptr == NULL) {
        return;
    }

    for (VMMEntry **curr = &vmm_table[vmm_hash(ptr)]; *curr; curr = &(*curr)->next) {
        VMMEntry *entry = *curr;
        if (entry->ptr != ptr) {
            continue;
        }

        CHECK_CU(cuMemUnmap((CUdeviceptr)ptr, entry->size));
        CHECK_CU(cuMemRelease(entry->handle));
        CHECK_CU(cuMemAddressFree((CUdeviceptr)ptr, entry->size));

        *curr = entry->next;
        free(entry);
        printf("Custom Free: ptr=%p, size=%zu, stream=%p\n", ptr, size, stream);
        return;
    }

    fprintf(stderr, "WARNING: free_fn could not find pointer %p in lookup table\n", ptr);
}
