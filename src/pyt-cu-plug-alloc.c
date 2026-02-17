#include "plat.h"
#include "vrambuf.h"

#define MIN_ALLOC_SHIFT 21
#define MIN_ALLOC       (1 << MIN_ALLOC_SHIFT) /*2 MB as per Cuda page sizes and pytorch cacheing alloc */

#define VMM_HASH_SIZE   (1 << 12) 

static VramBuffer* vmm_table[VMM_HASH_SIZE];

static inline unsigned int vmm_hash(CUdeviceptr ptr) {
    return ((uintptr_t)(void *)ptr >> MIN_ALLOC_SHIFT) % VMM_HASH_SIZE;
}

SHARED_EXPORT
void *alloc_fn(size_t size, int device, cudaStream_t stream) {
    CUresult err;
    VramBuffer *entry;
    size_t virt_size = size;

    log(VERBOSE, "%s (start): size=%zuk, device=%d\n", __func__, size / K, device);

    if (virt_size < MIN_ALLOC) {
        log(WARNING, "Unexpected small allocation from pytorch. Rounding up virt allocation to 2MB");
        virt_size = MIN_ALLOC;
    }

    entry = vrambuf_create(virt_size, device);
    if (!entry) {
        return NULL;
    }
    if (!vrambuf_grow(entry, size)) {
        vrambuf_destroy(entry);
        return NULL;
    }

    {
        unsigned int h = vmm_hash(vrambuf_get(entry));
        entry->next = vmm_table[h];
        vmm_table[h] = entry;
    }

    log(VERBOSE, "%s (return): ptr=%p\n", __func__, (void *)vrambuf_get(entry));
    return (void *)vrambuf_get(entry);
}

SHARED_EXPORT
void free_fn(void* ptr, size_t size, int device, cudaStream_t stream) {
    log_shot(DEBUG, "Pytorch is freeing VRAM ...\n");
    log(VERBOSE, "%s (start) ptr=%p size=%zuk, device=%d\n", __func__, ptr, size / K, device);
    if (ptr == NULL) {
        return;
    }

    for (VramBuffer **curr = &vmm_table[vmm_hash((CUdeviceptr)ptr)]; *curr; curr = &(*curr)->next) {
        VramBuffer *entry = *curr;
        if (vrambuf_get(entry) != (CUdeviceptr)ptr || entry->device != device) {
            continue;
        }

        *curr = entry->next;
        vrambuf_destroy(entry);
        log(VERBOSE, "Freed: ptr=%p, size=%zuk, stream=%p\n", ptr, size / K, stream);
        return;
    }

    log(ERROR, "%s could not find VRAM@%p\n", __func__, ptr);
}
