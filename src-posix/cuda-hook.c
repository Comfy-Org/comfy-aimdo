/*
 * cuda-hook.c — ELF GOT/PLT hooking for CUDA APIs on Linux.
 *
 * This is the POSIX equivalent of src-win/cuda-detour.c. It patches the
 * Global Offset Table (GOT) entries of all loaded shared libraries so that
 * CUDA memory allocation calls are redirected through aimdo's tracking.
 *
 * On Windows, Detours patches driver API entry points inside nvcuda.dll
 * which catches all callers (including libcudart). On Linux, libcudart
 * resolves driver symbols via dlsym at runtime — there are no GOT entries
 * to patch for the driver API. Instead we hook the CUDA *runtime* API
 * (cudaMalloc, cudaFree, cudaMallocAsync, cudaFreeAsync) which IS linked
 * normally by consumers such as libtorch_cuda.so.
 *
 * Runtime hooks call through to the REAL runtime API (preserving CUDA
 * memory pools, stream ordering, etc.) and wrap with aimdo's budget
 * checking and allocation tracking. Driver API hooks are also installed
 * for any library that links against libcuda.so directly.
 */

#define _GNU_SOURCE
#include "plat.h"

#include <dlfcn.h>
#include <link.h>
#include <elf.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <pthread.h>

/* ── Real (original) function pointers — Driver API ─────────────────── */

static CUresult (*true_cuMemAlloc_v2)(CUdeviceptr *, size_t);
static CUresult (*true_cuMemFree_v2)(CUdeviceptr);
static CUresult (*true_cuMemAllocAsync)(CUdeviceptr *, size_t, CUstream);
static CUresult (*true_cuMemFreeAsync)(CUdeviceptr, CUstream);

/* ── Real (original) function pointers — Runtime API ────────────────── */

static cudaError_t (*true_cudaMalloc)(void **, size_t);
static cudaError_t (*true_cudaFree)(void *);
static cudaError_t (*true_cudaMallocAsync)(void **, size_t, cudaStream_t);
static cudaError_t (*true_cudaFreeAsync)(void *, cudaStream_t);

/* ── Runtime hook allocation tracking ───────────────────────────────── *
 * Runtime hooks call through the real runtime API (not the driver API)
 * to preserve CUDA memory pool state. We maintain a separate size table
 * here so we can update total_vram_usage on alloc and free.
 */

#define RT_HASH_SIZE 1024
#define RT_MALLOC_HEADROOM (128 * M)

typedef struct RTSizeEntry {
    void              *ptr;
    size_t             size;
    struct RTSizeEntry *next;
} RTSizeEntry;

static RTSizeEntry         *rt_table[RT_HASH_SIZE];
static pthread_mutex_t      rt_lock = PTHREAD_MUTEX_INITIALIZER;

static inline unsigned int rt_hash(void *ptr) {
    return ((uintptr_t)ptr >> 10 ^ (uintptr_t)ptr >> 21) % RT_HASH_SIZE;
}

static void rt_account_alloc(void *ptr, size_t size) {
    unsigned int h = rt_hash(ptr);
    RTSizeEntry *entry;

    pthread_mutex_lock(&rt_lock);
    total_vram_usage += CUDA_ALIGN_UP(size);

    entry = (RTSizeEntry *)malloc(sizeof(*entry));
    if (entry) {
        entry->ptr  = ptr;
        entry->size = size;
        entry->next = rt_table[h];
        rt_table[h] = entry;
    }
    pthread_mutex_unlock(&rt_lock);
}

static void rt_account_free(void *ptr) {
    unsigned int h;
    RTSizeEntry *entry, **prev;

    if (!ptr) return;

    h = rt_hash(ptr);
    pthread_mutex_lock(&rt_lock);
    prev  = &rt_table[h];
    entry = rt_table[h];

    while (entry) {
        if (entry->ptr == ptr) {
            *prev = entry->next;
            total_vram_usage -= CUDA_ALIGN_UP(entry->size);
            pthread_mutex_unlock(&rt_lock);
            free(entry);
            return;
        }
        prev  = &entry->next;
        entry = entry->next;
    }
    pthread_mutex_unlock(&rt_lock);

    log(ERROR, "%s: could not account free at %p\n", __func__, ptr);
}

/* ── Hook wrappers — Driver API (mirrors src-win/cuda-detour.c) ─────── */

static CUresult hook_cuMemAlloc_v2(CUdeviceptr *dptr, size_t size) {
    return aimdo_cuda_malloc(dptr, size, true_cuMemAlloc_v2);
}

static CUresult hook_cuMemFree_v2(CUdeviceptr dptr) {
    return aimdo_cuda_free(dptr, true_cuMemFree_v2);
}

static CUresult hook_cuMemAllocAsync(CUdeviceptr *dptr, size_t size,
                                     CUstream hStream) {
    return aimdo_cuda_malloc_async(dptr, size, hStream, true_cuMemAllocAsync);
}

static CUresult hook_cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    return aimdo_cuda_free_async(dptr, hStream, true_cuMemFreeAsync);
}

/* ── Hook wrappers — Runtime API ────────────────────────────────────── *
 * These call through the REAL runtime API (preserving pool state) and
 * wrap with aimdo's budget eviction and allocation tracking.
 */

static cudaError_t hook_cudaMalloc(void **devPtr, size_t size) {
    cudaError_t status;

    log(VVERBOSE, "%s: size=%zuk\n", __func__, size / K);

    if (!devPtr) return 1; /* cudaErrorInvalidValue */

    vbars_free(budget_deficit(size + RT_MALLOC_HEADROOM));

    status = true_cudaMalloc(devPtr, size);
    if (status == 0) { rt_account_alloc(*devPtr, size); return 0; }

    vbars_free(size + RT_MALLOC_HEADROOM);
    status = true_cudaMalloc(devPtr, size);
    if (status == 0) { rt_account_alloc(*devPtr, size); return 0; }

    *devPtr = NULL;
    return status;
}

static cudaError_t hook_cudaFree(void *devPtr) {
    cudaError_t status;

    log(VVERBOSE, "%s: ptr=%p\n", __func__, devPtr);

    if (!devPtr) return 0;

    status = true_cudaFree(devPtr);
    if (status == 0) { rt_account_free(devPtr); }

    return status;
}

static cudaError_t hook_cudaMallocAsync(void **devPtr, size_t size,
                                        cudaStream_t stream) {
    cudaError_t status;

    log(VVERBOSE, "%s: size=%zuk stream=%p\n", __func__, size / K, (void *)stream);

    if (!devPtr) return 1; /* cudaErrorInvalidValue */

    vbars_free(budget_deficit(size));

    status = true_cudaMallocAsync(devPtr, size, stream);
    if (status == 0) { rt_account_alloc(*devPtr, size); return 0; }

    vbars_free(size);
    status = true_cudaMallocAsync(devPtr, size, stream);
    if (status == 0) { rt_account_alloc(*devPtr, size); return 0; }

    *devPtr = NULL;
    return status;
}

static cudaError_t hook_cudaFreeAsync(void *devPtr, cudaStream_t stream) {
    cudaError_t status;

    log(VVERBOSE, "%s: ptr=%p stream=%p\n", __func__, devPtr, (void *)stream);

    if (!devPtr) return 0;

    status = true_cudaFreeAsync(devPtr, stream);
    if (status == 0) { rt_account_free(devPtr); }

    return status;
}

/* ── Hook table ─────────────────────────────────────────────────────── */

typedef struct {
    const char *name;     /* Symbol name to find in GOT                  */
    void       *hook;     /* Address of our replacement                  */
} HookEntry;

static HookEntry hooks[] = {
    /* Driver API — catches direct libcuda.so consumers */
    { "cuMemAlloc_v2",    (void *)hook_cuMemAlloc_v2    },
    { "cuMemFree_v2",     (void *)hook_cuMemFree_v2     },
    { "cuMemAllocAsync",  (void *)hook_cuMemAllocAsync  },
    { "cuMemFreeAsync",   (void *)hook_cuMemFreeAsync   },
    /* Runtime API — catches PyTorch / libtorch_cuda.so consumers */
    { "cudaMalloc",       (void *)hook_cudaMalloc       },
    { "cudaFree",         (void *)hook_cudaFree         },
    { "cudaMallocAsync",  (void *)hook_cudaMallocAsync  },
    { "cudaFreeAsync",    (void *)hook_cudaFreeAsync    },
};

#define NUM_HOOKS (sizeof(hooks) / sizeof(hooks[0]))

/* ── ELF GOT patching helpers ───────────────────────────────────────── */

static long page_size;

static inline void *page_align(void *addr) {
    return (void *)((uintptr_t)addr & ~(uintptr_t)(page_size - 1));
}

/*
 * For a single shared object described by `info`, walk its JMPREL
 * (PLT relocations) and RELA.DYN relocations looking for GOT slots
 * that currently point to one of our hook targets.  Replace them.
 */
static int patch_got_callback(struct dl_phdr_info *info, size_t size,
                              void *data) {
    const char *obj_name = info->dlpi_name;
    ElfW(Addr)  base     = info->dlpi_addr;

    /* Skip our own library — we need the real pointers internally. */
    if (obj_name && strstr(obj_name, "aimdo"))
        return 0;

    /* Locate the DYNAMIC segment. */
    const ElfW(Dyn) *dyn = NULL;
    for (int i = 0; i < info->dlpi_phnum; i++) {
        if (info->dlpi_phdr[i].p_type == PT_DYNAMIC) {
            dyn = (const ElfW(Dyn) *)(base + info->dlpi_phdr[i].p_vaddr);
            break;
        }
    }
    if (!dyn)
        return 0;

    /* Extract the tables we need from the DYNAMIC section. */
    const ElfW(Sym)  *symtab  = NULL;
    const char       *strtab  = NULL;
    const ElfW(Rela) *jmprel  = NULL;
    size_t            jmprel_sz = 0;
    const ElfW(Rela) *rela    = NULL;
    size_t            rela_sz = 0;

    for (const ElfW(Dyn) *d = dyn; d->d_tag != DT_NULL; d++) {
        switch (d->d_tag) {
        case DT_SYMTAB:   symtab    = (const ElfW(Sym)  *)(d->d_un.d_ptr); break;
        case DT_STRTAB:   strtab    = (const char       *)(d->d_un.d_ptr); break;
        case DT_JMPREL:   jmprel    = (const ElfW(Rela) *)(d->d_un.d_ptr); break;
        case DT_PLTRELSZ: jmprel_sz = d->d_un.d_val;                       break;
        case DT_RELA:     rela      = (const ElfW(Rela) *)(d->d_un.d_ptr); break;
        case DT_RELASZ:   rela_sz   = d->d_un.d_val;                       break;
        }
    }

    if (!symtab || !strtab)
        return 0;

    /* Helper: scan a RELA table and patch matching GOT entries. */
    #define SCAN_RELA(table, table_sz) do {                                    \
        if (!(table) || !(table_sz)) break;                                    \
        size_t _n = (table_sz) / sizeof(ElfW(Rela));                           \
        for (size_t _i = 0; _i < _n; _i++) {                                  \
            const ElfW(Rela) *r = &(table)[_i];                               \
            unsigned long sym_idx = ELF64_R_SYM(r->r_info);                   \
            const char *sym_name = strtab + symtab[sym_idx].st_name;          \
            for (size_t _h = 0; _h < NUM_HOOKS; _h++) {                       \
                if (strcmp(sym_name, hooks[_h].name) != 0) continue;           \
                void **got_slot = (void **)(base + r->r_offset);              \
                /* Skip if already patched. */                                 \
                if (*got_slot == hooks[_h].hook) continue;                     \
                void *pg = page_align(got_slot);                               \
                if (mprotect(pg, page_size * 2, PROT_READ|PROT_WRITE) != 0) { \
                    log(WARNING, "%s: mprotect RW failed for %s in %s\n",      \
                        __func__, hooks[_h].name,                              \
                        obj_name ? obj_name : "<main>");                       \
                    continue;                                                  \
                }                                                              \
                *got_slot = hooks[_h].hook;                                    \
                /* Leave pages R+W so lazy resolution of other symbols on    \
                 * the same GOT page continues to work. */                    \
                log(DEBUG, "%s: patched %s in %s\n",                           \
                    __func__, hooks[_h].name,                                  \
                    obj_name ? obj_name : "<main>");                           \
            }                                                                  \
        }                                                                      \
    } while (0)

    SCAN_RELA(jmprel, jmprel_sz);
    SCAN_RELA(rela,   rela_sz);

    #undef SCAN_RELA

    return 0;  /* Continue iterating. */
}

/* ── Public API (called from plat_init / plat_cleanup) ──────────────── */

bool aimdo_setup_hooks() {
    page_size = sysconf(_SC_PAGESIZE);

    /* ── Resolve driver API symbols from libcuda.so ─────────────────── */
    void *libcuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!libcuda) {
        log(ERROR, "%s: libcuda.so.1 not loaded in process\n", __func__);
        return false;
    }

    true_cuMemAlloc_v2   = dlsym(libcuda, "cuMemAlloc_v2");
    true_cuMemFree_v2    = dlsym(libcuda, "cuMemFree_v2");
    true_cuMemAllocAsync = dlsym(libcuda, "cuMemAllocAsync");
    true_cuMemFreeAsync  = dlsym(libcuda, "cuMemFreeAsync");
    dlclose(libcuda);

    if (!true_cuMemAlloc_v2 || !true_cuMemFree_v2 ||
        !true_cuMemAllocAsync || !true_cuMemFreeAsync) {
        log(ERROR, "%s: failed to resolve CUDA driver symbols\n", __func__);
        return false;
    }

    log(INFO, "%s: resolved 4 CUDA driver API symbols\n", __func__);

    /* ── Resolve runtime API symbols via RTLD_DEFAULT ──────────────── *
     * PyTorch bundles libcudart under versioned sonames (e.g.
     * libcudart.so.12) so dlopen("libcudart.so", RTLD_NOLOAD) fails.
     * RTLD_DEFAULT searches all loaded libraries and always works.
     */
    true_cudaMalloc      = dlsym(RTLD_DEFAULT, "cudaMalloc");
    true_cudaFree        = dlsym(RTLD_DEFAULT, "cudaFree");
    true_cudaMallocAsync = dlsym(RTLD_DEFAULT, "cudaMallocAsync");
    true_cudaFreeAsync   = dlsym(RTLD_DEFAULT, "cudaFreeAsync");

    if (!true_cudaMalloc || !true_cudaFree ||
        !true_cudaMallocAsync || !true_cudaFreeAsync) {
        log(WARNING, "%s: some runtime API symbols not found — "
            "runtime hooks will be patched but non-functional\n", __func__);
    } else {
        log(INFO, "%s: resolved 4 CUDA runtime API symbols\n", __func__);
    }

    log(INFO, "%s: patching GOTs across %zu hook targets\n",
        __func__, NUM_HOOKS);

    dl_iterate_phdr(patch_got_callback, NULL);

    log(INFO, "%s: GOT patching complete\n", __func__);
    return true;
}

void aimdo_teardown_hooks() {
    /* Restore original GOT entries on teardown. */
    /* NOTE: In practice this is called at process exit and is best-effort.
     * A full restore would require a second dl_iterate_phdr pass replacing
     * hook pointers back with originals. For now we do nothing — the process
     * is about to exit anyway.
     */
    log(DEBUG, "%s: teardown (no-op on POSIX)\n", __func__);
}
