#include "dynbuf.h"
#include "plat.h"

#include <cuda.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#define MAX_BUFFER_RESERVATION (8ULL * 1024 * 1024 * 1024)

bool dynbuf_create(DynamicBuffer *buf) {
    buf->committed = 0;
#if defined(_WIN32) || defined(_WIN64)
    buf->base_ptr = VirtualAlloc(NULL, MAX_BUFFER_RESERVATION, MEM_RESERVE, PAGE_READWRITE);
#else
    buf->base_ptr = mmap(NULL, MAX_BUFFER_RESERVATION, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (buf->base_ptr == MAP_FAILED) {
        buf->base_ptr = NULL;
    }
#endif
    return buf->base_ptr != NULL;
}

bool dynbuf_grow(DynamicBuffer *buf, size_t required_size) {
    if (required_size <= buf->committed) {
        return true;
    }
    if (required_size > MAX_BUFFER_RESERVATION) {
        log(ERROR, "%s: Requested size %llu exceeds max reservation %llu\n",
            __func__, (ull)required_size, (ull)MAX_BUFFER_RESERVATION);
        return false;
    }

#if defined(_WIN32) || defined(_WIN64)
    if (!VirtualAlloc(buf->base_ptr, required_size, MEM_COMMIT, PAGE_READWRITE)) {
        log(ERROR, "%s: VirtualAlloc MEM_COMMIT failed for size %llu\n",
            __func__, (ull)required_size);
        return false;
    }
#endif

    if ((buf->committed > 0 && !CHECK_CU(cuMemHostUnregister(buf->base_ptr))) ||
        !CHECK_CU(cuMemHostRegister(buf->base_ptr, required_size, CU_MEMHOSTREGISTER_DEVICEMAP))) {
        buf->committed = 0;
        return false;
    }

    buf->committed = required_size;
    return true;
}

void dynbuf_destroy(DynamicBuffer *buf) {
    if (!buf->base_ptr) {
        return;
    }

    if (buf->committed > 0) {
        CHECK_CU(cuMemHostUnregister(buf->base_ptr));
    }

#if defined(_WIN32) || defined(_WIN64)
    VirtualFree(buf->base_ptr, 0, MEM_RELEASE);
#else
    munmap(buf->base_ptr, MAX_BUFFER_RESERVATION);
#endif
    buf->base_ptr = NULL;
    buf->committed = 0;
}
