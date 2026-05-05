#include "plat.h"
#include "hostbuf-plat.h"

#include <sys/mman.h>
#include <unistd.h>

size_t hostbuf_page_size(void) {
    static size_t page_size;

    if (!page_size) {
        page_size = (size_t)sysconf(_SC_PAGESIZE);
    }

    return page_size;
}

void *hostbuf_reserve_address_space(size_t size) {
    void *ptr = mmap(NULL, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);

    if (ptr == MAP_FAILED) {
        log(ERROR, "%s: mmap reserve failed for %zu bytes\n", __func__, size);
        return NULL;
    }

    return ptr;
}

bool hostbuf_commit_address_space(void *ptr, size_t size) {
    return mmap(ptr, size, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == ptr;
}

bool hostbuf_decommit_address_space(void *ptr, size_t size) {
    return mmap(ptr, size, PROT_NONE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0) == ptr;
}

void hostbuf_release_address_space(void *ptr, size_t size) {
    if (ptr) {
        munmap(ptr, size);
    }
}
