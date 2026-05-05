#include "plat.h"
#include "hostbuf-plat.h"

#include <windows.h>

size_t hostbuf_page_size(void) {
    static size_t page_size;

    if (!page_size) {
        SYSTEM_INFO info;

        GetSystemInfo(&info);
        page_size = info.dwPageSize;
    }

    return page_size;
}

void *hostbuf_reserve_address_space(size_t size) {
    return VirtualAlloc(NULL, size, MEM_RESERVE, PAGE_NOACCESS);
}

bool hostbuf_commit_address_space(void *ptr, size_t size) {
    return VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE) == ptr;
}

bool hostbuf_decommit_address_space(void *ptr, size_t size) {
    return VirtualFree(ptr, size, MEM_DECOMMIT);
}

void hostbuf_release_address_space(void *ptr, size_t size) {
    if (ptr) {
        VirtualFree(ptr, 0, MEM_RELEASE);
    }
}
