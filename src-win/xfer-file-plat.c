#include "plat.h"
#include "xfer-file.h"

#include <windows.h>

void xfer_file_prefetch(const void *ptr, size_t size) {
    WIN32_MEMORY_RANGE_ENTRY range = {
        .VirtualAddress = (PVOID)ptr,
        .NumberOfBytes = size,
    };

    PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
}

bool xfer_file_read_at(XferFileHandle file_handle, uint64_t offset, void *destination,
                       size_t size, bool mark_cold) {
    HANDLE handle = (HANDLE)(uintptr_t)file_handle;
    size_t done = 0;

    (void)mark_cold;
    while (done < size) {
        DWORD got = 0;
        HANDLE event = CreateEventW(NULL, TRUE, FALSE, NULL);
        OVERLAPPED overlapped = {
            .Offset = (DWORD)((offset + done) & 0xffffffffu),
            .OffsetHigh = (DWORD)((offset + done) >> 32),
            .hEvent = event,
        };

        if (!event) {
            return false;
        }
        if (!ReadFile(handle, (char *)destination + done,
                      (DWORD)MIN((uint64_t)0x7ffff000, (uint64_t)(size - done)),
                      &got, &overlapped)) {
            DWORD err = GetLastError();

            if (err != ERROR_IO_PENDING || !GetOverlappedResult(handle, &overlapped, &got, TRUE)) {
                CloseHandle(event);
                return false;
            }
        }
        CloseHandle(event);
        if (got == 0) {
            return false;
        }
        done += got;
    }

    return true;
}
