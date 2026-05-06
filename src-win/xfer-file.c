#include "plat.h"
#include "xfer-file.h"

#include <windows.h>

bool xfer_file_read_at(XferFileHandle file_handle, uint64_t offset, void *destination, size_t size) {
    HANDLE handle = (HANDLE)(uintptr_t)file_handle;
    size_t done = 0;

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
