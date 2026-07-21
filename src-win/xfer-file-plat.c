#include "plat.h"
#include "xfer-file.h"

#include <windows.h>

#define XFER_FILE_DIRECT_ALIGNMENT 4096

XferFileHandle xfer_file_open_direct(XferFileHandle file_handle) {
    HANDLE handle = ReOpenFile((HANDLE)(uintptr_t)file_handle, GENERIC_READ,
                               FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED);

    if (handle == INVALID_HANDLE_VALUE) {
        log(ERROR, "%s: ReOpenFile failed (error=%lu)\n", __func__, GetLastError());
        return 0;
    }
    return (XferFileHandle)(uintptr_t)handle;
}

void xfer_file_close_direct(XferFileHandle file_handle) {
    CloseHandle((HANDLE)(uintptr_t)file_handle);
}

bool xfer_file_direct_matches(XferFileHandle direct_handle,
                              XferFileHandle file_handle) {
    BY_HANDLE_FILE_INFORMATION direct_info;
    BY_HANDLE_FILE_INFORMATION source_info;

    return GetFileInformationByHandle((HANDLE)(uintptr_t)direct_handle, &direct_info) &&
           GetFileInformationByHandle((HANDLE)(uintptr_t)file_handle, &source_info) &&
           direct_info.dwVolumeSerialNumber == source_info.dwVolumeSerialNumber &&
           direct_info.nFileIndexHigh == source_info.nFileIndexHigh &&
           direct_info.nFileIndexLow == source_info.nFileIndexLow;
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

bool xfer_file_read_at_direct(XferFileHandle file_handle, uint64_t offset,
                              void *destination, size_t size) {
    HANDLE handle = (HANDLE)(uintptr_t)file_handle;
    DWORD got = 0;
    HANDLE event = CreateEventW(NULL, TRUE, FALSE, NULL);
    OVERLAPPED overlapped = {
        .Offset = (DWORD)(offset & 0xffffffffu),
        .OffsetHigh = (DWORD)(offset >> 32),
        .hEvent = event,
    };
    DWORD read_size = (DWORD)ALIGN_UP(size, XFER_FILE_DIRECT_ALIGNMENT);

    if (!event) {
        return false;
    }
    if (!ReadFile(handle, destination, read_size, &got, &overlapped)) {
        DWORD err = GetLastError();

        if (err != ERROR_IO_PENDING || !GetOverlappedResult(handle, &overlapped, &got, TRUE)) {
            log(ERROR, "%s: ReadFile failed at %llu size=%zu (error=%lu)\n",
                __func__, (ull)offset, size, err);
            CloseHandle(event);
            return false;
        }
    }
    CloseHandle(event);
    if ((size_t)got < size) {
        log(ERROR, "%s: short read at %llu size=%zu result=%lu\n",
            __func__, (ull)offset, size, got);
        return false;
    }
    return true;
}
