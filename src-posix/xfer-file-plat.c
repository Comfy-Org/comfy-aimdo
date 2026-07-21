#define _GNU_SOURCE

#include "plat.h"
#include "xfer-file.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#define XFER_FILE_DIRECT_ALIGNMENT 4096

XferFileHandle xfer_file_open_direct(XferFileHandle file_handle) {
    char path[64];
    int length = snprintf(path, sizeof(path), "/proc/self/fd/%d", (int)file_handle);
    int fd;

    if (length < 0 || (size_t)length >= sizeof(path)) {
        return 0;
    }
    fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        log(ERROR, "%s: open failed (errno=%d)\n", __func__, errno);
        return 0;
    }
    return (XferFileHandle)fd;
}

void xfer_file_close_direct(XferFileHandle file_handle) {
    close((int)file_handle);
}

bool xfer_file_read_at(XferFileHandle file_handle, uint64_t offset, void *destination,
                       size_t size, bool mark_cold) {
    int fd = (int)file_handle;
    size_t done = 0;

    while (done < size) {
        ssize_t n = pread(fd, (char *)destination + done, size - done,
                          (off_t)(offset + done));

        if (n <= 0) {
            log(ERROR, "%s: pread failed at %llu (errno=%d)\n", __func__,
                (ull)(offset + done), errno);
            return false;
        }
        if (mark_cold) {
            int err = posix_fadvise(fd, (off_t)(offset + done), (off_t)n, POSIX_FADV_DONTNEED);

            if (err) {
                log_shot(WARNING, "%s: posix_fadvise failed at %llu size=%zd errno=%d\n",
                         __func__, (ull)(offset + done), n, err);
            }
        }
        done += (size_t)n;
    }

    return true;
}

bool xfer_file_read_at_direct(XferFileHandle file_handle, uint64_t offset,
                              void *destination, size_t size) {
    int fd = (int)file_handle;
    size_t read_size = ALIGN_UP(size, XFER_FILE_DIRECT_ALIGNMENT);
    ssize_t n = pread(fd, destination, read_size, (off_t)offset);

    if (n < 0 || (size_t)n < size) {
        log(ERROR, "%s: pread failed at %llu size=%zu result=%zd (errno=%d)\n",
            __func__, (ull)offset, size, n, errno);
        return false;
    }
    return true;
}
