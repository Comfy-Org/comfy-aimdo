#define _GNU_SOURCE

#include "plat.h"
#include "xfer-file.h"

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

static bool xfer_file_transfer_at(XferFileHandle file_handle, uint64_t offset, void *buffer,
                                  size_t size, bool mark_cold, bool write) {
    int fd = (int)file_handle;
    size_t done = 0;

    while (done < size) {
        ssize_t n = write ? pwrite(fd, (char *)buffer + done, size - done, (off_t)(offset + done)) :
                            pread(fd, (char *)buffer + done, size - done, (off_t)(offset + done));

        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            log(AIMDO_LOG_ERROR, "%s: %s failed at %llu (errno=%d)\n", __func__,
                write ? "pwrite" : "pread", (ull)(offset + done), errno);
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

bool xfer_file_read_at(XferFileHandle file_handle, uint64_t offset, void *destination,
                       size_t size, bool mark_cold) {
    return xfer_file_transfer_at(file_handle, offset, destination, size, mark_cold, false);
}

bool xfer_file_write_at(XferFileHandle file_handle, uint64_t offset, void *source, size_t size) {
    return xfer_file_transfer_at(file_handle, offset, source, size, false, true);
}
