#define _GNU_SOURCE

#include "plat.h"
#include "xfer-file.h"

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

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
