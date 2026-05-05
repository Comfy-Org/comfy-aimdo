#include "plat.h"
#include "xfer-file.h"

#include <errno.h>
#include <unistd.h>

bool xfer_file_read_at(XferFileHandle file_handle, uint64_t offset, void *destination, size_t size) {
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
        done += (size_t)n;
    }

    return true;
}
