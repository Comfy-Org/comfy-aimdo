#include "plat.h"
#include "thread-plat.h"
#include "xfer-file.h"

#define XFER_FILE_THREADS 8
#define XFER_FILE_CHUNK_SIZE (2U * M)
#define XFER_FILE_QUEUE_CAP 256

typedef struct XferFileWait {
    Mutex mutex;
    CondVar condvar;
    size_t pending;
    bool failed;
} XferFileWait;

typedef struct {
    XferFileHandle file_handle;
    uint64_t offset;
    uint8_t *destination;
    size_t size;
    bool mark_cold;
    bool direct;
    XferFileWait *wait;
} XferFileTask;

typedef struct {
    Mutex mutex;
    CondVar has_items;
    CondVar has_space;

    Thread threads[XFER_FILE_THREADS];
    XferFileTask tasks[XFER_FILE_QUEUE_CAP];

    unsigned head;
    unsigned tail;
    unsigned count;

    bool stop;
    struct XferFileDirectHandle *direct_handles;
} XferFileReader;

typedef struct XferFileDirectHandle {
    XferFileHandle source;
    XferFileHandle direct;
    struct XferFileDirectHandle *next;
} XferFileDirectHandle;

static XferFileReader g_xfer_file_reader;

static XferFileHandle xfer_file_get_direct(XferFileHandle file_handle) {
    XferFileDirectHandle *entry;
    XferFileHandle direct;

    mutex_lock(g_xfer_file_reader.mutex);
    for (entry = g_xfer_file_reader.direct_handles; entry; entry = entry->next) {
        if (entry->source == file_handle &&
            xfer_file_direct_matches(entry->direct, file_handle)) {
            direct = entry->direct;
            mutex_unlock(g_xfer_file_reader.mutex);
            return direct;
        }
    }
    direct = xfer_file_open_direct(file_handle);
    if (!direct) {
        mutex_unlock(g_xfer_file_reader.mutex);
        return 0;
    }
    entry = malloc(sizeof(*entry));
    if (!entry) {
        xfer_file_close_direct(direct);
        mutex_unlock(g_xfer_file_reader.mutex);
        return 0;
    }
    entry->source = file_handle;
    entry->direct = direct;
    entry->next = g_xfer_file_reader.direct_handles;
    g_xfer_file_reader.direct_handles = entry;
    mutex_unlock(g_xfer_file_reader.mutex);
    return direct;
}

static bool xfer_file_task_pop(XferFileReader *reader, XferFileTask *task) {
    mutex_lock(reader->mutex);
    while (reader->count == 0 && !reader->stop) {
        condvar_wait(reader->has_items, reader->mutex);
    }
    if (reader->count == 0) {
        mutex_unlock(reader->mutex);
        return false;
    }
    *task = reader->tasks[reader->head];
    reader->head = (reader->head + 1) % XFER_FILE_QUEUE_CAP;
    reader->count--;
    condvar_signal(reader->has_space);
    mutex_unlock(reader->mutex);
    return true;
}

static THREAD_FUNC xfer_file_worker(void *arg) {
    XferFileTask task;
    bool ok;

    (void)arg;
    while (xfer_file_task_pop(&g_xfer_file_reader, &task)) {
        if (task.direct) {
            ok = xfer_file_read_at_direct(task.file_handle, task.offset,
                                          task.destination, task.size);
        } else {
            ok = xfer_file_read_at(task.file_handle, task.offset, task.destination,
                                   task.size, task.mark_cold);
        }
        mutex_lock(task.wait->mutex);
        task.wait->failed = !ok || task.wait->failed;
        if (--task.wait->pending == 0) {
            condvar_signal(task.wait->condvar);
        }
        mutex_unlock(task.wait->mutex);
    }
    return 0;
}

static bool xfer_file_read_impl(XferFileHandle file_handle, uint64_t offset,
                                void *destination, size_t size, bool mark_cold,
                                bool direct) {
    XferFileWait wait = {
        .mutex = mutex_create(),
        .condvar = condvar_create(),
        .pending = (size + XFER_FILE_CHUNK_SIZE - 1) / XFER_FILE_CHUNK_SIZE,
    };
    XferFileHandle direct_handle = direct ? xfer_file_get_direct(file_handle) : 0;
    bool ok = false;

    if ((direct && !direct_handle) || !wait.mutex || !wait.condvar) {
        goto fail;
    }
    for (size_t done = 0; done < size; done += XFER_FILE_CHUNK_SIZE) {
        XferFileTask task = {
            .file_handle = direct ? direct_handle : file_handle,
            .offset = offset + done,
            .destination = (uint8_t *)destination + done,
            .size = MIN(XFER_FILE_CHUNK_SIZE, size - done),
            .mark_cold = mark_cold,
            .direct = direct,
            .wait = &wait,
        };

        mutex_lock(g_xfer_file_reader.mutex);
        while (g_xfer_file_reader.count == XFER_FILE_QUEUE_CAP) {
            condvar_wait(g_xfer_file_reader.has_space, g_xfer_file_reader.mutex);
        }
        g_xfer_file_reader.tasks[g_xfer_file_reader.tail] = task;
        g_xfer_file_reader.tail = (g_xfer_file_reader.tail + 1) % XFER_FILE_QUEUE_CAP;
        g_xfer_file_reader.count++;
        condvar_signal(g_xfer_file_reader.has_items);
        mutex_unlock(g_xfer_file_reader.mutex);
    }
    mutex_lock(wait.mutex);
    while (wait.pending != 0) {
        condvar_wait(wait.condvar, wait.mutex);
    }
    mutex_unlock(wait.mutex);
    ok = !wait.failed;
fail:
    condvar_destroy(wait.condvar);
    mutex_destroy(wait.mutex);
    return ok;
}

bool xfer_file_read(XferFileHandle file_handle, uint64_t offset, void *destination,
                    size_t size, bool mark_cold) {
    return xfer_file_read_impl(file_handle, offset, destination, size, mark_cold, false);
}

bool xfer_file_read_direct(XferFileHandle file_handle, uint64_t offset, void *destination,
                           size_t size) {
    return xfer_file_read_impl(file_handle, offset, destination, size, false, true);
}

bool xfer_file_init(void) {
    memset(&g_xfer_file_reader, 0, sizeof(g_xfer_file_reader));
    g_xfer_file_reader.mutex = mutex_create();
    g_xfer_file_reader.has_items = condvar_create();
    g_xfer_file_reader.has_space = condvar_create();
    if (!g_xfer_file_reader.mutex || !g_xfer_file_reader.has_items || !g_xfer_file_reader.has_space) {
        xfer_file_cleanup();
        return false;
    }
    for (size_t i = 0; i < XFER_FILE_THREADS; i++) {
        if (!thread_create(&g_xfer_file_reader.threads[i], xfer_file_worker, NULL)) {
            xfer_file_cleanup();
            return false;
        }
    }

    return true;
}

void xfer_file_cleanup(void) {
    XferFileDirectHandle *entry;

    if (g_xfer_file_reader.mutex) {
        mutex_lock(g_xfer_file_reader.mutex);
        g_xfer_file_reader.stop = true;
        if (g_xfer_file_reader.has_items) {
            condvar_broadcast(g_xfer_file_reader.has_items);
        }
        mutex_unlock(g_xfer_file_reader.mutex);
    }
    for (size_t i = 0; i < XFER_FILE_THREADS; i++) {
        if (g_xfer_file_reader.threads[i]) {
            thread_join(g_xfer_file_reader.threads[i]);
        }
    }
    while ((entry = g_xfer_file_reader.direct_handles)) {
        g_xfer_file_reader.direct_handles = entry->next;
        xfer_file_close_direct(entry->direct);
        free(entry);
    }
    condvar_destroy(g_xfer_file_reader.has_space);
    condvar_destroy(g_xfer_file_reader.has_items);
    mutex_destroy(g_xfer_file_reader.mutex);
    memset(&g_xfer_file_reader, 0, sizeof(g_xfer_file_reader));
}
