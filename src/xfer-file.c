#include "plat.h"
#include "thread-plat.h"
#include "xfer-file.h"

#define XFER_FILE_THREADS 8
#define XFER_FILE_CHUNK_SIZE (2U * M)
#define XFER_FILE_QUEUE_CAP 256
#define XFER_COPY_THREADS 2
#define XFER_COPY_CHUNK_SIZE (32U * M)
#define XFER_COPY_QUEUE_CAP 256

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
    XferFileWait *wait;
} XferFileTask;

typedef struct XferCopyGroup {
    Mutex mutex;
    CondVar condvar;
    size_t pending;
} XferCopyGroup;

typedef struct {
    const uint8_t *source;
    uint8_t *destination;
    size_t size;
    XferCopyGroup *group;
} XferCopyTask;

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
} XferFileReader;

typedef struct {
    Mutex mutex;
    CondVar has_items;
    CondVar has_space;

    Thread threads[XFER_COPY_THREADS];
    XferCopyTask tasks[XFER_COPY_QUEUE_CAP];

    unsigned head;
    unsigned tail;
    unsigned count;

    bool stop;
} XferCopyPool;

static XferFileReader g_xfer_file_reader;
static XferCopyPool g_xfer_copy_pool;

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
        ok = xfer_file_read_at(task.file_handle, task.offset, task.destination,
                               task.size, task.mark_cold);
        mutex_lock(task.wait->mutex);
        task.wait->failed = !ok || task.wait->failed;
        if (--task.wait->pending == 0) {
            condvar_signal(task.wait->condvar);
        }
        mutex_unlock(task.wait->mutex);
    }
    return 0;
}

static bool xfer_copy_task_pop(XferCopyTask *task) {
    mutex_lock(g_xfer_copy_pool.mutex);
    while (g_xfer_copy_pool.count == 0 && !g_xfer_copy_pool.stop) {
        condvar_wait(g_xfer_copy_pool.has_items, g_xfer_copy_pool.mutex);
    }
    if (g_xfer_copy_pool.count == 0) {
        mutex_unlock(g_xfer_copy_pool.mutex);
        return false;
    }
    *task = g_xfer_copy_pool.tasks[g_xfer_copy_pool.head];
    g_xfer_copy_pool.head = (g_xfer_copy_pool.head + 1) % XFER_COPY_QUEUE_CAP;
    g_xfer_copy_pool.count--;
    condvar_signal(g_xfer_copy_pool.has_space);
    mutex_unlock(g_xfer_copy_pool.mutex);
    return true;
}

static THREAD_FUNC xfer_copy_worker(void *arg) {
    XferCopyTask task;

    (void)arg;
    while (xfer_copy_task_pop(&task)) {
        memcpy(task.destination, task.source, task.size);
        mutex_lock(task.group->mutex);
        if (--task.group->pending == 0) {
            condvar_signal(task.group->condvar);
        }
        mutex_unlock(task.group->mutex);
    }
    return 0;
}

bool xfer_file_read(XferFileHandle file_handle, uint64_t offset, void *destination,
                    size_t size, bool mark_cold) {
    XferFileWait wait = {
        .mutex = mutex_create(),
        .condvar = condvar_create(),
        .pending = (size + XFER_FILE_CHUNK_SIZE - 1) / XFER_FILE_CHUNK_SIZE,
    };
    bool ok = false;

    if (!wait.mutex || !wait.condvar) {
        goto fail;
    }
    for (size_t done = 0; done < size; done += XFER_FILE_CHUNK_SIZE) {
        XferFileTask task = {
            .file_handle = file_handle,
            .offset = offset + done,
            .destination = (uint8_t *)destination + done,
            .size = MIN(XFER_FILE_CHUNK_SIZE, size - done),
            .mark_cold = mark_cold,
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

void *xfer_copy_group_create(void) {
    XferCopyGroup *group = calloc(1, sizeof(*group));

    if (!group) {
        return NULL;
    }
    group->mutex = mutex_create();
    group->condvar = condvar_create();
    if (!group->mutex || !group->condvar) {
        condvar_destroy(group->condvar);
        mutex_destroy(group->mutex);
        free(group);
        return NULL;
    }
    return group;
}

static bool xfer_copy_group_add_sized(void *group_ptr, const void *source,
                                      void *destination, size_t size,
                                      size_t task_size) {
    XferCopyGroup *group = group_ptr;

    if (!group || !source || !destination || !size || !task_size) {
        return false;
    }
    for (size_t done = 0; done < size; done += task_size) {
        XferCopyTask task = {
            .source = (const uint8_t *)source + done,
            .destination = (uint8_t *)destination + done,
            .size = MIN(task_size, size - done),
            .group = group,
        };

        mutex_lock(group->mutex);
        group->pending++;
        mutex_unlock(group->mutex);

        mutex_lock(g_xfer_copy_pool.mutex);
        while (g_xfer_copy_pool.count == XFER_COPY_QUEUE_CAP) {
            condvar_wait(g_xfer_copy_pool.has_space, g_xfer_copy_pool.mutex);
        }
        g_xfer_copy_pool.tasks[g_xfer_copy_pool.tail] = task;
        g_xfer_copy_pool.tail = (g_xfer_copy_pool.tail + 1) % XFER_COPY_QUEUE_CAP;
        g_xfer_copy_pool.count++;
        condvar_signal(g_xfer_copy_pool.has_items);
        mutex_unlock(g_xfer_copy_pool.mutex);
    }
    return true;
}

bool xfer_copy_group_add(void *group, const void *source, void *destination,
                         size_t size) {
    return xfer_copy_group_add_sized(group, source, destination, size,
                                     XFER_COPY_CHUNK_SIZE);
}

bool xfer_copy_group_add_parallel(void *group, const void *source,
                                  void *destination, size_t size) {
    size_t task_size = size >= 8 * M ? ALIGN_UP((size + 1) / 2, 64) : size;

    return xfer_copy_group_add_sized(group, source, destination, size,
                                     task_size);
}

void xfer_copy_group_sync(void *group_ptr) {
    XferCopyGroup *group = group_ptr;

    if (!group) {
        return;
    }
    mutex_lock(group->mutex);
    while (group->pending) {
        condvar_wait(group->condvar, group->mutex);
    }
    mutex_unlock(group->mutex);
}

void xfer_copy_group_wait(void **group_ptr) {
    XferCopyGroup *group;

    if (!group_ptr || !(group = *group_ptr)) {
        return;
    }
    mutex_lock(group->mutex);
    while (group->pending) {
        condvar_wait(group->condvar, group->mutex);
    }
    mutex_unlock(group->mutex);
    condvar_destroy(group->condvar);
    mutex_destroy(group->mutex);
    free(group);
    *group_ptr = NULL;
}


bool xfer_file_init(void) {
    memset(&g_xfer_file_reader, 0, sizeof(g_xfer_file_reader));
    memset(&g_xfer_copy_pool, 0, sizeof(g_xfer_copy_pool));
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
    g_xfer_copy_pool.mutex = mutex_create();
    g_xfer_copy_pool.has_items = condvar_create();
    g_xfer_copy_pool.has_space = condvar_create();
    if (!g_xfer_copy_pool.mutex || !g_xfer_copy_pool.has_items ||
        !g_xfer_copy_pool.has_space) {
        xfer_file_cleanup();
        return false;
    }
    for (size_t i = 0; i < XFER_COPY_THREADS; i++) {
        if (!thread_create(&g_xfer_copy_pool.threads[i], xfer_copy_worker, NULL)) {
            xfer_file_cleanup();
            return false;
        }
    }

    return true;
}

void xfer_file_cleanup(void) {
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
    if (g_xfer_copy_pool.mutex) {
        mutex_lock(g_xfer_copy_pool.mutex);
        g_xfer_copy_pool.stop = true;
        if (g_xfer_copy_pool.has_items) {
            condvar_broadcast(g_xfer_copy_pool.has_items);
        }
        mutex_unlock(g_xfer_copy_pool.mutex);
    }
    for (size_t i = 0; i < XFER_COPY_THREADS; i++) {
        if (g_xfer_copy_pool.threads[i]) {
            thread_join(g_xfer_copy_pool.threads[i]);
        }
    }
    condvar_destroy(g_xfer_file_reader.has_space);
    condvar_destroy(g_xfer_file_reader.has_items);
    mutex_destroy(g_xfer_file_reader.mutex);
    condvar_destroy(g_xfer_copy_pool.has_space);
    condvar_destroy(g_xfer_copy_pool.has_items);
    mutex_destroy(g_xfer_copy_pool.mutex);
    memset(&g_xfer_file_reader, 0, sizeof(g_xfer_file_reader));
    memset(&g_xfer_copy_pool, 0, sizeof(g_xfer_copy_pool));
}
