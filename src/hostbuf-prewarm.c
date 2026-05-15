#include "plat.h"
#include "thread-plat.h"
#include "hostbuf-plat.h"
#include "hostbuf-prewarm.h"

#define HOSTBUF_PREWARM_THREADS 6

typedef struct {
    size_t index;
} PrewarmWorker;

typedef struct {
    Mutex mutex;
    CondVar work_cond;
    CondVar done_cond;
    volatile uint8_t *ptr;
    uint64_t generation;
    size_t remaining;
    size_t page_count;
    size_t page_span;
    size_t page_size;
} PrewarmPool;

static PrewarmPool g_prewarm_pool;
static Thread g_threads[HOSTBUF_PREWARM_THREADS];
static PrewarmWorker g_workers[HOSTBUF_PREWARM_THREADS];

static THREAD_FUNC hostbuf_prewarm_worker(void *arg) {
    PrewarmWorker *worker = (PrewarmWorker *)arg;
    volatile uint8_t *ptr;
    size_t page_end;
    size_t page_start;
    uint64_t generation = 0;

    for (;;) {
        mutex_lock(g_prewarm_pool.mutex);
        while (generation == g_prewarm_pool.generation) {
            condvar_wait(g_prewarm_pool.work_cond, g_prewarm_pool.mutex);
        }
        generation = g_prewarm_pool.generation;
        ptr = g_prewarm_pool.ptr;
        mutex_unlock(g_prewarm_pool.mutex);

        page_start = worker->index * g_prewarm_pool.page_span;
        page_end = MIN(page_start + g_prewarm_pool.page_span, g_prewarm_pool.page_count);
        for (size_t i = page_start; i < page_end; i++) {
            ptr[i * g_prewarm_pool.page_size] = 0;
        }

        mutex_lock(g_prewarm_pool.mutex);
        if (--g_prewarm_pool.remaining == 0) {
            condvar_broadcast(g_prewarm_pool.done_cond);
        }
        mutex_unlock(g_prewarm_pool.mutex);
    }
    return 0;
}

static bool hostbuf_prewarm_pool_init(void) {
    g_prewarm_pool.mutex = mutex_create();
    g_prewarm_pool.work_cond = condvar_create();
    g_prewarm_pool.done_cond = condvar_create();
    if (!g_prewarm_pool.mutex || !g_prewarm_pool.work_cond || !g_prewarm_pool.done_cond) {
        return false;
    }
    for (size_t i = 0; i < HOSTBUF_PREWARM_THREADS; i++) {
        g_workers[i].index = i;
        if (!thread_create(&g_threads[i], hostbuf_prewarm_worker, &g_workers[i])) {
            return false;
        }
    }
    return true;
}

bool hostbuf_prewarm_join(void) {
    if (!g_prewarm_pool.mutex) {
        return true;
    }

    mutex_lock(g_prewarm_pool.mutex);
    while (g_prewarm_pool.remaining) {
        condvar_wait(g_prewarm_pool.done_cond, g_prewarm_pool.mutex);
    }
    mutex_unlock(g_prewarm_pool.mutex);
    return true;
}

bool hostbuf_prewarm_start(void *ptr, size_t size) {
    size_t page_size = hostbuf_page_size();

    if (!size) {
        return hostbuf_prewarm_join();
    }

    if (!g_prewarm_pool.mutex && !hostbuf_prewarm_pool_init()) {
        return false;
    }

    mutex_lock(g_prewarm_pool.mutex);
    while (g_prewarm_pool.remaining) {
        condvar_wait(g_prewarm_pool.done_cond, g_prewarm_pool.mutex);
    }
    if (size) {
        g_prewarm_pool.ptr = ptr;
        g_prewarm_pool.page_size = page_size;
        g_prewarm_pool.page_count = (size + page_size - 1) / page_size;
        g_prewarm_pool.page_span = (g_prewarm_pool.page_count + HOSTBUF_PREWARM_THREADS - 1) /
                                   HOSTBUF_PREWARM_THREADS;
        g_prewarm_pool.remaining = HOSTBUF_PREWARM_THREADS;
        g_prewarm_pool.generation++;
        condvar_broadcast(g_prewarm_pool.work_cond);
    }
    mutex_unlock(g_prewarm_pool.mutex);
    return true;
}
