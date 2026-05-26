#include "plat.h"
#include "hostbuf-decommit.h"
#include "hostbuf-plat.h"

typedef struct DecommitJob {
    struct DecommitJob *next;
    HostbufDecommitState *state;
    void *ptr;
    size_t size;
    bool release_address_space;
} DecommitJob;

static Mutex g_decommit_mutex;
static CondVar g_decommit_ready;
static CondVar g_decommit_space;
static Thread g_decommit_thread;
static DecommitJob *g_decommit_head;
static DecommitJob **g_decommit_tail;
static size_t g_decommit_queued;
static bool g_decommit_started;

static THREAD_FUNC hostbuf_decommit_worker(void *arg) {
    (void)arg;

    mutex_lock(g_decommit_mutex);
    for (;;) {
        DecommitJob *job;

        while (!g_decommit_head) {
            condvar_wait(g_decommit_ready, g_decommit_mutex);
        }
        job = g_decommit_head;
        g_decommit_head = job->next;
        if (!g_decommit_head) {
            g_decommit_tail = &g_decommit_head;
        }
        mutex_unlock(g_decommit_mutex);

        if (job->release_address_space) {
            hostbuf_release_address_space(job->ptr, job->size);
        } else {
            hostbuf_decommit_address_space(job->ptr, job->size);
        }

        mutex_lock(g_decommit_mutex);
        if (!job->release_address_space) {
            g_decommit_queued -= job->size;
        }
        condvar_broadcast(g_decommit_space);

        mutex_lock(job->state->mutex);
        job->state->pending--;
        condvar_broadcast(job->state->done);
        mutex_unlock(job->state->mutex);
        free(job);
    }
    return 0;
}

static bool hostbuf_decommit_init(void) {
    if (g_decommit_started) {
        return true;
    }

    g_decommit_tail = &g_decommit_head;

    if (!g_decommit_mutex) {
        g_decommit_mutex = mutex_create();
        g_decommit_ready = condvar_create();
        g_decommit_space = condvar_create();
    }
    g_decommit_started = g_decommit_mutex && g_decommit_ready && g_decommit_space &&
                         thread_create(&g_decommit_thread, hostbuf_decommit_worker, NULL);
    return g_decommit_started;
}

bool hostbuf_decommit_async(HostbufDecommitState *state, void *ptr, size_t size,
                            bool release_address_space) {
    DecommitJob *job = malloc(sizeof(*job));

    if (!job || !hostbuf_decommit_init()) {
        free(job);
        return false;
    }
    *job = (DecommitJob){
        .state = state,
        .ptr = ptr,
        .size = size,
        .release_address_space = release_address_space,
    };

    mutex_lock(g_decommit_mutex);
    while (!release_address_space && g_decommit_queued + size > HOSTBUF_DECOMMIT_QUEUE_LIMIT) {
        condvar_wait(g_decommit_space, g_decommit_mutex);
    }
    mutex_lock(state->mutex);
    state->pending++;
    mutex_unlock(state->mutex);
    *g_decommit_tail = job;
    g_decommit_tail = &job->next;
    if (!release_address_space) {
        g_decommit_queued += size;
    }
    condvar_signal(g_decommit_ready);
    mutex_unlock(g_decommit_mutex);
    return true;
}
