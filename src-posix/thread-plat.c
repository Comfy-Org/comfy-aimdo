#include "thread-plat.h"

#include <stdlib.h>
#include <time.h>

Mutex mutex_create(void) {
    pthread_mutex_t *mutex = malloc(sizeof(*mutex));

    if (mutex && pthread_mutex_init(mutex, NULL) == 0) {
        return mutex;
    }

    free(mutex);
    return NULL;
}

void mutex_lock(Mutex mutex) {
    pthread_mutex_lock(mutex);
}

void mutex_unlock(Mutex mutex) {
    pthread_mutex_unlock(mutex);
}

void mutex_destroy(Mutex mutex) {
    if (!mutex) {
        return;
    }

    pthread_mutex_destroy(mutex);
    free(mutex);
}

CondVar condvar_create(void) {
    pthread_cond_t *condvar = malloc(sizeof(*condvar));

    if (condvar && pthread_cond_init(condvar, NULL) == 0) {
        return condvar;
    }

    free(condvar);
    return NULL;
}

void condvar_wait(CondVar condvar, Mutex mutex) {
    pthread_cond_wait(condvar, mutex);
}

void condvar_signal(CondVar condvar) {
    pthread_cond_signal(condvar);
}

void condvar_broadcast(CondVar condvar) {
    pthread_cond_broadcast(condvar);
}

void condvar_destroy(CondVar condvar) {
    if (!condvar) {
        return;
    }

    pthread_cond_destroy(condvar);
    free(condvar);
}

bool thread_create(Thread *thread, ThreadProc proc, void *arg) {
    return pthread_create(thread, NULL, proc, arg) == 0;
}

void thread_join(Thread thread) {
    pthread_join(thread, NULL);
}

void thread_sleep_ms(unsigned ms) {
    struct timespec delay = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&delay, NULL);
}
