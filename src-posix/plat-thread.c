#include "plat-thread.h"

#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

Mutex mutex_create(void) {
    pthread_mutex_t *m = malloc(sizeof(pthread_mutex_t));
    if (m && pthread_mutex_init(m, NULL) == 0) {
        return m;
    }
    free(m);
    return NULL;
}

void mutex_lock(Mutex m) {
    pthread_mutex_lock(m);
}

void mutex_unlock(Mutex m) {
    pthread_mutex_unlock(m);
}

void mutex_destroy(Mutex m) {
    pthread_mutex_destroy(m);
    free(m);
}

bool thread_create(Thread *t, THREAD_FUNC (*proc)(void *), void *arg) {
    return pthread_create(t, NULL, proc, arg) == 0;
}

void thread_join(Thread t) {
    pthread_join(t, NULL);
}

void sleep_us(unsigned int us) {
    usleep(us);
}
