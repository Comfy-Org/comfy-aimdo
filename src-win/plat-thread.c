#include "plat-thread.h"

Mutex mutex_create(void) {
    return CreateSemaphore(NULL, 1, 1, NULL);
}

void mutex_lock(Mutex m) {
    WaitForSingleObject(m, INFINITE);
}

void mutex_unlock(Mutex m) {
    ReleaseSemaphore(m, 1, NULL);
}

void mutex_destroy(Mutex m) {
    CloseHandle(m);
}

bool thread_create(Thread *t, ThreadProc proc, void *arg) {
    *t = CreateThread(NULL, 0, proc, arg, 0, NULL);
    return *t != NULL;
}

void thread_join(Thread t) {
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
}

void sleep_us(unsigned int us) {
    if (us < 1000) {
        YieldProcessor();
    } else {
        Sleep(us / 1000);
    }
}
