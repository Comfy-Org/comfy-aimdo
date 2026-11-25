#pragma once

#include <stdbool.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
typedef CRITICAL_SECTION *Mutex;
typedef CONDITION_VARIABLE *CondVar;
typedef HANDLE Thread;
typedef DWORD (WINAPI *ThreadProc)(void *);
#define THREAD_FUNC DWORD WINAPI
#else
#include <pthread.h>
typedef pthread_mutex_t *Mutex;
typedef pthread_cond_t *CondVar;
typedef pthread_t Thread;
typedef void *(*ThreadProc)(void *);
#define THREAD_FUNC void *
#endif

Mutex mutex_create(void);
void mutex_lock(Mutex mutex);
void mutex_unlock(Mutex mutex);
void mutex_destroy(Mutex mutex);

CondVar condvar_create(void);
void condvar_wait(CondVar condvar, Mutex mutex);
void condvar_signal(CondVar condvar);
void condvar_broadcast(CondVar condvar);
void condvar_destroy(CondVar condvar);

bool thread_create(Thread *thread, ThreadProc proc, void *arg);
void thread_join(Thread thread);
void thread_sleep_ms(unsigned ms);
