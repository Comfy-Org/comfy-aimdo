#ifndef PLAT_THREAD_H
#define PLAT_THREAD_H

#include <stdbool.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
typedef HANDLE Mutex;
typedef HANDLE Thread;
typedef DWORD (WINAPI *ThreadProc)(void*);
#else
#include <pthread.h>
#include <unistd.h>
typedef pthread_mutex_t* Mutex;
typedef pthread_t Thread;
typedef void* (*ThreadProc)(void*);
#endif

Mutex mutex_create(void);
void mutex_lock(Mutex m);
void mutex_unlock(Mutex m);
void mutex_destroy(Mutex m);

bool thread_create(Thread *t, ThreadProc proc, void *arg);
void thread_join(Thread t);

void sleep_us(unsigned int us);

#endif /* PLAT_THREAD_H */
