#pragma once

#ifdef _WIN32
#include <windows.h>
#define GET_TICK() GetTickCount64()
#else
#include <sys/time.h>
static inline uint64_t get_tick_linux() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}
#define GET_TICK() get_tick_linux()
#endif
