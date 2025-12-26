#pragma once

#if defined(_WIN32) || defined(_WIN64)
#define SHARED_EXPORT __declspec(dllexport)
#else
#define SHARED_EXPORT
#endif
