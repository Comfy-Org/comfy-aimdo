#pragma once

#include "gpu_dispatch.h"

void *malloc_graph_record(CUstream stream);
bool malloc_graph_push(void *graph, const char *name);
bool malloc_graph_pop(void *graph);
bool malloc_graph_replay(void *graph, CUstream stream);
void malloc_graph_destroy(void *graph);
size_t malloc_graph_stat(void *graph, int which);
bool malloc_graph_failed(void *graph);
bool malloc_graph_reject_external(CUstream stream);

CUresult malloc_graph_alloc(CUdeviceptr *ptr, size_t size, CUstream stream);
CUresult malloc_graph_free(CUdeviceptr ptr, CUstream stream);
