typedef int cudaError_t;
typedef struct CUstream_st *cudaStream_t;

static inline bool aimdo_setup_hooks() { return true; }
static inline void aimdo_teardown_hooks() {}
