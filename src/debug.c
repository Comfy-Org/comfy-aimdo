#include "plat.h"

int log_level;

static inline void set_log_level(int level) {
    log_level = level;
}

/* Assume we have python integration problems and start maxed out */
__attribute__((constructor)) void init_log() {
    log_level = DEBUG;
}

SHARED_EXPORT void set_log_level_none() { set_log_level(__NONE__); }
SHARED_EXPORT void set_log_level_critical() { set_log_level(CRITICAL); }
SHARED_EXPORT void set_log_level_error() { set_log_level(ERROR); }
SHARED_EXPORT void set_log_level_warning() { set_log_level(WARNING); }
SHARED_EXPORT void set_log_level_info() { set_log_level(INFO); }
SHARED_EXPORT void set_log_level_debug() { set_log_level(DEBUG); }

static const char *level_strs [] = {
    #define LEVEL_STR1(L) [L] = #L
    LEVEL_STR1(__NONE__),
    LEVEL_STR1(CRITICAL),
    LEVEL_STR1(ERROR),
    LEVEL_STR1(WARNING),
    LEVEL_STR1(INFO),
    LEVEL_STR1(DEBUG),
};

const char *get_level_str(int level) {
    if (level < 0 || level > DEBUG) {
        return "UNKNOWN";
    }
    return level_strs[level];
}
