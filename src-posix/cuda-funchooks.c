#define _GNU_SOURCE

#include "plat.h"

#include <funchook.h>

static funchook_t *funchook_state;

#include "cuda-hooks-shared.h"

static bool prepare_hook_entries(const HookEntry *entries, size_t count) {
    int status;

    for (size_t i = 0; i < count; i++) {
        const char *detail;

        if (!*entries[i].target_ptr) {
            continue;
        }

        *entries[i].true_ptr = *entries[i].target_ptr;
        status = funchook_prepare(funchook_state, entries[i].true_ptr, entries[i].hook_ptr);
        if (status != FUNCHOOK_ERROR_SUCCESS) {
            detail = funchook_error_message(funchook_state);
            log(ERROR, "%s: funchook_prepare(%s) failed: %d %s\n", __func__, entries[i].name,
                status, detail ? detail : "<unknown funchook error>");
            return false;
        }
    }
    return true;
}

bool aimdo_setup_hooks(void) {
    int status;

    if (!hooks[0].target_ptr || !*hooks[0].target_ptr) {
        log(ERROR, "%s: CUDA hook targets are not resolved\n", __func__);
        return false;
    }

    funchook_state = funchook_create();
    if (!funchook_state) {
        log(ERROR, "%s: funchook_create failed\n", __func__);
        return false;
    }

    if (!prepare_hook_entries(hooks, ARRAY_SIZE(hooks))) {
        goto fail_teardown;
    }
#if !defined(__HIP_PLATFORM_AMD__)
    if (!prepare_hook_entries(runtime_hooks, ARRAY_SIZE(runtime_hooks))) {
        goto fail_teardown;
    }
#endif

    status = funchook_install(funchook_state, 0);
    if (status != FUNCHOOK_ERROR_SUCCESS) {
        const char *detail = funchook_error_message(funchook_state);

        log(ERROR, "%s: funchook_install failed: %d %s\n", __func__, status,
            detail ? detail : "<unknown funchook error>");
        goto fail_teardown;
    }

    log(DEBUG, "%s: hooks successfully installed\n", __func__);
    return true;

fail_teardown:
    aimdo_teardown_hooks();
    return false;
}

void aimdo_teardown_hooks(void) {
    int status;

    if (!funchook_state) {
        return;
    }

    if (((status = funchook_uninstall(funchook_state, 0)) != FUNCHOOK_ERROR_SUCCESS &&
         status != FUNCHOOK_ERROR_NOT_INSTALLED) ||
        (status = funchook_destroy(funchook_state)) != FUNCHOOK_ERROR_SUCCESS) {
        const char *detail = funchook_error_message(funchook_state);

        log(ERROR, "%s: funchook teardown failed: %d %s\n", __func__, status,
            detail ? detail : "<unknown funchook error>");
    }

    funchook_state = NULL;
}
