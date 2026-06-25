/* src-posix/ze-funchooks.c
 *
 * Linux (funchook) installer for the Intel XPU build, mirroring
 * src-posix/cuda-funchooks.c but hooking the Level Zero loader instead of
 * libcuda. The platform-independent hook bodies live in
 * src-xpu/ze-hooks-shared.h (the Windows/Detours installer in
 * src-xpu/ze-detour.c shares them).
 */
#define _GNU_SOURCE

#include "plat.h"
#include "ze-shim.h"

#include <funchook.h>
#include <dlfcn.h>

#include "ze-hooks-shared.h"

static funchook_t *funchook_state;
static void *ze_handle;

static void *resolve_ze_symbol(const char *name) {
    void *sym = NULL;

    if (ze_handle) {
        sym = dlsym(ze_handle, name);
    }
    if (!sym) {
        /* ze_loader may already be loaded into the global scope by torch. */
        sym = dlsym(RTLD_DEFAULT, name);
    }
    return sym;
}

bool aimdo_setup_hooks(void) {
    int status;

    if (!ze_handle) {
        ze_handle = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!ze_handle) {
            ze_handle = dlopen("libze_loader.so", RTLD_NOW | RTLD_GLOBAL);
        }
    }
    if (!ze_handle) {
        log(ERROR, "%s: could not load libze_loader: %s\n", __func__, dlerror());
        return false;
    }

    funchook_state = funchook_create();
    if (!funchook_state) {
        log(ERROR, "%s: funchook_create failed\n", __func__);
        goto fail_teardown;
    }

    for (size_t i = 0; i < sizeof(hooks) / sizeof(hooks[0]); i++) {
        const char *detail;
        void *target = resolve_ze_symbol(hooks[i].name);

        if (!target) {
            log(ERROR, "%s: failed to resolve %s\n", __func__, hooks[i].name);
            goto fail_teardown;
        }

        *hooks[i].true_ptr = target;
        status = funchook_prepare(funchook_state, hooks[i].true_ptr, hooks[i].hook_ptr);
        if (status != FUNCHOOK_ERROR_SUCCESS) {
            detail = funchook_error_message(funchook_state);
            log(ERROR, "%s: funchook_prepare(%s) failed: %d %s\n", __func__, hooks[i].name,
                status, detail ? detail : "<unknown funchook error>");
            goto fail_teardown;
        }
    }

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

    if (funchook_state) {
        if (((status = funchook_uninstall(funchook_state, 0)) != FUNCHOOK_ERROR_SUCCESS &&
             status != FUNCHOOK_ERROR_NOT_INSTALLED) ||
            (status = funchook_destroy(funchook_state)) != FUNCHOOK_ERROR_SUCCESS) {
            const char *detail = funchook_error_message(funchook_state);

            log(ERROR, "%s: funchook teardown failed: %d %s\n", __func__, status,
                detail ? detail : "<unknown funchook error>");
        }
        funchook_state = NULL;
    }

    if (ze_handle) {
        dlclose(ze_handle);
        ze_handle = NULL;
    }
}
