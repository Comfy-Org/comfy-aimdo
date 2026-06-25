/* src-xpu/ze-detour.c
 *
 * Windows (Detours) installer for the Intel XPU build, mirroring
 * src-win/cuda-detour.c but hooking the Level Zero loader instead of nvcuda.
 * The platform-independent hook bodies live in src-xpu/ze-hooks-shared.h (the
 * Linux/funchook installer in src-posix/ze-funchooks.c shares them).
 */
#include "plat.h"
#include "ze-shim.h"

#include <windows.h>
#include <detours.h>

#include "ze-hooks-shared.h"

static inline bool install_hook_entrys(HMODULE h, const HookEntry *entries, size_t num) {
    int status;

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());

    for (size_t i = 0; i < num; i++) {
        *entries[i].true_ptr = (void *)GetProcAddress(h, entries[i].name);
        if (!*entries[i].true_ptr ||
            DetourAttach(entries[i].true_ptr, entries[i].hook_ptr) != 0) {
            log(ERROR, "%s: Hook %s failed %p", __func__, entries[i].name, *entries[i].true_ptr);
            DetourTransactionAbort();
            return false;
        }
    }

    status = (int)DetourTransactionCommit();
    if (status != 0) {
        log(ERROR, "%s: DetourTransactionCommit failed: %d", __func__, status);
        return false;
    }

    log(DEBUG, "%s: hooks successfully installed\n", __func__);
    return true;
}

bool aimdo_setup_hooks() {
    HMODULE h_ze = GetModuleHandleA("ze_loader.dll");
    if (h_ze == NULL) {
        h_ze = GetModuleHandleA("ze_loader");
    }
    if (h_ze == NULL) {
        log(ERROR, "%s: ze_loader not found in process memory\n", __func__);
        return false;
    }

    log(INFO, "%s: found ze_loader at %p, installing %zu hooks\n",
        __func__, h_ze, sizeof(hooks) / sizeof(HookEntry));

    return install_hook_entrys(h_ze, hooks, sizeof(hooks) / sizeof(HookEntry));
}

void aimdo_teardown_hooks() {
    int status;

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());

    for (size_t i = 0; i < sizeof(hooks) / sizeof(hooks[0]); i++) {
        if (*hooks[i].true_ptr) {
            DetourDetach(hooks[i].true_ptr, hooks[i].hook_ptr);
        }
    }

    status = (int)DetourTransactionCommit();
    if (status != 0) {
        log(ERROR, "%s: DetourDetach failed: %d", __func__, status);
    } else {
        log(DEBUG, "%s: hooks successfully removed\n", __func__);
    }
}
