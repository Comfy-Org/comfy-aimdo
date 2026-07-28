#include "plat.h"
#include <windows.h>
#include <detours.h>

#include "cuda-hooks-shared.h"

static inline bool install_hook_entries(HookEntry *hooks, size_t num_hooks) {
    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    int status;

    for (size_t i = 0; i < num_hooks; i++) {
        *hooks[i].true_ptr = *hooks[i].target_ptr;
        if (!*hooks[i].true_ptr ||
            DetourAttach(hooks[i].true_ptr, hooks[i].hook_ptr) != 0) {
            log(ERROR, "%s: Hook %s failed %p\n", __func__, hooks[i].name, *hooks[i].true_ptr);
            DetourTransactionAbort();
            return false;
        }
    }

    status = (int)DetourTransactionCommit();
    if (status != 0) {
        log(ERROR, "%s: DetourTransactionCommit failed: %d\n", __func__, status);
        return false;
    }

    log(DEBUG, "%s: hooks successfully installed\n", __func__);
    return true;
}

bool aimdo_setup_hooks() {
    if (!hooks[0].target_ptr || !*hooks[0].target_ptr) {
        log(ERROR, "%s: CUDA hook targets are not resolved\n", __func__);
        return false;
    }

    log(INFO, "%s: installing %zu hooks\n", __func__, sizeof(hooks) / sizeof(HookEntry));

    return install_hook_entries((HookEntry *)hooks, sizeof(hooks) / sizeof(hooks[0]));
}

void aimdo_teardown_hooks() {
    int status;

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());

    for (size_t i = 0; i < sizeof(hooks) / sizeof(hooks[0]); i++) {
        /* Only detach if we actually successfully resolved the pointer */
        if (*hooks[i].true_ptr) {
            DetourDetach(hooks[i].true_ptr, hooks[i].hook_ptr);
        }
    }

    status = (int)DetourTransactionCommit();
    if (status != 0) {
        log(ERROR, "%s: DetourDetach failed: %d\n", __func__, status);
    } else {
        log(DEBUG, "%s: hooks successfully removed\n", __func__);
    }
}
