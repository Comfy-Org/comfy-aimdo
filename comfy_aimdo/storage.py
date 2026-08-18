import ctypes
import platform

from . import control


def fast_disk(path):
    if platform.system() != "Windows" or control.lib is None:
        return None
    control.lib.aimdo_storage_fast_disk.argtypes = [ctypes.c_wchar_p]
    control.lib.aimdo_storage_fast_disk.restype = ctypes.c_int
    result = control.lib.aimdo_storage_fast_disk(str(path))
    if result < 0:
        return None
    return bool(result)
