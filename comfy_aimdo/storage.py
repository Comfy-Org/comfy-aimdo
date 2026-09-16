import ctypes
import os

from . import control

lib = control.lib if os.name == "nt" else None

if lib is not None:
    lib.aimdo_storage_fast_disk.argtypes = [ctypes.c_wchar_p]
    lib.aimdo_storage_fast_disk.restype = ctypes.c_int


def fast_disk(path):
    if lib is None:
        return None
    result = lib.aimdo_storage_fast_disk(str(path))
    if result < 0:
        return None
    return bool(result)
