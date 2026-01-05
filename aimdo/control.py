#This module does not import torch by design. It can used to detect the
#libraries existance via import before the first torch import for environment
#logic.

import os
import ctypes
import platform
from pathlib import Path

def get_lib_path():
    # Get the directory where this script/package is located
    base_path = Path(__file__).parent.resolve()

    # Determine extension based on OS
    system = platform.system()
    if system == "Windows":
        lib_name = "aimdo.dll"
    elif system == "Linux":
        lib_name = "aimdo.so"
    else:
        # MacOS usually uses .dylib, though often .so works
        lib_name = "aimdo.so"

    return str(base_path / lib_name)

# Load the library
lib_path = get_lib_path()
if not os.path.exists(lib_path):
    raise ImportError(f"Cannot find native library at {lib_path}")

lib = ctypes.CDLL(lib_path)
allocator = CUDAPluggableAllocator(lib, "alloc_fn", "free_fn")

lib.set_log_level_none.argtypes = []
lib.set_log_level_none.restype = None

lib.set_log_level_critical.argtypes = []
lib.set_log_level_critical.restype = None

lib.set_log_level_error.argtypes = []
lib.set_log_level_error.restype = None

lib.set_log_level_warning.argtypes = []
lib.set_log_level_warning.restype = None

lib.set_log_level_info.argtypes = []
lib.set_log_level_info.restype = None

lib.set_log_level_debug.argtypes = []
lib.set_log_level_debug.restype = None

def set_log_none(): lib.set_log_level_none()
def set_log_critical(): lib.set_log_level_critical()
def set_log_error(): lib.set_log_level_error()
def set_log_warning(): lib.set_log_level_warning()
def set_log_info(): lib.set_log_level_info()
def set_log_debug(): lib.set_log_level_debug()
