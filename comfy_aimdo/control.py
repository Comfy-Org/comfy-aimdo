import os
import ctypes
import platform
from pathlib import Path
import logging

lib = None

def get_lib_path():
    base_path = Path(__file__).parent.resolve()
    lib_name = None

    system = platform.system()
    if system == "Windows":
        lib_name = "aimdo.dll"
    elif system == "Linux":
        lib_name = "aimdo.so"

    return None if lib_name is None else str(base_path / lib_name)

def init():
    global lib

    if lib is not None:
        return True

    lib_path = get_lib_path()
    if lib_path is None:
        logging.info(f"Unsupported platform for comfy-aimdo: {platform.system()}")
        return False
    try:
        lib = ctypes.CDLL(lib_path)
    except Exception as e:
        logging.info(f"comfy-aimdo failed to load: {lib_path}: {e}")
        logging.info(f"NOTE: comfy-aimdo is currently only support for Nvidia GPUs")
        return False

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

    lib.set_log_level_verbose.argtypes = []
    lib.set_log_level_verbose.restype = None

    if platform.system() == "Windows":

        lib.wddm_init.argtypes = [ctypes.c_int]
        lib.wddm_init.restype = ctypes.c_bool

        lib.wddm_cleanup.argtypes = []
        lib.wddm_cleanup.restype = None

    return True

def init_device(device_id: int):
    if lib is None:
        return False
    if platform.system() == "Windows" and not lib.wddm_init(device_id):
        return False
    return True

def deinit():
    global lib
    if lib is not None and platform.system() == "Windows":
        lib.wddm_cleanup()
    lib = None


def set_log_none(): lib.set_log_level_none()
def set_log_critical(): lib.set_log_level_critical()
def set_log_error(): lib.set_log_level_error()
def set_log_warning(): lib.set_log_level_warning()
def set_log_info(): lib.set_log_level_info()
def set_log_debug(): lib.set_log_level_debug()
def set_log_verbose(): lib.set_log_level_verbose()
