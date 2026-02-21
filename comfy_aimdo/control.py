import os
import ctypes
import platform
from pathlib import Path
import logging

lib = None

def init():
    global lib

    if lib is not None:
        return True

    try:
        base_path = Path(__file__).parent.resolve()
        system = platform.system()
        if system == "Windows":
            lib = ctypes.CDLL(str(base_path / "aimdo.dll"))
        elif system == "Linux":
            lib = ctypes.CDLL(str(base_path / "aimdo.so"), mode=258)
        else:
            logging.info(f"comfy-aimdo os not supported {system}")
            logging.info(f"NOTE: comfy-aimdo is currently only support for Windows and Linux")
            return False
    except Exception as e:
        logging.info(f"comfy-aimdo failed to load: {e}")
        logging.info(f"NOTE: comfy-aimdo is currently only support for Nvidia GPUs")
        return False

    lib.get_total_vram_usage.argtypes = []
    lib.get_total_vram_usage.restype = ctypes.c_uint64

    lib.init.argtypes = [ctypes.c_int]
    lib.init.restype = ctypes.c_bool

    return True

def init_device(device_id: int):
    if lib is None:
        return False

    return lib.init(device_id)

def deinit():
    global lib
    if lib is not None:
        lib.cleanup()
    lib = None


def set_log_none(): lib.set_log_level_none()
def set_log_critical(): lib.set_log_level_critical()
def set_log_error(): lib.set_log_level_error()
def set_log_warning(): lib.set_log_level_warning()
def set_log_info(): lib.set_log_level_info()
def set_log_debug(): lib.set_log_level_debug()
def set_log_verbose(): lib.set_log_level_verbose()
def set_log_vverbose(): lib.set_log_level_vverbose()

def analyze():
    if lib is None:
        return
    lib.aimdo_analyze()

def get_total_vram_usage():
    return 0 if lib is None else lib.get_total_vram_usage()
