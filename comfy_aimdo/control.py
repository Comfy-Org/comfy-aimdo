import os
import ctypes
import platform
from pathlib import Path
import logging
import importlib.util
from enum import Enum

lib = None

class AimdoImpl(Enum):
    CUDA = "cuda"
    ROCM = "rocm"


def detect_vendor():
    version = ""
    try:
        torch_spec = importlib.util.find_spec("torch")
        for folder in torch_spec.submodule_search_locations:
            ver_file = Path(folder) / "version.py"
            if ver_file.is_file():
                spec = importlib.util.spec_from_file_location("torch_version_import", ver_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                version = module.__version__
    except Exception as e:
        logging.warning("Failed to detect Torch version")
        pass

    if '+cu' in version:
        return AimdoImpl.CUDA
    if '+rocm' in version:
        return AimdoImpl.ROCM
    return None


def init(implementation: AimdoImpl | None = None):
    global lib

    if lib is not None:
        return True

    if implementation is None:
        implementation = detect_vendor()

    if implementation is None:
        logging.warning("Could not autodetect AIMDO implementation, assuming Nvidia")
        implementation = AimdoImpl.CUDA

    impl = {
        AimdoImpl.CUDA: "aimdo",
        AimdoImpl.ROCM: "aimdo_rocm"
    }[implementation]

    try:
        base_path = Path(__file__).parent.resolve()
        system = platform.system()
        errors = []
        if system == "Windows":
            ext = "dll"
        elif system == "Linux":
            ext = "so"
        else:
            logging.info(f"comfy-aimdo unsupported operating system: {system}")
            logging.info(f"NOTE: comfy-aimdo currently only supports Windows and Linux")
            return False
        lib = ctypes.CDLL(str(base_path / f"{impl}.{ext}"), mode=258)
    except Exception as e:
        logging.info(f"comfy-aimdo failed to load: {e}")
        logging.info(f"NOTE: comfy-aimdo currently only supports Nvidia and AMD GPUs")
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
