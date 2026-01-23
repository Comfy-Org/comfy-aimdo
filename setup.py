import os
import sys
from setuptools import setup, Distribution

# This trick forces the wheel to be labeled with the platform (e.g., win_amd64)
# instead of "any", which is required for binary DLLs.
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return os.path.exists("comfy_aimdo/aimdo.so") or os.path.exists("comfy_aimdo/aimdo.dll")
    def get_tag(self):
        t = super().get_tag()
        return ("cp39", "abi3", t[2])

setup(
    distclass=BinaryDistribution,
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
