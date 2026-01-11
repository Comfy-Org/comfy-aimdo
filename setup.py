import os
import sys
from setuptools import setup, Distribution

# This trick forces the wheel to be labeled with the platform (e.g., win_amd64)
# instead of "any", which is required for binary DLLs.
class BinaryDistribution(Distribution):
    def has_ext_modules(foo): return True

setup(
    distclass=BinaryDistribution,
)
