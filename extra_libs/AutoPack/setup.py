"""Build the standalone AutoPack C++ extension."""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


setup(
    name="AutoPack",
    version="0.1.0",
    description="C++ rectangle packing extension for MIKU tools",
    python_requires=">=3.10",
    ext_modules=[Pybind11Extension("AutoPack", ["pack.cpp"], cxx_std=17)],
    cmdclass={"build_ext": build_ext},
)
