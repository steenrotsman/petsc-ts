from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "1.0"

ext_modules = [
    Pybind11Extension(
        "_petsc_miner",
        ["src/bind.cpp", "src/miner.cpp", "src/pattern.cpp"],
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

setup(
    name="_petsc_miner",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
