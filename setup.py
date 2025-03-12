from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.1"

ext_modules = [
    Pybind11Extension(
        "petsc_miner",
        ["src/bind.cpp", "src/miner.cpp", "src/pattern.cpp"],
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

setup(
    name="petsc_miner",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
