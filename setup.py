from setuptools import setup, Extension
import pybind11
import os

# Define the extension
ext_modules = [
    Extension(
        "chess_engine_cpp",
        ["cpp/bindings.cpp", "cpp/mcts.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
    ),
]

setup(
    name="chess_engine_cpp",
    version="0.1.0",
    description="C++ MCTS for Chess Engine",
    ext_modules=ext_modules,
)
