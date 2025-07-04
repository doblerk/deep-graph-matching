from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        name="gnnged.assignment.greedy_assignment",
        sources=["src/calc_greedy_assignment.cpp"],
        extra_compile_args=["-O3"],
    ),
]

setup(
    name='gnnged',
    version='2.0.0',
    packages=find_packages(),
    install_requires=[
        'optuna>=3.0.0',
        'scikit-learn>=1.0.0',
        'scipy>=1.0.0',
        'h5py>=3.0.0',
        'ortools>=9.13.0',
        'matplotlib>=3.0.0'
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)