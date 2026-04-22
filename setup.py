from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform

numpy_include_dir = np.get_include()
sysname = platform.system()

if sysname == "Windows":
    extra_compile_args = ["/O2", "/GL", "/Gw", "/Gy", "/fp:fast", "/DNDEBUG"]
    extra_link_args = ["/LTCG"]
elif sysname == "Darwin":
    extra_compile_args = ["-O3", "-flto", "-fvisibility=hidden", "-DNDEBUG"]
    extra_link_args = ["-flto"]
else:
    extra_compile_args = ["-O3", "-flto", "-fvisibility=hidden", "-fPIC", "-DNDEBUG"]
    extra_link_args = ["-flto"]

define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

extensions = [
    Extension(
        "marsopt.variable",
        ["marsopt/variable.pyx"],
        include_dirs=[numpy_include_dir],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    ),
    Extension(
        "marsopt.solver",
        ["marsopt/solver.pyx"],
        include_dirs=[numpy_include_dir],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]

ext_modules = cythonize(
    extensions,
    language_level=3,
    compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "nonecheck": False,
        "cdivision": True,
        "infer_types": True,
        "embedsignature": True,
    },
)

setup(ext_modules=ext_modules)
