from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension(
    'evection_solver',
    ['evection_solver.pyx'],
    libraries=['m', ],
    extra_compile_args=['-Ofast'],
)]

setup(
    name='evection_solver',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    include_dirs=np.get_include(),
)
