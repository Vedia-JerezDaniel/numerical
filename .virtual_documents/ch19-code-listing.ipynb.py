import numba
import pyximport
import cython


import numpy as np


get_ipython().run_line_magic("matplotlib", " inline")
import matplotlib.pyplot as plt


np.random.seed(0)


data = np.random.randn(50000)


def py_sum(data):
    s = 0
    for d in data:
        s += d
    return s


def py_cumsum(data):
    out = np.zeros(len(data), dtype=np.float64)
    s = 0 
    for n in range(len(data)):
        s += data[n]
        out[n] = s

    return out


get_ipython().run_line_magic("timeit", " py_sum(data)")


assert abs(py_sum(data) - np.sum(data)) < 1e-10


get_ipython().run_line_magic("timeit", " np.sum(data)")


get_ipython().run_line_magic("timeit", " py_cumsum(data)")


assert np.allclose(np.cumsum(data), py_cumsum(data))


get_ipython().run_line_magic("timeit", " np.cumsum(data)")


@numba.jit
def jit_sum(data):
    s = 0 
    for d in data:
        s += d

    return s


assert abs(jit_sum(data) - np.sum(data)) < 1e-10


get_ipython().run_line_magic("timeit", " jit_sum(data)")


jit_cumsum = numba.jit()(py_cumsum)


assert np.allclose(np.cumsum(data), jit_cumsum(data))


get_ipython().run_line_magic("timeit", " jit_cumsum(data)")


def py_julia_fractal(z_re, z_im, j):
    for m in range(len(z_re)):
        for n in range(len(z_im)):
            z = z_re[m] + 1j * z_im[n]
            for t in range(256):
                z = z ** 2 - 0.05 + 1.5j
                if np.abs(z) > 2.0:
                #if (z.real * z.real + z.imag * z.imag) > 4.0:  # a bit faster
                    j[m, n] = t
                    break


jit_julia_fractal = numba.jit(nopython=True)(py_julia_fractal)


N = 3240
j = np.zeros((N, N), np.int64)
z_real = np.linspace(-1.5, 1.5, N)
z_imag = np.linspace(-1.5, 1.5, N)


jit_julia_fractal(z_real, z_imag, j)


fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(j, cmap=plt.cm.RdBu_r,
          extent=[-1.5, 1.5, -1.5, 1.5])
ax.set_xlabel("$\mathrm{Re}(z)$", fontsize=18)
ax.set_ylabel("$\mathrm{Im}(z)$", fontsize=18)
fig.tight_layout()
# fig.savefig("ch19-numba-julia-fractal.pdf")


get_ipython().run_line_magic("timeit", " py_julia_fractal(z_real, z_imag, j)")


get_ipython().run_line_magic("timeit", " jit_julia_fractal(z_real, z_imag, j)")
254/1000 * 450 


def py_Heaviside(x):
    if x == 0.0:
        return 0.5
    if x < 0.0:
        return 0.0
    else:
        return 1.0


x = np.linspace(-2, 2, 100001)


get_ipython().run_line_magic("timeit", " [py_Heaviside(xx) for xx in x]")


np_vec_Heaviside = np.vectorize(py_Heaviside)


np_vec_Heaviside(x)


get_ipython().run_line_magic("timeit", " np_vec_Heaviside(x)")


def np_Heaviside(x):
    return (x > 0.0) + (x == 0.0)/2.0


get_ipython().run_line_magic("timeit", " np_Heaviside(x)")


@numba.vectorize([numba.float32(numba.float32), numba.float64(numba.float64)])

def jit_Heaviside(x):
    if x == 0.0:
        return 0.5
    if x < 0:
        return 0.0
    else:
        return 1.0


get_ipython().run_line_magic("timeit", " jit_Heaviside(x)")


jit_Heaviside([-1, -0.5, 0.0, 0.5, 1.0])


get_ipython().getoutput("rm cy_sum.*")


get_ipython().run_cell_magic("writefile", " cy_sum.pyx", """
def cy_sum(data):
    s = 0.0
    for d in data:
        s += d
    return s""")


get_ipython().getoutput("cython cy_sum.pyx")


# 5 lines of python code -> 1470 lines of C code ...
get_ipython().getoutput("wc cy_sum.c")


get_ipython().run_cell_magic("writefile", " setup.py", """
from distutils.core import setup
from Cython.Build import cythonize

import numpy as np
setup(ext_modules=cythonize('cy_sum.pyx'),
      include_dirs=[np.get_include()],
      requires=['Cython', 'numpy'] )""")


get_ipython().getoutput("python setup.py build_ext --inplace > /dev/null")


from cy_sum import cy_sum


cy_sum(data)


get_ipython().run_line_magic("timeit", " cy_sum(data)")


get_ipython().run_line_magic("timeit", " py_sum(data)")


get_ipython().run_cell_magic("writefile", " cy_cumsum.pyx", """
cimport numpy
import numpy

def cy_cumsum(data):
    out = numpy.zeros_like(data)
    s = 0 
    for n in range(len(data)):
        s += data[n]
        out[n] = s

    return out""")


pyximport.install(setup_args={'include_dirs': np.get_include()});


pyximport.install(setup_args=dict(include_dirs=np.get_include()));


from cy_cumsum import cy_cumsum


get_ipython().run_line_magic("timeit", " cy_cumsum(data)")


get_ipython().run_line_magic("timeit", " py_cumsum(data)")


get_ipython().run_line_magic("load_ext", " cython")


get_ipython().run_cell_magic("cython", " -a", """def cy_sum(data):
    s = 0.0
    for d in data:
        s += d
    return s""")


get_ipython().run_line_magic("timeit", " cy_sum(data)")


get_ipython().run_line_magic("timeit", " py_sum(data)")


assert np.allclose(np.sum(data), cy_sum(data))


get_ipython().run_cell_magic("cython", " -a", """cimport numpy
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_sum(numpy.ndarray[numpy.float64_t, ndim=1] data):
    cdef numpy.float64_t s = 0.0
    #cdef int n, N = data.shape[0]
    cdef int n, N = len(data)
    for n in range(N):
        s += data[n]
    return s""")


get_ipython().run_line_magic("timeit", " cy_sum(data)")


get_ipython().run_line_magic("timeit", " jit_sum(data)")


get_ipython().run_line_magic("timeit", " np.sum(data)")


get_ipython().run_cell_magic("cython", " -a", """cimport numpy
import numpy
cimport cython

ctypedef numpy.float64_t FTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_cumsum(numpy.ndarray[FTYPE_t, ndim=1] data):
    cdef int n, N = data.size
    cdef numpy.ndarray[FTYPE_t, ndim=1] out = numpy.zeros(N, dtype=data.dtype)
    cdef numpy.float64_t s = 0.0
    for n in range(N):
        s += data[n]
        out[n] = s
    return out""")


get_ipython().run_line_magic("timeit", " py_cumsum(data)")


get_ipython().run_line_magic("timeit", " cy_cumsum(data)")


get_ipython().run_line_magic("timeit", " jit_cumsum(data)")


get_ipython().run_line_magic("timeit", " np.cumsum(data)")


assert np.allclose(cy_cumsum(data), np.cumsum(data))


py_sum([1.0, 2.0, 3.0, 4.0, 5.0])


py_sum([1, 2, 3, 4, 5])


cy_sum(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))


cy_sum(np.array([1, 2, 3, 4, 5]))


get_ipython().run_cell_magic("cython", " -a", """cimport numpy
cimport cython

ctypedef fused I_OR_F_t:
    numpy.int64_t 
    numpy.float64_t 

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_fused_sum(numpy.ndarray[I_OR_F_t, ndim=1] data):
    cdef I_OR_F_t s = 0
    cdef int n, N = data.size
    for n in range(N):
        s += data[n]
    return s""")


cy_fused_sum(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))


cy_fused_sum(np.array([1, 2, 3, 4, 5]))


get_ipython().run_cell_magic("cython", " -a", """cimport numpy
cimport cython

ctypedef numpy.int64_t ITYPE_t
ctypedef numpy.float64_t FTYPE_t

cpdef inline double abs2(double complex z):
    return z.real * z.real + z.imag * z.imag

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_julia_fractal(numpy.ndarray[FTYPE_t, ndim=1] z_re, 
                     numpy.ndarray[FTYPE_t, ndim=1] z_im, 
                     numpy.ndarray[ITYPE_t, ndim=2] j):
    cdef int m, n, t, M = z_re.size, N = z_im.size
    cdef double complex z
    for m in range(M):
        for n in range(N):
            z = z_re[m] + 1.0j * z_im[n]
            for t in range(256):
                z = z ** 2 - 0.05 + 0.68j
                if abs2(z) > 4.0:
                    j[m, n] = t
                    break""")


N = 1024


j = np.zeros((N, N), dtype=np.int64)


z_real = np.linspace(-1.5, 1.5, N)


z_imag = np.linspace(-1.5, 1.5, N)


get_ipython().run_line_magic("timeit", " cy_julia_fractal(z_real, z_imag, j)")


get_ipython().run_line_magic("timeit", " jit_julia_fractal(z_real, z_imag, j)")


j1 = np.zeros((N, N), dtype=np.int64)


cy_julia_fractal(z_real, z_imag, j1)


j2 = np.zeros((N, N), dtype=np.int64)


jit_julia_fractal(z_real, z_imag, j2)


assert np.allclose(j1, j2)


get_ipython().run_cell_magic("cython", "", """
cdef extern from "math.h":
     double acos(double)

def cy_acos1(double x):
    return acos(x)""")


get_ipython().run_line_magic("timeit", " cy_acos1(0.5)")


get_ipython().run_cell_magic("cython", "", """
from libc.math cimport acos

def cy_acos2(double x):
    return acos(x)""")


get_ipython().run_line_magic("timeit", " cy_acos2(0.5)")


from numpy import arccos


get_ipython().run_line_magic("timeit", " arccos(0.5)")


from math import acos


get_ipython().run_line_magic("timeit", " acos(0.5)")


assert cy_acos1(0.5) == acos(0.5)


assert cy_acos2(0.5) == acos(0.5)


get_ipython().run_line_magic("reload_ext", " version_information")


get_ipython().run_line_magic("version_information", " numpy, cython, numba, matplotlib")



