'''
to-be-cythonized utils, use keywords instead of getters to pass params

params = eps_gw, eps_gr, eps_oct
'''
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin, cos, sqrt, pow

ctypedef double FLT

@cython.boundscheck(False)
@cython.wraparound(False)
def dadt(FLT t, np.ndarray[FLT, ndim=1] y, FLT eps_gw, FLT eps_gr, FLT eps_oct,
         FLT eta):
    cdef FLT a = y[0], e = y[1]
    cdef FLT esq = e * e
    return -64 / 5 * eps_gw / (pow(1 - esq, 3.5) * pow(a, 3)) * (
        1 + 73 * esq / 24 + 37 * esq * esq / 96)
@cython.boundscheck(False)
@cython.wraparound(False)
def dedt(FLT t, np.ndarray[FLT, ndim=1] y, FLT eps_gw, FLT eps_gr, FLT eps_oct,
         FLT eta):
    cdef FLT a = y[0], e = y[1], I = y[2], W = y[3], w = y[4]
    cdef FLT Itot = I
    cdef FLT esq = e * e
    cdef FLT j = sqrt(1 - esq)
    return j * pow(a, 1.5) / 64 * (
        120 * e * pow(sin(Itot), 2) * sin(2 * w)
    ) - 304 / 15 * eps_gw * e / (pow(1 - esq, 2.5) * pow(a, 4)) * (
        1 + 121 * esq / 304)
@cython.boundscheck(False)
@cython.wraparound(False)
def dIdt(FLT t, np.ndarray[FLT, ndim=1] y, FLT eps_gw, FLT eps_gr, FLT eps_oct,
         FLT eta):
    cdef FLT a = y[0], e = y[1], I = y[2], W = y[3], w = y[4]
    cdef FLT Itot = I
    cdef FLT j = sqrt(1 - e * e)
    return -3 * e * pow(a, 1.5) / (32 * j) * (
        10 * sin(2 * Itot) * e * sin(2 * w))
@cython.boundscheck(False)
@cython.wraparound(False)
def dWdt(FLT t, np.ndarray[FLT, ndim=1] y, FLT eps_gw, FLT eps_gr, FLT eps_oct,
         FLT eta):
    cdef FLT a = y[0], e = y[1], I = y[2], W = y[3], w = y[4]
    cdef FLT Itot = I
    cdef FLT e_sq = e * e
    cdef FLT j = sqrt(1 - e_sq)
    return -3 * pow(a, 1.5) / (32 * sin(I) * j) * (
        2 * (2 + 3 * e_sq - 5 * e_sq * cos(2 * w)))
@cython.boundscheck(False)
@cython.wraparound(False)
def dwdt(FLT t, np.ndarray[FLT, ndim=1] y, FLT eps_gw, FLT eps_gr, FLT eps_oct,
         FLT eta):
    cdef FLT a = y[0], e = y[1], I = y[2], W = y[3], w = y[4]
    cdef FLT Itot = I
    cdef FLT e_sq = e * e
    cdef FLT j = sqrt(1 - e_sq)
    return 3 * pow(a, 1.5) / 8 * (
        (4 * pow(cos(Itot), 2) + (5 * cos(2 * w) - 1) *
         (1 - e_sq - pow(cos(Itot), 2))) / j
    ) + eps_gr / (pow(a, 2.5) * (1 - e_sq))

@cython.boundscheck(False)
@cython.wraparound(False)
def dydt(FLT t, np.ndarray[FLT, ndim=1] y, FLT eps_gw, FLT eps_gr, FLT eps_oct,
         FLT eta):
    return [
        dadt(t, y, eps_gw, eps_gr, eps_oct, eta),
        dedt(t, y, eps_gw, eps_gr, eps_oct, eta),
        dIdt(t, y, eps_gw, eps_gr, eps_oct, eta),
        dWdt(t, y, eps_gw, eps_gr, eps_oct, eta),
        dwdt(t, y, eps_gw, eps_gr, eps_oct, eta),
    ]

##########################################
### VECTOR IMPLEMENTATION (incomplete) ###
### just used to check orb els ###########
##########################################

def dot(FLT x1, FLT y1, FLT z1, FLT x2, FLT y2, FLT z2):
    return x1 * x2 + y1 * y2 + z1 * z2
def cross(FLT x1, FLT y1, FLT z1, FLT x2, FLT y2, FLT z2):
    return [
        y1 * z2 - z1 * y2,
        z1 * x2 - x1 * z2,
        x1 * y2 - y1 * x2,
    ]

@cython.boundscheck(False)
@cython.wraparound(False)
def dydt_vec(FLT t, np.ndarray[FLT, ndim=1] y, FLT eps_gw, FLT eps_gr, FLT
             eps_oct, FLT eta):
    '''
    dydt for all useful of 10 orbital elements + spin, eps_oct = 0 in LML15.
    eta = L / Lout
    '''
    cdef FLT a = y[0], jx = y[1], jy = y[2], jz=y[3],\
            ex = y[4], ey=y[5], ez=y[6]
    cdef FLT e_sq = ex * ex + ey * ey + ez * ez # scalar
    cdef FLT e = sqrt(e_sq)
    cdef FLT x1 = 1 - e_sq
    cdef FLT n2x = 0, n2y = 0, n2z = 1 # e2 = 0

    # orbital evolution
    dadt = (
        -eps_gw * (64 * (1 + 73 * e_sq / 24 + 37 * pow(e_sq, 2) / 96)) / (
            5 * pow(a, 3) * pow(x1, 3.5))
    )
    exn2 = cross(ex, ey, ez, n2x, n2y, n2z)
    edn2 = dot(ex, ey, ez, n2x, n2y, n2z)
    jxn2 = cross(jx, jy, jz, n2x, n2y, n2z)
    jdn2 = dot(jx, jy, jz, n2x, n2y, n2z)
    jxe = cross(jx, jy, jz, ex, ey, ez)
    djdtx = 3 * pow(a, 1.5) / 4 * (
        jdn2 * jxn2[0]
        - 5 * edn2 * exn2[0]
    )
    djdty = 3 * pow(a, 1.5) / 4 * (
        jdn2 * jxn2[1]
        - 5 * edn2 * exn2[1]
    )
    djdtz = 3 * pow(a, 1.5) / 4 * (
        jdn2 * jxn2[2]
        - 5 * edn2 * exn2[2]
    )
    dedt_gw = -(
        eps_gw * (304 / 15) * (1 + 121 / 304 * e_sq) /
        (pow(a, 4) * pow(x1, 2.5))
    )
    dedtx = 3 * pow(a, 1.5) / 4 * (
        jdn2 * exn2[0]
        - 5 * edn2 * jxn2[0]
        + 2 * jxe[0]
    ) + dedt_gw * ex + (
        eps_gr * jxe[0] / (pow(x1, 1.5) * pow(a, 2.5))
    )
    dedty = 3 * pow(a, 1.5) / 4 * (
        jdn2 * exn2[1]
        - 5 * edn2 * jxn2[1]
        + 2 * jxe[1]
    ) + dedt_gw * ey + (
        eps_gr * jxe[1] / (pow(x1, 1.5) * pow(a, 2.5))
    )
    dedtz = 3 * pow(a, 1.5) / 4 * (
        jdn2 * exn2[2]
        - 5 * edn2 * jxn2[2]
        + 2 * jxe[2]
    ) + dedt_gw * ez + (
        eps_gr * jxe[2] / (pow(x1, 1.5) * pow(a, 2.5))
    )
    ret = [dadt, djdtx, djdty, djdtz, dedtx, dedty, dedtz]
    return ret
