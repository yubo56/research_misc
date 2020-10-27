'''
to-be-cythonized utils, use keywords instead of getters to pass params

params = eps_gw, eps_gr, eps_oct, eta
'''
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin, cos, sqrt, pow

ctypedef np.float64_t FLT

@cython.boundscheck(False)
@cython.wraparound(False)
def dadt(FLT t, np.ndarray[FLT, ndim=1] y, FLT eps_gw, FLT eps_gr, FLT eps_oct,
         FLT eta):
    cdef FLT a = y[0]
    cdef FLT e = y[1]
    cdef FLT esq = e * e
    return -64 / 5 * eps_gw / (pow(1 - esq, 3.5) * pow(a, 3)) * (
        1 + 73 * esq / 24 + 37 * esq * esq / 96)
@cython.boundscheck(False)
@cython.wraparound(False)
def dedt(FLT t, np.ndarray[FLT, ndim=1] y, FLT eps_gw, FLT eps_gr, FLT eps_oct,
         FLT eta):
    cdef FLT a = y[0]
    cdef FLT e = y[1]
    cdef FLT I = y[2]
    cdef FLT w = y[4]
    cdef FLT e2 = y[5]
    cdef FLT I2 = y[6]
    cdef FLT w2 = y[8]
    cdef FLT Itot = I +I2
    cdef FLT esq = e * e
    cdef FLT j = sqrt(1 - esq)
    return j * pow(a, 1.5) / 64 * (
        120 * e * pow(sin(Itot), 2) * sin(2 * w)
        + 15 * eps_oct / 8 * cos(w2) * (
            (4 + 3 * esq) * (3 + 5 * cos(2 * Itot)) * sin(w)
            + 210 * esq * pow(sin(Itot), 2) * sin(3 * w))
        - 15 * eps_oct / 4 * cos(Itot) * cos(w) * (
            15 * (2 + 5 * esq) * cos(2 * Itot)
            + 7 * (30 * esq * cos(2 * w) * pow(sin(Itot), 2)
                   - 2 - 9 * esq)) * sin(w2)
    ) - 304 / 15 * eps_gw * e / (pow(1 - esq, 2.5) * pow(a, 4)) * (
        1 + 121 * esq / 304)
@cython.boundscheck(False)
@cython.wraparound(False)
def dIdt(FLT t, np.ndarray[FLT, ndim=1] y, FLT eps_gw, FLT eps_gr, FLT eps_oct,
         FLT eta):
    cdef FLT a = y[0]
    cdef FLT e = y[1]
    cdef FLT I = y[2]
    cdef FLT w = y[4]
    cdef FLT e2 = y[5]
    cdef FLT I2 = y[6]
    cdef FLT w2 = y[8]
    cdef FLT Itot = I +I2
    cdef FLT esq = e * e
    cdef FLT j = sqrt(1 - esq)
    return -3 * e * pow(a, 1.5) / (32 * j) * (
        10 * sin(2 * Itot) * (e * sin(2 * w)
        + 5 * eps_oct / 8 * (2 + 5 * esq + 7 * esq * cos(2 * w)) * cos(w2) *
            sin(w))
        + 5 * eps_oct * cos(w) / 8 * (
            26 + 37 * esq - 35 * esq * cos(2 * w)
            - 15 * cos(2 * Itot) * (7 * esq * cos(2 * w) - 2 - 5 * esq))
        * sin(Itot) * sin(w2)
    )
@cython.boundscheck(False)
@cython.wraparound(False)
def dWdt(FLT t, np.ndarray[FLT, ndim=1] y, FLT eps_gw, FLT eps_gr, FLT eps_oct,
         FLT eta):
    cdef FLT a = y[0]
    cdef FLT e = y[1]
    cdef FLT I = y[2]
    cdef FLT w = y[4]
    cdef FLT e2 = y[5]
    cdef FLT I2 = y[6]
    cdef FLT w2 = y[8]
    cdef FLT Itot = I +I2
    cdef FLT esq = e * e
    cdef FLT j = sqrt(1 - esq)
    return -3 * pow(a, 1.5) / (32 * sin(I) * j) * (
        2 * (
            (2 + 3 * esq - 5 * esq * cos(2 * w))
            + 25 * eps_oct * e * cos(w) / 8 * (
                2 + 5 * esq - 7 * esq * cos(2 * w)
            ) * cos(w2)
        ) * sin(2 * Itot)
        - 5 * eps_oct * e / 8 * (
            35 * esq * (1 + 3 * cos(2 * Itot)) * cos(2 * w)
            - 46 - 17 * esq - 15 * (6 + esq) * cos(2 * Itot)
        ) * sin(Itot) * sin(w) * sin(w2)
    )
@cython.boundscheck(False)
@cython.wraparound(False)
def dwdt(FLT t, np.ndarray[FLT, ndim=1] y, FLT eps_gw, FLT eps_gr, FLT eps_oct,
         FLT eta):
    cdef FLT a = y[0]
    cdef FLT e = y[1]
    cdef FLT I = y[2]
    cdef FLT w = y[4]
    cdef FLT e2 = y[5]
    cdef FLT I2 = y[6]
    cdef FLT w2 = y[8]
    cdef FLT Itot = I +I2
    cdef FLT esq = e * e
    cdef FLT j = sqrt(1 - esq)
    cdef FLT jout = sqrt(1 - e2 * e2)
    cdef B = 2 + 5 * esq - 7 * esq * cos(2 * w)
    cdef A = 4 + 3 * esq - 5 * B * pow(sin(Itot), 2) / 2
    cdef cosQ = -cos(w) * cos(w2) - cos(Itot) * sin(w) * sin(w2)
    return 3 * pow(a, 1.5) / 8 * (
        (4 * pow(cos(Itot), 2) + (5 * cos(2 * w) - 1) *
         (1 - esq - pow(cos(Itot), 2))) / j
        + eta * cos(Itot) / jout * (2 + esq * (3 - 5 * cos(2 * w)))
    ) + eps_gr / (pow(a, 2.5) * (1 - esq)) + 15 * eps_oct * pow(a, 1.5) / 64 * (
        (cos(Itot) / j + eta / jout) * e * (
            sin(w) * sin(w2) * (
                10 * (3 * pow(cos(Itot), 2) - 1) * (1 - esq) + A)
            - 5 * B * cos(Itot) * cosQ
        ) - j / e * (
            10 * sin(w) * sin(w2) * cos(Itot) * pow(sin(Itot), 2)
                * (1 - 3 * esq)
            + cosQ * (3 * A - 10 * pow(cos(Itot), 2) + 2)
        )
    )

@cython.boundscheck(False)
@cython.wraparound(False)
def de2dt(FLT t, np.ndarray[FLT, ndim=1] y, FLT eps_gw, FLT eps_gr, FLT eps_oct,
          FLT eta):
    cdef FLT a = y[0]
    cdef FLT e = y[1]
    cdef FLT I = y[2]
    cdef FLT w = y[4]
    cdef FLT e2 = y[5]
    cdef FLT I2 = y[6]
    cdef FLT w2 = y[8]
    cdef FLT Itot = I +I2
    cdef FLT esq = e * e
    cdef FLT j = sqrt(1 - esq)
    cdef FLT jout = sqrt(1 - e2 * e2)
    return (
        15 * e * eta * jout * eps_oct * pow(a, 1.5) / (256 * e2)
    ) * (
        cos(w) * (
            6 - 13 * esq + 5 * (2 + 5 * esq) * cos(2 * Itot)
            + 70 * esq * cos(2 * w) * pow(sin(Itot), 2)
        ) * sin(w2) - cos(Itot) * cos(w2) * (
            5 * (6 + esq) * cos(2 * Itot)
            + 7 * (10 * esq * cos(2 * w) * pow(sin(Itot), 2) - 2 + esq)
        ) * sin(w)
    )

@cython.boundscheck(False)
@cython.wraparound(False)
def dI2dt(FLT t, np.ndarray[FLT, ndim=1] y, FLT eps_gw, FLT eps_gr, FLT eps_oct,
          FLT eta):
    cdef FLT a = y[0]
    cdef FLT e = y[1]
    cdef FLT I = y[2]
    cdef FLT w = y[4]
    cdef FLT e2 = y[5]
    cdef FLT I2 = y[6]
    cdef FLT w2 = y[8]
    cdef FLT Itot = I +I2
    cdef FLT esq = e * e
    cdef FLT j = sqrt(1 - esq)
    cdef FLT jout = sqrt(1 - e2 * e2)
    return -(
        3 * e * eta * pow(a, 1.5) / (32 * jout)
    ) * (
        10 * (
            2 * e * sin(Itot) * sin(2 * w)
            + 5 * eps_oct * cos(w) / 8 * (2 + 5 * esq - 7 * esq * cos(2 * w))
            * sin(2 * Itot) * sin(w2)
        ) + 5 * eps_oct / 8 * (
            26 + 107 * esq + 5 * (6 + esq) * cos(2 * Itot)
            - 35 * esq * (cos(2 * Itot) - 5) * cos(2 * w)
        ) * cos(w2) * sin(Itot) * sin(w)
    )
@cython.boundscheck(False)
@cython.wraparound(False)
def dw2dt(FLT t, np.ndarray[FLT, ndim=1] y, FLT eps_gw, FLT eps_gr, FLT eps_oct,
          FLT eta):
    cdef FLT a = y[0]
    cdef FLT e = y[1]
    cdef FLT I = y[2]
    cdef FLT w = y[4]
    cdef FLT e2 = y[5]
    cdef FLT I2 = y[6]
    cdef FLT w2 = y[8]
    cdef FLT Itot = I +I2
    cdef FLT esq = e * e
    cdef FLT j = sqrt(1 - esq)
    cdef FLT jout = sqrt(1 - e2 * e2)
    cdef B = 2 + 5 * esq - 7 * esq * cos(2 * w)
    cdef A = 4 + 3 * esq - 5 * B * pow(sin(Itot), 2) / 2
    cdef cosQ = -cos(w) * cos(w2) - cos(Itot) * sin(w) * sin(w2)
    return 3 * pow(a, 1.5) / 16 * (
        2 * cos(Itot) / j * (2 + esq * (3 - 5 * cos(2 * w)))
        + eta / jout * (
            4 + 6 * esq + (5 * pow(cos(Itot), 2) - 3)
            * (2 + esq * (3 - 5 * cos(2 * w)))
        )
    ) - 15 * eps_oct * e * pow(a, 1.5) / (64 * e2) * (
        sin(w) * sin(w2) * (
            eta * (4 * pow(e2, 2) + 1) / (e2 * jout)
            * 10 * cos(Itot) * pow(sin(Itot), 2) * (1 - esq)
            - e2 * (1 / j + eta * cos(Itot) / jout)
            * (A + 10 * (3 * pow(cos(Itot), 2) - 1) * (1 - esq))
        ) + cosQ * (
            5 * B * cos(Itot) * e2 * (1 / j + eta * cos(Itot) / jout)
            + eta * (4 * pow(e2, 2) + 1) / (e2 * jout) * A
        )
    )

def dydt(FLT t, np.ndarray[FLT, ndim=1] y, FLT eps_gw, FLT eps_gr, FLT eps_oct,
         FLT eta):
    return [
        dadt(t, y, eps_gw, eps_gr, eps_oct, eta),
        dedt(t, y, eps_gw, eps_gr, eps_oct, eta),
        dIdt(t, y, eps_gw, eps_gr, eps_oct, eta),
        dWdt(t, y, eps_gw, eps_gr, eps_oct, eta),
        dwdt(t, y, eps_gw, eps_gr, eps_oct, eta),
        de2dt(t, y, eps_gw, eps_gr, eps_oct, eta),
        dI2dt(t, y, eps_gw, eps_gr, eps_oct, eta),
        dWdt(t, y, eps_gw, eps_gr, eps_oct, eta), # = dWdt
        dw2dt(t, y, eps_gw, eps_gr, eps_oct, eta),
    ]

##########################################
### VECTOR IMPLEMENTATION ################
### just used to check orb els ###########
##########################################
# - No octupole or eta

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
    cdef FLT a = y[0]
    cdef FLT jx = y[1]
    cdef FLT jy = y[2]
    cdef FLT jz = y[3]
    cdef FLT ex = y[4]
    cdef FLT ey = y[5]
    cdef FLT ez = y[6]
    cdef FLT esq = ex * ex + ey * ey + ez * ez # scalar
    cdef FLT e = sqrt(esq)
    cdef FLT x1 = 1 - esq
    cdef FLT n2x = 0, n2y = 0, n2z = 1 # e2 = 0

    # orbital evolution
    dadt = (
        -eps_gw * (64 * (1 + 73 * esq / 24 + 37 * pow(esq, 2) / 96)) / (
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
        eps_gw * (304 / 15) * (1 + 121 / 304 * esq) /
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
