#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
from libc.math cimport sin, cos, sqrt, pow
cimport numpy as np
cimport cython

cdef double G = 39.4751488
cdef double c = 6.32397263*10**4
def dydt_cython(double t, double[:] y, double m1, double m2, double m3,
                double fsec, double fsl, double fgwa, double fgwe):
    cdef double m12 = m1 + m2
    cdef double mu = (m2 * m1) / m12
    cdef double m123 = m12 + m3
    cdef double muout = (m12 * m3) / m123

    cdef double Linx = y[0]
    cdef double Liny = y[1]
    cdef double Linz = y[2]
    cdef double einx = y[3]
    cdef double einy = y[4]
    cdef double einz = y[5]
    cdef double routx = y[6]
    cdef double routy = y[7]
    cdef double routz = y[8]
    cdef double voutx = y[9]
    cdef double vouty = y[10]
    cdef double voutz = y[11]

    cdef double Linmag = sqrt(Linx**2 + Liny**2 + Linz**2)
    cdef double e = sqrt(einx**2 + einy**2 + einz**2)
    cdef double jsq = 1 - e**2
    cdef double routmag = sqrt(routx**2 + routy**2 + routz**2)

    cdef double ain = Linmag**2 / (mu**2 * G * (m12) * jsq)
    cdef double phi0 = G * m123 / routmag
    cdef double n = sqrt(G * m12 / ain**3)
    cdef double WSA = fsec * 3./2. * (m3 / m12) * (ain / routmag)**3 * n
    cdef double phiQ = fsec * m3 / (4 * routmag) * (mu / muout) * (ain / routmag)**2
    cdef double WSL = fsl * 3. * G * n * m12 / (c**2 * ain * (1 - e**2))
    cdef double WGWL = fgwa * 32. / 5. * G**(7./2) / c**5 * (
        mu**2 * m12**(5./2) / ain**(7./2)
        * (1. + 7. * e**2 / 8.) / (1. - e**2)**2
    )
    cdef double WGWe = fgwe * 304. / 15 * G**3 / (c**5) * (
        mu * m12**2 / ain**4
        * (1 + 121. * e**2 / 304) / (1 - e**2)**(5./2)
    )

    cdef double Lhatdotrouthat = (
            Linx * routx + Liny * routy + Linz * routz
    ) / (Linmag * routmag)
    cdef double edotrouthat = (
            routx * einx + routy * einy + routz * einz
    ) / routmag
    cdef double dLindtx = Linmag / sqrt(1 - e**2) * WSA * (
        -(1 - e**2) * Lhatdotrouthat *
            (Liny * routz - Linz * routy) / (Linmag * routmag)
        + 5 * edotrouthat * (einy * routz - einz * routy) / routmag
    ) - WGWL * Linx / Linmag
    cdef double dLindty = Linmag / sqrt(1 - e**2) * WSA * (
        -(1 - e**2) * Lhatdotrouthat *
            (Linz * routx - Linx * routz) / (Linmag * routmag)
        + 5 * edotrouthat * (einz * routx - einx * routz) / routmag
    ) - WGWL * Liny / Linmag
    cdef double dLindtz = Linmag / sqrt(1 - e**2) * WSA * (
        -(1 - e**2) * Lhatdotrouthat *
            (Linx * routy - Liny * routx) / (Linmag * routmag)
        + 5 * edotrouthat * (einx * routy - einy * routx) / routmag
    ) - WGWL * Linz / Linmag
    cdef double deindtx = WSA * sqrt(1 - e**2) * (
        -Lhatdotrouthat * (einy * routz - einz * routy) / routmag
        -2 * (Liny * einz - Linz * einy) / Linmag
        + 5 * edotrouthat * (Liny * routz - Linz * routy) / (Linmag * routmag)
    ) + WSL * (Liny * einz - Linz * einy) / Linmag - WGWe * einx / e
    cdef double deindty = WSA * sqrt(1 - e**2) * (
        -Lhatdotrouthat * (einz * routx - einx * routz) / routmag
        -2 * (Linz * einx - Linx * einz) / Linmag
        + 5 * edotrouthat * (Linz * routx - Linx * routz) / (Linmag * routmag)
    ) + WSL * (Linz * einx - Linx * einz) / Linmag - WGWe * einy / e
    cdef double deindtz = WSA * sqrt(1 - e**2) * (
        -Lhatdotrouthat * (einx * routy - einy * routx) / routmag
        -2 * (Linx * einy - Liny * einx) / Linmag
        + 5 * edotrouthat * (Linx * routy - Liny * routx) / (Linmag * routmag)
    ) + WSL * (Linx * einy - Liny * einx) / Linmag - WGWe * einz / e

    cdef double dvoutdtx = -phi0 * routx / routmag**2 - phiQ * (
        -3 * (routx / routmag**2) * (
            -1 + 6*e**2 + 3*(1 - e**2) * Lhatdotrouthat
            - 15 * edotrouthat
        ) + 6 * (1 - e**2) / routmag**2 * Lhatdotrouthat * (
            Linx / Linmag - Lhatdotrouthat * routx
        ) - 30 * edotrouthat / routmag**2 * (
            einx - edotrouthat * routx
        )
    )
    cdef double dvoutdty = -phi0 * routy / routmag**2 - phiQ * (
        -3 * (routy / routmag**2) * (
            -1 + 6*e**2 + 3*(1 - e**2) * Lhatdotrouthat
            - 15 * edotrouthat
        ) + 6 * (1 - e**2) / routmag**2 * Lhatdotrouthat * (
            Liny / Linmag - Lhatdotrouthat * routy
        ) - 30 * edotrouthat / routmag**2 * (
            einy - edotrouthat * routy
        )
    )
    cdef double dvoutdtz = -phi0 * routz / routmag**2 - phiQ * (
        -3 * (routz / routmag**2) * (
            -1 + 6*e**2 + 3*(1 - e**2) * Lhatdotrouthat
            - 15 * edotrouthat
        ) + 6 * (1 - e**2) / routmag**2 * Lhatdotrouthat * (
            Linz / Linmag - Lhatdotrouthat * routz
        ) - 30 * edotrouthat / routmag**2 * (
            einz - edotrouthat * routz
        )
    )
    return [
        dLindtx,
        dLindty,
        dLindtz,
        deindtx,
        deindty,
        deindtz,
        voutx,
        vouty,
        voutz,
        dvoutdtx,
        dvoutdty,
        dvoutdtz,
    ]

def ain_thresh_cython(t, y, m1, m2, m3, f1, f2, f3, f4):
    cdef double m12 = m1 + m2
    cdef double mu = (m2 * m1) / m12

    cdef double Linx = y[0]
    cdef double Liny = y[1]
    cdef double Linz = y[2]
    cdef double einx = y[3]
    cdef double einy = y[4]
    cdef double einz = y[5]

    cdef double Linmag = sqrt(Linx**2 + Liny**2 + Linz**2)
    cdef double e = sqrt(einx**2 + einy**2 + einz**2)
    cdef double jsq = 1 - e**2

    cdef double ain = Linmag**2 / (mu**2 * G * (m12) * jsq)
    return ain - 0.0015
