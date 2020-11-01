'''
to-be-cythonized utils, use keywords instead of getters to pass params

params = eps_gw, eps_gr, eps_oct, eta
'''
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin, cos, sqrt, pow

@cython.boundscheck(False)
@cython.wraparound(False)
def dadt(double t, np.ndarray[double, ndim=1] y, double eps_gw, double eps_gr, double eps_oct,
         double eta):
    cdef double a = y[0]
    cdef double e = y[1]
    cdef double esq = e * e
    return -64 / 5 * eps_gw / (pow(1 - esq, 3.5) * pow(a, 3)) * (
        1 + 73 * esq / 24 + 37 * esq * esq / 96)
@cython.boundscheck(False)
@cython.wraparound(False)
def dedt(double t, np.ndarray[double, ndim=1] y, double eps_gw, double eps_gr, double eps_oct,
         double eta):
    cdef double a = y[0]
    cdef double e = y[1]
    cdef double I = y[2]
    cdef double w = y[4]
    cdef double e2 = y[5]
    cdef double I2 = y[6]
    cdef double w2 = y[8]
    cdef double Itot = I +I2
    cdef double esq = e * e
    cdef double j = sqrt(1 - esq)
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
def dIdt(double t, np.ndarray[double, ndim=1] y, double eps_gw, double eps_gr, double eps_oct,
         double eta):
    cdef double a = y[0]
    cdef double e = y[1]
    cdef double I = y[2]
    cdef double w = y[4]
    cdef double e2 = y[5]
    cdef double I2 = y[6]
    cdef double w2 = y[8]
    cdef double Itot = I +I2
    cdef double esq = e * e
    cdef double j = sqrt(1 - esq)
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
def dWdt(double t, np.ndarray[double, ndim=1] y, double eps_gw, double eps_gr, double eps_oct,
         double eta):
    cdef double a = y[0]
    cdef double e = y[1]
    cdef double I = y[2]
    cdef double w = y[4]
    cdef double e2 = y[5]
    cdef double I2 = y[6]
    cdef double w2 = y[8]
    cdef double Itot = I +I2
    cdef double esq = e * e
    cdef double j = sqrt(1 - esq)
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
def dwdt(double t, np.ndarray[double, ndim=1] y, double eps_gw, double eps_gr, double eps_oct,
         double eta):
    cdef double a = y[0]
    cdef double e = y[1]
    cdef double I = y[2]
    cdef double w = y[4]
    cdef double e2 = y[5]
    cdef double I2 = y[6]
    cdef double w2 = y[8]
    cdef double Itot = I +I2
    cdef double esq = e * e
    cdef double j = sqrt(1 - esq)
    cdef double jout = sqrt(1 - e2 * e2)
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
def de2dt(double t, np.ndarray[double, ndim=1] y, double eps_gw, double eps_gr, double eps_oct,
          double eta):
    cdef double a = y[0]
    cdef double e = y[1]
    cdef double I = y[2]
    cdef double w = y[4]
    cdef double e2 = y[5]
    cdef double I2 = y[6]
    cdef double w2 = y[8]
    cdef double Itot = I +I2
    cdef double esq = e * e
    cdef double j = sqrt(1 - esq)
    cdef double jout = sqrt(1 - e2 * e2)
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
def dI2dt(double t, np.ndarray[double, ndim=1] y, double eps_gw, double eps_gr, double eps_oct,
          double eta):
    cdef double a = y[0]
    cdef double e = y[1]
    cdef double I = y[2]
    cdef double w = y[4]
    cdef double e2 = y[5]
    cdef double I2 = y[6]
    cdef double w2 = y[8]
    cdef double Itot = I +I2
    cdef double esq = e * e
    cdef double j = sqrt(1 - esq)
    cdef double jout = sqrt(1 - e2 * e2)
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
def dw2dt(double t, np.ndarray[double, ndim=1] y, double eps_gw, double eps_gr, double eps_oct,
          double eta):
    cdef double a = y[0]
    cdef double e = y[1]
    cdef double I = y[2]
    cdef double w = y[4]
    cdef double e2 = y[5]
    cdef double I2 = y[6]
    cdef double w2 = y[8]
    cdef double Itot = I +I2
    cdef double esq = e * e
    cdef double j = sqrt(1 - esq)
    cdef double jout = sqrt(1 - e2 * e2)
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

def dydt(double t, np.ndarray[double, ndim=1] y, double eps_gw, double eps_gr, double eps_oct,
         double eta):
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
###### BIN IMPLEMENTATION ################
##########################################

@cython.boundscheck(False)
@cython.wraparound(False)
def dydt_vec_bin(double t, np.ndarray[double, ndim=1] y, double m, double mm, double l, double ll,
                 double M1, double M2, double M3, double Itot, double INTain, double a2, double N1,
                 double Mu, double J1, double J2, double T):
    cdef double k = 39.4751488
    cdef double c = 6.32397263e4
    cdef double L1x = y[0]
    cdef double L1y = y[1]
    cdef double L1z = y[2]
    cdef double e1x = y[3]
    cdef double e1y = y[4]
    cdef double e1z = y[5]
    cdef double L2x = y[6]
    cdef double L2y = y[7]
    cdef double L2z = y[8]
    cdef double e2x = y[9]
    cdef double e2y = y[10]
    cdef double e2z = y[11]

    cdef double LIN = sqrt(L1x**2 + L1y**2 + L1z**2)
    cdef double LOUT = sqrt(L2x**2 + L2y**2 + L2z**2)
    cdef double E1 = sqrt(e1x**2 + e1y**2 + e1z**2)
    cdef double E2 = sqrt(e2x**2 + e2y**2 + e2z**2)
    cdef double AIN = LIN**2/((Mu**2)*k*(M1 + M2)*(1 - E1**2) )
    cdef double linx = L1x/LIN
    cdef double liny = L1y/LIN
    cdef double linz = L1z/LIN
    cdef double loutx = L2x/LOUT
    cdef double louty = L2y/LOUT
    cdef double loutz = L2z/LOUT
    cdef double n1 = sqrt((k*(M1 + M2))/AIN**3)
    cdef double tk = 1/n1*((M1 + M2)/M3)*(a2/AIN)**3*(1 - E2**2)**(3.0/2)
    cdef double WLK = m*1/tk
    cdef double Epsilonoct = mm*(M1 - M2)/(M1 + M2)*AIN/a2*E2/(1 - E2**2)
    cdef double WGR = l*(3*k**2*(M1 + M2)**2)/((AIN**2)*(c**2)*sqrt(AIN*k*(M1 + M2))*(1 - E1**2))
    cdef double JGW = -ll*32/5*k**(7.0/2)/c**5*Mu**2/(AIN)**(7.0/2)*(M1 + M2)**(5.0/2)*(
       1.0 + 7.0/8*E1**2)/(1 - E1**2)**2
    cdef double EGW = -ll*(304*k**3*M1*M2*(M1 + M2))/(
       15*c**5*AIN**4*(1 - E1**2)**(5.0/2))*(1 + 121.0/304*E1**2)

    return [
      (3*WLK*LIN)/(
        4*sqrt(1 -
          E1**2))*((1 - E1**2)*(-linz*louty + liny*loutz)*(linx*loutx +
             liny*louty + linz*loutz) -
          5*(loutz*e1y - louty*e1z)*(loutx*e1x + louty*e1y +
              loutz*e1z)) +
        JGW*linx - (75*Epsilonoct*WLK*LIN)/(
        64*sqrt(1 -
          E1**2))*(2*(sqrt(1 - E1**2)*linx*loutx +
             sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(loutx*e1x + louty*e1y +
             loutz*e1z)*(-((sqrt(1 - E1**2)*linz*e2y)/E2) + (
             sqrt(1 - E1**2)*liny*e2z)/E2) + (-(1.0/5) + (8*E1**2)/
             5 + (sqrt(1 - E1**2)*linx*loutx +
               sqrt(1 - E1**2)*liny*louty +
               sqrt(1 - E1**2)*linz*loutz)**2 -
             7*(loutx*e1x + louty*e1y + loutz*e1z)**2)*(-((
              e1z*e2y)/E2) + (e1y*e2z)/E2) +
          2*(-sqrt(1 - E1**2)*linz*louty +
             sqrt(1 - E1**2)
              *liny*loutz)*((loutx*e1x + louty*e1y +
                loutz*e1z)*((sqrt(1 - E1**2)*linx*e2x)/E2 + (
                sqrt(1 - E1**2)*liny*e2y)/E2 + (
                sqrt(1 - E1**2)*linz*e2z)/
                E2) + (sqrt(1 - E1**2)*linx*loutx +
                sqrt(1 - E1**2)*liny*louty +
                sqrt(1 - E1**2)*linz*loutz)*((e1x*e2x)/E2 + (
                e1y*e2y)/E2 + (e1z*e2z)/E2)) +
          2*(loutz*e1y -
             louty*e1z)*((sqrt(1 - E1**2)*linx*loutx +
                sqrt(1 - E1**2)*liny*louty +
                sqrt(1 - E1**2)*linz*loutz)*((
                sqrt(1 - E1**2)*linx*e2x)/E2 + (
                sqrt(1 - E1**2)*liny*e2y)/E2 + (
                sqrt(1 - E1**2)*linz*e2z)/E2) -
             7*(loutx*e1x + louty*e1y + loutz*e1z)*((
                e1x*e2x)/E2 + (e1y*e2y)/E2 + (
                e1z*e2z)/E2))),
      (3*WLK*LIN)/(
        4*sqrt(1 -
          E1**2))*((1 - E1**2)*(linz*loutx - linx*loutz)*(linx*loutx +
             liny*louty + linz*loutz) -
          5*(-loutz*e1x + loutx*e1z)*(loutx*e1x +
             louty*e1y + loutz*e1z)) +
        JGW*liny - (75*Epsilonoct*WLK*LIN)/(
        64*sqrt(1 -
          E1**2))*(2*(sqrt(1 - E1**2)*linx*loutx +
             sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(loutx*e1x + louty*e1y +
             loutz*e1z)*((sqrt(1 - E1**2)*linz*e2x)/E2 - (
             sqrt(1 - E1**2)*linx*e2z)/E2) + (-(1.0/5) + (8*E1**2)/
             5 + (sqrt(1 - E1**2)*linx*loutx +
               sqrt(1 - E1**2)*liny*louty +
               sqrt(1 - E1**2)*linz*loutz)**2 -
             7*(loutx*e1x + louty*e1y + loutz*e1z)**2)*((
             e1z*e2x)/E2 - (e1x*e2z)/E2) +
          2*(sqrt(1 - E1**2)*linz*loutx -
             sqrt(1 - E1**2)
              *linx*loutz)*((loutx*e1x + louty*e1y +
                loutz*e1z)*((sqrt(1 - E1**2)*linx*e2x)/E2 + (
                sqrt(1 - E1**2)*liny*e2y)/E2 + (
                sqrt(1 - E1**2)*linz*e2z)/
                E2) + (sqrt(1 - E1**2)*linx*loutx +
                sqrt(1 - E1**2)*liny*louty +
                sqrt(1 - E1**2)*linz*loutz)*((e1x*e2x)/E2 + (
                e1y*e2y)/E2 + (e1z*e2z)/E2)) +
          2*(-loutz*e1x +
             loutx*e1z)*((sqrt(1 - E1**2)*linx*loutx +
                sqrt(1 - E1**2)*liny*louty +
                sqrt(1 - E1**2)*linz*loutz)*((
                sqrt(1 - E1**2)*linx*e2x)/E2 + (
                sqrt(1 - E1**2)*liny*e2y)/E2 + (
                sqrt(1 - E1**2)*linz*e2z)/E2) -
             7*(loutx*e1x + louty*e1y + loutz*e1z)*((
                e1x*e2x)/E2 + (e1y*e2y)/E2 + (
                e1z*e2z)/E2))),
      (3*WLK*LIN)/(
        4*sqrt(1 -
          E1**2))*((1 - E1**2)*(-liny*loutx + linx*louty)*(linx*loutx +
             liny*louty + linz*loutz) -
          5*(louty*e1x - loutx*e1y)*(loutx*e1x + louty*e1y +
              loutz*e1z)) +
        JGW*linz - (75*Epsilonoct*WLK*LIN)/(
        64*sqrt(1 -
          E1**2))*(2*(sqrt(1 - E1**2)*linx*loutx +
             sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(loutx*e1x + louty*e1y +
             loutz*e1z)*(-((sqrt(1 - E1**2)*liny*e2x)/E2) + (
             sqrt(1 - E1**2)*linx*e2y)/E2) + (-(1.0/5) + (8*E1**2)/
             5 + (sqrt(1 - E1**2)*linx*loutx +
               sqrt(1 - E1**2)*liny*louty +
               sqrt(1 - E1**2)*linz*loutz)**2 -
             7*(loutx*e1x + louty*e1y + loutz*e1z)**2)*(-((
              e1y*e2x)/E2) + (e1x*e2y)/E2) +
          2*(-sqrt(1 - E1**2)*liny*loutx +
             sqrt(1 - E1**2)
              *linx*louty)*((loutx*e1x + louty*e1y +
                loutz*e1z)*((sqrt(1 - E1**2)*linx*e2x)/E2 + (
                sqrt(1 - E1**2)*liny*e2y)/E2 + (
                sqrt(1 - E1**2)*linz*e2z)/
                E2) + (sqrt(1 - E1**2)*linx*loutx +
                sqrt(1 - E1**2)*liny*louty +
                sqrt(1 - E1**2)*linz*loutz)*((e1x*e2x)/E2 + (
                e1y*e2y)/E2 + (e1z*e2z)/E2)) +
          2*(louty*e1x -
             loutx*e1y)*((sqrt(1 - E1**2)*linx*loutx +
                sqrt(1 - E1**2)*liny*louty +
                sqrt(1 - E1**2)*linz*loutz)*((
                sqrt(1 - E1**2)*linx*e2x)/E2 + (
                sqrt(1 - E1**2)*liny*e2y)/E2 + (
                sqrt(1 - E1**2)*linz*e2z)/E2) -
             7*(loutx*e1x + louty*e1y + loutz*e1z)*((
                e1x*e2x)/E2 + (e1y*e2y)/E2 + (
                e1z*e2z)/E2))),
      WGR*(-linz*e1y + liny*e1z)
        + (3*WLK*sqrt(1 - E1**2))/
        4*(2*(-linz*e1y + liny*e1z) + (linx*loutx + liny*louty +
             linz*loutz)*(loutz*e1y - louty*e1z) -
          5*(-linz*louty + liny*loutz)*(loutx*e1x + louty*e1y +
             loutz*e1z)) +
        EGW*e1x - (75*Epsilonoct*WLK)/
        64*((-(1.0/5) + (8*E1**2)/
             5 + (sqrt(1 - E1**2)*linx*loutx +
               sqrt(1 - E1**2)*liny*louty +
               sqrt(1 - E1**2)*linz*loutz)**2 -
             7*(loutx*e1x + louty*e1y + loutz*e1z)**2)*(-((
              sqrt(1 - E1**2)*linz*e2y)/E2) + (
             sqrt(1 - E1**2)*liny*e2z)/E2) +
          2*(sqrt(1 - E1**2)*linx*loutx + sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(loutx*e1x + louty*e1y +
             loutz*e1z)*(-((e1z*e2y)/E2) + (e1y*e2z)/
             E2) + 16/
           5*(-sqrt(1 - E1**2)*linz*e1y +
             sqrt(1 - E1**2)*liny*e1z)*((e1x*e2x)/E2 + (
             e1y*e2y)/E2 + (e1z*e2z)/E2) +
          2*(loutz*e1y -
             louty*e1z)*((loutx*e1x + louty*e1y + loutz*e1z)*((
                sqrt(1 - E1**2)*linx*e2x)/E2 + (
                sqrt(1 - E1**2)*liny*e2y)/E2 + (
                sqrt(1 - E1**2)*linz*e2z)/
                E2) + (sqrt(1 - E1**2)*linx*loutx +
                sqrt(1 - E1**2)*liny*louty +
                sqrt(1 - E1**2)*linz*loutz)*((e1x*e2x)/E2 + (
                e1y*e2y)/E2 + (e1z*e2z)/E2)) +
          2*(-sqrt(1 - E1**2)*linz*louty +
             sqrt(1 - E1**2)
               *liny*loutz)*((sqrt(1 - E1**2)*linx*loutx +
                sqrt(1 - E1**2)*liny*louty +
                sqrt(1 - E1**2)*linz*loutz)*((
                sqrt(1 - E1**2)*linx*e2x)/E2 + (
                sqrt(1 - E1**2)*liny*e2y)/E2 + (
                sqrt(1 - E1**2)*linz*e2z)/E2) -
             7*(loutx*e1x + louty*e1y + loutz*e1z)*((
                e1x*e2x)/E2 + (e1y*e2y)/E2 + (
                e1z*e2z)/E2))),
      WGR*(linz*e1x - linx*e1z)
         + (3*WLK*sqrt(1 - E1**2))/
        4*(2*(linz*e1x - linx*e1z) + (linx*loutx + liny*louty +
             linz*loutz)*(-loutz*e1x + loutx*e1z) -
          5*(linz*loutx - linx*loutz)*(loutx*e1x + louty*e1y +
             loutz*e1z)) +
        EGW*e1y - (75*Epsilonoct*WLK)/
        64*((-(1.0/5) + (8*E1**2)/
             5 + (sqrt(1 - E1**2)*linx*loutx +
               sqrt(1 - E1**2)*liny*louty +
               sqrt(1 - E1**2)*linz*loutz)**2 -
             7*(loutx*e1x + louty*e1y + loutz*e1z)**2)*((
             sqrt(1 - E1**2)*linz*e2x)/E2 - (
             sqrt(1 - E1**2)*linx*e2z)/E2) +
          2*(sqrt(1 - E1**2)*linx*loutx + sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(loutx*e1x + louty*e1y +
             loutz*e1z)*((e1z*e2x)/E2 - (e1x*e2z)/E2) +
          16/
           5*(sqrt(1 - E1**2)*linz*e1x -
             sqrt(1 - E1**2)*linx*e1z)*((e1x*e2x)/E2 + (
             e1y*e2y)/E2 + (e1z*e2z)/E2) +
          2*(-loutz*e1x +
             loutx*e1z)*((loutx*e1x + louty*e1y + loutz*e1z)*((
                sqrt(1 - E1**2)*linx*e2x)/E2 + (
                sqrt(1 - E1**2)*liny*e2y)/E2 + (
                sqrt(1 - E1**2)*linz*e2z)/
                E2) + (sqrt(1 - E1**2)*linx*loutx +
                sqrt(1 - E1**2)*liny*louty +
                sqrt(1 - E1**2)*linz*loutz)*((e1x*e2x)/E2 + (
                e1y*e2y)/E2 + (e1z*e2z)/E2)) +
          2*(sqrt(1 - E1**2)*linz*loutx -
             sqrt(1 - E1**2)
               *linx*loutz)*((sqrt(1 - E1**2)*linx*loutx +
                sqrt(1 - E1**2)*liny*louty +
                sqrt(1 - E1**2)*linz*loutz)*((
                sqrt(1 - E1**2)*linx*e2x)/E2 + (
                sqrt(1 - E1**2)*liny*e2y)/E2 + (
                sqrt(1 - E1**2)*linz*e2z)/E2) -
             7*(loutx*e1x + louty*e1y + loutz*e1z)*((
                e1x*e2x)/E2 + (e1y*e2y)/E2 + (
                e1z*e2z)/E2))),
      WGR*(-liny*e1x +
          linx*e1y) + (
        3*WLK*sqrt(1 - E1**2))/
        4*(2*(-liny*e1x + linx*e1y) + (linx*loutx + liny*louty +
             linz*loutz)*(louty*e1x - loutx*e1y) -
          5*(-liny*loutx + linx*louty)*(loutx*e1x + louty*e1y +
             loutz*e1z)) +
        EGW*e1z - (75*Epsilonoct*WLK)/
        64*((-(1.0/5) + (8*E1**2)/
             5 + (sqrt(1 - E1**2)*linx*loutx +
               sqrt(1 - E1**2)*liny*louty +
               sqrt(1 - E1**2)*linz*loutz)**2 -
             7*(loutx*e1x + louty*e1y + loutz*e1z)**2)*(-((
              sqrt(1 - E1**2)*liny*e2x)/E2) + (
             sqrt(1 - E1**2)*linx*e2y)/E2) +
          2*(sqrt(1 - E1**2)*linx*loutx + sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(loutx*e1x + louty*e1y +
             loutz*e1z)*(-((e1y*e2x)/E2) + (e1x*e2y)/
             E2) + 16/
           5*(-sqrt(1 - E1**2)*liny*e1x +
             sqrt(1 - E1**2)*linx*e1y)*((e1x*e2x)/E2 + (
             e1y*e2y)/E2 + (e1z*e2z)/E2) +
          2*(louty*e1x -
             loutx*e1y)*((loutx*e1x + louty*e1y + loutz*e1z)*((
                sqrt(1 - E1**2)*linx*e2x)/E2 + (
                sqrt(1 - E1**2)*liny*e2y)/E2 + (
                sqrt(1 - E1**2)*linz*e2z)/
                E2) + (sqrt(1 - E1**2)*linx*loutx +
                sqrt(1 - E1**2)*liny*louty +
                sqrt(1 - E1**2)*linz*loutz)*((e1x*e2x)/E2 + (
                e1y*e2y)/E2 + (e1z*e2z)/E2)) +
          2*(-sqrt(1 - E1**2)*liny*loutx +
             sqrt(1 - E1**2)
               *linx*louty)*((sqrt(1 - E1**2)*linx*loutx +
                sqrt(1 - E1**2)*liny*louty +
                sqrt(1 - E1**2)*linz*loutz)*((
                sqrt(1 - E1**2)*linx*e2x)/E2 + (
                sqrt(1 - E1**2)*liny*e2y)/E2 + (
                sqrt(1 - E1**2)*linz*e2z)/E2) -
             7*(loutx*e1x + louty*e1y + loutz*e1z)*((
                e1x*e2x)/E2 + (e1y*e2y)/E2 + (
                e1z*e2z)/E2))),
      (3*WLK*LIN)/(
        4*sqrt(1 -
          E1**2))*((1 - E1**2)*(linz*louty - liny*loutz)*(linx*loutx +
             liny*louty + linz*loutz) -
          5*(-loutz*e1y + louty*e1z)*(loutx*e1x +
             louty*e1y + loutz*e1z)) - (
        75*Epsilonoct*WLK)/64*LIN/sqrt(1 - E1**2)*(2*(sqrt(1 - E1**2)*linx*loutx +
             sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(loutx*e1x + louty*e1y +
             loutz*e1z)*((sqrt(1 - E1**2)*linz*e2y)/E2 - (
             sqrt(1 - E1**2)*liny*e2z)/E2) +
          2*(sqrt(1 - E1**2)*linx*loutx + sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(-loutz*e1y +
             louty*e1z)*((sqrt(1 - E1**2)*linx*e2x)/E2 + (
             sqrt(1 - E1**2)*liny*e2y)/E2 + (
             sqrt(1 - E1**2)*linz*e2z)/E2) +
          2*(sqrt(1 - E1**2)*linz*louty -
             sqrt(1 - E1**2)*liny*loutz)*(loutx*e1x + louty*e1y +
             loutz*e1z)*((sqrt(1 - E1**2)*linx*e2x)/E2 + (
             sqrt(1 - E1**2)*liny*e2y)/E2 + (
             sqrt(1 - E1**2)*linz*e2z)/E2) + (-(1.0/5) + (8*E1**2)/
             5 + (sqrt(1 - E1**2)*linx*loutx +
               sqrt(1 - E1**2)*liny*louty +
               sqrt(1 - E1**2)*linz*loutz)**2 -
             7*(loutx*e1x + louty*e1y + loutz*e1z)**2)*((
             e1z*e2y)/E2 - (e1y*e2z)/E2) +
          2*(sqrt(1 - E1**2)*linz*louty -
             sqrt(1 - E1**2)*liny*loutz)*(sqrt(1 - E1**2)*linx*loutx +
             sqrt(1 - E1**2)*liny*louty + sqrt(1 - E1**2)*linz*loutz)*((
             e1x*e2x)/E2 + (e1y*e2y)/E2 + (e1z*e2z)/
             E2) - 14*(-loutz*e1y + louty*e1z)*(loutx*e1x +
             louty*e1y + loutz*e1z)*((e1x*e2x)/E2 + (
             e1y*e2y)/E2 + (e1z*e2z)/E2)),
      (3*WLK*LIN)/(
        4*sqrt(1 -
          E1**2))*((1 - E1**2)*(-linz*loutx + linx*loutz)*(linx*loutx +
             liny*louty + linz*loutz) -
          5*(loutz*e1x - loutx*e1z)*(loutx*e1x + louty*e1y +
              loutz*e1z)) - (75*Epsilonoct*WLK)/
        64*LIN/sqrt(1 - E1**2)*(2*(sqrt(1 - E1**2)*linx*loutx +
             sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(loutx*e1x + louty*e1y +
             loutz*e1z)*(-((sqrt(1 - E1**2)*linz*e2x)/E2) + (
             sqrt(1 - E1**2)*linx*e2z)/E2) +
          2*(sqrt(1 - E1**2)*linx*loutx + sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(loutz*e1x - loutx*e1z)*((
             sqrt(1 - E1**2)*linx*e2x)/E2 + (
             sqrt(1 - E1**2)*liny*e2y)/E2 + (
             sqrt(1 - E1**2)*linz*e2z)/E2) +
          2*(-sqrt(1 - E1**2)*linz*loutx +
             sqrt(1 - E1**2)*linx*loutz)*(loutx*e1x + louty*e1y +
             loutz*e1z)*((sqrt(1 - E1**2)*linx*e2x)/E2 + (
             sqrt(1 - E1**2)*liny*e2y)/E2 + (
             sqrt(1 - E1**2)*linz*e2z)/E2) + (-(1.0/5) + (8*E1**2)/
             5 + (sqrt(1 - E1**2)*linx*loutx +
               sqrt(1 - E1**2)*liny*louty +
               sqrt(1 - E1**2)*linz*loutz)**2 -
             7*(loutx*e1x + louty*e1y + loutz*e1z)**2)*(-((
              e1z*e2x)/E2) + (e1x*e2z)/E2) +
          2*(-sqrt(1 - E1**2)*linz*loutx +
             sqrt(1 - E1**2)*linx*loutz)*(sqrt(1 - E1**2)*linx*loutx +
             sqrt(1 - E1**2)*liny*louty + sqrt(1 - E1**2)*linz*loutz)*((
             e1x*e2x)/E2 + (e1y*e2y)/E2 + (e1z*e2z)/
             E2) - 14*(loutz*e1x - loutx*e1z)*(loutx*e1x +
             louty*e1y + loutz*e1z)*((e1x*e2x)/E2 + (
             e1y*e2y)/E2 + (e1z*e2z)/E2)),
      (3*WLK*LIN)/(
        4*sqrt(1 - E1**2))*((1 - E1**2)*(liny*loutx - linx*louty)*(linx*loutx +
             liny*louty + linz*loutz) -
          5*(-louty*e1x + loutx*e1y)*(loutx*e1x +
             louty*e1y + loutz*e1z)) - (
        75*Epsilonoct*WLK)/64*LIN/sqrt(1 - E1**2)*(2*(sqrt(1 - E1**2)*linx*loutx +
             sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(loutx*e1x + louty*e1y +
             loutz*e1z)*((sqrt(1 - E1**2)*liny*e2x)/E2 - (
             sqrt(1 - E1**2)*linx*e2y)/E2) + (-(1.0/5) + (8*E1**2)/
             5 + (sqrt(1 - E1**2)*linx*loutx +
               sqrt(1 - E1**2)*liny*louty +
               sqrt(1 - E1**2)*linz*loutz)**2 -
             7*(loutx*e1x + louty*e1y + loutz*e1z)**2)*((
             e1y*e2x)/E2 - (e1x*e2y)/E2) +
          2*(sqrt(1 - E1**2)*linx*loutx + sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(-louty*e1x +
             loutx*e1y)*((sqrt(1 - E1**2)*linx*e2x)/E2 + (
             sqrt(1 - E1**2)*liny*e2y)/E2 + (
             sqrt(1 - E1**2)*linz*e2z)/E2) +
          2*(sqrt(1 - E1**2)*liny*loutx -
             sqrt(1 - E1**2)*linx*louty)*(loutx*e1x + louty*e1y +
             loutz*e1z)*((sqrt(1 - E1**2)*linx*e2x)/E2 + (
             sqrt(1 - E1**2)*liny*e2y)/E2 + (
             sqrt(1 - E1**2)*linz*e2z)/E2) +
          2*(sqrt(1 - E1**2)*liny*loutx -
             sqrt(1 - E1**2)*linx*louty)*(sqrt(1 - E1**2)*linx*loutx +
             sqrt(1 - E1**2)*liny*louty + sqrt(1 - E1**2)*linz*loutz)*((
             e1x*e2x)/E2 + (e1y*e2y)/E2 + (e1z*e2z)/
             E2) - 14*(-louty*e1x + loutx*e1y)*(loutx*e1x +
             louty*e1y + loutz*e1z)*((e1x*e2x)/E2 + (
             e1y*e2y)/E2 + (e1z*e2z)/E2)),
      (3*WLK*LIN)/(4*sqrt(1 - E1**2))*1/(
        J2*sqrt(1 -
          E2**2))*((1 - E1**2)*(linx*loutx + liny*louty +
             linz*loutz)*(linz*e2y - liny*e2z) - (1.0/2 - 3*E1**2 -
             5.0/2*(1 - E1**2)*(linx*loutx + liny*louty + linz*loutz)**2 +
             25.0/2*(loutx*e1x + louty*e1y +
                loutz*e1z)**2)*(-loutz*e2y + louty*e2z) -
          5*(loutx*e1x + louty*e1y +
             loutz*e1z)*(e1z*e2y - e1y*e2z)) - (
        75*Epsilonoct*WLK )/64*LIN/sqrt(1 - E1**2)*1/(
        J2*sqrt(1 - E2**2))*(1/
           E2*2*(1 - E2**2)*(sqrt(1 - E1**2)*linz*louty -
             sqrt(1 - E1**2)*liny*loutz)*(sqrt(1 - E1**2)*linx*loutx +
             sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(loutx*e1x + louty*e1y +
             loutz*e1z) +
          1/E2*(1 - E2**2)*(-loutz*e1y + louty*e1z)*(-(1.0/5) + (
             8*E1**2)/
             5 + (sqrt(1 - E1**2)*linx*loutx +
               sqrt(1 - E1**2)*liny*louty +
               sqrt(1 - E1**2)*linz*loutz)**2 -
             7*(loutx*e1x + louty*e1y + loutz*e1z)**2) +
          2*(loutx*e1x + louty*e1y + loutz*e1z)*((
             sqrt(1 - E1**2)*linz*e2y)/E2 - (
             sqrt(1 - E1**2)*liny*e2z)/
             E2)*(sqrt(1 - E1**2)*linx*e2x +
             sqrt(1 - E1**2)*liny*e2y + sqrt(1 - E1**2)*linz*e2z) -
          14*(sqrt(1 - E1**2)*linx*loutx + sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(loutx*e1x + louty*e1y +
             loutz*e1z)*((sqrt(1 - E1**2)*linx*e2x)/E2 + (
             sqrt(1 - E1**2)*liny*e2y)/E2 + (
             sqrt(1 - E1**2)*linz*e2z)/E2)*(loutz*e2y -
             louty*e2z) +
          2*(sqrt(1 - E1**2)*linx*loutx + sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(sqrt(1 - E1**2)*linx*e2x +
             sqrt(1 - E1**2)*liny*e2y + sqrt(1 - E1**2)*linz*e2z)*((
             e1z*e2y)/E2 - (e1y*e2z)/E2) +
          2*(sqrt(1 - E1**2)*linx*loutx + sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*((sqrt(1 - E1**2)*linz*e2y)/
             E2 - (sqrt(1 - E1**2)*liny*e2z)/E2)*(e1x*e2x +
             e1y*e2y + e1z*e2z) -
          14*(loutx*e1x + louty*e1y + loutz*e1z)*((
             e1z*e2y)/E2 - (e1y*e2z)/E2)*(e1x*e2x +
             e1y*e2y + e1z*e2z) -
          2*(1.0/5 - (8*E1**2)/5)*(loutz*e2y - louty*e2z)*((
             e1x*e2x)/E2 + (e1y*e2y)/E2 + (e1z*e2z)/
             E2) - 7*(-(1.0/5) + (8*E1**2)/
             5 + (sqrt(1 - E1**2)*linx*loutx +
               sqrt(1 - E1**2)*liny*louty +
               sqrt(1 - E1**2)*linz*loutz)**2 -
             7*(loutx*e1x + louty*e1y +
                loutz*e1z)**2)*(loutz*e2y - louty*e2z)*((
             e1x*e2x)/E2 + (e1y*e2y)/E2 + (e1z*e2z)/
             E2)),
      (3*WLK*LIN)/(4*sqrt(1 - E1**2))*1/(
        J2*sqrt(1 -
          E2**2))*((1 - E1**2)*(linx*loutx + liny*louty +
             linz*loutz)*(-linz*e2x + linx*e2z) - (1.0/2 - 3*E1**2 -
             5.0/2*(1 - E1**2)*(linx*loutx + liny*louty + linz*loutz)**2 +
             25.0/2*(loutx*e1x + louty*e1y +
                loutz*e1z)**2)*(loutz*e2x - loutx*e2z) -
          5*(loutx*e1x + louty*e1y +
             loutz*e1z)*(-e1z*e2x + e1x*e2z)) - (
        75*Epsilonoct*WLK )/64*LIN/sqrt(1 - E1**2)*1/(
        J2*sqrt(1 -
          E2**2))*(1/
           E2*2*(1 - E2**2)*(-sqrt(1 - E1**2)*linz*loutx +
             sqrt(1 - E1**2)*linx*loutz)*(sqrt(1 - E1**2)*linx*loutx +
             sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(loutx*e1x + louty*e1y +
             loutz*e1z) +
          1/E2*(1 - E2**2)*(loutz*e1x - loutx*e1z)*(-(1.0/5) + (
             8*E1**2)/
             5 + (sqrt(1 - E1**2)*linx*loutx +
               sqrt(1 - E1**2)*liny*louty +
               sqrt(1 - E1**2)*linz*loutz)**2 -
             7*(loutx*e1x + louty*e1y + loutz*e1z)**2) +
          2*(loutx*e1x + louty*e1y +
             loutz*e1z)*(-((sqrt(1 - E1**2)*linz*e2x)/E2) + (
             sqrt(1 - E1**2)*linx*e2z)/
             E2)*(sqrt(1 - E1**2)*linx*e2x +
             sqrt(1 - E1**2)*liny*e2y + sqrt(1 - E1**2)*linz*e2z) -
          14*(sqrt(1 - E1**2)*linx*loutx + sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(loutx*e1x + louty*e1y +
             loutz*e1z)*((sqrt(1 - E1**2)*linx*e2x)/E2 + (
             sqrt(1 - E1**2)*liny*e2y)/E2 + (
             sqrt(1 - E1**2)*linz*e2z)/E2)*(-loutz*e2x +
             loutx*e2z) +
          2*(sqrt(1 - E1**2)*linx*loutx + sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(sqrt(1 - E1**2)*linx*e2x +
             sqrt(1 - E1**2)*liny*e2y +
             sqrt(1 - E1**2)*linz*e2z)*(-((e1z*e2x)/E2) + (
             e1x*e2z)/E2) +
          2*(sqrt(1 - E1**2)*linx*loutx + sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(-((sqrt(1 - E1**2)*linz*e2x)/
              E2) + (sqrt(1 - E1**2)*linx*e2z)/E2)*(e1x*e2x +
             e1y*e2y + e1z*e2z) -
          14*(loutx*e1x + louty*e1y +
             loutz*e1z)*(-((e1z*e2x)/E2) + (e1x*e2z)/
             E2)*(e1x*e2x + e1y*e2y + e1z*e2z) -
          2*(1.0/5 - (8*E1**2)/5)*(-loutz*e2x + loutx*e2z)*((
             e1x*e2x)/E2 + (e1y*e2y)/E2 + (e1z*e2z)/
             E2) - 7*(-(1.0/5) + (8*E1**2)/
             5 + (sqrt(1 - E1**2)*linx*loutx +
               sqrt(1 - E1**2)*liny*louty +
               sqrt(1 - E1**2)*linz*loutz)**2 -
             7*(loutx*e1x + louty*e1y +
                loutz*e1z)**2)*(-loutz*e2x + loutx*e2z)*((
             e1x*e2x)/E2 + (e1y*e2y)/E2 + (e1z*e2z)/
             E2)),
      (3*WLK*LIN)/(4*sqrt(1 - E1**2))*1/(
        J2*sqrt(1 -
          E2**2))*((1 - E1**2)*(linx*loutx + liny*louty +
             linz*loutz)*(liny*e2x - linx*e2y) - (1.0/2 - 3*E1**2 -
             5.0/2*(1 - E1**2)*(linx*loutx + liny*louty + linz*loutz)**2 +
             25.0/2*(loutx*e1x + louty*e1y +
                loutz*e1z)**2)*(-louty*e2x + loutx*e2y) -
          5*(loutx*e1x + louty*e1y +
             loutz*e1z)*(e1y*e2x - e1x*e2y)) - (
        75*Epsilonoct*WLK )/64*LIN/sqrt(1 - E1**2)*1/(
        J2*sqrt(1 -
          E2**2))*(1/
           E2*2*(1 - E2**2)*(sqrt(1 - E1**2)*liny*loutx -
             sqrt(1 - E1**2)*linx*louty)*(sqrt(1 - E1**2)*linx*loutx +
             sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(loutx*e1x + louty*e1y +
             loutz*e1z) +
          1/E2*(1 - E2**2)*(-louty*e1x + loutx*e1y)*(-(1.0/5) + (
             8*E1**2)/
             5 + (sqrt(1 - E1**2)*linx*loutx +
               sqrt(1 - E1**2)*liny*louty +
               sqrt(1 - E1**2)*linz*loutz)**2 -
             7*(loutx*e1x + louty*e1y + loutz*e1z)**2) +
          2*(loutx*e1x + louty*e1y + loutz*e1z)*((
             sqrt(1 - E1**2)*liny*e2x)/E2 - (
             sqrt(1 - E1**2)*linx*e2y)/
             E2)*(sqrt(1 - E1**2)*linx*e2x +
             sqrt(1 - E1**2)*liny*e2y + sqrt(1 - E1**2)*linz*e2z) +
          2*(sqrt(1 - E1**2)*linx*loutx + sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*((e1y*e2x)/E2 - (
             e1x*e2y)/E2)*(sqrt(1 - E1**2)*linx*e2x +
             sqrt(1 - E1**2)*liny*e2y + sqrt(1 - E1**2)*linz*e2z) -
          14*(sqrt(1 - E1**2)*linx*loutx + sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*(loutx*e1x + louty*e1y +
             loutz*e1z)*(louty*e2x - loutx*e2y)*((
             sqrt(1 - E1**2)*linx*e2x)/E2 + (
             sqrt(1 - E1**2)*liny*e2y)/E2 + (
             sqrt(1 - E1**2)*linz*e2z)/E2) +
          2*(sqrt(1 - E1**2)*linx*loutx + sqrt(1 - E1**2)*liny*louty +
             sqrt(1 - E1**2)*linz*loutz)*((sqrt(1 - E1**2)*liny*e2x)/
             E2 - (sqrt(1 - E1**2)*linx*e2y)/E2)*(e1x*e2x +
             e1y*e2y + e1z*e2z) -
          14*(loutx*e1x + louty*e1y + loutz*e1z)*((
             e1y*e2x)/E2 - (e1x*e2y)/E2)*(e1x*e2x +
             e1y*e2y + e1z*e2z) -
          2*(1.0/5 - (8*E1**2)/5)*(louty*e2x - loutx*e2y)*((
             e1x*e2x)/E2 + (e1y*e2y)/E2 + (e1z*e2z)/
             E2) - 7*(-(1.0/5) + (8*E1**2)/
             5 + (sqrt(1 - E1**2)*linx*loutx +
               sqrt(1 - E1**2)*liny*louty +
               sqrt(1 - E1**2)*linz*loutz)**2 -
             7*(loutx*e1x + louty*e1y +
                loutz*e1z)**2)*(louty*e2x - loutx*e2y)*((
             e1x*e2x)/E2 + (e1y*e2y)/E2 + (e1z*e2z)/
             E2)),
     ]

def get_ain_vec_bin(double t, np.ndarray[double, ndim=1] y, double m, double mm, double l, double ll,
                 double M1, double M2, double M3, double Itot, double INTain, double a2, double N1,
                 double Mu, double J1, double J2, double T):
    cdef double k = 39.4751488
    cdef double L1x = y[0]
    cdef double L1y = y[1]
    cdef double L1z = y[2]
    cdef double e1x = y[3]
    cdef double e1y = y[4]
    cdef double e1z = y[5]

    cdef double LIN = sqrt(L1x**2 + L1y**2 + L1z**2)
    cdef double E1 = sqrt(e1x**2 + e1y**2 + e1z**2)
    cdef double AIN = LIN**2/((Mu**2)*k*(M1 + M2)*(1 - E1**2) )
    return AIN

##########################################
######## SYMPY VEC IMPLEMENTATION ########
##########################################
# UNUSED EXCEPT IN EXPLORATORY

@cython.boundscheck(False)
@cython.wraparound(False)
def dydt_vec_sympy(double t, np.ndarray[double, ndim=1] y, double eps_gw, double eps_gr, double
                   epsilon_oct, double eta):
    cdef double j_x = y[0]
    cdef double j_y = y[1]
    cdef double j_z = y[2]
    cdef double e_x = y[3]
    cdef double e_y = y[4]
    cdef double e_z = y[5]
    cdef double j_2x = y[6]
    cdef double j_2y = y[7]
    cdef double j_2z = y[8]
    cdef double e_2x = y[9]
    cdef double e_2y = y[10]
    cdef double e_2z = y[11]
    cdef double e2 = e_x * e_x + e_y * e_y + e_z * e_z
    cdef double e2_2 = e_2x * e_2x + e_2y * e_2y + e_2z * e_2z
    cdef double j2 = 1 - e2
    cdef double j2_2 = 1 - e2_2

    cdef double x_0 = pow(j2_2, -1.0/2.0)
    cdef double x_1 = j_2x*x_0
    cdef double x_2 = j_2y*x_0
    cdef double x_3 = j_2z*x_0
    cdef double x_4 = j_x*x_1 + j_y*x_2 + j_z*x_3
    cdef double x_5 = (3.0/4.0)*x_4/pow(j2_2, 2)
    cdef double x_6 = j_2z*x_5
    cdef double x_7 = pow(e_2x, 2)
    cdef double x_8 = pow(e_2y, 2)
    cdef double x_9 = pow(e_2z, 2)
    cdef double x_10 = x_7 + x_8 + x_9
    cdef double x_11 = pow(x_10, -1.0/2.0)
    cdef double x_12 = e_x*x_1
    cdef double x_13 = e_y*x_2
    cdef double x_14 = e_z*x_3
    cdef double x_15 = 10*x_12 + 10*x_13 + 10*x_14
    cdef double x_16 = x_15*x_4
    cdef double x_17 = x_11*x_16
    cdef double x_18 = e_x*x_11
    cdef double x_19 = e_y*x_11
    cdef double x_20 = e_z*x_11
    cdef double x_21 = e_2x*x_18 + e_2y*x_19 + e_2z*x_20
    cdef double x_22 = 10*x_4
    cdef double x_23 = x_21*x_22
    cdef double x_24 = j_x*x_11
    cdef double x_25 = j_y*x_11
    cdef double x_26 = j_z*x_11
    cdef double x_27 = e_2x*x_24 + e_2y*x_25 + e_2z*x_26
    cdef double x_28 = x_15*x_27
    cdef double x_29 = pow(j2_2, -3.0/2.0)
    cdef double x_30 = (15.0/64.0)*epsilon_oct*x_29
    cdef double x_31 = x_30*(e_2z*x_17 + x_23*x_3 + x_28*x_3)
    cdef double x_32 = x_31 - x_6
    cdef double x_33 = j_2y*x_5
    cdef double x_34 = x_30*(e_2y*x_17 + x_2*x_23 + x_2*x_28)
    cdef double x_35 = x_33 - x_34
    cdef double x_36 = x_12 + x_13 + x_14
    cdef double x_37 = x_3*x_36
    cdef double x_38 = (1.0/8.0)*x_29
    cdef double x_39 = x_38*(-12*e_z + 30*x_37)
    cdef double x_40 = x_22*x_27
    cdef double x_41 = 8*pow(e_x, 2) + 8*pow(e_y, 2) + 8*pow(e_z, 2) - 35*pow(x_36, 2) + 5*pow(x_4, 2) - 1
    cdef double x_42 = x_11*x_41
    cdef double x_43 = x_30*(e_2z*x_42 + x_21*(16*e_z - 70*x_37) + x_3*x_40)
    cdef double x_44 = x_39 + x_43
    cdef double x_45 = x_2*x_36
    cdef double x_46 = x_38*(-12*e_y + 30*x_45)
    cdef double x_47 = x_30*(e_2y*x_42 + x_2*x_40 + x_21*(16*e_y - 70*x_45))
    cdef double x_48 = -x_46 - x_47
    cdef double x_49 = -x_31 + x_6
    cdef double x_50 = j_2x*x_5
    cdef double x_51 = x_30*(e_2x*x_17 + x_1*x_23 + x_1*x_28)
    cdef double x_52 = -x_50 + x_51
    cdef double x_53 = -x_39 - x_43
    cdef double x_54 = x_1*x_36
    cdef double x_55 = x_38*(-12*e_x + 30*x_54)
    cdef double x_56 = x_30*(e_2x*x_42 + x_1*x_40 + x_21*(16*e_x - 70*x_54))
    cdef double x_57 = x_55 + x_56
    cdef double x_58 = -x_33 + x_34
    cdef double x_59 = x_50 - x_51
    cdef double x_60 = x_46 + x_47
    cdef double x_61 = -x_55 - x_56
    cdef double x_62 = pow(x_10, -3.0/2.0)
    cdef double x_63 = e_2x*e_2z*x_62
    cdef double x_64 = e_2y*x_62
    cdef double x_65 = e_2z*x_64
    cdef double x_66 = x_62*x_9
    cdef double x_67 = x_30*(x_16*(-j_x*x_63 - j_y*x_65 - j_z*x_66 + x_26) + x_41*(-e_x*x_63 - e_y*x_65 - e_z*x_66 + x_20))
    cdef double x_68 = e_2x*x_64
    cdef double x_69 = x_62*x_8
    cdef double x_70 = x_16*(-j_x*x_68 - j_y*x_69 - j_z*x_65 + x_25) + x_41*(-e_x*x_68 - e_y*x_69 - e_z*x_65 + x_19)
    cdef double x_71 = e_2z*x_30
    cdef double x_72 = x_0*x_36
    cdef double x_73 = e_z*x_72
    cdef double x_74 = j_z*x_0
    cdef double x_75 = x_4*x_74
    cdef double x_76 = x_38*(30*x_73 - 6*x_75)
    cdef double x_77 = x_0*x_40
    cdef double x_78 = x_30*(e_z*x_77 + x_21*(-70*x_73 + 10*x_75) + x_28*x_74)
    cdef double x_79 = x_76 + x_78
    cdef double x_80 = e_y*x_72
    cdef double x_81 = j_y*x_0
    cdef double x_82 = x_4*x_81
    cdef double x_83 = x_38*(30*x_80 - 6*x_82)
    cdef double x_84 = x_30*(e_y*x_77 + x_21*(-70*x_80 + 10*x_82) + x_28*x_81)
    cdef double x_85 = -x_83 - x_84
    cdef double x_86 = x_62*x_7
    cdef double x_87 = x_16*(-j_x*x_86 - j_y*x_68 - j_z*x_63 + x_24) + x_41*(-e_x*x_86 - e_y*x_68 - e_z*x_63 + x_18)
    cdef double x_88 = -x_76 - x_78
    cdef double x_89 = e_x*x_72
    cdef double x_90 = j_x*x_0
    cdef double x_91 = x_4*x_90
    cdef double x_92 = x_38*(30*x_89 - 6*x_91)
    cdef double x_93 = x_30*(e_x*x_77 + x_21*(-70*x_89 + 10*x_91) + x_28*x_90)
    cdef double x_94 = x_92 + x_93
    cdef double x_95 = x_30*x_70
    cdef double x_96 = x_30*x_87
    cdef double x_97 = x_83 + x_84
    cdef double x_98 = -x_92 - x_93

    return [
            -e_y*x_44 - e_z*x_48 - j_y*x_32 - j_z*x_35,
            -e_x*x_53 - e_z*x_57 - j_x*x_49 - j_z*x_52,
            -e_x*x_60 - e_y*x_61 - j_x*x_58 - j_y*x_59,
            -e_y*x_32 - e_z*x_35 - j_y*x_44 - j_z*x_48,
            -e_x*x_49 - e_z*x_52 - j_x*x_53 - j_z*x_57,
            -e_x*x_58 - e_y*x_59 - j_x*x_60 - j_y*x_61,
            -eta*(e_2y*x_67 + j_2y*x_79 + j_2z*x_85 - x_70*x_71),
            -eta*(-e_2x*x_67 + j_2x*x_88 + j_2z*x_94 + x_71*x_87),
            -eta*(e_2x*x_95 - e_2y*x_96 + j_2x*x_97 + j_2y*x_98),
            -eta*(e_2y*x_79 + e_2z*x_85 + j_2y*x_67 - j_2z*x_95),
            -eta*(e_2x*x_88 + e_2z*x_94 - j_2x*x_67 + j_2z*x_96),
            -eta*(e_2x*x_97 + e_2y*x_98 + j_2x*x_95 - j_2y*x_96),
    ]
