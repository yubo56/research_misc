'''
to-be-cythonized utils, use keywords instead of getters to pass params

params = eps_gw, eps_gr, eps_oct, eta
'''
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin, cos, sqrt, pow

##########################################
###### ORB IMPLEMENTATION ################
##########################################
# unused for now, seems wrong

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
