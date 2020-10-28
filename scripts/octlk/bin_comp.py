'''
copied Bin's mathematica notebook and did minimal edits to get it to compile in
Python
'''
import time
from multiprocessing import Pool
from numpy import sqrt, sin, cos, abs, radians, pi
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('lines', lw=3.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

m = 1 # (*Kozai*)
mm = 1 # (* turn on oct Kozai *)
l = 1 # (*GR*)
ll = 1 # (*GW*)

k = 39.4751488
c = 6.32397263*10**4
# length = AU
# c = 1AU / 499s
# unit of time = 499s * 6.32e4 = 1yr
# unit of mass = solar mass, solve for M using N1 + distance in correct units
M1 = 30
M2 = 20
M3 = 30
Itot = 93.5
INTain = 100
a2 = 6000
N1 = sqrt((k*(M1 + M2))/INTain ** 3)
Mu = (M1*M2)/(M1 + M2)
J1 = (M2*M1)/(M2 + M1)*sqrt(k*(M2 + M1)*INTain)
J2 = ((M2 + M1)*M3)/(M3 + M1 + M2) * sqrt(k*(M3 + M1 + M2)*a2 )
T = 1e8

def get_emax(seed):
    np.random.seed(seed)
    w1 = np.random.rand() * 2 * pi
    w2 = np.random.rand() * 2 * pi
    W = np.random.rand() * 2 * pi

    E10 = 0.001
    E20 = 0.6
    GTOT = sqrt(
        (J1*sqrt(1 - E10**2))**2 + (J2*sqrt(1 - E20**2))**2 +
         2*J1*sqrt(1 - E10**2)*J2*sqrt(1 - E20**2)*cos(radians(Itot)))
    def f(y):
        i1, i2 = y
        return [
            J1*sqrt(1 - E10**2)*cos(radians(90 - i1)) -
          J2*sqrt(1 - E20**2)*cos(radians(90 - i2)),
         J1*sqrt(1 - E10**2)*sin(radians(90 - i1)) +
           J2*sqrt(1 - E20**2)*sin(radians(90 - i2)) - GTOT
        ]
    I1, I2 = root(f, [60, 0]).x

    L1x00 = sin(radians(I1))*sin(W)
    L1y00 = -sin(radians(I1))*cos(W)
    L1z00 = cos(radians(I1))
    e1x00 = cos(w1)*cos(W) - sin(w1)*cos(radians(I1))*sin(W)
    e1y00 = cos(w1)*sin(W) + sin(w1)*cos(radians(I1))*cos(W)
    e1z00 = sin(w1)*sin(radians(I1))
    L2x00 = sin(radians(I2))*sin(W - pi)
    L2y00 = -sin(radians(I2))*cos(W - pi)
    L2z00 = cos(radians(I2))
    e2x00 = cos(w2)*cos(W - pi) - sin(w2)*cos(radians(I2))*sin(W - pi)
    e2y00 = cos(w2)*sin(W - pi) + sin(w2)*cos(radians(I2))*cos(W - pi)
    e2z00 = sin(w2)*sin(radians(I2))

    L1x0 = J1*sqrt(1 - E10**2)*(L1x00)
    L1y0 = J1*sqrt(1 - E10**2)*(L1y00)
    L1z0 = J1*sqrt(1 - E10**2)*(L1z00)
    e1x0 = E10*(e1x00)
    e1y0 = E10*(e1y00)
    e1z0 = E10*(e1z00)
    L2x0 = J2*sqrt(1 - E20**2)*(L2x00)
    L2y0 = J2*sqrt(1 - E20**2)*(L2y00)
    L2z0 = J2*sqrt(1 - E20**2)*(L2z00)
    e2x0 = E20*(e2x00)
    e2y0 = E20*(e2y00)
    e2z0 = E20*(e2z00)

    y0 = [L1x0, L1y0, L1z0, e1x0, e1y0, e1z0, L2x0, L2y0, L2z0, e2x0, e2y0, e2z0]
    start = time.time()
    ret = solve_ivp(dydt, (0, T), y0, method='DOP853', atol=1e-10, rtol=1e-10)
    ein_vec = ret.y[3:6]
    evec_mags = np.sqrt(np.sum(ein_vec**2, axis=0))
    delta_emax_log10 = np.log10(1 - np.max(evec_mags))
    print('Took', time.time() - start, 'log10(1 - emax) is', delta_emax_log10,
          W, w1, w2)
    return delta_emax_log10
    # plt.semilogy(ret.t, 1 - evec_mags)
    # plt.savefig('/tmp/bin_comp', dpi=300)

def dydt(t, y):
    L1x, L1y, L1z, e1x, e1y, e1z, L2x, L2y, L2z, e2x, e2y, e2z = y

    LIN = sqrt(L1x**2 + L1y**2 + L1z**2)
    LOUT = sqrt(L2x**2 + L2y**2 + L2z**2)
    E1 = sqrt(e1x**2 + e1y**2 + e1z**2)
    E2 = sqrt(e2x**2 + e2y**2 + e2z**2)
    AIN = LIN**2/((Mu**2)*k*(M1 + M2)*(1 - E1**2) )
    linx = L1x/LIN
    liny = L1y/LIN
    linz = L1z/LIN
    loutx = L2x/LOUT
    louty = L2y/LOUT
    loutz = L2z/LOUT
    n1 = sqrt((k*(M1 + M2))/AIN**3)
    tk = 1/n1*((M1 + M2)/M3)*(a2/AIN)**3*(1 - E2**2)**(3/2)
    WLK = m*1/tk
    Epsilonoct = mm*(M1 - M2)/(M1 + M2)*AIN/a2*E2/(1 - E2**2)
    WGR = l*(3*k**2*(M1 + M2)**2)/((AIN**2)*(c**2)*sqrt(AIN*k*(M1 + M2))*(1 - E1**2))
    JGW = -ll*32/5*k**(7/2)/c**5*Mu**2/(AIN)**(7/2)*(M1 + M2)**(5/2)*(
       1 + 7/8*E1**2)/(1 - E1**2)**2
    EGW = -ll*(304*k**3*M1*M2*(M1 + M2))/(
       15*c**5*AIN**4*(1 - E1**2)**(5/2))*(1 + 121/304*E1**2)

    return [
        (
        3*WLK*LIN)/(
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
             sqrt(1 - E1**2)*liny*e2z)/E2) + (-(1/5) + (8*E1**2)/
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
      (
        3*WLK*LIN)/(
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
             sqrt(1 - E1**2)*linx*e2z)/E2) + (-(1/5) + (8*E1**2)/
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
      (
        3*WLK*LIN)/(
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
             sqrt(1 - E1**2)*linx*e2y)/E2) + (-(1/5) + (8*E1**2)/
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
        64*((-(1/5) + (8*E1**2)/
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
        64*((-(1/5) + (8*E1**2)/
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
        64*((-(1/5) + (8*E1**2)/
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
             sqrt(1 - E1**2)*linz*e2z)/E2) + (-(1/5) + (8*E1**2)/
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
             sqrt(1 - E1**2)*linz*e2z)/E2) + (-(1/5) + (8*E1**2)/
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
             sqrt(1 - E1**2)*linx*e2y)/E2) + (-(1/5) + (8*E1**2)/
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
             linz*loutz)*(linz*e2y - liny*e2z) - (1/2 - 3*E1**2 -
             5/2*(1 - E1**2)*(linx*loutx + liny*louty + linz*loutz)**2 +
             25/2*(loutx*e1x + louty*e1y +
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
          1/E2*(1 - E2**2)*(-loutz*e1y + louty*e1z)*(-(1/5) + (
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
          2*(1/5 - (8*E1**2)/5)*(loutz*e2y - louty*e2z)*((
             e1x*e2x)/E2 + (e1y*e2y)/E2 + (e1z*e2z)/
             E2) - 7*(-(1/5) + (8*E1**2)/
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
             linz*loutz)*(-linz*e2x + linx*e2z) - (1/2 - 3*E1**2 -
             5/2*(1 - E1**2)*(linx*loutx + liny*louty + linz*loutz)**2 +
             25/2*(loutx*e1x + louty*e1y +
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
          1/E2*(1 - E2**2)*(loutz*e1x - loutx*e1z)*(-(1/5) + (
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
          2*(1/5 - (8*E1**2)/5)*(-loutz*e2x + loutx*e2z)*((
             e1x*e2x)/E2 + (e1y*e2y)/E2 + (e1z*e2z)/
             E2) - 7*(-(1/5) + (8*E1**2)/
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
             linz*loutz)*(liny*e2x - linx*e2y) - (1/2 - 3*E1**2 -
             5/2*(1 - E1**2)*(linx*loutx + liny*louty + linz*loutz)**2 +
             25/2*(loutx*e1x + louty*e1y +
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
          1/E2*(1 - E2**2)*(-louty*e1x + loutx*e1y)*(-(1/5) + (
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
          2*(1/5 - (8*E1**2)/5)*(louty*e2x - loutx*e2y)*((
             e1x*e2x)/E2 + (e1y*e2y)/E2 + (e1z*e2z)/
             E2) - 7*(-(1/5) + (8*E1**2)/
             5 + (sqrt(1 - E1**2)*linx*loutx +
               sqrt(1 - E1**2)*liny*louty +
               sqrt(1 - E1**2)*linz*loutz)**2 -
             7*(loutx*e1x + louty*e1y +
                loutz*e1z)**2)*(louty*e2x - loutx*e2y)*((
             e1x*e2x)/E2 + (e1y*e2y)/E2 + (e1z*e2z)/
             E2)),
     ]

def get_emax_dist(num_trials=1000):
    p = Pool(10)
    delta_emax_log10s = p.map(get_emax, range(num_trials))
    plt.hist(delta_emax_log10s, bins=50)
    plt.xlabel(r'$\log_{10} (1 - e_{\max})$ in $10^8\;\mathrm{yr}$')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig('1emaxdist_bin')
    plt.close()

if __name__ == '__main__':
    # get_emax()
    get_emax_dist()
    pass
