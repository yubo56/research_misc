import numpy as np
import os
try: # can't compile cython on MacOS...
    from cython_utils import *
except:
    pass

from scipy.integrate import solve_ivp
import scipy.optimize as opt

def cosd(ang):
    return np.cos(np.radians(ang))

def inverse_permutation(a):
    b = np.arange(a.shape[0])
    b[a] = b.copy()
    return b

def reg(z):
    return np.minimum(np.maximum(z, -1), 1)

def ts_dot(x, y):
    ''' dot product of two time series (is there a better way?) '''
    z = np.zeros(np.shape(x)[1])
    for x1, y1 in zip(x, y):
        z += x1 * y1
    return reg(z)

def ts_cross(x, y):
    return np.array([
        x[1] * y[2] - x[2] * y[1],
        -x[0] * y[2] + x[2] * y[0],
        x[0] * y[1] - x[1] * y[0],
    ])

def get_I1(I0, eta):
    ''' given total inclination between Lout and L, returns I_tot '''
    def I2_constr(_I2):
        return np.sin(_I2) - eta * np.sin(I0 - _I2)
    I2 = opt.brenth(I2_constr, 0, np.pi, xtol=1e-12)
    return np.degrees(I0 - I2)

def get_emax(eta=0, eps_gr=0, I=0, eps_tide=0):
    def jmin_criterion(j): # eq 42, satisfied when j = jmin
        emax_sq = 1 - j**2
        return (
            3/8 * (j**2 - 1) / j**2 * (
                5 * (np.cos(I) + eta / 2)**2
                - (3 + 4 * eta * np.cos(I) + 9 * eta**2 / 4) * j**2
                + eta**2 * j**4)
            + eps_gr * (1 - 1 / j)
            + eps_tide/15 * (1 - (1 + 3 * emax_sq + 3 * emax_sq**2 / 8) / j**9)
        )
    jmin = opt.brenth(jmin_criterion, 1e-15, 1 - 1e-15, xtol=1e-14)
    return np.sqrt(1 - jmin**2)

def get_elim(eta=0, eps_gr=0, eps_tide=0):
    def jlim_criterion(j): # eq 44, satisfied when j = jlim
        emax_sq = 1 - j**2
        return (
            3/8 * (j**2 - 1) * (
                - 3 + eta**2 / 4 * (4 * j**2 / 5 - 1))
            + eps_gr * (1 - 1 / j)
            + eps_tide/15 * (1 - (1 + 3 * emax_sq + 3 * emax_sq**2 / 8) / j**9)
        )
    jlim = opt.brenth(jlim_criterion, 1e-15, 1 - 1e-15, xtol=1e-14)
    return np.sqrt(1 - jlim**2)

def get_Ilim(eta=0, eps_gr=0):
    elim = get_elim(eta=eta, eps_gr=eps_gr)
    jlim = np.sqrt(1 - elim**2)
    Ilim = np.arccos(eta / 2 * (4 * jlim**2 / 5 - 1))
    return np.degrees(Ilim)

# by convention, use solar masses, AU, and set c = 1, in which case G = 9.87e-9
# NB: slight confusion here: to compute epsilon + timescales, we use AU as the
# unit of length, but during the calculation, a0 is the unit of length
G = 9.87e-9
S_PER_UNIT = 499 # 1AU / c, in seconds
S_PER_YR = 3.154e7 # seconds per year
def get_eps(m1, m2, m3, a0, a2, e2):
    m12 = m1 + m2
    mu = m1 * m2 / m12
    m123 = m12 + m3
    mu_out = m12 * m3 / m123
    n = np.sqrt(G * m12 / a0**3)
    a2eff = a2 * np.sqrt(1 - e2**2)
    eps_gw = (1 / n) * (m12 / m3) * (a2eff**3 / a0**7) * G**3 * mu * m12**2
    eps_gr = (m12 / m3) * (a2eff**3 / a0**4) * 3 * G * m12
    eps_oct = ((m2 - m1) / m12) * (a0 / a2) * (e2 / (1 - e2**2))
    eta = (mu / mu_out) * np.sqrt((m12 * a0) / (m123 * a2 * (1 - e2**2)))
    return [eps_gw, eps_gr, eps_oct, eta]

def get_eps_eta0(m1, m2, m3, a0, a2, e2):
    m12 = m1 + m2
    mu = m1 * m2 / m12
    m123 = m12 + m3
    mu_out = m12 * m3 / m123
    n = np.sqrt(G * m12 / a0**3)
    a2eff = a2 * np.sqrt(1 - e2**2)
    eps_gw = (1 / n) * (m12 / m3) * (a2eff**3 / a0**7) * G**3 * mu * m12**2
    eps_gr = (m12 / m3) * (a2eff**3 / a0**4) * 3 * G * m12
    eps_oct = ((m2 - m1) / m12) * (a0 / a2) * (e2 / (1 - e2**2))
    eta0 = (mu / mu_out) * np.sqrt((m12 * a0) / (m123 * a2))
    return [eps_gw, eps_gr, eps_oct, eta0]

def get_eps_tide(m1, m2, m3, a0, a2, e2, k2, R1):
    return (
        15 * m1 * (m1 + m2) * (a2**3) * (1 - e2**2)**(3/2) * k2 * R1**5
    ) / (2 * a0**8 * m2 * m3)

def get_tlk0(m1, m2, m3, a0, a2eff):
    m12 = m1 + m2
    m123 = m12 + m3

    # calculate lk time
    n = np.sqrt(G * m12 / a0**3)
    t_lk0 = (1 / n) * (m12 / m3) * (a2eff / a0)**3

    return (t_lk0 * S_PER_UNIT) / S_PER_YR

def mkdirp(path):
    if not os.path.exists(path):
        os.mkdir(path)

k = 39.4751488
c = 6.32397263*10**4
# length = AU
# c = 1AU / 499s
# unit of time = 499s * 6.32e4 = 1yr
# unit of mass = solar mass, solve for M using N1 + distance in correct units
def run_vec(
        m=1, mm=1, l=1, ll=1,
        M1=30, M2=20, M3=30, Itot=93.5, INTain=100, a2=6000,
        T=1e10, w1=0, w2=0.7, W=0, E10=1e-3, E20=0.6,
        TOL=1e-11, AF=5e-3, method='LSODA',
        k2=0, R2=0,
):
    N1 = np.sqrt((k*(M1 + M2))/INTain ** 3)
    Mu = (M1*M2)/(M1 + M2)
    J1 = (M2*M1)/(M2 + M1)*np.sqrt(k*(M2 + M1)*INTain)
    J2 = ((M2 + M1)*M3)/(M3 + M1 + M2) * np.sqrt(k*(M3 + M1 + M2)*a2 )

    GTOT = np.sqrt(
        (J1*np.sqrt(1 - E10**2))**2 + (J2*np.sqrt(1 - E20**2))**2 +
         2*J1*np.sqrt(1 - E10**2)*J2*np.sqrt(1 - E20**2)*np.cos(np.radians(Itot)))
    def f(y):
        i1, i2 = y
        return [
            J1*np.sqrt(1 - E10**2)*np.cos(np.radians(90 - i1)) -
          J2*np.sqrt(1 - E20**2)*np.cos(np.radians(90 - i2)),
         J1*np.sqrt(1 - E10**2)*np.sin(np.radians(90 - i1)) +
           J2*np.sqrt(1 - E20**2)*np.sin(np.radians(90 - i2)) - GTOT
        ]
    I1, I2 = opt.root(f, [Itot, 0]).x

    L1x00 = np.sin(np.radians(I1))*np.sin(W)
    L1y00 = -np.sin(np.radians(I1))*np.cos(W)
    L1z00 = np.cos(np.radians(I1))
    e1x00 = np.cos(w1)*np.cos(W) - np.sin(w1)*np.cos(np.radians(I1))*np.sin(W)
    e1y00 = np.cos(w1)*np.sin(W) + np.sin(w1)*np.cos(np.radians(I1))*np.cos(W)
    e1z00 = np.sin(w1)*np.sin(np.radians(I1))
    L2x00 = np.sin(np.radians(I2))*np.sin(W - np.pi)
    L2y00 = -np.sin(np.radians(I2))*np.cos(W - np.pi)
    L2z00 = np.cos(np.radians(I2))
    e2x00 = np.cos(w2)*np.cos(W - np.pi) - np.sin(w2)*np.cos(np.radians(I2))*np.sin(W - np.pi)
    e2y00 = np.cos(w2)*np.sin(W - np.pi) + np.sin(w2)*np.cos(np.radians(I2))*np.cos(W - np.pi)
    e2z00 = np.sin(w2)*np.sin(np.radians(I2))

    L1x0 = J1*np.sqrt(1 - E10**2)*(L1x00)
    L1y0 = J1*np.sqrt(1 - E10**2)*(L1y00)
    L1z0 = J1*np.sqrt(1 - E10**2)*(L1z00)
    e1x0 = E10*(e1x00)
    e1y0 = E10*(e1y00)
    e1z0 = E10*(e1z00)
    L2x0 = J2*np.sqrt(1 - E20**2)*(L2x00)
    L2y0 = J2*np.sqrt(1 - E20**2)*(L2y00)
    L2z0 = J2*np.sqrt(1 - E20**2)*(L2z00)
    e2x0 = E20*(e2x00)
    e2y0 = E20*(e2y00)
    e2z0 = E20*(e2z00)

    y0 = np.array([L1x0, L1y0, L1z0, e1x0, e1y0, e1z0, L2x0, L2y0, L2z0, e2x0,
                   e2y0, e2z0])
    def a_term_event(*args):
        ain = get_ain_vec_bin(*args)
        return ain - AF * INTain
    a_term_event.terminal = True
    args = [m, mm, l, ll, M1, M2, M3, Itot, INTain, a2, N1, Mu, J1, J2, T, k2,
            R2]
    ret = solve_ivp(dydt_vec_bin, (0, T), y0, args=args,
                    events=[a_term_event] if ll == 1 else [],
                    method=method, atol=TOL, rtol=TOL)
    return ret

def get_I2(e1, I1, e2, eta0):
    arg = eta0 * np.sqrt((1 - e1**2) / (1 - e2**2)) * np.sin(I1)
    assert arg <= 1
    return np.arcsin(arg)
def ltot(e1, I1, e2, eta0):
    ''' returns Ltot**2 / G2**2 '''
    Itot = I1 + get_I2(e1, I1, e2, eta0)
    return (
        (1 - e2**2)
        + eta0**2 * (1 - e1**2)
        + 2 * eta0 * np.sqrt((1 - e2**2) * (1 - e1**2)) * np.cos(Itot)
    )
def get_eI2(e1, I1, eta0, ltot_i):
    ''' uses law of sines + conservation of Ltot to give e2, I2 '''
    opt_func = lambda e2: ltot(e1, I1, e2, eta0) - ltot_i
    e2max = np.sqrt(1 - (eta0 * np.sin(I1))**2 * (1 - e1**2))
    e2 = opt.brenth(opt_func, 0, e2max - 1e-7)
    I2 = get_I2(e1, I1, e2, eta0)
    return e2, I2

# Python version of orbital elements code
# ran into bugs on Radau/BDF, maybe repro later
def dydtP(t, y, eps_gw, eps_gr, eps_oct, eta):
    ''' python version for debugging '''
    # ret = [
    #     Pdadt(t, y, eps_gw, eps_gr, eps_oct, eta),
    #     Pdedt(t, y, eps_gw, eps_gr, eps_oct, eta),
    #     PdIdt(t, y, eps_gw, eps_gr, eps_oct, eta),
    #     PdWdt(t, y, eps_gw, eps_gr, eps_oct, eta),
    #     Pdwdt(t, y, eps_gw, eps_gr, eps_oct, eta),
    #     Pde2dt(t, y, eps_gw, eps_gr, eps_oct, eta),
    #     PdI2dt(t, y, eps_gw, eps_gr, eps_oct, eta),
    #     PdWdt(t, y, eps_gw, eps_gr, eps_oct, eta), # = dWdt
    #     Pdw2dt(t, y, eps_gw, eps_gr, eps_oct, eta),
    # ]
    import cython_utils as cu
    ret = [
        cu.dadt(t, y, eps_gw, eps_gr, eps_oct, eta),
        cu.dedt(t, y, eps_gw, eps_gr, eps_oct, eta),
        cu.dIdt(t, y, eps_gw, eps_gr, eps_oct, eta),
        cu.dWdt(t, y, eps_gw, eps_gr, eps_oct, eta),
        cu.dwdt(t, y, eps_gw, eps_gr, eps_oct, eta),
        cu.de2dt(t, y, eps_gw, eps_gr, eps_oct, eta),
        cu.dI2dt(t, y, eps_gw, eps_gr, eps_oct, eta),
        cu.dWdt(t, y, eps_gw, eps_gr, eps_oct, eta), # = dWdt
        cu.dw2dt(t, y, eps_gw, eps_gr, eps_oct, eta),
    ]
    return ret
sin = np.sin
cos = np.cos
sqrt = np.sqrt
def Pdadt(t, y, eps_gw, eps_gr, eps_oct, eta):
    a = y[0]
    e = y[1]
    esq = pow(e, 2)
    val = -64 / 5 * eps_gw / (pow(1 - esq, 3.5) * pow(a, 3)) * (
        1 + 73 * esq / 24 + 37 * esq * esq / 96)
    return val
def Pdedt(t, y, eps_gw, eps_gr, eps_oct, eta):
    a = y[0]
    e = y[1]
    I = y[2]
    W = y[3]
    w = y[4]
    e2 = y[5]
    I2 = y[6]
    w2 = y[8]
    Itot = I +I2
    esq = pow(e, 2)
    j = sqrt(1 - esq)
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
def PdIdt(t, y, eps_gw, eps_gr, eps_oct, eta):
    a = y[0]
    e = y[1]
    I = y[2]
    W = y[3]
    w = y[4]
    e2 = y[5]
    I2 = y[6]
    w2 = y[8]
    Itot = I +I2
    esq = pow(e, 2)
    j = sqrt(1 - esq)
    return -3 * e * pow(a, 1.5) / (32 * j) * (
        10 * sin(2 * Itot) * e * sin(2 * w)
        + 5 * eps_oct / 8 * (2 + 5 * esq + 7 * esq * cos(2 * w)) * cos(w2) *
            sin(w)
        + 5 * eps_oct * cos(w) / 8 * (
            26 + 37 * esq - 35 * esq * cos(2 * w)
            - 15 * cos(2 * Itot) * (7 * esq * cos(2 * w) - 2 - 5 * esq))
        * sin(Itot) * sin(w2)
    )
def PdWdt(t, y, eps_gw, eps_gr, eps_oct, eta):
    a = y[0]
    e = y[1]
    I = y[2]
    W = y[3]
    w = y[4]
    e2 = y[5]
    I2 = y[6]
    w2 = y[8]
    Itot = I +I2
    esq = pow(e, 2)
    j = sqrt(1 - esq)
    return 0 # TODO
def Pdwdt(t, y, eps_gw, eps_gr, eps_oct, eta):
    a = y[0]
    e = y[1]
    I = y[2]
    W = y[3]
    w = y[4]
    e2 = y[5]
    I2 = y[6]
    w2 = y[8]
    Itot = I +I2
    esq = pow(e, 2)
    j = sqrt(1 - esq)
    B = 2 + 5 * esq - 7 * esq * cos(2 * w)
    A = 4 + 3 * esq - 5 * B * pow(sin(Itot), 2) / 2
    cosQ = -cos(w) * cos(w2) - cos(Itot) * sin(w) * sin(w2)
    return 3 * pow(a, 1.5) / 8 * (
        (4 * pow(cos(Itot), 2) + (5 * cos(2 * w) - 1) *
         (1 - esq - pow(cos(Itot), 2))) / j
    ) + eps_gr / (pow(a, 2.5) * (1 - esq)) + 15 * eps_oct * pow(a, 1.5) / 64 * (
        cos(Itot) / j * e * (
            sin(w) * sin(w2) * (
                10 * (3 * pow(cos(Itot), 2) - 1) * (1 - esq) + A)
            - 5 * B * cos(Itot) * cosQ)
        - j / e * (
            10 * sin(w) * sin(w2) * cos(Itot) * pow(sin(Itot), 2)
                * (1 - 3 * esq)
            + cosQ * (3 * A - 10 * pow(cos(Itot), 2) + 2)))
def Pde2dt(t, y, eps_gw, eps_gr, eps_oct, eta):
    return 0
def PdI2dt(t, y, eps_gw, eps_gr, eps_oct, eta):
    return 0
def Pdw2dt(t, y, eps_gw, eps_gr, eps_oct, eta):
    a = y[0]
    e = y[1]
    I = y[2]
    W = y[3]
    w = y[4]
    e2 = y[5]
    I2 = y[6]
    w2 = y[8]
    Itot = I +I2
    esq = pow(e, 2)
    j = sqrt(1 - esq)
    B = 2 + 5 * esq - 7 * esq * cos(2 * w)
    A = 4 + 3 * esq - 5 * B * pow(sin(Itot), 2) / 2
    cosQ = -cos(w) * cos(w2) - cos(Itot) * sin(w) * sin(w2)
    return 3 * pow(a, 1.5) / 16 * (
        2 * cos(Itot) / j * (2 + esq * (3 - 5 * cos(2 * w)))
    ) + 15 * eps_oct * e * pow(a, 1.5) / (64 * e2) * (
        sin(w) * sin(w2) * (-e2 / j) * (
            A + 10 * (3 * pow(cos(Itot), 2)) * (1 - esq))
        + cosQ * (5 * B * cos(Itot) * e2 / j)
    )
