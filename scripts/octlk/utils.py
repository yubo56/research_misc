import numpy as np

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
    eps_gw = (1 / n) * (m12 / m3) * (a2**3 / a0**7) * G**3 * mu * m12**2
    eps_gr = (m12 / m3) * (a2**3 / a0**4) * 3 * G * m12
    eps_oct = ((m2 - m1) / m12) * (a0 / a2) * (e2 / (1 - e2**2))
    eta = (mu / mu_out) * np.sqrt((m12 * a0) / (m123 * a2))
    return [eps_gw, eps_gr, eps_oct, eta]

def get_tlk0(m1, m2, m3, a0, a2):
    ''' calculates a bunch of physically relevant values '''
    m12 = m1 + m2
    m123 = m12 + m3
    mu = m1 * m2 / m12
    mu123 = m12 * m3 / m123

    # calculate lk time
    n = np.sqrt(G * m12 / a0**3)
    t_lk0 = (1 / n) * (m12 / m3) * (a2 / a0)**3

    return (t_lk0 * S_PER_UNIT) / S_PER_YR

# implementation of vector octupole equations for eta = 0
def dydt_vecP(t, y, eps_gw, eps_gr, eps_oct, eta):
    a, j1, e1, j2, e2 = y[0], y[1:4], y[4:7], y[7:10], y[10:13]
    esq = np.dot(e1, e1)
    e = np.sqrt(esq)
    x1 = 1 - esq
    esq2 = np.dot(e2, e2)
    n2 = j2 / np.sqrt(1 - esq2)
    u2 = e2 / np.sqrt(esq2)

    dadt = (
        -eps_gw * (64 * (1 + 73 * esq / 24 + 37 * pow(esq, 2) / 96)) / (
            5 * pow(a, 3) * pow(x1, 3.5))
    )
    exn2 = np.cross(e1, n2)
    edn2 = np.dot(e1, n2)
    jxn2 = np.cross(j1, n2)
    jdn2 = np.dot(j1, n2)
    jxe = np.cross(j1, e1)
    djdt = 3 * pow(a, 1.5) / 4 * (
        jdn2 * jxn2
        - 5 * edn2 * exn2
    ) - 75 * eps_oct * a**(3/2) / 64 * (
        np.cross(
            2 * (np.dot(e1, u2) * np.dot(j1, n2)
                 + np.dot(e1, n2) * np.dot(j1, u2)) * j1
            + 2 * (np.dot(j1, u2) * np.dot(j1, n2)
                   - 7 * (np.dot(e1, u2) * np.dot(e1, n2))) * e1
        , n2) + np.cross(
            2 * np.dot(e1, n2) * np.dot(j1, n2) * j1
            + (8 * esq / 5 - 1/5 - 7 * np.dot(e1, n2)**2 + np.dot(j1, n2)**2) *
            e1, u2
        )
    )
    dedt_gw = -(
        eps_gw * (304 / 15) * (1 + 121 / 304 * esq) /
        (pow(a, 4) * pow(x1, 2.5))
    )
    dedt = 3 * pow(a, 1.5) / 4 * (
        jdn2 * exn2
        - 5 * edn2 * jxn2
        + 2 * jxe
    ) + dedt_gw * e1 + (
        eps_gr * jxe / (pow(x1, 1.5) * pow(a, 2.5))
    ) - 75 * eps_oct * a**(3/2) / 64 * (
        np.cross(
            2 * np.dot(e1, n2) * np.dot(j1, n2) * e1
            + (8 * esq / 5 - 1/5 - 7 * np.dot(e1, n2)**2 + np.dot(j1, n2)**2)
            * j1, u2
        ) + np.cross(
            2 * (np.dot(e1, u2) * np.dot(j1, n2)
                 + np.dot(e1, n2) * np.dot(j1, u2)) * e1
            + 2 * (np.dot(j1, n2) * np.dot(j1, u2)
                   - 7 * np.dot(e1, n2) * np.dot(e1, u2)) * j1,
            n2
        ) + 16 / 5 * np.dot(e1, u2) * np.cross(j1, e1)
    )
    dj2dt = 3 * a**(3/2) * eta / 4 * (
        jdn2 * (-jxn2)
        - 5 * edn2 * (-exn2)
    ) - 75 * eps_oct * a**(3/2) * eta / 64 * (
        2 * np.cross(
            np.dot(e1, n2) * np.dot(j1, u2) * n2
            + np.dot(e1, u2) * np.dot(j1, n2) * n2
            + np.dot(e1, n2) * np.dot(j1, n2) * u2
        , j1)
        + np.cross(
            2 * np.dot(j1, u2) * np.dot(j1, n2) * n2
            - 14 * np.dot(e1, u2) * np.dot(e1, n2) * n2
            + (8 * esq / 5 - 1/5 - 7 * np.dot(e1, n2)**2
               + np.dot(j1, n2)**2) * u2
        , e1)
    )
    de2dt = 3 * a**(3/2) * eta / (4 * np.sqrt(1 - esq2)) * (
        np.dot(j1, n2) * np.cross(e2, j1)
        - 5 * np.dot(e1, n2) * np.cross(e2, e1)
        - (0.5 - 3 * esq + 25 * np.dot(e1, n2)**2 / 2
           - 2.5 * np.dot(j1, n2)**2) * np.cross(n2, e2)
    ) - 75 * eta * eps_oct * a**(3/2) / (64 * np.sqrt(1 - esq2)) * (
        2 * np.cross(
            np.dot(e1, n2) * np.dot(j1, e2) * u2
            + np.dot(j1, 2) * np.dot(e1, e2) * u2
            + (1 - esq2) / np.sqrt(esq2) * np.dot(e1, n2)
                * np.dot(j1, n2) * n2
        , j1) + np.cross(
            2 * np.dot(j1, e2) * np.dot(j1, n2) * u2
            - 14 * np.dot(e1, e2) * np.dot(e1, n2) * u2
            + (1 - esq2) / np.sqrt(esq2) * (
                8 * esq / 5 - 1/5 - 7 * np.dot(e1, n2)**2
                + np.dot(j1, n2)**2
            ) * n2
        , e1) - np.cross(
            2 * (1 / 5 - 8 * esq / 5) * np.dot(e1, u2) * e2
            + 14 * np.dot(e1, n2) * np.dot(j1, u2) * np.dot(j1, n2) * e2
            + 7 * np.dot(e1, u2) * (
                8 * esq / 5 - 1/5 - 7 * np.dot(e1, n2)**2
                + np.dot(j1, n2)**2
            ) * e2
        , n2)
    )

    ret = [dadt, *djdt, *dedt, *dj2dt, *de2dt]
    return ret

# PYTHON VERSIONS FOR DEBUGGING Radau/BDF (maybe repro later)
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
    # return -3 * pow(a, 1.5) / (32 * sin(I) * j) * (
    #     2 * (2 + 3 * esq - 5 * esq * cos(2 * w))
    # )
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
