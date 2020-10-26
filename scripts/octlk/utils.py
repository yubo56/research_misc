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
