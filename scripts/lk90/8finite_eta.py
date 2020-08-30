'''
self-contained file for the finite-eta simulations
'''
import os
import pickle
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    plt.rc('lines', lw=3.5)
    plt.rc('xtick', direction='in', top=True, bottom=True)
    plt.rc('ytick', direction='in', left=True, right=True)
except:
    pass

from utils import ts_dot, get_vals
from multiprocessing import Pool

from scipy.integrate import solve_ivp
from scipy.optimize import brenth

m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0

# by convention, use solar masses, AU, and set c = 1, in which case G = 9.87e-9
G = 9.87e-9
def get_eps(m1, m2, m3, a0, a2, e2):
    m12 = m1 + m2
    mu = m1 * m2 / m12
    n = np.sqrt(G * m12 / a0**3)
    eps_gw = (1 / n) * (m12 / m3) * (a2**3 / a0**7) * G**3 * mu * m12**2
    eps_gr = (m12 / m3) * (a2**3 / a0**4) * (1 - e2**2)**(3/2) * 3 * G * m12
    eps_sl = (m12 / m3) * (a2**3 / a0**4) * (1 - e2**2)**(3/2) * (
        3 * G * (m2 + mu / 3) / 2)
    L1 = mu * np.sqrt(G * (m12) * a0)
    L2 = m3 * m12 / (m3 + m12) * np.sqrt(G * (m3 + m12) * a2)
    eta = L1 / L2
    return {'eps_gw': eps_gw, 'eps_gr': eps_gr, 'eps_sl': eps_sl,
            'eta': eta, 'e2': e2}

def get_Ilimd(eta=0, eps_gr=0, **kwargs):
    def jlim_criterion(j): # eq 44, satisfied when j = jlim
        return (
            3/8 * (j**2 - 1) * (
                - 3 + eta**2 / 4 * (4 * j**2 / 5 - 1))
            + eps_gr * (1 - 1 / j))
    jlim = brenth(jlim_criterion, 1e-15, 1 - 1e-15)
    Ilim = np.arccos(eta / 2 * (4 * jlim**2 / 5 - 1))
    Ilimd = np.degrees(Ilim)
    return Ilimd

def get_dydt(eps_gw=0, eps_gr=0, eps_sl=0, eta=0, e2=0):
    def dydt(t, y):
        '''
        dydt for all useful of 10 orbital elements + spin, eps_oct = 0 in LML15.
        eta = L / Lout
        '''
        a1, e1, W, I1, w1, I2, sx, sy, sz = y
        # print(t, a1, 1 - e1)
        Itot = I1 + I2
        x1 = 1 - e1**2
        x2 = 1 - e2**2

        # orbital evolution
        da1dt =  (
            -eps_gw * (64 * (1 + 73 * e1**2 / 24 + 37 * e1**4 / 96)) / (
                5 * a1**3 * x1**(7/2))
        )
        de1dt = (
            15 * a1**(3/2) * e1 * np.sqrt(x1) * np.sin(2 * w1)
                    * np.sin(Itot)**2 / 8
                - eps_gw * 304 * e1 * (1 + 121 * e1**2 / 304)
                    / (15 * a1**4 * x1**(5/2))
        )
        dWdt = (
            3 * a1**(3/2) * np.sin(2 * Itot) / np.sin(I1) *
                    (5 * e1**2 * np.cos(w1)**2 - 4 * e1**2 - 1)
                / (8 * np.sqrt(x1))
        )
        dI1dt = (
            -15 * a1**(3/2) * e1**2 * np.sin(2 * w1)
                * np.sin(2 * Itot) / (16 * np.sqrt(x1))
        )
        dI2dt = eta * (
            -15 * a1**(3/2) * e1**2 * np.sin(2 * w1)
                * 2 * np.sin(Itot) / (16 * np.sqrt(x2))
        )
        dw1dt = (
            3 * a1**(3/2) / 8 * (
                (4 * np.cos(Itot)**2 +
                 (5 * np.cos(2 * w1) - 1) * (1 - e1**2 - np.cos(Itot)**2)) /
                    np.sqrt(x1)
                + eta * np.cos(Itot) * (
                    2 + e1**2 * (3 - 5 * np.cos(2 * w1))) / np.sqrt(x2)
            )
            + eps_gr / (a1**(5/2) * x1)
        )

        # spin evolution
        Lhat = [np.sin(I1) * np.cos(W), np.sin(I1) * np.sin(W), np.cos(I1)]
        s = [sx, sy, sz]

        dsdt = eps_sl * np.cross(Lhat, s) / (a1**(5/2) * x1)

        return (da1dt, de1dt, dWdt, dI1dt, dw1dt, dI2dt, *dsdt)
    return dydt

def get_qslf_for_I0(I0, tf=np.inf, plot=False):
    print('Running for', np.degrees(I0))
    af = 5e-3
    getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    dydt = get_dydt(**getter_kwargs)

    # a1, e1, W, I1, w1, I2, sx, sy, sz = y
    # NB: y0 has Lout pointing up, no impact on dynamics
    s0 = [np.sin(I0), 0, np.cos(I0)] # initial alignment
    y0 = [1, 1e-3, 0, I0, 0, 0, *s0]

    a_term_event = lambda t, y: y[0] - af
    a_term_event.terminal = True
    ret = solve_ivp(dydt, (0, tf), y0, events=[a_term_event],
                    method='BDF', atol=1e-8, rtol=1e-8)

    _, _, W_arr, I_arr, _, _, *s_arr = ret.y
    Lhat_arr = [np.sin(I_arr) * np.cos(W_arr),
                np.sin(I_arr) * np.sin(W_arr),
                np.cos(I_arr)]
    qslfd = np.degrees(np.arccos(ts_dot(Lhat_arr, s_arr)))
    print('Ran for', np.degrees(I0), qslfd[-1])

    if plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
        ax1.semilogy(ret.t, ret.y[0], 'k', alpha=0.7, lw=0.7)
        ax2.semilogy(ret.t, 1 - ret.y[1], 'k', alpha=0.7, lw=0.7)
        ax3.plot(ret.t, np.degrees(ret.y[3]), 'k', alpha=0.7, lw=0.7)
        ax4.plot(ret.t, qslfd, 'k', alpha=0.7, lw=0.7)
        plt.savefig('8sim', dpi=200)
        plt.close()

    return qslfd[-1]

def ensemble_run(npts=200):
    pkl_fn = '8finite_qslfs'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)

        getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
        Ilimd = get_Ilimd(**getter_kwargs)
        incs1 = np.radians(np.linspace(Ilimd + 0.5, Ilimd, npts))
        incs2 = np.radians(
            np.linspace(Ilimd - 0.5, Ilimd, npts - 1, endpoint=False)
        )
        incs = np.array(list(zip(incs1, incs2))).flatten()
        with Pool(64) as p:
            qslfds = p.map(get_qslf_for_I0, incs)
        with open(pkl_fn, 'wb') as f:
            pickle.dump((incs, qslfds), f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            incs, qslfds = pickle.load(f)

if __name__ == '__main__':
    # getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    # Ilimd = get_Ilimd(**getter_kwargs)
    # get_qslf_for_I0(np.radians(Ilimd))

    ensemble_run()
    pass
