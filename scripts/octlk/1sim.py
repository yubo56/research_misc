import time
from utils import *
import numpy as np
from cython_utils import *

from scipy.integrate import solve_ivp
from scipy.optimize import brenth
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('lines', lw=1.0)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

TF = 100

def get_I1(I0, eta):
    ''' given total inclination between Lout and L, returns I_tot '''
    def I2_constr(_I2):
        return np.sin(_I2) - eta * np.sin(I0 - _I2)
    I2 = brenth(I2_constr, 0, np.pi, xtol=1e-12)
    return np.degrees(I0 - I2)

def test_orbs():
    m1, m2, m3, a, a2, e2 = 20, 30, 30, 100, 6000, 0.6
    # Seems like Radau & BDF do some invalid memory access, when dadt neq 0, e2
    # grows without bound
    eps = get_eps(m1, m2, m3, a, a2, e2)
    tlk0 = get_tlk0(m1, m2, m3, a, a2) / 1e8
    I0 = np.radians(93.5)
    I1 = np.radians(get_I1(I0, eps[3]))
    I2 = I0 - I1
    y0 = np.array([1.0, 1e-3, I1, 0, 0, e2, I2, 0, 0],
                  dtype=np.float64)
    def a_term_event(_t, y, *_args):
        return y[0] - 5e-3
    a_term_event.terminal = True
    ret = solve_ivp(dydt, (0, TF), y0, args=eps,
                    events=[a_term_event],
                    method='LSODA', atol=1e-12, rtol=1e-12)
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 8),
        gridspec_kw={'height_ratios': [1, 1]},
        sharex=True)
    ax1.semilogy(ret.t * tlk0, ret.y[0])
    ax1.set_ylabel(r'$a$')
    ax2.semilogy(ret.t * tlk0, 1 - ret.y[1])
    ax2.set_ylabel(r'$1 - e$')
    ax2.set_xlabel(r'Time $(10^8 \;\mathrm{yr})$')
    # plt.semilogy(ret.t * tlk0, 1 - ret.y[1])
    plt.tight_layout()
    plt.savefig('/tmp/foo', dpi=300)
    plt.clf()

def test_vec():
    m1, m2, m3, a, a2, e2 = 20, 30, 30, 100, 6000, 0.6
    a0, e0, I0, W0, w0 = (1, 1e-3, np.radians(93.5), 0, 0)
    W20, w20 = (0, 0.7)
    j0 = np.sqrt(1 - e0**2)
    j2 = np.sqrt(1 - e2**2)
    y0 = [
        a0,
        j0 * np.sin(W0) * np.sin(I0),
        -j0 * np.cos(W0) * np.sin(I0),
        j0 * np.cos(I0),
        e0 * (np.cos(w0) * np.cos(W0) - np.sin(w0) * np.cos(I0) * np.sin(W0)),
        e0 * (np.cos(w0) * np.sin(W0) + np.sin(w0) * np.cos(I0) * np.cos(W0)),
        e0 * np.sin(w0) * np.sin(I0),
        0,
        0,
        j2,
        e2 * (np.cos(w20) * np.cos(W20) - np.sin(w20) * np.sin(W20)),
        e2 * (np.cos(w20) * np.sin(W20) + np.sin(w20) * np.cos(W20)),
        0,
    ]
    eps = get_eps(m1, m2, m3, a, a2, e2)
    tlk0 = get_tlk0(m1, m2, m3, a, a2) / 1e8
    # tlk0 = 1
    def a_term_event(_t, y, *_args):
        return y[0] - 0.9
    a_term_event.terminal = True
    ret = solve_ivp(dydt_vecP, (0, TF), y0, args=eps,
                    events=[a_term_event],
                    method='Radau', atol=1e-9, rtol=1e-9)
    evec = ret.y[4:7, :]
    evec_mags = np.sqrt(np.sum(evec**2, axis=0))
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 8),
        gridspec_kw={'height_ratios': [1, 1]},
        sharex=True)
    ax1.semilogy(ret.t * tlk0, ret.y[0])
    ax1.set_ylabel(r'$a$')
    ax2.semilogy(ret.t * tlk0, 1 - evec_mags)
    ax2.set_ylabel(r'$1 - e$')
    ax2.set_xlabel(r'Time $(10^8 \;\mathrm{yr})$')
    plt.tight_layout()
    plt.savefig('/tmp/foo_vec', dpi=300)
    plt.clf()

def timing_tests():
    '''
    To run 90.35 to coalescence, using params from SLL20, orbs = 21s, vecs = 62,
    and native python orbs ~ 50
    '''
    start = time.time()
    test_orbs()
    print('Orbs used', time.time() - start)
    # start = time.time()
    # test_vec()
    # print('Vecs used', time.time() - start)

def sweep(m12=50, e0=1e-3, num_trials=5, ilow=91, ihigh=95, num_i=20,
          t_hubb_gyr=10):
    a0 = 1
    m3, a, a2, e2 = 30, 100, 6000, 0.6
    I0 = np.radians(93.5)
    def a_term_event(_t, y, *_args):
        return y[0] - 0.1
    a_term_event.terminal = True

    for q, fn in [
            [1, '1equaldist'],
            [0.2, '1p2dist']
            [0.5, '1p5dist'],
            [0.7, '1p7dist'],
            [0.4, '1p4dist'],
            [0.3, '1p3dist'],
    ]:

        m2 = m12 / (1 + q)
        m1 = m12 - m2
        eps = get_eps(m1, m2, m3, a, a2, e2)
        tlk0 = get_tlk0(m1, m2, m3, a, a2) / 1e9
        t_final = t_hubb_gyr / tlk0

        I0s = np.radians(np.linspace(ilow, ihigh, num_i))
        I_plots = []
        tmerges = []
        for I0 in I0s:
            I1 = np.radians(get_I1(I0, eps[3]))
            I2 = I0 - I1
            for _ in range(num_trials):
                W, w0, w20 = np.random.random(3) * 2 * np.pi
                y0 = [a0, e0, I1, W, w0, e2, I2, W + np.pi, w20]
                ret = solve_ivp(dydt, (0, t_final), y0, args=eps,
                                events=[a_term_event],
                                method='LSODA', atol=1e-12, rtol=1e-12)
                tmerges.append(ret.t[-1] * tlk0)
                I_plots.append(np.degrees(I0))
                print(q, np.degrees(I0), tmerges[-1])
        plt.plot(I_plots, tmerges, 'bx', ms=1)
        plt.xlabel(r'$I_0$')
        plt.ylabel(r'$T_m$ (Gyr)')
        plt.savefig(fn, dpi=300)
        plt.close()

if __name__ == '__main__':
    # timing_tests()
    sweep()
    pass
