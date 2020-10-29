import time
from utils import *
import numpy as np
from cython_utils import *
from multiprocessing import Pool

import os
import pickle

from scipy.integrate import solve_ivp
from scipy.optimize import brenth, root
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    plt.rc('lines', lw=1.0)
    plt.rc('xtick', direction='in', top=True, bottom=True)
    plt.rc('ytick', direction='in', left=True, right=True)
except:
    plt = None

AF = 5e-3 # in units of the initial a

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
    y0 = np.array([1.0, 1e-3, I1, 0, 0, e2, I2, 0, 0.7],
                  dtype=np.float64)
    def a_term_event(_t, y, *_args):
        return y[0] - AF
    a_term_event.terminal = True
    ret = solve_ivp(dydt, (0, 100 / tlk0), y0, args=eps,
                    events=[a_term_event],
                    method='LSODA', atol=1e-12, rtol=1e-12)
    fig, axs = plt.subplots(
        3, 1,
        figsize=(8, 12),
        sharex=True)

    t = ret.t * tlk0
    axs[0].semilogy(t, ret.y[0] * a)
    axs[0].set_ylabel(r'$a$ (AU)')
    axs[1].semilogy(t, 1 - ret.y[1])
    axs[1].set_ylabel(r'$1 - e$')
    axs[2].plot(t, np.degrees(ret.y[2]))
    axs[2].set_ylabel(r'$I_{\rm tot}$')
    axs[2].set_xlabel(r'Time $(10^8 \;\mathrm{yr})$')

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.03)
    plt.savefig('1fiducial', dpi=300)
    plt.clf()

def test_vec():
    k = 39.4751488
    c = 6.32397263*10**4
    m = 1
    mm = 1
    l = 1
    ll = 1

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
    N1 = np.sqrt((k*(M1 + M2))/INTain ** 3)
    Mu = (M1*M2)/(M1 + M2)
    J1 = (M2*M1)/(M2 + M1)*np.sqrt(k*(M2 + M1)*INTain)
    J2 = ((M2 + M1)*M3)/(M3 + M1 + M2) * np.sqrt(k*(M3 + M1 + M2)*a2 )
    T = 1e10

    w1 = 0
    w2 = 0.7
    W = 0

    E10 = 1e-3
    E20 = 0.6
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
    I1, I2 = root(f, [60, 0]).x

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
    args = [m, mm, l, ll, M1, M2, M3, Itot, INTain, a2, N1, Mu, J1, J2, T]
    ret = solve_ivp(dydt_vec_bin, (0, T), y0, args=args,
                    events=[a_term_event],
                    method='LSODA', atol=1e-12, rtol=1e-12)
    lin = ret.y[ :3, :]
    lin_mag = np.sqrt(np.sum(lin**2, axis=0))
    evec = ret.y[3:6, :]
    evec_mags = np.sqrt(np.sum(evec**2, axis=0))
    a = lin_mag**2/((Mu**2)*k*(M1 + M2)*(1 - evec_mags**2))
    I = np.degrees(np.arccos(ret.y[2] / lin_mag))

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(8, 12),
        sharex=True)

    ax1.semilogy(ret.t / 1e8, a)
    ax1.set_ylabel(r'$a$ (AU)')
    ax2.semilogy(ret.t / 1e8, 1 - evec_mags)
    ax2.set_ylabel(r'$1 - e$')
    ax3.plot(ret.t / 1e8, I)
    ax3.set_ylabel(r'$I_{\rm tot}$ (Deg)')
    ax3.set_xlabel(r'Time $(10^8 \;\mathrm{yr})$')

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.03)
    plt.savefig('1fiducial_vec', dpi=300)
    plt.clf()

def timing_tests():
    '''
    orb els is ~20% faster
    '''
    start = time.time()
    test_vec()
    print('Bin vecs used', time.time() - start)
    start = time.time()
    test_orbs()
    print('Orbs used', time.time() - start)

def sweeper(q, t_final, tlk0, a0, e0, I0, e2, eps):
    def a_term_event(_t, y, *_args):
        return y[0] - AF
    a_term_event.terminal = True

    I1 = np.radians(get_I1(I0, eps[3]))
    I2 = I0 - I1
    W, w0, w20 = np.random.random(3) * 2 * np.pi
    y0 = [a0, e0, I1, W, w0, e2, I2, W + np.pi, w20]
    ret = solve_ivp(dydt, (0, t_final), y0, args=eps,
                    events=[a_term_event],
                    method='LSODA', atol=1e-12, rtol=1e-12)
    tf = ret.t[-1] * tlk0
    print(q, np.degrees(I0), tf)
    return tf

def sweeper_bin(q, t_final, _tlk0, a0, e0, I0, e2, _eps):
    k = 39.4751488
    c = 6.32397263*10**4
    m = 1
    mm = 1
    l = 1
    ll = 1

    # length = AU
    # c = 1AU / 499s
    # unit of time = 499s * 6.32e4 = 1yr
    # unit of mass = solar mass, solve for M using N1 + distance in correct units
    M1 = 30
    M2 = 20
    M3 = 30
    Itot = np.degrees(I0)
    INTain = a0
    a2 = 6000
    N1 = np.sqrt((k*(M1 + M2))/INTain ** 3)
    Mu = (M1*M2)/(M1 + M2)
    J1 = (M2*M1)/(M2 + M1)*np.sqrt(k*(M2 + M1)*INTain)
    J2 = ((M2 + M1)*M3)/(M3 + M1 + M2) * np.sqrt(k*(M3 + M1 + M2)*a2 )
    T = 1e10

    w1 = np.random.rand() * 2 * np.pi
    w2 = np.random.rand() * 2 * np.pi
    W = np.random.rand() * 2 * np.pi

    E10 = e0
    E20 = e2
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
    I1, I2 = root(f, [60, 0]).x

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
    args = [m, mm, l, ll, M1, M2, M3, Itot, INTain, a2, N1, Mu, J1, J2, T]
    def a_term_event(*args):
        return get_ain_vec_bin(*args) - AF * INTain
    a_term_event.terminal = True
    ret = solve_ivp(dydt_vec_bin, (0, T), y0, args=args,
                    events=[a_term_event],
                    method='LSODA', atol=1e-12, rtol=1e-12)
    tf = ret.t[-1]
    print(q, np.degrees(I0), tf, T)
    return tf

def sweep(num_trials=20, num_i=200, t_hubb_gyr=10,
          folder='1sweep', func=sweeper, nthreads=10):
    mkdirp(folder)
    a0 = 1
    m12, m3, a, a2, e0, e2 = 50, 30, 100, 4500, 1e-3, 0.6
    p = Pool(nthreads)

    for q, base_fn, ilow, ihigh in [
            [1.0, '1equaldist', 92, 93.5],
            # [0.7, '1p7dist', 91, 95],
            # [0.5, '1p5dist', 91, 98],
            # [0.4, '1p4dist', 90.5, 98],
            # [0.3, '1p3dist', 90.5, 100],
            # [0.2, '1p2dist', 90.5, 105],
    ]:
        fn = '%s/%s' % (folder, base_fn)
        pkl_fn = fn + '.pkl'
        if not os.path.exists(pkl_fn):
            print('Running %s' % pkl_fn)
            m2 = m12 / (1 + q)
            m1 = m12 - m2
            eps = get_eps(m1, m2, m3, a, a2, e2)
            tlk0 = get_tlk0(m1, m2, m3, a, a2) / 1e9
            t_final = t_hubb_gyr / tlk0

            I0s = np.radians(np.linspace(ilow, ihigh, num_i))
            I_plots = np.repeat(I0s, num_trials)
            args = [
                (q, t_final, tlk0, a0, e0, I0, e2, eps)
                for I0 in I_plots
            ]
            tmerges = p.starmap(func, args)
            with open(pkl_fn, 'wb') as f:
                pickle.dump((I_plots, tmerges), f)
        else:
            with open(pkl_fn, 'rb') as f:
                print('Loading %s' % pkl_fn)
                I_plots, tmerges = pickle.load(f)

        tmerges = np.array(tmerges)
        merged = np.where(tmerges < 9.9)[0]
        nmerged = np.where(tmerges > 9.9)[0]
        if plt is not None:
            plt.semilogy(np.degrees(I_plots[merged]), tmerges[merged], 'go', ms=1)
            plt.semilogy(np.degrees(I_plots[nmerged]), tmerges[nmerged], 'b^', ms=1)
            plt.xlabel(r'$I_{\rm tot, 0}$')
            plt.ylabel(r'$T_m$ (Gyr)')
            plt.savefig(fn, dpi=300)
            plt.close()

def get_emax(t_final, y0, eps):
    ret = solve_ivp(dydt, (0, t_final), y0, args=eps,
                    method='LSODA', atol=1e-12, rtol=1e-12)
    return ret.y[1].max()

def emax_dist(num_trials=1000):
    a0 = 1
    m1, m2, m3, a, a2, e0, e2 = 20, 30, 30, 100, 4500, 1e-3, 0.6
    eps = get_eps(m1, m2, m3, a, a2, e2)
    eps[0] = 0 # eps_gw
    I0 = np.radians(93.5)
    I1 = np.radians(get_I1(I0, eps[3]))
    I2 = I0 - I1

    tlk0 = get_tlk0(m1, m2, m3, a, a2) / 1e9 # physical units, Gyr
    t_final = 0.1 / tlk0 # just run to 1e8 yr

    p = Pool(10)
    args = []
    for _ in range(num_trials):
        W, w0, w20 = np.random.random(3) * 2 * np.pi
        y0 = [a0, e0, I1, W, w0, e2, I2, W + np.pi, w20]
        args.append((t_final, y0, eps))
    emaxes = p.starmap(get_emax, args)
    emaxes = np.array(emaxes)
    plt.hist(np.log10(1 - emaxes), bins=50)
    plt.xlabel(r'$\log_{10} (1 - e_{\max})$ in $10^8\;\mathrm{yr}$')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig('1emaxdist')
    plt.close()

if __name__ == '__main__':
    # timing_tests()
    # emax_dist()

    # sweep(nthreads=20)
    sweep(folder='1sweepbin', func=sweeper_bin, nthreads=20)
    pass
