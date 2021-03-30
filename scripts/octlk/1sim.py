'''
TODO: redo one-shot-merger calculation, e_eff is correct
'''
from collections import defaultdict
import time
from utils import *
import numpy as np
try: # can't compile cython on MacOS...
    from cython_utils import *
except:
    pass
from multiprocessing import Pool

import os
import pickle

from scipy.integrate import solve_ivp
import scipy.optimize as opt
from scipy.interpolate import interp1d

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=20)
    plt.rc('lines', lw=1.0)
    plt.rc('xtick', direction='in', top=True, bottom=True)
    plt.rc('ytick', direction='in', left=True, right=True)
except:
    plt = None

AF = 5e-3 # in units of the initial a
TOL = 1e-11
legfs = 16

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
    ret = solve_ivp(dydt, (0, 20 / tlk0), y0, args=eps,
                    events=[a_term_event],
                    method='LSODA', atol=TOL / 1e3, rtol=TOL / 1e3)
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
    axs[2].set_ylabel(r'$I$')
    axs[2].set_xlabel(r'Time $(10^8 \;\mathrm{yr})$')

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.03)
    plt.savefig('1fiducial', dpi=300)
    plt.clf()

def test_vec(fn='1fiducial_vec', **kwargs):
    ret = run_vec(**kwargs)
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
    ax3.set_ylabel(r'$I$ (Deg)')
    ax3.set_xlabel(r'Time $(10^8 \;\mathrm{yr})$')

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.03)
    plt.savefig(fn, dpi=300)
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

    I1 = np.radians(get_I1(I0, eps[3]))
    I2 = I0 - I1
    np.random.seed(idx + int(time.time()))
    W, w0, w20 = np.random.random(3) * 2 * np.pi
    y0 = [a0, e0, I1, W, w0, e2, I2, W + np.pi, w20]
    ret = solve_ivp(dydt, (0, t_final), y0, args=eps,
                    events=[a_term_event],
                    method='LSODA', atol=TOL, rtol=TOL)
    tf = ret.t[-1] * tlk0
    print(idx, q, np.degrees(I0), tf)
    return tf

# manually codify a, M12
def sweeper_bin(idx, q, t_final, a0, a2, e0, e2, I0, return_final=False):
    np.random.seed(idx + int(time.time()))
    M1 = 50 / (1 + q)
    AF = 3 / a0
    ret = run_vec(
        T=t_final,
        M1=M1,
        M2=50 - M1,
        Itot=np.degrees(I0),
        INTain=a0,
        a2=a2,
        E10=e0,
        E20=e2,
        method='Radau',
        TOL=1e-8,
        AF=AF,
        w1=np.random.rand() * 2 * np.pi,
        w2=np.random.rand() * 2 * np.pi,
        W=np.random.rand() * 2 * np.pi,
    )
    tf = ret.t[-1]
    print(idx, q, t_final / 1e9, tf / 1e9, a0, a2, e0, e2, np.degrees(I0))
    if return_final:
        return tf, ret.y[ :, -1]
    return tf

bin_aeff = 700 * np.sqrt(1 - 0.9**2)
def sweep(num_trials=20, num_trials_purequad=4, num_i=200, t_hubb_gyr=10,
# def sweep(num_trials=3, num_trials_purequad=1, num_i=200, t_hubb_gyr=10,
          folder='1sweepbin', nthreads=60):
    mkdirp(folder)
    m12, m3, e0 = 50, 30, 1e-3

    # q, e2, filename, ilow, ihigh, a0, a2eff
    run_cfgs = [
        # exploratory, find the right inclination range to restrict to
        # [0.01, 0.6, 'explore_tp', 50, 130, 100, 3600],
        # [0.01, 0.6, 'tp', 68, 112, 100, 3600],
        # [0.2, 0.6, 'explore_1p2dist', 50, 130, 100, 3600],
        # [0.3, 0.6, 'explore_1p3dist', 50, 130, 100, 3600],
        # [0.4, 0.6, 'explore_1p4dist', 50, 130, 100, 3600],
        # [0.5, 0.6, 'explore_1p5dist', 50, 130, 100, 3600],
        # [0.7, 0.6, 'explore_1p7dist', 50, 130, 100, 3600],
        # [1.0, 0.6, 'explore_1equaldist', 50, 130, 100, 3600],

        # [0.2, 0.8, 'explore_e81p2dist', 50, 130, 100, 3600],
        # [0.3, 0.8, 'explore_e81p3dist', 50, 130, 100, 3600],
        # [0.4, 0.8, 'explore_e81p4dist', 50, 130, 100, 3600],
        # [0.5, 0.8, 'explore_e81p5dist', 50, 130, 100, 3600],
        # [0.7, 0.8, 'explore_e81p7dist', 50, 130, 100, 3600],
        # [1.0, 0.8, 'explore_e81equaldist', 50, 130, 100, 3600],

        # [0.2, 0.9, 'explore_e91p2dist', 50, 130, 100, 3600],
        # [0.3, 0.9, 'explore_e91p3dist', 50, 130, 100, 3600],
        # [0.4, 0.9, 'explore_e91p4dist', 50, 130, 100, 3600],
        # [0.5, 0.9, 'explore_e91p5dist', 50, 130, 100, 3600],
        # [0.7, 0.9, 'explore_e91p7dist', 50, 130, 100, 3600],
        # [1.0, 0.9, 'explore_e91equaldist', 50, 130, 100, 3600],

        # [0.2, 0.6, '1p2dist', 89.5, 105, 100, 3600],
        # [0.2, 0.6, '1p2distp2', 66, 87, 100, 3600],
        # [0.3, 0.6, '1p3dist', 90.5, 100, 100, 3600],
        # [0.3, 0.6, '1p3distp2', 73, 86, 100, 3600],
        # [0.4, 0.6, '1p4dist', 90.5, 98, 100, 3600],
        # [0.5, 0.6, '1p5dist', 91, 98, 100, 3600],
        # [0.7, 0.6, '1p7dist', 91, 95, 100, 3600],
        # [1.0, 0.6, '1equaldist', 92.0, 93.5, 100, 3600],

        # [0.2, 0.8, 'e81p2dist', 89, 107, 100, 3600],
        # [0.2, 0.8, 'e81p2distp2', 57, 86.5, 100, 3600],
        # [0.3, 0.8, 'e81p3dist', 90.5, 103, 100, 3600],
        # [0.3, 0.8, 'e81p3distp2', 63, 86.5, 100, 3600],
        # [0.4, 0.8, 'e81p4dist', 90.5, 100, 100, 3600],
        [0.4, 0.8, 'e81p4distp2', 76, 84, 100, 3600],
        # [0.5, 0.8, 'e81p5dist', 91, 98, 100, 3600],
        # [0.7, 0.8, 'e81p7dist', 91, 95, 100, 3600],
        # [1.0, 0.8, 'e81equaldist', 92.1, 93.5, 100, 3600],

        # [0.2, 0.9, 'e91p2dist', 89.5, 112, 100, 3600],
        # [0.2, 0.9, 'e91p2distp2', 54, 86.5, 100, 3600],
        # [0.3, 0.9, 'e91p3dist', 90, 107, 100, 3600],
        # [0.3, 0.9, 'e91p3distp2', 60, 84, 100, 3600],
        # [0.4, 0.9, 'e91p4dist', 90.5, 103, 100, 3600],
        # [0.4, 0.9, 'e91p4distp2', 69, 83, 100, 3600],
        # [0.5, 0.9, 'e91p5dist', 90.5, 101.5, 100, 3600],
        # [0.7, 0.9, 'e91p7dist', 91, 98, 100, 3600],
        # [1.0, 0.9, 'e91equaldist', 92.1, 93.5, 100, 3600],

        # Bin's case
        # [0.4, 0.9, 'bindist', 70, 110, 10, bin_aeff],
        # [1.0, 0.9, 'bindistequal', 70, 110, 10, bin_aeff], # TODO
    ]
    for cfg in run_cfgs:
        q, e2, base_fn, ilow, ihigh, a0, a2eff = cfg
        a2 = a2eff / np.sqrt(1 - e2**2)

        I0s = np.radians(np.linspace(ilow, ihigh, num_i))
        if q == 1.0:
            I_plots = np.repeat(I0s, num_trials_purequad)
        else:
            I_plots = np.repeat(I0s, num_trials)

        fn = '%s/%s' % (folder, base_fn)
        pkl_fn = fn + '.pkl'
        if not os.path.exists(pkl_fn):
            # print('Not exists: %s' % pkl_fn)
            # continue
            print('Running %s' % pkl_fn)
            p = Pool(nthreads)
            m2 = m12 / (1 + q)
            m1 = m12 - m2
            args = [
                (idx, q, t_hubb_gyr * 1e9, a0, a2, e0, e2, I0)
                for idx, I0 in enumerate(I_plots)
            ]
            tmerges = p.starmap(sweeper_bin, args)
            with open(pkl_fn, 'wb') as f:
                pickle.dump((I_plots, tmerges), f)
        else:
            with open(pkl_fn, 'rb') as f:
                print('Loading %s' % pkl_fn)
                I_plots, tmerges = pickle.load(f)
# I(min(deltaK)), gap_left, gap_right
# [1.0, 0.6, '1equaldist', 100, 3600] 89.91983967935872 None 92.8056112224449
# [0.2, 0.6, '1p2dist', 100, 3600] 88.31663326653307 85.4308617234469 91.04208416833667
# [0.3, 0.6, '1p3dist', 100, 3600] 87.83567134268537 83.9879759519038 91.52304609218436
# [0.4, 0.6, '1p4dist', 100, 3600] 87.51503006012024 None 92.00400801603206
# [0.5, 0.6, '1p5dist', 100, 3600] 87.35470941883767 None 92.16432865731463
# [0.7, 0.6, '1p7dist', 100, 3600] 87.03406813627254 None 92.64529058116233
# [1.0, 0.8, 'e81equaldist', 100, 3600] 88.31663326653307 None 93.28657314629258
# [0.2, 0.8, 'e81p2dist', 100, 3600] 88.15631262525051 85.11022044088176 90.72144288577155
# [0.3, 0.8, 'e81p3dist', 100, 3600] 87.6753507014028 83.50701402805612 91.52304609218436
# [0.4, 0.8, 'e81p4dist', 100, 3600] 87.1943887775551 82.22444889779558 92.00400801603206
# [0.5, 0.8, 'e81p5dist', 100, 3600] 86.87374749498997 None 92.3246492985972
# [0.7, 0.8, 'e81p7dist', 100, 3600] 86.55310621242485 None 92.96593186372746
# [1.0, 0.9, 'e91equaldist', 100, 3600] 87.99599198396794 None 93.76753507014027
# [0.2, 0.9, 'e91p2dist', 100, 3600] 87.1943887775551 84.78957915831663 90.56112224448898
# [0.3, 0.9, 'e91p3dist', 100, 3600] 87.35470941883767 82.70541082164328 91.36272545090179
# [0.4, 0.9, 'e91p4dist', 100, 3600] 86.7134268537074 81.2625250501002 92.16432865731463
# [0.5, 0.9, 'e91p5dist', 100, 3600] 86.39278557114228 None 92.64529058116233
# [0.7, 0.9, 'e91p7dist', 100, 3600] 86.07214428857715 None 93.28657314629258
# [0.4, 0.9, 'bindist', 10, 305.1229260478471] 86.23246492985972 80.62124248496994 91.84368737474949
# [0.01, 0.6, 'tp', 100, 3600] 89.11823647294588 89.91983967935872 90.08016032064128
EMAX_CFGS = [
    # a2 = 4500, e2 = 0.6
    [1.0, 0.6, '1equaldist', 100, 3600],
    [0.2, 0.6, '1p2dist', 100, 3600],
    [0.3, 0.6, '1p3dist', 100, 3600],
    [0.4, 0.6, '1p4dist', 100, 3600],
    [0.5, 0.6, '1p5dist', 100, 3600],
    [0.7, 0.6, '1p7dist', 100, 3600],

    [1.0, 0.8, 'e81equaldist', 100, 3600],
    [0.2, 0.8, 'e81p2dist', 100, 3600],
    [0.3, 0.8, 'e81p3dist', 100, 3600],
    [0.4, 0.8, 'e81p4dist', 100, 3600],
    [0.5, 0.8, 'e81p5dist', 100, 3600],
    [0.7, 0.8, 'e81p7dist', 100, 3600],

    [1.0, 0.9, 'e91equaldist', 100, 3600],
    [0.2, 0.9, 'e91p2dist', 100, 3600],
    [0.3, 0.9, 'e91p3dist', 100, 3600],
    [0.4, 0.9, 'e91p4dist', 100, 3600],
    [0.5, 0.9, 'e91p5dist', 100, 3600],
    [0.7, 0.9, 'e91p7dist', 100, 3600],

    # Bin's weird case
    # [1.0, 0.9, 'bindistequal', 10, 700 * np.sqrt(1 - 0.9**2)],
    [0.4, 0.9, 'bindist', 10, 700 * np.sqrt(1 - 0.9**2)],

    # misc cases
    [0.01, 0.6, 'tp', 100, 3600],
    # [0.3, 0.6, '1p3dist_long', 100, 3600, {'tf_mult': 2000}],
]
def run_emax_sweep(num_trials=5, num_trials_purequad=1, num_i=1000,
                   folder='1sweepbin_emax_long', nthreads=1, run_cfgs=EMAX_CFGS,
                   tf_mult_default=500):
    mkdirp(folder)
    m12, m3, e0 = 50, 30, 1e-3

    # q, e2, filename, ilow, ihigh, a0, a2eff
    for cfg in run_cfgs:
        q, e2, base_fn, a0, a2eff = cfg[ :5]
        override_kwargs = {} if len(cfg) == 5 else cfg[5]
        override_kwargs.setdefault('tf_mult', tf_mult_default)
        a2 = a2eff / np.sqrt(1 - e2**2)

        I0s = np.linspace(50, 130, num_i)
        fn = '%s/%s' % (folder, base_fn)
        pkl_fn = fn + '.pkl'

        m2 = m12 / (1 + q)
        m1 = m12 - m2

        if q == 1.0:
            I_plots = np.repeat(I0s, num_trials_purequad)
        else:
            I_plots = np.repeat(I0s, num_trials)

        # auto-determine tf
        args = [
            (idx, q, I0, None, dict(a0=a0, a2=a2, e2=e2, **override_kwargs))
            for idx, I0 in enumerate(I_plots)
        ]
        if not os.path.exists(pkl_fn):
            # print('Not exists %s' % pkl_fn)
            # continue
            print('Running %s' % pkl_fn)
            p = Pool(nthreads)
            rets = p.starmap(get_emax_series, args)
            with open(pkl_fn, 'wb') as f:
                pickle.dump(rets, f)
        else:
            with open(pkl_fn, 'rb') as f:
                print('Loading %s' % pkl_fn)
                rets = pickle.load(f)

        eta0 = get_eps_eta0(m1, m2, m3, a0, a2, e2)[3]
        if q == 1.0:
            I_plots = I0s
        else:
            I_plots = np.repeat(I0s, num_trials)
        k_pklfn = fn + 'kvals.pkl'
        if not os.path.exists(k_pklfn):
            print('Running %s' % k_pklfn)
            emaxes = []
            Kmaxes = [] # K = j * cos(I) - eta0 * e^2 / (2 * j2)
            Kmins = []
            K0s = []
            emeans = []
            for I0d, ret in zip(I_plots, rets):
                I0 = np.radians(I0d)
                e_vals = np.array(ret[1])
                I1_vals = np.radians(np.array(ret[2]))
                if len(e_vals) == 0:
                    emaxes.append(e0)
                    Imaxes.append(I0)
                    emeans.append(e0)
                    continue
                where_idx = np.where(e_vals >= 0.3)[0]
                e_vals = e_vals[where_idx]
                I1_vals = I1_vals[where_idx]
                emaxes.append(np.max(e_vals))
                # approx 1 + 73e^2/24... \approx 4.427 is constant
                jmean = np.mean((1 - e_vals**2)**(-3))**(-1/6)
                emean = np.sqrt(1 - jmean**2)
                emeans.append(emean)

                # calculate K using I1_vals and e_vals, at emax
                ltot_i = ltot(e0, I0, e2, eta0)
                e2_vals, I2_vals = np.array([
                    get_eI2(emax, Imax, eta0, ltot_i)
                    for emax, Imax in zip(e_vals, I1_vals)
                ]).T
                # e2_vals = np.full_like(e_vals, e2) # temp
                # I2_vals = np.zeros_like(e_vals)
                K_vals = (
                    np.sqrt(1 - e_vals**2) * np.cos(I1_vals + I2_vals)
                    - eta0 * e_vals**2 / (2 * np.sqrt(1 - e2_vals**2))
                )
                Kmins.append(K_vals.min())
                Kmaxes.append(K_vals.max())
                K0s.append(
                    np.sqrt(1 - e0**2) * np.cos(I0)
                    - eta0 * e0**2 / (2 * np.sqrt(1 - e2**2))
                )
            with open(k_pklfn, 'wb') as f:
                pickle.dump((emaxes, Kmaxes, Kmins, K0s, emeans), f)
        else:
            with open(k_pklfn, 'rb') as f:
                print('Loading %s' % k_pklfn)
                emaxes, Kmaxes, Kmins, K0s, emeans = pickle.load(f)


        I0_search = []
        Kdiffs = []
        for I0 in I0s:
            if I0 < 85 or I0 > 90:
                continue
            Kmax_here = np.array(Kmaxes)[np.where(I_plots == I0)[0]]
            Kmin_here = np.array(Kmins)[np.where(I_plots == I0)[0]]
            I0_search.append(I0)
            Kdiffs.append(np.max(Kmax_here) - np.min(Kmin_here))
        Imin = I0_search[np.argmin(Kdiffs)]
        emaxes = np.array(emaxes)
        left_idxs = np.where(np.logical_and(
            I_plots < 90,
            (1 - emaxes) / (1 - emaxes.max()) < 2,
        ))[0]
        right_idxs = np.where(np.logical_and(
            I_plots > 90,
            (1 - emaxes) / (1 - emaxes.max()) < 2,
        ))[0]
        if left_idxs.size > 0:
            print(cfg, Imin, I_plots[left_idxs[-1]], I_plots[right_idxs[0]])
        else:
            print(cfg, Imin, 'None', I_plots[right_idxs[0]])

        m2 = m12 / (1 + q)
        m1 = m12 - m2
        j_eff_crit = 0.01461 * (100 / a0)**(2/3)
        e_eff_crit = np.sqrt(1 - j_eff_crit**2)
        j_os = (256 * k**3 * q / (1 + q)**2 * m12**3 * a2eff**3 / (
            c**5 * a0**4 * np.sqrt(k * m12 / a0**3) * m3 * a0**3))**(1/6)
        e_os = np.sqrt(1 - j_os**2)
        _, eps_gr, eps_oct, eta = get_eps(m1, m2, m3, a0, a2, e2)
        Ilimd = get_Ilim(eta, eps_gr)
        elim = get_elim(eta, eps_gr)

        if plt is None:
            continue

        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(9, 7),
            sharex=True)
        ax1.semilogy(I_plots, 1 - np.array(emaxes), 'bo', ms=0.5,
                     label=r'$e_{\max}$')
        # ax1.semilogy(I_plots, 1 - np.array(emeans), 'go', ms=0.5,
        #              label=r'$e_{\rm eff}$')
        # ax1.axhline(1 - e_eff_crit, c='g', ls=':')
        # ax1.axhline(1 - e_os, c='b')
        ax1.axhline(1 - elim, c='r', ls='--')

        # overplot MLL fit for reference
        # ilimd_MLL_L = np.degrees(np.arccos(np.sqrt(
        #     0.26 * (eps_oct / 0.1)
        #     - 0.536 * (eps_oct / 0.1)**2
        #     + 12.05 * (eps_oct / 0.1)**3
        #     -16.78 * (eps_oct / 0.1)**4
        # )))
        # ilimd_MLL_R = np.degrees(np.arccos(-np.sqrt(
        #     0.26 * (eps_oct / 0.1)
        #     - 0.536 * (eps_oct / 0.1)**2
        #     + 12.05 * (eps_oct / 0.1)**3
        #     -16.78 * (eps_oct / 0.1)**4
        # )))
        # test using Antognini-like criterion? Doesn't work
        # def get_cross_lambda(crit):
        #     return lambda I_test: np.sqrt(1 - e0**2) * (
        #         - np.cos(I_test)
        #         + eta0 * np.sqrt(1 - e0**2) / (2 * np.sqrt(1 - e2**2))
        #     )**2 - crit
        # ilimd_MLL_L = np.degrees(opt.brenth(
        #     get_cross_lambda(0.05), 0, np.pi / 2))
        # ilimd_MLL_R = np.degrees(opt.brenth(
        #     get_cross_lambda(0.05), np.pi / 2, np.pi))
        # ax1.axvline(ilimd_MLL_L, c='m', lw=1.0)
        # ax1.axvline(ilimd_MLL_R, c='m', lw=1.0)

        # overplot emax due to quadrupole
        emaxes4 = []
        for I in I0s:
            emaxes4.append(get_emax(eta=eta, eps_gr=eps_gr, I=np.radians(I)))

        ax1.set_ylabel(r'$1 - e$')
        ticks = [50, 70, 90, 110, 130]
        ax1.set_xticks(ticks)
        ax1.set_xticklabels([r'$%d$' % d for d in ticks])
        # ax1.legend(fontsize=legfs)
        ax1.set_ylim(bottom=(1 - elim) / 5)
        ax1.axvline(Imin, c='k')

        Kcrit = (
            np.sqrt(1 - e0**2) * np.cos(np.radians(Ilimd))
            - eta0 * e0**2 / (2 * np.sqrt(1 - e2**2))
        )
        ax2.plot(I_plots, Kmins, 'bo', ms=0.5,
                 alpha=0.5)
        ax2.plot(I_plots, Kmaxes, 'go', ms=0.5,
                 alpha=0.5)
        xlims = ax2.get_xlim()
        ax2.plot(0, 0.1, 'bo', label=r'$K_{\min}$', ms=3)
        ax2.plot(0, 0.1, 'go', label=r'$K_{\max}$', ms=3)
        ax2.set_xlim(xlims)
        ax2.axvline(Imin, c='k')
        ax2.axhline(Kcrit, c='r', ls='--', lw=1.0)
        ax2.set_xlabel(r'$I_0$ (Deg)')
        ax2.set_ylabel(r'$K$')
        ax2.legend(fontsize=legfs, loc='upper right')

        rbound1, rbound2 = I_plots[right_idxs[0]], I_plots[right_idxs[-1]]
        ylim1 = ax1.get_ylim()
        ylim2 = ax2.get_ylim()
        ax1.fill_betweenx([1, 1e-10], [rbound1, rbound1], [rbound2, rbound2],
                          alpha=0.2, color='purple')
        ax2.fill_betweenx([-1, 1], [rbound1, rbound1], [rbound2, rbound2],
                          alpha=0.2, color='purple')
        if len(left_idxs):
            lbound1, lbound2 = I_plots[left_idxs[0]], I_plots[left_idxs[-1]]
            ax1.fill_betweenx([1, 1e-10],
                              [lbound1, lbound1],
                              [lbound2, lbound2],
                              alpha=0.2, color='purple')
            ax2.fill_betweenx([-1, 1],
                              [lbound1, lbound1],
                              [lbound2, lbound2],
                              alpha=0.2, color='purple')
        # plot last for visibility
        emaxes4.append(get_emax(eta=eta, eps_gr=eps_gr, I=np.radians(Ilimd)))
        emaxes4 = np.array(emaxes4)
        I0s_emaxes4 = np.concatenate((I0s, np.array([Ilimd])))
        idxs4 = np.argsort(I0s_emaxes4)
        ax1.plot(I0s_emaxes4[idxs4], 1 - np.array(emaxes4[idxs4]), 'k--', lw=2.0)
        ax2.plot(I_plots, K0s, 'k--', label=r'$K_0$', lw=2.0)
        ax1.set_ylim(ylim1)
        ax2.set_ylim(ylim2)

        plt.tight_layout()
        fig.subplots_adjust(hspace=0.03, wspace=0.03)
        plt.savefig(folder + '/' + base_fn, dpi=300)
        plt.close()

# default tf is 500 tk, if tf == None
def get_emax_series(idx, q, I0, tf, kwargs={}):
    np.random.seed(idx + int(time.time()))
    M1 = kwargs.get('M12', 50) / (1 + q)
    M2 = kwargs.get('M12', 50) - M1
    M3 = kwargs.get('M3', 30)
    ain = kwargs.get('a0', 100)
    a2 = kwargs.get('a2', 4500)
    E2 = kwargs.get('e2', 0.6)
    n1 = np.sqrt((k*(M1 + M2))/ain ** 3)
    tk = 1/n1*((M1 + M2)/M3)*(a2/ain)**3*(1 - E2**2)**(3.0/2)
    k2 = kwargs.get('k2', 0)
    R2 = kwargs.get('R2', 0)
    l = kwargs.get('l', 1)

    tgw = k**3 * M1 * M2 * (M1 + M2) / (c**5 * ain**4)

    if tf == None:
        tf = kwargs.get('tf_mult', 500) * tk

    ret = run_vec(
        ll=0,
        mm=kwargs.get('mm', 1),
        l=l,
        T=tf,
        M1=M1,
        M2=M2,
        M3=M3,
        Itot=I0,
        INTain=ain,
        a2=a2,
        E20=E2,
        TOL=1e-9,
        k2=k2,
        R2=R2,
        method='Radau',
        w1=kwargs.get('w1', np.random.rand() * 2 * np.pi),
        w2=kwargs.get('w2', np.random.rand() * 2 * np.pi),
        W=kwargs.get('W', np.random.rand() * 2 * np.pi),
    )
    evec = ret.y[3:6, :]
    evec_mags = np.sqrt(np.sum(evec**2, axis=0))

    lin = ret.y[ :3, :]
    lin_mag = np.sqrt(np.sum(lin**2, axis=0))
    I = np.degrees(np.arccos(ret.y[2] / lin_mag))

    # extract emaxes by looking in windows where de/dt is small. assume emaxes
    # are well separated by > 0.1 * tk
    ts = []
    emaxes = []
    Ivals = []
    dx=10
    demag = evec_mags[2 * dx: ] - evec_mags[ :-2 * dx]
    demag_ts = ret.t[dx:-dx]
    t_idxs = np.where(abs(demag) < 1e-4)[0] + 10

    if len(t_idxs) == 0:
        print('Ran for', idx, q, 'no maxima??')
        return np.array(ts), np.array(emaxes)

    blockstartidx = t_idxs[0]
    for t_idx, next_t_idx in zip(t_idxs, np.concatenate((t_idxs[1: ], [-1]))):
        if next_t_idx != -1 and ret.t[next_t_idx] - ret.t[t_idx] < 0.1 * tk:
            continue
        # t_idx is the last in its block
        emax_idx = np.argmax(evec_mags[blockstartidx:next_t_idx]) + blockstartidx
        emax = evec_mags[emax_idx]
        # if emax < 0.3: # eliminate minima from calculation
        #     continue
        ts.append(ret.t[emax_idx])
        emaxes.append(emax)
        Ivals.append(I[emax_idx])
        blockstartidx = next_t_idx
    print('Ran for (%d, %.1f, %.3f Gyr, emax=%.7f)' %
          (idx, q, tf / 10**9, np.max(emaxes)))
    return np.array(ts), np.array(emaxes), np.array(Ivals)

def plot_emax_dq(I0=93.5, fn='q_sweep_935', tf=3e9, num_reps=100):
    folder = '1emax_q'
    mkdirp(folder)

    q_arr = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

    eps = get_eps(25, 25, 30, 100, 4500, 0.6)
    # my epsilons above assume e2 = 0
    elim = get_elim(eps[3], eps[1])

    filename = folder + '/' + fn
    pkl_fn = filename + '.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        p = Pool(64)

        # just run 1.0 once
        args = [(q, I0, tf) for q in q_arr[ :-1]]
        args_full = [(idx, *args[idx // num_reps])
                     for idx in range(num_reps * len(args))]
        args_full.append((-1, 1.0, I0, tf))
        ret = p.starmap(get_emax_series, args_full)

        q_full = np.repeat(q_arr, num_reps)
        ts = [k[0] for k in ret]
        emax_arr = [k[1] for k in ret]
        I_arr = [k[2] for k in ret]
        dat = (q_full, ts, emax_arr, I_arr)
        with open(pkl_fn, 'wb') as f:
            pickle.dump(dat, f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            dat = pickle.load(f)
    if plt is None:
        return
    fig, _axs = plt.subplots(
        3, 2,
        figsize=(12, 9),
        sharex=True, sharey=True)
    axs = _axs.flat
    axmap = {
        0.2: axs[0],
        0.3: axs[1],
        0.4: axs[2],
        0.5: axs[3],
        0.7: axs[4],
        1.0: axs[5],
    }

    colors = ['k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan']
    # plot actual trajectories (too messy for longer times)
    # num_plotted = {q:0 for q in q_arr}
    # for q, t, emaxes, I_vals in zip(*dat):
    #     ax = axmap[q]
    #     idx = num_plotted[q]
    #     if q == 1.0:
    #         if idx == 0:
    #             emax_quad = np.median(emaxes)
    #         elif idx > 0:
    #             continue # do not plot q=1.0 more than once
    #     num_plotted[q] += 1
    #     c = colors[idx % len(colors)]
    #     ax.semilogy(t / tf * 10, 1 - emaxes,
    #                 c=c, marker='o', ms=2.5, lw=0, ls='')
    #     ax.plot(t / tf * 10, 1 - emaxes,
    #             c=c, ls=':', lw=0.7)
    # for q, ax in axmap.items():
    #     ax.axhline(1 - elim, c='k', ls='--', lw=1.0)
    #     ax.axhline(1 - emax_quad, c='b', ls='--', lw=1.0)
    # # text last, after lims are set
    # for q, ax in axmap.items():
    #     ax.text(ax.get_xlim()[0] + 0.2, ax.get_ylim()[1] / 3, 'q=%.1f' % q)
    # axs[0].set_ylabel(r'$1 - e_{\max}$')
    # axs[2].set_ylabel(r'$1 - e_{\max}$')
    # axs[4].set_ylabel(r'$1 - e_{\max}$')
    # axs[4].set_xlabel(r'$t$ ($10^{%d}$ Gyr)' % (np.round(np.log10(tf)) - 1))
    # axs[5].set_xlabel(r'$t$ ($10^{%d}$ Gyr)' % (np.round(np.log10(tf)) - 1))

    # plt.suptitle(r'$I = %.1f$' % I0, fontsize=20)

    # plt.tight_layout()
    # fig.subplots_adjust(hspace=0.03, wspace=0.03)
    # plt.savefig(filename, dpi=300)
    # plt.close()

    # hist the emaxes
    fig, _axs = plt.subplots(
        3, 2,
        figsize=(12, 9),
        sharex=True, sharey=True)
    axs = _axs.flat
    axmap = {
        0.2: axs[0],
        0.3: axs[1],
        0.4: axs[2],
        0.5: axs[3],
        0.7: axs[4],
        1.0: axs[5],
    }
    hist_vals = defaultdict(list)
    hist_incs = defaultdict(list)
    for q, t, emaxes, Ivals in zip(*dat):
        ax = axmap[q]
        if q == 1.0:
            emax_quad = np.median(emaxes)
        hist_vals[q].extend(np.log10(1 - emaxes))
        hist_incs[q].extend(Ivals)

    # use global hist bins
    _, bin_edges = np.histogram([v for x in hist_vals.values() for v in x],
                                bins=100)
    for q, ax in axmap.items():
        ax.hist(hist_vals[q], bins=bin_edges)
        ax.axvline(np.log10(1 - elim), c='k', ls='--', lw=1.0)
        ax.axvline(np.log10(1 - emax_quad), c='b', ls='--', lw=1.0)
    for q, ax in axmap.items():
        xlim = ax.get_xlim()
        xpos = xlim[1] - (xlim[1] - xlim[0]) / 8
        ax.text(xpos, ax.get_ylim()[1] * 0.9, 'q=%.1f' % q)
    axs[4].set_xlabel(r'$\log_{10}(1 - e_{\max})$')
    axs[5].set_xlabel(r'$\log_{10}(1 - e_{\max})$')
    plt.suptitle(r'$I = %.1f$' % I0, fontsize=20)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.03, wspace=0.03)
    plt.savefig(filename + 'hist', dpi=300)
    plt.close()

    # hist the inclinations
    fig, _axs = plt.subplots(
        3, 2,
        figsize=(12, 9),
        sharex=True, sharey=True)
    axs = _axs.flat
    axmap = {
        0.2: axs[0],
        0.3: axs[1],
        0.4: axs[2],
        0.5: axs[3],
        0.7: axs[4],
        1.0: axs[5],
    }
    # use global hist bins
    _, inc_bin_edges = np.histogram([v for x in hist_incs.values() for v in x],
                                bins=100)
    for q, ax in axmap.items():
        ax.hist(hist_incs[q], bins=inc_bin_edges)
    for q, ax in axmap.items():
        xlim = ax.get_xlim()
        xpos = xlim[1] - (xlim[1] - xlim[0]) / 8
        ax.text(xpos, ax.get_ylim()[1] * 0.9, 'q=%.1f' % q)
    axs[4].set_xlabel(r'$I(e_{\max})$ (Deg)')
    axs[5].set_xlabel(r'$I(e_{\max})$ (Deg)')
    plt.suptitle(r'$I = %.1f$' % I0, fontsize=20)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.03, wspace=0.03)
    plt.savefig(filename + 'histinc', dpi=300)
    plt.close()

    # plot delay time distributions
    fig, _axs = plt.subplots(
        3, 2,
        figsize=(12, 9),
        sharex=True, sharey=True)
    axs = _axs.flat
    axmap = {
        0.2: axs[0],
        0.3: axs[1],
        0.4: axs[2],
        0.5: axs[3],
        0.7: axs[4],
        1.0: axs[5],
    }
    # q -> map[eccentricity_idx] -> [first_time_hit]
    times_map = {
        0.2: [list() for _ in range(len(bin_edges))],
        0.3: [list() for _ in range(len(bin_edges))],
        0.4: [list() for _ in range(len(bin_edges))],
        0.5: [list() for _ in range(len(bin_edges))],
        0.7: [list() for _ in range(len(bin_edges))],
        1.0: [list() for _ in range(len(bin_edges))],
    }
    for q, t, emaxes, I_vals in zip(*dat):
        ax = axmap[q]
        for idx, ecc in enumerate(bin_edges):
            idxs = np.where((np.log10(1 - emaxes)) < ecc)[0]
            if len(idxs) == 0:
                times_map[q][idx].append(np.inf)
                continue
            times_map[q][idx].append(t[idxs[0]])
    # y-axis = median time to arrive at eccentricity x
    pow10_yr = int(np.log10(tf)) - 1
    for q, ax in axmap.items():
        median_times = np.array([
            np.median(l) / 10**(pow10_yr)
            if len(l) > 0 else tf / 10**(pow10_yr)
            for l in times_map[q]
        ])
        wherenoninf = np.where(median_times < np.inf)[0]
        ax.plot(-bin_edges[wherenoninf], median_times[wherenoninf])
        ax.axvline(-np.log10(1 - elim), c='k', ls='--', lw=1.0)
        ax.axvline(-np.log10(1 - emax_quad), c='b', ls='--', lw=1.0)
    for q, ax in axmap.items():
        xlim = ax.get_xlim()
        xpos = xlim[0] + (xlim[1] - xlim[0]) / 8
        ax.text(xpos, ax.get_ylim()[1] * 0.9, 'q=%.1f' % q)
        ax.set_ylim(top=31)
    axs[0].set_ylabel(r'$t$ ($10^{%d}$ Gyr)' % pow10_yr)
    axs[2].set_ylabel(r'$t$ ($10^{%d}$ Gyr)' % pow10_yr)
    axs[4].set_ylabel(r'$t$ ($10^{%d}$ Gyr)' % pow10_yr)
    axs[4].set_xlabel(r'$-\log_{10}(1 - e_{\max})$')
    axs[5].set_xlabel(r'$-\log_{10}(1 - e_{\max})$')
    plt.suptitle(r'$I = %.1f$' % I0, fontsize=20)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.03, wspace=0.03)
    plt.savefig(filename + 'delays', dpi=300)
    plt.close()

def run_nogw_vec(fn='1nogw_vec', q=2/3, M12=50, M3=30, a0=100, e2=0.6,
                 a2=4500, plot_full=True, ret=None, **kwargs):
    Itot = kwargs.get('Itot', 93.5)
    M1 = M12 / (1 + q)
    M2 = M12 - M1
    eps = get_eps(M2, M1, M3, a0, a2, e2)
    eta_ecc = eps[3]

    if ret is None:
        ret = run_vec(a2=a2, M1=M1, M2=M2, **kwargs)
    lin = ret.y[ :3, :]
    lin_mag = np.sqrt(np.sum(lin**2, axis=0))
    lout = ret.y[6:9, :]
    lout_mag = np.sqrt(np.sum(lout**2, axis=0))
    evec = ret.y[3:6, :]
    evec_mags = np.sqrt(np.sum(evec**2, axis=0))
    eoutvec = ret.y[9:12, :]
    eoutvec_mags = np.sqrt(np.sum(eoutvec**2, axis=0))
    Mu = M1 * M2 / M12
    a = lin_mag**2/((Mu**2)*k*50*(1 - evec_mags**2))
    I = np.degrees(np.arccos(ret.y[2] / lin_mag))
    Iout = np.degrees(np.arccos(ret.y[8] / lout_mag))
    Ie = np.degrees(np.arccos(ret.y[5] / evec_mags))

    n2hat = lout / lout_mag
    u2hat = eoutvec / eoutvec_mags
    v2hat = ts_cross(n2hat, u2hat)
    We = np.degrees(np.arctan2(ts_dot(evec, v2hat), ts_dot(evec, u2hat)))

    w1 = np.unwrap(np.arcsin(evec[2] / (evec_mags * np.sin(np.radians(I)))))

    # kozai constant (LL18.37)
    eta = eps[3]
    K = (
        np.sqrt(1 - evec_mags**2) * np.cos(np.radians(I + Iout))
        - eta * evec_mags**2/2
    )

    if plot_full:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            4, 1,
            figsize=(8, 11),
            sharex=True)
        ax4.plot(ret.t / 1e8, We, 'ko', ms=0.7)
        ax4.set_ylabel(r'$\Omega_{\rm e}$ (Deg)')
        ax4.set_xlabel(r'Time $(10^8 \;\mathrm{yr})$')
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1,
            figsize=(8, 8),
            sharex=True)
        ax3.set_xlabel(r'Time $(10^8 \;\mathrm{yr})$')

    ax1.semilogy(ret.t / 1e8, 1 - evec_mags, 'k', lw=1)
    ax1.axhline(1 - get_elim(eta, eps[1]), c='k', ls='--', lw=2)
    ax1.set_ylabel(r'$1 - e$')
    ax3.plot(ret.t / 1e8, K, 'k', lw=1)
    ax3.axhline(-eta / 2, c='k', ls=':', lw=2)
    ax3.set_ylabel(r'$K$')
    if (1 - np.max(evec_mags)) > 1e-4:
        ax1.set_yticks([1e-3, 1e-2, 1e-1, 1])
        ax1.set_yticklabels([r'$10^{%d}$' % d for d in [-3, -2, -1, 0]])
    else:
        ax1.set_yticks([1e-6, 1e-4, 1e-2, 1])
        ax1.set_yticklabels([r'$10^{%d}$' % d for d in [-6, -4, -2, 0]])
    ax2.plot(ret.t / 1e8, I + Iout, 'k', lw=1)
    ax2.set_ylabel(r'$I$ (Deg)')

    # try to predict eout_max & bounds of oscillation of K?
    # emax = get_emax(eta_ecc, eps[1] * (1 - 0.6**2)**(3/2), I=np.radians(Itot))
    # elim = get_elim(eta_ecc, eps[1] * (1 - 0.6**2)**(3/2))
    # jmin = np.sqrt(1 - emax**2)
    # jlim = np.sqrt(1 - elim**2)
    # delta_jout = np.cos(np.radians(np.max(I))) * eta_ecc * (jmin - jlim)
    # jout0 = np.sqrt(1 - 0.6**2)
    # jout_min = delta_jout + jout0
    # eout_max = np.sqrt(1 - jout_min**2)
    # eta_min = eps[3] / np.sqrt(1 - eout_max**2)
    # Kmin = -eta_min / 2
    # ax1.axhline(eout_max, c='r', ls=':', lw=2)
    # ax4.axhline(Kmin, c='r', ls=':', lw=2)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.03)
    print('Saving', fn)
    plt.savefig(fn, dpi=300)
    plt.clf()
    return ret

def emax_omega_sweep(fn='1sweep/emax_omega_sweep'):
    q, I0, tf = 0.234, 98, 3e9
    inits = {'w2':0, 'W': 0}
    pkl_fn = fn + '.pkl'
    num_pts = 100
    w1s = np.arange(num_pts) / num_pts * np.pi
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        p = Pool(10)
        args = [(idx, q, I0, tf, dict(inits, w1=w1s[idx]))
                for idx in range(num_pts)]
        rets = p.starmap(get_emax_series, args)
        with open(pkl_fn, 'wb') as f:
            pickle.dump((rets), f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            rets = pickle.load(f)

    fig, axs = plt.subplots(
        2, 2,
        figsize=(8, 8),
        sharex=True, sharey=True)
    times = np.array([1, 2, 4, 8]) * tf / 8
    for t_end, ax in zip(times, axs.flat):
        for w1, (t, emaxes, _) in zip(w1s, rets):
            emaxes_before_end = emaxes[np.where(t < t_end)[0]]
            ax.semilogy(w1, 1 - np.max(emaxes_before_end), 'bo')
    axs[1][0].set_xlabel(r'$\omega_{1,0}$')
    axs[1][1].set_xlabel(r'$\omega_{1,0}$')
    axs[0][0].set_ylabel(r'$1 - \max(e_{\max})$')
    axs[1][0].set_ylabel(r'$1 - \max(e_{\max})$')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.03, wspace=0.03)
    plt.savefig(fn, dpi=300)

def k_sweep_runner(idx, q, I0, tf):
    np.random.seed(idx + int(time.time()))
    M1 = 50 / (1 + q)
    M2 = 50 - M1
    M3 = 30
    ain = 100
    a2 = 4500
    E2 = 0.6
    n1 = np.sqrt((k*(M1 + M2))/ain ** 3)
    ret = run_vec(
        ll=0,
        T=tf,
        M1=M1,
        M2=M2,
        M3=M3,
        Itot=I0,
        INTain=ain,
        a2=a2,
        E20=E2,
        TOL=1e-9,
        method='Radau',
        w1=np.random.rand() * 2 * np.pi,
        w2=np.random.rand() * 2 * np.pi,
        W=np.random.rand() * 2 * np.pi,
    )

    lin = ret.y[ :3, :]
    lin_mag = np.sqrt(np.sum(lin**2, axis=0))
    lout = ret.y[6:9, :]
    lout_mag = np.sqrt(np.sum(lout**2, axis=0))
    evec = ret.y[3:6, :]
    evec_mags = np.sqrt(np.sum(evec**2, axis=0))
    eoutvec = ret.y[9:12, :]
    eoutvec_mags = np.sqrt(np.sum(eoutvec**2, axis=0))
    Mu = 30 * 20 / 50
    a = lin_mag**2/((Mu**2)*k*50*(1 - evec_mags**2))
    I = np.degrees(np.arccos(ret.y[2] / lin_mag))
    Iout = np.degrees(np.arccos(ret.y[8] / lout_mag))

    # kozai constant (LL18.37)
    eps = get_eps(20, 30, 30, 100, a2, 0.6)
    eta_ecc = eps[3]
    eta = eps[3] * np.sqrt(1 - 0.6**2) / np.sqrt(1 - eoutvec_mags**2)
    K = (
        np.sqrt(1 - evec_mags**2) * np.cos(np.radians(I + Iout))
        - eta * evec_mags**2/2
    )
    deltaK = np.max(K) - np.min(K)
    print('Ran for (%d, %.1f, %.1f)' % (idx, q, I0), deltaK)
    return deltaK

def k_sweep(fn='1sweep/ksweep', n_pts=30, tf=1e9, n_reps=3):
    '''
    try to see whether delta K has any discernable scaling with either I or q
    '''
    q_vals = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    I_vals = [90, 92, 94, 96, 98, 100]
    # equally spaced in eps_oct =  (1-q)/(1+q) \in [0,
    eps_arr = np.linspace(0, (1 - q_vals[0]) / (1 + q_vals[0]), n_pts)
    q_arr = (1 - eps_arr) / (1 + eps_arr)
    I_arr = np.linspace(np.min(I_vals), np.max(I_vals), n_pts)
    args = []
    for q in q_vals:
        for rep in range(n_reps):
            args.extend([(q, I, tf) for idx, I in enumerate(I_arr)])
    for I in I_vals:
        for rep in range(n_reps):
            args.extend([(q, I, tf) for idx, q in enumerate(q_arr)])

    pkl_fn = fn + '.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        p = Pool(10)
        full_args = [(idx, *q) for idx, q in enumerate(args)]
        dat = p.starmap(k_sweep_runner, full_args)
        with open(pkl_fn, 'wb') as f:
            pickle.dump((dat), f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            dat = pickle.load(f)

    args = np.array(args)
    dat = np.array(dat)
    for q in q_vals:
        to_plot_idxs = np.where([arg[0] == q for arg in args])[0]
        plt.plot(args[to_plot_idxs, 1],
                 dat[to_plot_idxs],
                 ls='',
                 marker='o',
                 ms=2.0,
                 label=str(q))
    plt.legend(loc='upper left', ncol=2, fontsize=legfs)
    plt.xlabel(r'$I_0$ (Deg)')
    plt.ylabel(r'$\Delta K$')
    plt.savefig(fn + 'vsq', dpi=300)
    plt.close()

    for I in I_vals:
        to_plot_idxs = np.where([arg[1] == I for arg in args])[0]
        q_dat = args[to_plot_idxs, 0]
        eps_dat = (1 - q_dat) / (1 + q_dat)
        plt.plot(eps_dat,
                 dat[to_plot_idxs],
                 ls='',
                 marker='o',
                 ms=2.0,
                 label=str(I))
    plt.legend(loc='upper left', ncol=2, fontsize=legfs)
    plt.xlabel(r'$\epsilon_{\rm oct} / \epsilon_{\rm oct}(q = 0)$')
    plt.ylabel(r'$\Delta K$')
    plt.savefig(fn + 'vsI', dpi=300)
    plt.close()

COMPOSITE_CFGS = [
    [
        [0.2, 0.6, 'explore_1p2dist', 50, 130, 100, 3600],
        [0.2, 0.6, '1p2dist', 89.5, 105, 100, 3600],
        [0.2, 0.6, '1p2distp2', 66, 87, 100, 3600],
    ],
    [
        [0.3, 0.6, 'explore_1p3dist', 50, 130, 100, 3600],
        [0.3, 0.6, '1p3dist', 90.5, 100, 100, 3600],
        [0.3, 0.6, '1p3distp2', 73, 86, 100, 3600],
    ],
    [
        [0.4, 0.6, 'explore_1p4dist', 50, 130, 100, 3600],
        [0.4, 0.6, '1p4dist', 90.5, 98, 100, 3600],
    ],
    [
        [0.5, 0.6, 'explore_1p5dist', 50, 130, 100, 3600],
        [0.5, 0.6, '1p5dist', 91, 98, 100, 3600],
    ],
    [
        [0.7, 0.6, 'explore_1p7dist', 50, 130, 100, 3600],
        [0.7, 0.6, '1p7dist', 91, 95, 100, 3600],
    ],
    [
        [1.0, 0.6, 'explore_1equaldist', 50, 130, 100, 3600],
        [1.0, 0.6, '1equaldist', 92.0, 93.5, 100, 3600],
    ],
    [
        [0.2, 0.8, 'explore_e81p2dist', 50, 130, 100, 3600],
        [0.2, 0.8, 'e81p2dist', 89, 107, 100, 3600],
        [0.2, 0.8, 'e81p2distp2', 57, 86.5, 100, 3600],
    ],
    [
        [0.3, 0.8, 'explore_e81p3dist', 50, 130, 100, 3600],
        [0.3, 0.8, 'e81p3dist', 90.5, 103, 100, 3600],
        [0.3, 0.8, 'e81p3distp2', 63, 86.5, 100, 3600],
    ],
    [
        [0.4, 0.8, 'explore_e81p4dist', 50, 130, 100, 3600],
        [0.4, 0.8, 'e81p4dist', 90.5, 100, 100, 3600],
        [0.4, 0.8, 'e81p4distp2', 76, 84, 100, 3600],
    ],
    [
        [0.5, 0.8, 'explore_e81p5dist', 50, 130, 100, 3600],
        [0.5, 0.8, 'e81p5dist', 91, 98, 100, 3600],
    ],
    [
        [0.7, 0.8, 'explore_e81p7dist', 50, 130, 100, 3600],
        [0.7, 0.8, 'e81p7dist', 91, 95, 100, 3600],
    ],
    [
        [1.0, 0.8, 'explore_e81equaldist', 50, 130, 100, 3600],
        [1.0, 0.8, 'e81equaldist', 92.1, 93.5, 100, 3600],
    ],
    [
        [0.2, 0.9, 'explore_e91p2dist', 50, 130, 100, 3600],
        [0.2, 0.9, 'e91p2dist', 89.5, 112, 100, 3600],
        [0.2, 0.8, 'e91p2distp2', 54, 86.5, 100, 3600],
    ],
    [
        [0.3, 0.9, 'explore_e91p3dist', 50, 130, 100, 3600],
        [0.3, 0.9, 'e91p3dist', 90, 107, 100, 3600],
        [0.3, 0.8, 'e91p3distp2', 60, 84, 100, 3600],
    ],
    [
        [0.4, 0.9, 'explore_e91p4dist', 50, 130, 100, 3600],
        [0.4, 0.9, 'e91p4dist', 90.5, 103, 100, 3600],
        [0.4, 0.8, 'e91p4distp2', 69, 83, 100, 3600],
    ],
    [
        [0.5, 0.9, 'explore_e91p5dist', 50, 130, 100, 3600],
        [0.5, 0.9, 'e91p5dist', 90.5, 101.5, 100, 3600],
    ],
    [
        [0.7, 0.9, 'explore_e91p7dist', 50, 130, 100, 3600],
        [0.7, 0.9, 'e91p7dist', 91, 98, 100, 3600],
    ],
    [
        [1.0, 0.9, 'explore_e91equaldist', 50, 130, 100, 3600],
        [1.0, 0.9, 'e91equaldist', 92.1, 93.5, 100, 3600],
    ],
    [
        [0.01, 0.6, 'explore_tp', 50, 130, 100, 3600],
        [0.01, 0.6, 'tp', 68, 112, 100, 3600],
    ],
    [
        [0.4, 0.9, 'bindist', 70, 110, 10, bin_aeff],
    ],
]
def plot_composite(fldr='1sweepbin', emax_fldr='1sweepbin_emax', num_trials=5,
                   plot_single=True,
                   get_mergerfracs=False, plot_all=False):
    # explore_pkl (emax_pkl just has explore removed, new folder), *zoom_pkls
    labelfs = 20
    m12, m3, e0 = 50, 30, 1e-3
    total_merger_fracs = []
    total_mergers_nogw = []
    all_cfgs = (
        COMPOSITE_CFGS if plot_single else
        [c for c in COMPOSITE_CFGS if c[0][0] > 0.1 and c[0][-1] == 3600]
    )
    for cfgs in COMPOSITE_CFGS:
        # load everything first
        explore_cfg = cfgs[0]
        q, e2, explore_fn, _, _, a0, a2eff = explore_cfg

        a2 = a2eff / np.sqrt(1 - e2**2)
        other_cfgs = cfgs[1: ]
        with open('%s/%s.pkl' % (fldr, explore_fn), 'rb') as f:
            explore_ret = pickle.load(f)
        emax_fn = '%s/%s.pkl' % (emax_fldr, explore_fn.replace('explore_', ''))
        with open(emax_fn, 'rb') as f:
            emax_ret = pickle.load(f)
        emaxlong_fn = '%s_long/%s.pkl' % (emax_fldr, explore_fn.replace('explore_', ''))
        with open(emaxlong_fn, 'rb') as f:
            emaxlong_ret = pickle.load(f)
        other_rets = []
        for cfg in other_cfgs:
            with open('%s/%s.pkl' % (fldr, cfg[2]), 'rb') as f:
                other_rets.append(pickle.load(f))

        # join explore_ret w/ other_rets
        I_plots = []
        tmerges = []
        # first, join in all the explores not in other_rets intervals
        starts = [explore_ret[0].min()] + [r[0].max() for r in other_rets]
        ends = [r[0].min() for r in other_rets] + [explore_ret[0].max()]
        for start, end in zip(starts, ends):
            in_interval = np.where(np.logical_and(
                explore_ret[0] < end,
                explore_ret[0] > start,
            ))[0]
            I_plots.extend(np.array(explore_ret[0])[in_interval])
            tmerges.extend(np.array(explore_ret[1])[in_interval])
        # add in other_rets
        for other_incs, other_merges in other_rets:
            I_plots.extend(other_incs)
            tmerges.extend(other_merges)

        # sort them for plotting
        I_plots = np.array(I_plots)
        tmerges = np.array(tmerges)
        sort_idx = np.argsort(I_plots)
        I_plots = I_plots[sort_idx]
        tmerges = tmerges[sort_idx]
        merged = np.where(tmerges < 9.9e9)[0]
        nmerged = np.where(tmerges > 9.9e9)[0]

        # compute stuff for emax
        I0s_emax = np.linspace(50, 130, 1000)
        I0s_emaxlong = np.linspace(50, 130, 500)
        if explore_cfg[0] == 1.0:
            I_plotsemax = I0s_emax
            I_plotsemaxlong = I0s_emaxlong
        else:
            I_plotsemax = np.repeat(I0s_emax, 5)
            I_plotsemaxlong = np.repeat(I0s_emaxlong, 20)
        emaxes = []
        emeans = []
        emaxeslong = []
        emeanslong = []
        for ret in emax_ret:
            e_vals = np.array(ret[1])
            if len(e_vals) == 0:
                emaxes.append(e0)
                emeans.append(e0)
                continue
            e_vals = e_vals[np.where(e_vals >= np.median(e_vals))[0]]
            emaxes.append(np.max(e_vals))
            # approx 1 + 73e^2/24... \approx 4.427 is constant
            jmean = np.mean((1 - e_vals**2)**(-3))**(-1/6)
            emean = np.sqrt(1 - jmean**2)
            emeans.append(emean)

        for retlong in emaxlong_ret:
            e_vals = np.array(retlong[1])
            if len(e_vals) == 0:
                emaxeslong.append(e0)
                emeanslong.append(e0)
                continue
            e_vals = e_vals[np.where(e_vals >= np.median(e_vals))[0]]
            emaxeslong.append(np.max(e_vals))
            # approx 1 + 73e^2/24... \approx 4.427 is constant
            jmean = np.mean((1 - e_vals**2)**(-3))**(-1/6)
            emean = np.sqrt(1 - jmean**2)
            emeanslong.append(emean)
        emaxes = np.array(emaxes)
        emeans = np.array(emeans)
        emaxeslong = np.array(emaxeslong)
        emeanslong = np.array(emeanslong)

        m2 = m12 / (1 + q)
        m1 = m12 - m2
        j_eff_crit = 0.01461 * (100 / a0)**(2/3)
        e_eff_crit = np.sqrt(1 - j_eff_crit**2)
        j_os = (256 * k**3 * q / (1 + q)**2 * m12**3 * a2eff**3 / (
            c**5 * a0**4 * np.sqrt(k * m12 / a0**3) * m3 * a0**3))**(1/6)
        e_os = np.sqrt(1 - j_os**2)
        _, eps_gr, eps_oct, eta = get_eps(m1, m2, m3, a0, a2, e2)
        Ilimd = get_Ilim(eta, eps_gr)
        elim = get_elim(eta, eps_gr)

        ilimd_MLL_L = np.degrees(np.arccos(np.sqrt(
            0.26 * (eps_oct / 0.1)
            - 0.536 * (eps_oct / 0.1)**2
            + 12.05 * (eps_oct / 0.1)**3
            -16.78 * (eps_oct / 0.1)**4
        )))
        ilimd_MLL_R = np.degrees(np.arccos(-np.sqrt(
            0.26 * (eps_oct / 0.1)
            - 0.536 * (eps_oct / 0.1)**2
            + 12.05 * (eps_oct / 0.1)**3
            -16.78 * (eps_oct / 0.1)**4
        )))
        merged_nogw = np.where(np.logical_or(
            emaxes > e_os,
            emeans > e_eff_crit
        ))[0]
        merged_nogwlong = np.where(np.logical_or(
            emaxeslong > e_os,
            emeanslong > e_eff_crit
        ))[0]

        # compute merger probabilities (only due to emax > eos)
        I0s = np.unique(I_plots)
        merge_probs = []
        weights = -np.gradient(np.cos(I0s))
        weights[0] /= 2
        weights[-1] /= 2
        for I in I0s:
            merge_probs.append(
                len(np.where(np.abs(I_plots[merged] - I) < 1e-6)[0]) /
                len(np.where(np.abs(I_plots - I) < 1e-6)[0]))
        weights_nogw = -np.gradient(np.cos(np.radians(I0s_emax)))
        weights_nogw[0] /= 2
        weights_nogw[-1] /= 2
        merge_probs_nogw = []

        weights_nogwlong = -np.gradient(np.cos(np.radians(I0s_emaxlong)))
        weights_nogwlong[0] /= 2
        weights_nogwlong[-1] /= 2
        merge_probs_nogwlong = []
        for I in I0s_emax:
            merge_probs_nogw.append(
                len(np.where(np.abs(I_plotsemax[merged_nogw] - I) < 1e-6)[0]) /
                len(np.where(np.abs(I_plotsemax - I) < 1e-6)[0]))
        for I in I0s_emaxlong:
            merge_probs_nogwlong.append(
                len(np.where(np.abs(I_plotsemaxlong[merged_nogwlong]
                                    - I) < 1e-6)[0]) /
                len(np.where(np.abs(I_plotsemaxlong - I) < 1e-6)[0]))
        merge_probs_nogw = np.array(merge_probs_nogw)
        merge_probs_nogwlong = np.array(merge_probs_nogwlong)
        total_merger_fracs.append(
            np.sum(np.array(merge_probs) * weights) / 2
        )
        total_mergers_nogw.append(
            np.sum(merge_probs_nogwlong * weights_nogwlong) / 2
        )
        if not plot_single:
            continue

        if q != 0.01:
            fig, (ax3, ax2, ax1) = plt.subplots(
                3, 1,
                figsize=(7, 10),
                gridspec_kw={'height_ratios': [1, 1, 0.7]},
                sharex=True)

            ax1.plot(np.degrees(I0s), merge_probs, 'k', lw=1.5,
                     label=r'$P_{\rm merger}$')
            merge_probs_nogw_smooth = (
                np.concatenate((merge_probs_nogw[ :2], merge_probs_nogw[ :-2])) +
                np.concatenate((merge_probs_nogw[ :1], merge_probs_nogw[ :-1])) +
                merge_probs_nogw +
                np.concatenate((merge_probs_nogw[1: ], merge_probs_nogw[-1: ])) +
                np.concatenate((merge_probs_nogw[2: ], merge_probs_nogw[-2: ]))
            ) / 5
            merge_probs_nogwlong_smooth = (
                np.concatenate((merge_probs_nogwlong[ :1],
                                merge_probs_nogwlong[ :-1])) +
                merge_probs_nogwlong +
                np.concatenate((merge_probs_nogwlong[1: ],
                                merge_probs_nogwlong[-1: ]))
            ) / 3
            ax1.plot(I0s_emaxlong[::3], merge_probs_nogwlong_smooth[::3], 'g',
                     lw=1.5, alpha=1,
                     label=r'$P_{\rm merger}^{\rm an}$')
            ax1.plot(I0s_emax[::5], merge_probs_nogw_smooth[::5], 'g',
                     lw=0.7, alpha=0.3)
            ax1.set_ylabel('Probability', fontsize=labelfs)
            ax1.legend(fontsize=legfs)
            ax1.set_ylim(bottom=0)

            # plot actual merger times
            ax2.semilogy(np.degrees(I_plots[merged]), tmerges[merged], 'ko',
                         ms=1)
            ax2.semilogy(np.degrees(I_plots[nmerged]), tmerges[nmerged], 'k^',
                         ms=1)
            ax2.set_ylabel(r'$T_{\rm m}$ (yr)', fontsize=labelfs)
            tlk0 = get_tlk0(m1, m2, m3, a0, a2eff)
            ax2.axhline(tlk0, c='k', ls='--')
            ax2.text(50, 1.3 * tlk0, r'$t_{\rm ZLK}$')
            if eps_oct > 0:
                t_ezlk = tlk0 * 128 * np.sqrt(10) / (15 * np.pi * np.sqrt(eps_oct))
                ax2.axhline(t_ezlk, c='k', ls='-.')
                ax2.text(50, 1.3 * t_ezlk, r'$t_{\rm ZLK, oct}$', c='k')

            # fill in to the left and right edges (bindist)
            min_Iplot = np.degrees(np.min(I_plots))
            max_Iplot = np.degrees(np.max(I_plots))
            ax1.plot([50, min_Iplot], [0, 0], 'k')
            ax1.plot([max_Iplot, 130], [0, 0], 'k')
            ax2.plot(np.linspace(50, min_Iplot, 50), np.full(50, 1e10), 'k^', ms=1)
            ax2.plot(np.linspace(max_Iplot, 130, 50), np.full(50, 1e10), 'k^', ms=1)
            ax1.set_xlabel(r'$I_0$ (Deg)', fontsize=labelfs)
            ax3.semilogy(I_plotsemaxlong, 1 - np.array(emeanslong), 'go',
                         ms=0.5, alpha=0.5)
            ax3.axhline(1 - e_eff_crit, c='g')
            ax3.text(50, 1.3 * (1 - e_eff_crit), r'$e_{\rm eff, c}$', c='g')
            ax3.set_ylabel(r'$1 - e$', fontsize=labelfs)
            ax3.axhline(1 - e_os, c='b')
            ratio = (1 - elim) / (1 - e_os)
            if ratio > 3 or ratio < 1/3:
                ax3.text(50, 1.3 * (1 - elim), r'$e_{\lim}$', c='r')
                ax3.text(50, 1.3 * (1 - e_os), r'$e_{\rm os}$', c='b')
            else:
                ax3.text(50, 1.3 * (1 - min(e_os, elim)), r'$e_{\lim}$', c='r')
                ax3.text(58, 1.3 * (1 - min(e_os, elim)), r'$e_{\rm os}$', c='b')
        else:
            fig, ax3 = plt.subplots(1, 1, figsize=(7, 5))
            ax3.set_ylabel(r'$1 - e_{\max}$', fontsize=labelfs)

        ax3.set_xlabel(r'$I_0$ (Deg)', fontsize=labelfs)
        ax3.semilogy(I_plotsemaxlong, 1 - np.array(emaxeslong), 'bo', ms=0.5,
                     alpha=0.5)
        ax3.axhline(1 - elim, c='r', ls='--')

        # overplot emax due to quadrupole
        emaxes4 = []
        for I in I0s_emax:
            emaxes4.append(get_emax(eta=eta, eps_gr=eps_gr, I=np.radians(I)))
        emaxes4.append(get_emax(eta=eta, eps_gr=eps_gr, I=np.radians(Ilimd)))
        emaxes4 = np.array(emaxes4)
        I0s_emaxes4 = np.concatenate((I0s_emax, np.array([Ilimd])))
        idxs4 = np.argsort(I0s_emaxes4)
        ax3.plot(I0s_emaxes4[idxs4], 1 - np.array(emaxes4[idxs4]), 'k--', lw=1.0)

        ticks = [50, 70, 90, 110, 130]
        ax3.set_xticks(ticks)
        ax3.set_xticklabels(labels=[r'$%d$' % d for d in ticks])
        ylim = ax3.get_ylim()
        # overplot MLL fit for reference
        # ax3.axvline(ilimd_MLL_L, c='m', lw=1.0)
        # ax3.axvline(ilimd_MLL_R, c='m', lw=1.0)
        if q != 0.01:
            # hacky way to make legend plot; I'm too lazy to figure out the
            # correct syntax
            ax3.semilogy(90, 1e-10, 'go', ms=3,
                         label=r'$e_{\rm eff}$')
            ax3.semilogy(90, 1e-10, 'bo', ms=3,
                         label=r'$e_{\max}$')
            ax3.legend(fontsize=legfs)
        ax3.set_ylim(ylim)

        plt.tight_layout()
        fig.subplots_adjust(hspace=0.02)
        composite_fn = fldr + '/' + explore_fn.replace('explore', 'composite')
        print('Saving', composite_fn, eps_oct, eta)
        plt.savefig(composite_fn, dpi=300)
        plt.close()

    if not plot_all:
        return
    total_fn = fldr + '/' + 'total_merger_fracs'
    qs = np.array([cfgs[0][0] for cfgs in all_cfgs])
    e2s = np.array([cfgs[0][1] for cfgs in all_cfgs])
    a0s = np.array([cfgs[0][5] for cfgs in all_cfgs])
    a2effs = np.array([cfgs[0][6] for cfgs in all_cfgs])
    eps_octs = np.array([get_eps(q / (1 + q) * m12,
                                 m12 / (1 + q),
                                 m3,
                                 a0,
                                 a2eff / np.sqrt(1 - e2**2),
                                 e2)[2]
                         for q, e2, a0, a2eff in zip(qs, e2s, a0s, a2effs)])
    # group by e2
    total_merger_fracs = np.array(total_merger_fracs)
    total_mergers_nogw = np.array(total_mergers_nogw)
    if get_mergerfracs:
        return total_merger_fracs
    sorted_qs = sorted(np.unique(qs))
    colors = ['r', 'b', 'k']

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(12, 6),
        sharey=True)
    for e2, color in zip(np.unique(e2s), colors):
        plot_idxs = np.where(e2s == e2)[0][::-1]
        ax1.plot(qs[plot_idxs],
                 100 * total_merger_fracs[plot_idxs],
                 c=color,
                 lw=1.0,
                 marker='o',
                 label=r'$e_{\rm out} = %.1f$' % e2)
        ax1.plot(qs[plot_idxs],
                 100 * total_mergers_nogw[plot_idxs],
                 c=color,
                 alpha=0.5,
                 lw=0.7,
                 ls=':',
                 marker='x')
    ax1.legend(fontsize=legfs)
    ax1.set_xlabel(r'$q$')
    ax1.set_ylabel(r'$f_{\rm merger}$ [\%]')

    for e2, color in zip(np.unique(e2s), colors):
        plot_idxs = np.where(e2s == e2)[0][::-1]
        ax2.plot(eps_octs[plot_idxs],
                 100 * total_merger_fracs[plot_idxs],
                 c=color,
                 lw=1.0,
                 marker='o')
        ax2.plot(eps_octs[plot_idxs],
                 100 * total_mergers_nogw[plot_idxs],
                 c=color,
                 alpha=0.5,
                 lw=0.7,
                 ls=':',
                 marker='x')
    ax2.set_xlabel(r'$\epsilon_{\rm oct}$')

    plt.tight_layout()
    print('Saving', total_fn)
    plt.savefig(total_fn, dpi=300)
    plt.close()

def plot_massratio_sample():
    total_merger_fracs = plot_composite(get_mergerfracs=True, plot_single=False)
    qs = np.array([cfgs[0][0] for cfgs in COMPOSITE_CFGS])
    e2s = np.array([cfgs[0][1] for cfgs in COMPOSITE_CFGS])

    fig = plt.figure(figsize=(6, 6))
    e2_uniqs = np.unique(e2s)
    interps = [] # array of q => merger frac, for each e2 in e2_uniqs
    for e2 in e2_uniqs:
        to_get_idxs = np.where(e2s == e2)[0][::-1]
        q_vals = qs[to_get_idxs]
        merger_fracs = total_merger_fracs[to_get_idxs]
        interps.append(interp1d(q_vals, merger_fracs))

    q_dist = np.linspace(0, 1, 1000)
    dq = np.mean(np.diff(q_dist))
    q_pdf_base = (q_dist > 0.2).astype(np.float64)
    plt.plot(q_dist, q_pdf_base / q_pdf_base.sum() / dq,
             label='Primordial')
    for e2, interp in zip(e2_uniqs, interps):
        q_pdf = np.array([interp(q) if q > 0.2 else 0 for q in q_dist])
        plt.plot(q_dist, q_pdf / q_pdf.sum() / dq,
                 label='%.1f' % e2)
    plt.legend(fontsize=legfs)

    plt.xlabel(r'$q$')
    plt.ylabel(r'Probability Density')
    plt.savefig('1massratio', dpi=300)
    plt.close()

# replot for the 1p3dist case, using long emax sweep
def plot_long_composite(fldr='1sweepbin', emax_fldr='1sweepbin_emax',
                        num_trials=5, num_i=1000, plot_single=True,
                        get_mergerfracs=False, plot_all=False):
    # explore_pkl (emax_pkl just has explore removed, new folder), *zoom_pkls
    m12, m3, e0 = 50, 30, 1e-3
    cfgs = [
        [0.3, 0.6, 'explore_1p3dist', 50, 130, 100, 3600],
        [0.3, 0.6, '1p3dist', 90.5, 100, 100, 3600],
        [0.3, 0.6, '1p3distp2', 73, 86, 100, 3600],
    ]
    # load everything first
    explore_cfg = cfgs[0]
    q, e2, explore_fn, _, _, a0, a2eff = explore_cfg

    a2 = a2eff / np.sqrt(1 - e2**2)
    other_cfgs = cfgs[1: ]
    with open('%s/%s.pkl' % (fldr, explore_fn), 'rb') as f:
        explore_ret = pickle.load(f)
    emax_fn = '%s/%s_long.pkl' % (emax_fldr, explore_fn.replace('explore_', ''))
    with open(emax_fn, 'rb') as f:
        emax_ret = pickle.load(f)
    emax_shortfn = '%s/%s.pkl' % (emax_fldr, explore_fn.replace('explore_', ''))
    with open(emax_shortfn, 'rb') as f:
        emax_shortret = pickle.load(f)
    other_rets = []
    for cfg in other_cfgs:
        with open('%s/%s.pkl' % (fldr, cfg[2]), 'rb') as f:
            other_rets.append(pickle.load(f))

    # join explore_ret w/ other_rets
    I_plots = []
    tmerges = []
    # first, join in all the explores not in other_rets intervals
    starts = [explore_ret[0].min()] + [r[0].max() for r in other_rets]
    ends = [r[0].min() for r in other_rets] + [explore_ret[0].max()]
    for start, end in zip(starts, ends):
        in_interval = np.where(np.logical_and(
            explore_ret[0] < end,
            explore_ret[0] > start,
        ))[0]
        I_plots.extend(np.array(explore_ret[0])[in_interval])
        tmerges.extend(np.array(explore_ret[1])[in_interval])
    # add in other_rets
    for other_incs, other_merges in other_rets:
        I_plots.extend(other_incs)
        tmerges.extend(other_merges)

    # sort them for plotting
    I_plots = np.array(I_plots)
    tmerges = np.array(tmerges)
    sort_idx = np.argsort(I_plots)
    I_plots = I_plots[sort_idx]
    tmerges = tmerges[sort_idx]
    merged = np.where(tmerges < 9.9e9)[0]
    nmerged = np.where(tmerges > 9.9e9)[0]

    # compute stuff for emax
    I0s_emax = np.linspace(50, 130, num_i)
    if explore_cfg[0] == 1.0:
        I_plotsemax = I0s_emax
    else:
        I_plotsemax = np.repeat(I0s_emax, 5)
    emaxes = []
    emeans = []
    emaxes_short = []
    emeans_short = []
    for ret_short, ret in zip(emax_shortret, emax_ret):
        e_vals = np.array(ret[1])
        if len(e_vals) == 0:
            emaxes.append(e0)
            emeans.append(e0)
            continue
        e_vals = e_vals[np.where(e_vals >= np.median(e_vals))[0]]
        emaxes.append(np.max(e_vals))
        # approx 1 + 73e^2/24... \approx 4.427 is constant
        jmean = np.mean((1 - e_vals**2)**(-3))**(-1/6)
        emean = np.sqrt(1 - jmean**2)
        emeans.append(emean)

        e_shortvals = np.array(ret_short[1])
        if len(e_shortvals) == 0:
            emaxes_short.append(e0)
            emeans_short.append(e0)
            continue
        e_shortvals = e_shortvals[
            np.where(e_shortvals >= np.median(e_shortvals))[0]]
        emaxes_short.append(np.max(e_shortvals))
        # approx 1 + 73e^2/24... \approx 4.427 is constant
        jmeanshort = np.mean((1 - e_shortvals**2)**(-3))**(-1/6)
        emeanshort = np.sqrt(1 - jmeanshort**2)
        emeans_short.append(emeanshort)
    emaxes = np.array(emaxes)
    emeans = np.array(emeans)
    emaxes_short = np.array(emaxes_short)
    emeans_short = np.array(emeans_short)

    m2 = m12 / (1 + q)
    m1 = m12 - m2
    j_eff_crit = 0.01461 * (100 / a0)**(2/3)
    e_eff_crit = np.sqrt(1 - j_eff_crit**2)
    j_os = (256 * k**3 * q / (1 + q)**2 * m12**3 * a2eff**3 / (
        c**5 * a0**4 * np.sqrt(k * m12 / a0**3) * m3 * a0**3))**(1/6)
    e_os = np.sqrt(1 - j_os**2)
    _, eps_gr, eps_oct, eta = get_eps(m1, m2, m3, a0, a2, e2)
    Ilimd = get_Ilim(eta, eps_gr)
    elim = get_elim(eta, eps_gr)

    ilimd_MLL_L = np.degrees(np.arccos(np.sqrt(
        0.26 * (eps_oct / 0.1)
        - 0.536 * (eps_oct / 0.1)**2
        + 12.05 * (eps_oct / 0.1)**3
        -16.78 * (eps_oct / 0.1)**4
    )))
    ilimd_MLL_R = np.degrees(np.arccos(-np.sqrt(
        0.26 * (eps_oct / 0.1)
        - 0.536 * (eps_oct / 0.1)**2
        + 12.05 * (eps_oct / 0.1)**3
        -16.78 * (eps_oct / 0.1)**4
    )))
    merged_nogw = np.where(np.logical_or(
        emaxes > e_os,
        emeans > e_eff_crit
    ))[0]
    merged_nogw_short = np.where(np.logical_or(
        emaxes_short > e_os,
        emeans_short > e_eff_crit
    ))[0]

    # compute merger probabilities (only due to emax > eos)
    I0s = np.unique(I_plots)
    merge_probs = []
    weights = -np.gradient(np.cos(I0s))
    weights[0] /= 2
    weights[-1] /= 2
    for I in I0s:
        merge_probs.append(
            len(np.where(np.abs(I_plots[merged] - I) < 1e-6)[0]) /
            len(np.where(np.abs(I_plots - I) < 1e-6)[0]))
    weights_nogw = -np.gradient(np.cos(np.radians(I0s_emax)))
    weights_nogw[0] /= 2
    weights_nogw[-1] /= 2
    merge_probs_nogw = []
    merge_probs_nogw_short = []
    for I in I0s_emax:
        merge_probs_nogw.append(
            len(np.where(np.abs(I_plotsemax[merged_nogw] - I) < 1e-6)[0]) /
            len(np.where(np.abs(I_plotsemax - I) < 1e-6)[0]))
        merge_probs_nogw_short.append(
            len(np.where(np.abs(I_plotsemax[merged_nogw_short] - I) < 1e-6)[0]) /
            len(np.where(np.abs(I_plotsemax - I) < 1e-6)[0]))
    merge_probs_nogw = np.array(merge_probs_nogw)
    merge_probs_nogw_short = np.array(merge_probs_nogw_short)

    fig, (ax3, ax1) = plt.subplots(
        2, 1,
        figsize=(7, 6),
        gridspec_kw={'height_ratios': [1, 0.3]},
        sharex=True)

    ax1.plot(np.degrees(I0s), merge_probs, 'k', label='Simulation')
    merge_probs_nogw_smooth = (
        np.concatenate((merge_probs_nogw[ :1], merge_probs_nogw[ :-1])) +
        merge_probs_nogw +
        np.concatenate((merge_probs_nogw[1: ], merge_probs_nogw[-1: ]))
    ) / 3
    merge_probs_nogw_short_smooth = (
        np.concatenate((merge_probs_nogw_short[ :1],
                        merge_probs_nogw_short[ :-1])) +
        merge_probs_nogw_short +
        np.concatenate((merge_probs_nogw_short[1: ],
                        merge_probs_nogw_short[-1: ]))
    ) / 3
    ax1.plot(I0s_emax[::3], merge_probs_nogw_smooth[::3], 'g',
             label=r'SA ($t_{\rm f} = 2000t_{\rm ZLK}$)')
    ax1.plot(I0s_emax[::3], merge_probs_nogw_short_smooth[::3], 'r',
             label=r'SA ($t_{\rm f} = 500t_{\rm ZLK}$)')
    ax1.set_ylabel(r'$P_{\rm merger}$')
    ax1.legend(fontsize=legfs)

    # fill in to the left and right edges (bindist)
    min_Iplot = np.degrees(np.min(I_plots))
    max_Iplot = np.degrees(np.max(I_plots))
    ax1.plot([50, min_Iplot], [0, 0], 'k')
    ax1.plot([max_Iplot, 130], [0, 0], 'k')
    ax1.set_xlabel(r'$I_0$ (Deg)')
    ax3.semilogy(I_plotsemax, 1 - np.array(emeans), 'go', ms=0.5,
                 alpha=0.5)
    ax3.axhline(1 - e_eff_crit, c='g')
    ax3.text(50, 1.3 * (1 - e_eff_crit), r'$e_{\rm eff, c}$', c='g')
    ax3.axhline(1 - e_os, c='b')
    ax3.text(50, 1.3 * (1 - e_os), r'$e_{\rm os}$', c='b')

    ax3.set_xlabel(r'$I_0$ (Deg)')
    ax3.semilogy(I_plotsemax, 1 - np.array(emaxes), 'bo', ms=0.5, alpha=0.5)
    ax3.axhline(1 - elim, c='r', ls='--')
    ax3.text(50, 1.3 * (1 - elim), r'$e_{\lim}$', c='r')

    # overplot MLL fit for reference
    ax3.axvline(ilimd_MLL_L, c='m', lw=1.0)
    ax3.axvline(ilimd_MLL_R, c='m', lw=1.0)

    # overplot emax due to quadrupole
    emaxes4 = []
    for I in I0s_emax:
        emaxes4.append(get_emax(eta=eta, eps_gr=eps_gr, I=np.radians(I)))
    ax3.plot(I0s_emax, 1 - np.array(emaxes4), 'k--', lw=1.0)

    ax3.set_ylabel(r'$1 - e$')
    ticks = [50, 70, 90, 110, 130]
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(labels=[r'$%d$' % d for d in ticks])

    # hacky way to make legend plot; I'm too lazy to figure out the
    # correct syntax
    ylim = ax3.get_ylim()
    ax3.semilogy(90, 1e-10, 'go', ms=3,
                 label=r'$e_{\rm eff}$')
    ax3.semilogy(90, 1e-10, 'bo', ms=3,
                 label=r'$e_{\max}$')
    ax3.set_ylim(ylim)
    ax3.legend(fontsize=legfs)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    composite_fn = fldr + '/' + explore_fn.replace('explore', 'composite_long')
    print('Saving', composite_fn, eps_oct, eta)
    plt.savefig(composite_fn, dpi=300)
    plt.close()

def plot_completeness(fldr='1sweepbin', emax_fldr='1sweepbin_emax_long',
                      num_trials=20, num_i=500, plot_single=True,
                      get_mergerfracs=False, plot_all=False, num_ts=20):
    # explore_pkl (emax_pkl just has explore removed, new folder), *zoom_pkls
    m12, m3, e0 = 50, 30, 1e-3
    num_run = 0
    completenesses = np.zeros(num_ts)
    times = np.linspace(0, np.sqrt(2000), num_ts)**2
    for cfgs in COMPOSITE_CFGS:
        # load everything first
        explore_cfg = cfgs[0]
        q, e2, explore_fn, _, _, a0, a2eff = explore_cfg
        tlk0 = get_tlk0(m12 / 2, m12 / 2, m3, a0, a2eff)

        a2 = a2eff / np.sqrt(1 - e2**2)
        other_cfgs = cfgs[1: ]
        if q > 0.9:
            continue
        try:
            with open('%s/%s.pkl' % (fldr, explore_fn), 'rb') as f:
                explore_ret = pickle.load(f)
            emax_fn = '%s/%s.pkl' % (emax_fldr, explore_fn.replace('explore_', ''))
            with open(emax_fn, 'rb') as f:
                emax_ret = pickle.load(f)
            other_rets = []
            for cfg in other_cfgs:
                with open('%s/%s.pkl' % (fldr, cfg[2]), 'rb') as f:
                    other_rets.append(pickle.load(f))
        except:
            print(explore_fn, 'not available, continuing')
            continue

        # join explore_ret w/ other_rets
        I_plots = []
        tmerges = []
        # first, join in all the explores not in other_rets intervals
        starts = [explore_ret[0].min()] + [r[0].max() for r in other_rets]
        ends = [r[0].min() for r in other_rets] + [explore_ret[0].max()]
        for start, end in zip(starts, ends):
            in_interval = np.where(np.logical_and(
                explore_ret[0] < end,
                explore_ret[0] > start,
            ))[0]
            I_plots.extend(np.array(explore_ret[0])[in_interval])
            tmerges.extend(np.array(explore_ret[1])[in_interval])
        # add in other_rets
        for other_incs, other_merges in other_rets:
            I_plots.extend(other_incs)
            tmerges.extend(other_merges)

        # sort them for plotting
        I_plots = np.array(I_plots)
        tmerges = np.array(tmerges)
        sort_idx = np.argsort(I_plots)
        I_plots = I_plots[sort_idx]
        tmerges = tmerges[sort_idx]
        merged = np.where(tmerges < 9.9e9)[0]
        nmerged = np.where(tmerges > 9.9e9)[0]

        I0s = np.unique(I_plots)
        merge_probs = []
        weights = -np.gradient(np.cos(I0s))
        weights[0] /= 2
        weights[-1] /= 2
        for I in I0s:
            merge_probs.append(
                len(np.where(np.abs(I_plots[merged] - I) < 1e-6)[0]) /
                len(np.where(np.abs(I_plots - I) < 1e-6)[0]))
        merge_prob_gw = np.sum(np.array(merge_probs) * weights) / 2

        m2 = m12 / (1 + q)
        m1 = m12 - m2
        j_eff_crit = 0.01461 * (100 / a0)**(2/3)
        e_eff_crit = np.sqrt(1 - j_eff_crit**2)
        j_os = (256 * k**3 * q / (1 + q)**2 * m12**3 * a2eff**3 / (
            c**5 * a0**4 * np.sqrt(k * m12 / a0**3) * m3 * a0**3))**(1/6)
        e_os = np.sqrt(1 - j_os**2)
        _, eps_gr, eps_oct, eta = get_eps(m1, m2, m3, a0, a2, e2)
        Ilimd = get_Ilim(eta, eps_gr)
        elim = get_elim(eta, eps_gr)

        # compute stuff for emax
        I0s_emax = np.linspace(50, 130, num_i)
        if explore_cfg[0] == 1.0:
            I_plotsemax = I0s_emax
        else:
            I_plotsemax = np.repeat(I0s_emax, 5)
        will_merges = np.zeros((len(emax_ret), num_ts))
        for idx0, ret in enumerate(emax_ret):
            ts = ret[0]
            e_vals = np.array(ret[1])
            for idx1, t in enumerate(times):
                t_idx = np.searchsorted(ts, t * tlk0)
                if t_idx == 0:
                    continue

                curr_es = e_vals[ :t_idx]
                curr_es = curr_es[np.where(curr_es >= np.median(curr_es))[0]]
                curr_emax = np.max(curr_es)
                # approx 1 + 73e^2/24... \approx 4.427 is constant
                jmean = np.mean((1 - curr_es**2)**(-3))**(-1/6)
                if jmean <= j_eff_crit or curr_emax > e_os:
                    will_merges[idx0, idx1] = 1
        merge_fracs = np.sum(will_merges, axis=0) / len(emax_ret)
        merge_fracs *= (cosd(50) - cosd(130)) / 2
        print(merge_fracs[-1], merge_prob_gw)
        merge_fracs /= merge_prob_gw

        plt.plot(times, merge_fracs, 'k', alpha=0.3, lw=0.5)
        completenesses += merge_fracs
        num_run += 1
        print('Done collecting for', explore_fn)
    plt.plot(times, completenesses / num_run, 'k', lw=2.5)
    plt.xlabel(r'Integration Time ($t_{\rm ZLK}$)')
    plt.ylabel(r'$f_{\rm merger}^{\rm an} / f_{\rm merger}$')
    plt.ylim(bottom=0)

    print('Saving 1completeness, Total Int time:', 2000 * tlk0 / 1e9)
    plt.tight_layout()
    plt.savefig('%s/completeness' % fldr, dpi=300)
    plt.close()

def plot_emaxgrid(folder='1sweepbin_emax', nthreads=1, q=0.3, e2=0.6,
                  a2eff=3600, a0=100, base_fn='1p3grid',
                  num_i=200, num_w=40):
    mkdirp(folder)
    m12, m3, e0 = 50, 30, 1e-3
    a2 = a2eff / np.sqrt(1 - e2**2)

    I0s = np.linspace(50, 130, num_i)
    w1s = np.linspace(0, np.pi, num_w)
    fn = '%s/%s' % (folder, base_fn)
    pkl_fn = fn + '.pkl'

    args = [
        (0, q, I0, None, dict(a0=a0, a2=a2, e2=e2, w1=w1, w2=0, W=0, tf_mult=100))
        for I0 in I0s
        for w1 in w1s
    ]
    I_grid = [
        I0
        for I0 in I0s
        for _ in w1s
    ]
    w1_grid = [
        w1
        for _ in I0s
        for w1 in w1s
    ]

    # emin, emax = np.log10(1 - e)
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        p = Pool(nthreads)
        rets = p.starmap(get_emax_series, args)
        emaxes = [
            np.min(np.log10(1 - ret[1])) if len(ret[1]) > 0 else 0
            for ret in rets
        ]
        with open(pkl_fn, 'wb') as f:
            pickle.dump(emaxes, f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            emaxes = pickle.load(f)
    I_grid = np.reshape(I_grid, (num_i, num_w))
    w1_grid = np.degrees(np.reshape(w1_grid, (num_i, num_w)))
    emax_grid = np.reshape(emaxes, (num_i, num_w))
    plt.pcolormesh(I_grid, w1_grid, emax_grid, shading='nearest')
    plt.xlabel(r'$I_0$ (Deg)')
    plt.ylabel(r'$\omega_1$ (Deg)')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fn, dpi=300)
    plt.close()

def get_e_mergers(tmerge, yf, M1, M2):
    ''' given yf of vec simulation, calculate eccentricity @ 0.1Hz, 10Hz '''
    Lx, Ly, Lz, ex, ey, ez = yf[ :6]
    L = np.sqrt(Lx**2 + Ly**2 + Lz**2)
    e0 = np.sqrt(ex**2 + ey**2 + ez**2)
    mu = M1 * M2 / (M1 + M2)
    m12 = M1 + M2
    a0 = L**2 / ((mu**2) * k * (M1 + M2) * (1 - e0**2))
    # GW frequency = 2x orbital freq = Omega / pi = sqrt(k * m12 / a**3) / pi
    f_LIGO = 3.154e8 # 10Hz to 1/yr
    f_LISA = 3.154e6 # 0.1Hz to 1/yr
    a_LIGO = (k * m12 / (np.pi * f_LIGO)**2)**(1/3)
    a_LISA = (k * m12 / (np.pi * f_LISA)**2)**(1/3)
    if tmerge > 9.9e9:
        return -1, -1

    def dydt(t, y):
        a, e = y
        return [(
            -64 * k**3 * mu * m12**2
            * (1 + 73 / 24 * e**2 + 37 * e**4 / 96)
            / (5 * c**5 * a**3 * (1 - e**2)**(7/2))
        ), (
            -304 * k**3 * mu * m12**2 * e
            * (1 + 121 * e**2 / 304)
            / (15 * c**5 * a**4 * (1 - e**2)**(5/2))
        )]
    def term_event(t, y):
        return y[0] - a_LIGO
    term_event.terminal = True
    try:
        ret = solve_ivp(dydt, (0, np.inf), [a0, e0], events=[term_event],
                        dense_output=True, method='Radau', atol=1e-9, rtol=1e-9)
    except ValueError:
        print(a0, e0)
        return -1, -1
    e_LIGO = ret.y[1, -1]
    try:
        t_LISA = opt.brenth(lambda t: ret.sol(t)[0] - a_LISA, 0, ret.t[-1])
    except ValueError:
        print(ret.sol(0), ret.sol(ret.t[-1]))
        return -1, -1
    e_LISA = ret.sol(t_LISA)[1]
    return e_LIGO, e_LISA

def pop_synth(a2eff=3600, ntrials=19000, base_fn='a2eff3600', nthreads=32,
              n_qs=19, q_min=0.2, to_plot=True):
    '''
    Observation: fix m12 = 50, m3 = 30, a = 100, pick a few abarouteff (5-10),
    sample e_out in [0, 0.9], q in [0.2, 1], scan over cos(I),
    merger fraction(abarouteff), histograms of observed q
    calculate using uniform grid in cos I
    store e_in @ 0.5AU (can postprocess to get merger e)
    '''

    folder = '1popsynth'
    t_hubb_gyr = 10
    m12 = 50
    m3 = 30
    a0 = 100
    e0 = 1e-3

    mkdirp(folder)

    fn = '%s/%s' % (folder, base_fn)
    pkl_fn = fn + '.pkl'
    q_vals = np.linspace(q_min, 1, n_qs)
    if not os.path.exists(pkl_fn):
        print('Skipping %s' % pkl_fn)
        return
        print('Running %s' % pkl_fn)
        qs = np.repeat(q_vals, ntrials // n_qs)
        # only draw from ~50-130
        Imin, Imax = np.radians([50, 130])
        e2s = np.random.uniform(0, 0.9, ntrials)
        I0s = np.arccos(
            np.random.uniform(np.cos(Imax), np.cos(Imin), ntrials)
        )
        m2 = m12 / (1 + qs)
        m1 = m12 - m2
        a2s = a2eff / np.sqrt(1 - e2s**2)

        p = Pool(nthreads)
        args2 = [
            (idx, q, np.degrees(I0), None, dict(a0=a0, a2=a2, e2=e2))
            for idx, (q, a2, e2, I0) in enumerate(zip(qs, a2s, e2s, I0s))
        ]
        start = time.time()
        emax_rets = p.starmap(get_emax_series, args2)
        emax_time_elapsed = time.time() - start

        args = [
            (idx, q, t_hubb_gyr * 1e9, a0, a2, e0, e2, I0, True)
            for idx, (q, a2, e2, I0) in enumerate(zip(qs, a2s, e2s, I0s))
        ]
        start = time.time()
        tmerge_rets = p.starmap(sweeper_bin, args)
        tmerge_time_elapsed = time.time() - start

        with open(pkl_fn, 'wb') as f:
            ret = (args, tmerge_rets, emax_rets, tmerge_time_elapsed,
                   emax_time_elapsed)
            pickle.dump(ret, f)
    else:
        print('Loading %s' % pkl_fn)
        with open(pkl_fn, 'rb') as f:
            ret = pickle.load(f)
    args, tmerge_rets, emax_rets, tmerge_time_elapsed, emax_time_elapsed = ret

    pkl_eccs = pkl_fn.replace('.pkl', '_eccs.pkl')
    if not os.path.exists(pkl_eccs):
        print('Running %s' % pkl_eccs)
        args_ecc = [
            (tmerge_ret[0], tmerge_ret[1],
             m12 / (1 + arg[1]), arg[1] * m12 / (1 + arg[1]))
            for arg, tmerge_ret in zip(args, tmerge_rets)
        ]
        # rets = [get_e_mergers(*_arg) for _arg in args_ecc]
        p = Pool(8)
        rets = p.starmap(get_e_mergers, args_ecc)
        e_LIGOs, e_LISAs = np.array(rets).T
        with open(pkl_eccs, 'wb') as f:
            pickle.dump((e_LIGOs, e_LISAs), f)
    else:
        with open(pkl_eccs, 'rb') as f:
            print('Loading %s' % pkl_eccs)
            e_LIGOs, e_LISAs = pickle.load(f)
    print('Emax took', emax_time_elapsed)
    print('TMerge took', tmerge_time_elapsed)

    # PDF of q among merged systems
    j_eff_crit = 0.01461 * (100 / a0)**(2/3)
    e_eff_crit = np.sqrt(1 - j_eff_crit**2)

    # convention: +e2 if merged, -e2 if not
    q_counts = defaultdict(list)
    q_counts_nogw = defaultdict(list)
    q_counts_emaxonly = defaultdict(list)
    merged_times = defaultdict(list)
    for arg, tmerge_ret, emax_ret in zip(args, tmerge_rets, emax_rets):
        _, q, _, _, _, _, e2, I0d, _ = arg
        tmerge, yf = tmerge_ret

        merged_times[q].append(tmerge)
        if tmerge < 9.9e9:
            q_counts[q].append(e2)
        else:
            q_counts[q].append(-e2)
        I0 = np.radians(I0d)
        e_vals = np.array(emax_ret[1])
        where_idx = np.where(e_vals >= 0.3)[0]
        e_vals = e_vals[where_idx]

        if len(e_vals) == 0:
            continue
        j_os = (256 * k**3 * q / (1 + q)**2 * m12**3 * a2eff**3 / (
            c**5 * a0**4 * np.sqrt(k * m12 / a0**3) * m3 * a0**3))**(1/6)
        e_os = np.sqrt(1 - j_os**2)

        # approx 1 + 73e^2/24... \approx 4.427 is constant
        jmean = np.mean((1 - e_vals**2)**(-3))**(-1/6)
        emean = np.sqrt(1 - jmean**2)
        if np.max(e_vals) > e_os or emean > e_eff_crit:
            q_counts_nogw[q].append(e2)
        else:
            q_counts_nogw[q].append(-e2)
        if np.max(e_vals) > e_os:
            q_counts_emaxonly[q].append(e2)
        else:
            q_counts_emaxonly[q].append(-e2)

    f_cos = 100 * (np.cos(np.radians(50)) - np.cos(np.radians(130))) / 2
    merged_fracs = []
    merged_fracs_thermal = []
    uncerts = []
    merged_fracs_nogw = []
    merged_fracs_emaxonly = []
    tmerges_gyr = [] # median merger times
    for arr, arr_nogw, arr_emaxonly, tmerge_vals in zip(
            q_counts.values(), q_counts_nogw.values(),
            q_counts_emaxonly.values(), merged_times.values()):
        # ceil = merged vs not merged weighting
        merged_fracs.append(
            np.sum(np.ceil(arr)) / len(arr))
        uncerts.append(
            np.sqrt(np.sum(np.ceil(arr))) / len(arr))
        # abs = thermal total, max(e2, 0) = e2 or zero, thermal weighting
        merged_fracs_thermal.append(
            np.sum(np.maximum(arr, np.zeros_like(arr))) /
            np.sum(np.abs(arr)))
        merged_fracs_nogw.append(
            np.sum(np.ceil(arr_nogw)) / len(arr_nogw))
        merged_fracs_emaxonly.append(
            np.sum(np.ceil(arr_emaxonly)) / len(arr_emaxonly))
        tmerges_gyr.append([
            t for e2, t in zip(arr, tmerge_vals) if e2 > 0])
    merged_fracs = np.array(merged_fracs)
    merged_fracs_thermal = np.array(merged_fracs_thermal)
    uncerts = np.array(uncerts)
    merged_fracs_nogw = np.array(merged_fracs_nogw)
    merged_fracs_emaxonly = np.array(merged_fracs_emaxonly)

    tot_perc = np.mean(merged_fracs) * f_cos # all bins are equal

    if to_plot and plt is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1,
            figsize=(6, 9),
            gridspec_kw={'height_ratios': [0.5, 0.25, 0.25]},
            sharex=True)
        # ax1.errorbar(q_counts.keys(), merged_fracs * f_cos,
        #              yerr=np.sqrt(uncerts * f_cos),
        #              fmt='ko', lw=1.0, ms=3.0, label='Uniform')
        ax1.plot(q_counts.keys(), merged_fracs * f_cos,
                 'ko', ls='-', lw=1.0, ms=3.0, alpha=0.7, label='Uniform')
        ax1.plot(q_counts.keys(), merged_fracs_thermal * f_cos, 'ro',
                 ls='-', label='Thermal', alpha=0.7, ms=3.5)
        # ax1.plot(q_counts_nogw.keys(), merged_fracs_nogw * f_cos, 'go',
        #          label='No-GW', ms=3.5)

        nogw_fn = pkl_fn.replace(str(a2eff), "_nogw%d" % a2eff)
        if os.path.exists(nogw_fn):
            _, qplots, frac_plots = pop_synth_nogw(
                a2eff=a2eff, base_fn='a2eff_nogw%d' % a2eff, to_plot=False,
                n_qs=31, q_min=0.2, n_e2s=17, n_I0s=41, stride=1, reps=1
            )
            ax1.plot(qplots, frac_plots, 'bx', lw=1.0, ls=':',
                     alpha=0.5, label='Uniform (Analytic)')
            p = np.polyfit(np.log(qplots), np.log(frac_plots), 1)
            print(p[0])
            ax1.plot(qplots, np.exp(p[1]) * qplots**(p[0]), 'g', lw=4, alpha=0.3)
        ax1.legend(fontsize=legfs, loc='upper right')
        ax1.set_ylabel(r'$\eta_{\rm merger}$ [\%]')
        ax1.set_ylim(bottom=-0.5)

        for q, tm_lst in zip(q_counts.keys(), tmerges_gyr):
            ax2.semilogy(np.full_like(tm_lst, q), tm_lst, 'ko', ms=0.5, alpha=0.3)
            ax2.plot(q, np.median(tm_lst), 'ko', ms=3.0)
        ax2.set_ylabel(r'$T_{\rm m}$ (yr)')

        q_arr = [arg[1] for e_LIGO, arg in zip(e_LIGOs, args)
                 if e_LIGO > 0]
        e_LIGOsm = e_LIGOs[np.where(e_LIGOs > 0)[0]]
        e_LISAsm = e_LISAs[np.where(e_LIGOs > 0)[0]]
        ax3.plot(q_arr, e_LIGOsm, 'ko', ms=0.5, alpha=0.3)
        ax3.plot(q_arr, e_LISAsm, 'go', ms=0.5, alpha=0.3)
        q = q_vals[0]
        ax3.plot(q, np.median(e_LIGOsm[np.where(q_arr == q)[0]]),
                 'ko', label='10 Hz', ms=3.0)
        ax3.plot(q, np.median(e_LISAsm[np.where(q_arr == q)[0]]),
                 'go', label='0.1 Hz', ms=3.0)
        for q in q_vals[1: ]:
            ax3.plot(q, np.median(e_LIGOsm[np.where(q_arr == q)[0]]),
                     'ko', ms=3.0)
            ax3.plot(q, np.median(e_LISAsm[np.where(q_arr == q)[0]]),
                     'go', ms=3.0)
        ax3.legend(ncol=2, loc='lower left', fontsize=legfs)
        ax3.set_ylabel(r'$e_{\rm m}$')
        ax3.set_xlabel(r'$q$')
        ax3.set_yscale('log')
        yticks = [1e-1, 1e-3, 1e-5]
        ax3.set_yticks(yticks)
        ax3.set_yticklabels([r'$10^{%d}$' % d for d in np.log10(yticks)])
        # plt.suptitle(
        #     r'$a_{\rm out, eff} = %d\;\mathrm{AU}, N_{\rm trials} = %d$' %
        #     (a2eff, ntrials))
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.03)

        plt.savefig('%s/%s' % (folder, base_fn), dpi=300)
        plt.close()
    return tot_perc

def pop_synth_nogw(a2eff=3600, base_fn='a2eff_nogw3600',
                   nthreads=32, n_qs=41, q_min=0.2, q_max=1, to_plot=True,
                   q_spread=np.linspace, n_e2s=31, n_I0s=57, stride=10, reps=3):
    '''
    Observation: fix m12 = 50, m3 = 30, a = 100, pick a few abarouteff (5-10),
    sample e_out in [0, 0.9], q in [0.2, 1], scan over cos(I),
    merger fraction(abarouteff), histograms of observed q
    calculate using uniform grid in cos I
    store e_in @ 0.5AU (can postprocess to get merger e)
    '''

    folder = '1popsynth'
    t_hubb_gyr = 10
    m12 = 50
    m3 = 30
    a0 = 100
    e0 = 1e-3

    mkdirp(folder)

    fn = '%s/%s' % (folder, base_fn)
    pkl_fn = fn + '.pkl'
    q_vals = q_spread(q_min, q_max, n_qs)
    ntrials_perq = n_e2s * n_I0s * reps // stride
    Imin, Imax = np.radians([50, 130])
    _e2s = np.linspace(1e-3, 0.9, n_e2s)
    _I0s = np.arccos(np.linspace(np.cos(Imax), np.cos(Imin), n_I0s))
    e2s_perq = np.repeat(
        np.outer(_e2s, np.ones_like(_I0s)).flatten(), reps)[::stride]
    I0s_perq = np.repeat(
        np.outer(np.ones_like(_e2s), _I0s).flatten(), reps)[::stride]
    e2s = np.tile(e2s_perq, n_qs)
    I0s = np.tile(I0s_perq, n_qs)

    qs = np.repeat(q_vals, ntrials_perq)
    m2 = m12 / (1 + qs)
    m1 = m12 - m2
    a2s = a2eff / np.sqrt(1 - e2s**2)

    args2 = [
        (idx, q, np.degrees(I0), None, dict(a0=a0, a2=a2, e2=e2, tf_mult=2000))
        for idx, (q, a2, e2, I0) in enumerate(zip(qs, a2s, e2s, I0s))
    ]

    if not os.path.exists(pkl_fn):
        print('Skipping %s' % pkl_fn)
        return
        print('Running %s' % pkl_fn)
        # only draw from ~50-130
        p = Pool(nthreads)
        start = time.time()
        emax_rets = p.starmap(get_emax_series, args2)

        with open(pkl_fn, 'wb') as f:
            pickle.dump(emax_rets, f)
    else:
        print('Loading %s' % pkl_fn)
        with open(pkl_fn, 'rb') as f:
            emax_rets = pickle.load(f)

    # PDF of q among merged systems
    j_eff_crit = 0.01461 * (100 / a0)**(2/3)
    e_eff_crit = np.sqrt(1 - j_eff_crit**2)

    q_counts_nogw = defaultdict(list)
    q_counts_emaxonly = defaultdict(list)
    for arg, emax_ret in zip(args2, emax_rets):
        _, q, I0d, _, argdict = arg
        e2 = argdict['e2']

        e_vals = np.array(emax_ret[1])
        where_idx = np.where(e_vals >= 0.3)[0]
        e_vals = e_vals[where_idx]

        if len(e_vals) == 0:
            continue
        j_os = 3 * (256 * k**3 * q / (1 + q)**2 * m12**3 * a2eff**3 / (
            c**5 * a0**4 * np.sqrt(k * m12 / a0**3) * m3 * a0**3))**(1/6)
        e_os = np.sqrt(1 - j_os**2)

        # approx 1 + 73e^2/24... \approx 4.427 is constant
        jmean = np.mean((1 - e_vals**2)**(-3))**(-1/6)
        emean = np.sqrt(1 - jmean**2)
        if np.max(e_vals) > e_os or emean > e_eff_crit:
            q_counts_nogw[q].append(1)
        else:
            q_counts_nogw[q].append(0)
        if np.max(e_vals) > e_os:
            q_counts_emaxonly[q].append(1)
        else:
            q_counts_emaxonly[q].append(0)

    f_cos = 100 * (np.cos(np.radians(50)) - np.cos(np.radians(130))) / 2
    merged_fracs_nogw = np.array(
        [np.sum(arr) / len(arr) for arr in q_counts_nogw.values()])
    # merged_fracs_emaxonly = np.array(
    #     [np.sum(arr) / len(arr) for arr in q_counts_emaxonly.values()])
    tot_perc = np.mean(merged_fracs_nogw) * f_cos # all bins are equal
    if to_plot and plt is not None:
        plt.figure(figsize=(6, 6))
        plt.ylabel(r'$f_{\rm merger}$ [\%]')
        plt.xlabel(r'$q$')
        if q_spread == np.geomspace:
            # steal the full data
            ret = pop_synth_nogw(a2eff=a2eff, base_fn='a2eff_nogw%d' % a2eff,
                                 to_plot=False, n_qs=31, q_min=0.2,
                                 n_e2s=17, n_I0s=41, stride=1, reps=1)
            plt.xscale('log')

            x = np.concatenate((list(ret[1]), list(q_counts_nogw.keys())))
            y = np.concatenate((list(ret[2]), list(merged_fracs_nogw * f_cos)))
            idx_s = np.argsort(x)
            plt.plot(x[idx_s], y[idx_s], 'bx', ls=':', ms=3.5)
        else:
            plt.plot(q_counts_nogw.keys(), merged_fracs_nogw * f_cos, 'bx',
                     ls=':', ms=3.5)
        # plt.title(r'$a_{\rm out, eff} = %d\;\mathrm{AU}$' % a2eff)
        plt.tight_layout()

        plt.savefig('%s/%s' % (folder, base_fn), dpi=300)
        plt.close()
    return tot_perc, np.array(list(q_counts_nogw.keys())), merged_fracs_nogw * f_cos

# num_i total inclinations, use stride + offsets to control which ones to run
def run_laetitia(num_i=2000, ntrials=3, stride=10, offsets=[0],
                 folder='1laetitia', base_fn='e2_6', nthreads=4,
                 M1=1,
                 M2=1e-3,
                 M3=1e-3,
                 a0=5,
                 a2=50,
                 E10=1e-3,
                 e2=0.6,
                 k2=0.37,
                 R2=4.779e-4,
                 **kwargs,
                 ):
    plot_k = False
    mkdirp(folder)
    M12 = M1 + M2
    q = M2 / M1
    kwargs_dict = dict(
        M12=M12,
        M3=M3,
        a0=a0,
        a2=a2,
        e2=e2,
        k2=k2,
        R2=R2,
        **kwargs,
    )
    I0d_vals_tot = np.linspace(40, 140, num_i)

    I0d_plot = []
    I0d_plot2 = []
    m1_emaxes = []
    m1_emaxes2 = [] # emax over first fifth of sim
    Kmaxes = []
    Kmins = []
    K0s = []
    eta0 = get_eps_eta0(M1, M2, M3, a0, a2, e2)[3]
    for offset in offsets:
        _I0d_vals = I0d_vals_tot[offset::stride]
        I0d_vals = np.repeat(_I0d_vals, ntrials)
        pkl_fn = '%s/%s_%d.pkl' % (folder, base_fn, offset)
        if not os.path.exists(pkl_fn):
            # print('Skipping %s' % pkl_fn)
            # continue
            print('Running %s' % pkl_fn)
            args = [
                (idx, q, I0d, None, kwargs_dict)
                for idx, I0d in enumerate(I0d_vals)
            ]
            p = Pool(nthreads)
            emax_rets = p.starmap(get_emax_series, args)
            with open(pkl_fn, 'wb') as f:
                pickle.dump((emax_rets), f)
        else:
            with open(pkl_fn, 'rb') as f:
                # print('Loading %s' % pkl_fn)
                emax_rets = pickle.load(f)
        for I0d_val, emax_ret in zip(I0d_vals, emax_rets):
            if len(emax_ret[1]) == 0:
                continue
            first_fifth_idx = np.where(emax_ret[0] < emax_ret[0][-1] / 5)[0]
            I0d_plot.append(I0d_val)
            m1_emaxes.append(1 - np.max(emax_ret[1]))
            if len(emax_ret[1][first_fifth_idx]) > 0:
                I0d_plot2.append(I0d_val)
                m1_emaxes2.append(1 - np.max(emax_ret[1][first_fifth_idx]))

            if plot_k:
                I0 = np.radians(I0d_val)
                e_vals = np.array(emax_ret[1])
                I1_vals = np.radians(np.array(emax_ret[2]))
                ltot_i = ltot(E10, I0, e2, eta0)

                e2_vals, I2_vals = np.array([
                    get_eI2(emax, Imax, eta0, ltot_i)
                    for emax, Imax in zip(e_vals, I1_vals)
                ]).T
                K_vals = (
                    np.sqrt(1 - e_vals**2) * np.cos(I1_vals + I2_vals)
                    - eta0 * e_vals**2 / (2 * np.sqrt(1 - e2_vals**2))
                )
                Kmins.append(K_vals.min())
                Kmaxes.append(K_vals.max())
                K0s.append(
                    np.sqrt(1 - E10**2) * np.cos(I0)
                    - eta0 * E10**2 / (2 * np.sqrt(1 - e2**2))
                )

    eps_oct = a0 / a2 * e2 / (1 - e2**2)
    eta = eta0 / np.sqrt(1 - e2**2)
    print(base_fn, 'eps_oct, eta', eps_oct, eta, a2)
    MLL_expr = (
        0.26 * (eps_oct / 0.1)
        - 0.536 * (eps_oct / 0.1)**2
        + 12.05 * (eps_oct / 0.1)**3
        -16.78 * (eps_oct / 0.1)**4
    ) if eps_oct < 0.05 else 0.45
    ilimd_MLL_L = np.degrees(np.arccos(np.sqrt(MLL_expr)))
    ilimd_MLL_R = np.degrees(np.arccos(-np.sqrt(MLL_expr)))

    _, eps_gr, eps_oct, eta = get_eps(M1, M2, M3, a0, a2, e2)
    Ilimd = get_Ilim(eta, eps_gr)
    Kcrit = (
        np.sqrt(1 - E10**2) * np.cos(np.radians(Ilimd))
        - eta0 * E10**2 / (2 * np.sqrt(1 - e2**2))
    )

    if plot_k:
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(7, 8),
            gridspec_kw={'height_ratios': [1, 1]},
            sharex=True)
    else:
        fig = plt.figure(figsize=(7, 5))
        ax1 = plt.gca()
    ax1.set_yscale('log')
    ax1.scatter(I0d_plot, m1_emaxes, marker='o', s=1.0, edgecolors='tab:blue',
                facecolors='none', alpha=0.5)
    # ax1.semilogy(I0d_plot2, m1_emaxes2, 'b,', alpha=0.5)
    ax1.axvline(ilimd_MLL_L, c='tab:purple', lw=2.5, ls='--')
    ax1.axvline(ilimd_MLL_R, c='tab:purple', lw=2.5, ls='--')
    ax1.set_ylabel(r'$1 - e_{\max}$')

    eps_tide = get_eps_tide(M1, M2, M3, a0, a2, e2, k2, R2)
    elim_tide = get_elim(eta=eta, eps_gr=eps_gr, eps_tide=eps_tide)
    ax1.axhline(1 - elim_tide, c='k', ls=':', lw=2.5)
    emaxes_arr = []
    I0d_emax_arr = sorted(I0d_plot)

    for I0d in I0d_emax_arr:
        try:
            emaxes_arr.append(
                get_emax(eta=eta, eps_gr=eps_gr, I=np.radians(I0d),
                         eps_tide=eps_tide)
            )
        except:
            emaxes_arr.append(1e-3)
    emaxes_arr = np.array(emaxes_arr)
    plt.plot(I0d_emax_arr, 1 - emaxes_arr, 'g', lw=2.0)

    if plot_k:
        ax2.plot(I0d_plot, Kmins, 'bo', label=r'$K_{\min}$', ms=0.5,
                 alpha=0.5)
        ax2.plot(I0d_plot, Kmaxes, 'go', label=r'$K_{\max}$', ms=0.5,
                 alpha=0.5)
        sort_idx = np.argsort(I0d_plot)
        ax2.plot(np.array(I0d_plot)[sort_idx],
                 np.array(K0s)[sort_idx],
                 'k--',
                 label=r'$K_0$')
        ax2.axhline(Kcrit, c='r', ls='--', lw=1.0)
        ax2.set_ylabel(r'$K = j\cos(I) - \eta e^2/2$')
        ax2.set_xlabel(r'$I_0$ (Deg)')

    plt.tight_layout()
    if plot_k:
        fig.subplots_adjust(hspace=0.02)
    plt.savefig('%s/%s' % (folder, base_fn), dpi=300)
    plt.close()

# e2_6 eps_oct, eta 0.09356268731268731 0.39508701612647323 50.00000000000001
# e2_6_m2 eps_oct, eta 0.07426075413256578 0.17607905619931238 62.996052494743665
# e2_6_m3 eps_oct, eta 0.06487274410679486 0.10977015800917482 72.11247851537041
# e2_6_m5 eps_oct, eta 0.054715791467432315 0.060547035219609725 85.49879733383486
# e2_6_m10 eps_oct, eta 0.04342795246733734 0.027037579406008757 107.7217345015942
# e2_6tp eps_oct, eta 0.009356268731268732 0.00017655598377444502 500.0
def plot_laetitia():
    folder='1laetitia'
    num_i = 2000
    ntrials = 3
    M1 = 1
    M2 = 1e-3
    a0 = 5
    a2 = 50
    E10 = 1e-3
    e2 = 0.6
    k2 = 0.37
    R2 = 4.779e-4
    stride = 10
    offsets = np.arange(stride)

    mkdirp(folder)
    M12 = M1 + M2
    q = M2 / M1

    fig, _axs = plt.subplots(
        2, 3,
        figsize=(12, 6),
        sharex=True, sharey=True)
    axs = _axs.flat
    files = ['e2_6', 'e2_6_m2', 'e2_6_m3', 'e2_6_m5', 'e2_6_m10', 'e2_6tp']
    labels = ['(i)', '(ii)', '(iii)', '(iv)', '(v)', '(vi)']
    M3s = np.concatenate((np.array([1, 2, 3, 5, 10]) * 1e-3, [1]))
    for ax, base_fn, M3, label in zip(axs, files, M3s, labels):
        a2 = 50 * M3**(1/3) * 10
        I0d_vals_tot = np.linspace(40, 140, num_i)
        I0d_plot = []
        m1_emaxes = []
        for offset in offsets:
            _I0d_vals = I0d_vals_tot[offset::stride]
            I0d_vals = np.repeat(_I0d_vals, ntrials)
            pkl_fn = '%s/%s_%d.pkl' % (folder, base_fn, offset)
            if not os.path.exists(pkl_fn):
                continue
            with open(pkl_fn, 'rb') as f:
                # print('Loading %s' % pkl_fn)
                emax_rets = pickle.load(f)
            for I0d_val, emax_ret in zip(I0d_vals, emax_rets):
                if len(emax_ret[1]) == 0:
                    continue
                I0d_plot.append(I0d_val)
                m1_emaxes.append(1 - np.max(emax_ret[1]))

        _, eps_gr, eps_oct, eta = get_eps(M2, M1, M3, a0, a2, e2)
        print(base_fn, 'eps_oct, eta', eps_oct, eta, a2)
        MLL_expr = (
            0.26 * (eps_oct / 0.1)
            - 0.536 * (eps_oct / 0.1)**2
            + 12.05 * (eps_oct / 0.1)**3
            -16.78 * (eps_oct / 0.1)**4
        ) if eps_oct < 0.05 else 0.45
        ilimd_MLL_L = np.degrees(np.arccos(np.sqrt(MLL_expr)))
        ilimd_MLL_R = np.degrees(np.arccos(-np.sqrt(MLL_expr)))

        ax.set_yscale('log')
        ax.scatter(I0d_plot, m1_emaxes, marker='o', s=1.0,
                   edgecolors='tab:blue', facecolors='none', alpha=0.5)
        ax.axvline(ilimd_MLL_L, c='tab:purple', lw=2.0, ls='--')
        ax.axvline(ilimd_MLL_R, c='tab:purple', lw=2.0, ls='--')

        eps_tide = get_eps_tide(M1, M2, M3, a0, a2, e2, k2, R2)
        elim_tide = get_elim(eta=eta, eps_gr=eps_gr, eps_tide=eps_tide)
        ax.axhline(1 - elim_tide, c='k', ls=':', lw=2.0)
        emaxes_arr = []
        I0d_emax_arr = sorted(I0d_plot)

        I0d_emax_plot = []
        for I0d in I0d_emax_arr:
            try:
                emaxes_arr.append(
                    get_emax(eta=eta, eps_gr=eps_gr, I=np.radians(I0d),
                             eps_tide=eps_tide)
                )
                I0d_emax_plot.append(I0d)
            except:
                continue
        emaxes_arr = np.array(emaxes_arr)
        ax.plot(I0d_emax_plot, 1 - emaxes_arr, 'g', lw=2.0)

        if M3 < 1:
            ax.text(55, 0.6,
                    # r'%s $m_{\rm p} = %d\,\mathrm{M_{Jup}}$' % (label, 1000 * M3),
                    r'%s $l = %.2f$' % (label, eta),
                    fontsize=12)
        else:
            ax.text(40, 0.003,
                    # r'%s $m_{\rm p} = \mathrm{M}_{\odot}$' % label,
                    r'%s $l = %.4f$' % (label, eta),
                    fontsize=12)
    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = _axs
    for ax in _axs.flat:
        ax.set_xticks([50, 90, 130])
        ax.set_xticklabels([r'$50$', r'$90$', r'$130$'])
    ax1.set_ylabel(r'$1 - e_{\rm J, \max}$')
    ax4.set_ylabel(r'$1 - e_{\rm J, \max}$')
    ax4.set_xlabel(r'$i_{\rm p}$ (deg)')
    ax5.set_xlabel(r'$i_{\rm p}$ (deg)')
    ax6.set_xlabel(r'$i_{\rm p}$ (deg)')
    fig.subplots_adjust(hspace=0, wspace=0)
    plt.savefig('%s/Su_HJs' % folder, dpi=400)
    plt.close()

def make_nogw_plots():
    mkdirp('1nogw_sims')
    # run_nogw_vec(ll=0, q=1, T=1e9, method='Radau', TOL=1e-9,
    #              fn='1nogw_sims/1nogw_vec_q1', Itot=93.5,
    #              plot_full=False)
    ret = run_nogw_vec(ll=0, q=0.2, T=3e9, method='Radau', TOL=1e-9,
                 fn='1nogw_sims/1nogw_vec_q02_short', Itot=93.5, plot_full=False)
    run_nogw_vec(ll=0, q=0.2, T=3e9, method='Radau', TOL=1e-9,
                 fn='1nogw_sims/1nogw_vec_q02', Itot=93.5, ret=ret)
    run_nogw_vec(ll=0, q=0.2, T=3e9, method='Radau', TOL=1e-9,
                 fn='1nogw_sims/1nogw_vec88', Itot=88, w1=np.pi / 2)
    return
    run_nogw_vec(ll=0, q=0.2, T=1e9, method='Radau', TOL=1e-9,
                 fn='1nogw_sims/1nogw_vec80', Itot=80)
    laetitia_kwargs = dict(
        ll=0,
        M12=1+1e-3,
        q=1e-3,
        M3=1,
        INTain=5,
        a2=500,
        E10=1e-3,
        e2=0.6,
        method='Radau',
        T=3e7,
        TOL=1e-9,
        k2=0.37,
        R2=4.67e-4,
    )
    run_nogw_vec(**laetitia_kwargs,
                 w1=0,
                 w2=0,
                 W=0,
                 fn='1nogw_sims/1laetitia_tp_90',
                 Itot=89.9)
    run_nogw_vec(**laetitia_kwargs,
                 w1=0,
                 w2=0,
                 W=0,
                 fn='1nogw_sims/1laetitia_tp_88',
                 Itot=88)
    run_nogw_vec(**laetitia_kwargs,
                 w1=0,
                 w2=0,
                 W=0,
                 fn='1nogw_sims/1laetitia_tp_87',
                 Itot=87)
    run_nogw_vec(**laetitia_kwargs,
                 w1=0,
                 w2=0,
                 W=0,
                 fn='1nogw_sims/1laetitia_tp_86',
                 Itot=86)
    tp_nosrf_kwargs = dict(
        ll=0,
        l=0,
        M12=1+1e-3,
        q=1e-3,
        M3=1,
        INTain=5,
        a2=500,
        E10=1e-3,
        e2=0.6,
        method='Radau',
        T=3e7,
        TOL=1e-9,
    )
    run_nogw_vec(**tp_nosrf_kwargs,
                 w1=0,
                 w2=0,
                 W=0,
                 fn='1nogw_sims/1nosrf_tp_90',
                 Itot=89.9)
    run_nogw_vec(**tp_nosrf_kwargs,
                 w1=0,
                 w2=0,
                 W=0,
                 fn='1nogw_sims/1nosrf_tp_88',
                 Itot=88)
    run_nogw_vec(**tp_nosrf_kwargs,
                 w1=0,
                 w2=0,
                 W=0,
                 fn='1nogw_sims/1nosrf_tp_87',
                 Itot=87)
    run_nogw_vec(**tp_nosrf_kwargs,
                 w1=0,
                 w2=0,
                 W=0,
                 fn='1nogw_sims/1nosrf_tp_86',
                 Itot=86)
    bbh_nosrf_kwargs = dict(
        ll=0,
        l=0,
        M12=50,
        q=0.3,
        M3=30,
        INTain=100,
        a2=4500,
        E10=1e-3,
        e2=0.6,
        method='Radau',
        T=3e8,
        TOL=1e-9,
    )
    run_nogw_vec(**bbh_nosrf_kwargs,
                 w1=0,
                 w2=0,
                 W=0,
                 fn='1nogw_sims/1nosrf_bbh_90',
                 Itot=89.9)
    run_nogw_vec(**bbh_nosrf_kwargs,
                 w1=0,
                 w2=0,
                 W=0,
                 fn='1nogw_sims/1nosrf_bbh_88',
                 Itot=88)
    run_nogw_vec(**bbh_nosrf_kwargs,
                 w1=0,
                 w2=0,
                 W=0,
                 fn='1nogw_sims/1nosrf_bbh_87',
                 Itot=87)
    run_nogw_vec(**bbh_nosrf_kwargs,
                 w1=0,
                 w2=0,
                 W=0,
                 fn='1nogw_sims/1nosrf_bbh_86',
                 Itot=86)
    run_nogw_vec(**bbh_nosrf_kwargs,
                 w1=np.pi / 2,
                 w2=0,
                 W=0,
                 fn='1nogw_sims/1nosrf_bbh_86_2',
                 Itot=86)
    run_nogw_vec(**bbh_nosrf_kwargs,
                 w1=np.pi / 4,
                 w2=0,
                 W=0,
                 fn='1nogw_sims/1nosrf_bbh_86_3',
                 Itot=86)
    run_nogw_vec(**bbh_nosrf_kwargs,
                 w1=3 * np.pi / 4,
                 w2=0,
                 W=0,
                 fn='1nogw_sims/1nosrf_bbh_86_4',
                 Itot=86)
    run_nogw_vec(**bbh_nosrf_kwargs,
                 mm=0,
                 w1=0,
                 w2=0,
                 W=0,
                 fn='1nogw_sims/1nosrf_bbh_86_nooct',
                 Itot=86)

def plot_total_fracs_simple(ain=50, fldr='1sweepbin_simple', num_Is=500,
                            nruns=2, Imin=50, Imax=130, a2eff=1800,
                            nthreads=48, suffix=''):
    mkdirp(fldr)
    e0 = 1e-3
    m12 = 50
    m3 = 30
    qs = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    e2s = [0.6, 0.8, 0.9]
    I0s = np.arccos(
        np.linspace(np.cos(np.radians(Imax)), np.cos(np.radians(Imin)), num_Is)
    )
    # for plotting
    colors = ['r', 'b', 'k']

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(12, 6),
        sharey=True)
    for color, e2 in zip(colors, e2s):
        pkl_fn = '%s/e%d%s.pkl' % (fldr, 10 * e2, suffix)
        a2 = a2eff / np.sqrt(1 - e2**2)
        args = [
            (idx, q, 1e10, ain, a2, e0, e2, I0, True)
            for idx in range(nruns)
            for I0 in I0s
            for q in qs
        ]
        if not os.path.exists(pkl_fn):
            print('Running %s' % pkl_fn)
            p = Pool(nthreads)
            rets = p.starmap(sweeper_bin, args)
            with open(pkl_fn, 'wb') as f:
                pickle.dump((rets), f)
        else:
            with open(pkl_fn, 'rb') as f:
                print('Loading %s' % pkl_fn)
                rets = pickle.load(f)

        pkl_fn_nogw = '%s/e%d%s_nogw.pkl' % (fldr, 10 * e2, suffix)
        args_nogw = [
            (idx, q, np.degrees(I0), None,
             dict(a0=ain, a2=a2, e2=e2, tf_mult=2000))
            for idx in range(nruns)
            for I0 in I0s
            for q in qs
        ]
        if not os.path.exists(pkl_fn_nogw):
            # print('Not exists %s' % pkl_fn_nogw)
            # continue
            print('Running %s' % pkl_fn_nogw)
            p = Pool(nthreads)
            rets_nogw = p.starmap(get_emax_series, args_nogw)
            with open(pkl_fn_nogw, 'wb') as f:
                pickle.dump((rets_nogw), f)
        else:
            with open(pkl_fn_nogw, 'rb') as f:
                print('Loading %s' % pkl_fn_nogw)
                rets_nogw = pickle.load(f)
        I0ds = np.degrees([a[-2] for a in args])

        emaxes = []
        emeans = []
        for I0d, ret in zip(I0ds, rets_nogw):
            I0 = np.radians(I0d)
            e_vals = np.array(ret[1])
            if len(e_vals) == 0:
                emaxes.append(e0)
                emeans.append(e0)
                continue
            where_idx = np.where(e_vals >= 0.3)[0]
            e_vals = e_vals[where_idx]
            if len(e_vals) == 0:
                emaxes.append(e0)
                emeans.append(e0)
                continue
            emaxes.append(np.max(e_vals))
            # approx 1 + 73e^2/24... \approx 4.427 is constant
            jmean = np.mean((1 - e_vals**2)**(-3))**(-1/6)
            emean = np.sqrt(1 - jmean**2)
            emeans.append(emean)
        emeans = np.array(emeans)
        emaxes = np.array(emaxes)

        q_arr = []
        epsoct_arr = []
        frac_arr = []
        frac_arr_nogw = []
        q_vals = np.array([a[1] for a in args])
        tmerges = np.array([r[0] for r in rets])

        for q in qs:
            q_idxs = np.where(q_vals == q)[0]
            q_tmerges = tmerges[q_idxs]
            num_merges = len(np.where(q_tmerges < 9.9e9)[0])
            q_arr.append(q)
            frac_arr.append(num_merges / len(q_tmerges)
                            * (np.cos(Imin) - np.cos(Imax)) / 2 * 100)
            epsoct_arr.append((1 - q) / (1 + q)
                           * ain / a2
                           * e2 / (1 - e2**2))
            m2 = m12 / (1 + q)
            m1 = m12 - m2
            j_eff_crit = 0.01461 * (100 / ain)**(2/3)
            e_eff_crit = np.sqrt(1 - j_eff_crit**2)
            j_os = (256 * k**3 * q / (1 + q)**2 * m12**3 * a2eff**3 / (
                c**5 * ain**4 * np.sqrt(k * m12 / ain**3) * m3 * ain**3))**(1/6)
            e_os = np.sqrt(1 - j_os**2)

            num_merges_nogw = len(np.where(np.logical_or(
                emaxes[q_idxs] > e_os,
                emeans[q_idxs] > e_eff_crit
            ))[0])
            frac_arr_nogw.append(num_merges_nogw / len(q_tmerges)
                                 * (np.cos(Imin) - np.cos(Imax)) / 2 * 100)

        ax1.plot(q_arr, frac_arr, c=color, lw=1.0, marker='o',
                 label=r'$e_{\rm out} = %.1f$' % e2)
        ax1.legend()
        ax1.set_xlabel(r'$q$')
        ax1.set_ylabel(r'$f_{\rm merger}$ [\%]')

        ax2.plot(epsoct_arr, frac_arr, c=color, lw=1.0, marker='o')
        ax2.set_xlabel(r'$\epsilon_{\rm oct}$')
        if suffix == '':
            ax1.plot(q_arr, frac_arr_nogw, c=color, alpha=0.5, lw=1.0, marker='x',
                     ls=':')
            ax2.plot(epsoct_arr, frac_arr_nogw, c=color, alpha=0.5, lw=1.0,
                     marker='x', ls=':')
    ax1.set_ylim(bottom=0)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.03)
    plt.savefig(fldr + '/simple%s' % suffix, dpi=300)

if __name__ == '__main__':
    # UNUSED
    # timing_tests()

    # testing elim calculation
    # emaxes = get_emax_series(0, 1, 92.8146, 2e7)[1]
    # print(1 - np.mean(emaxes))

    # sweep(folder='1sweepbin', nthreads=64)
    # run_emax_sweep(nthreads=64, tf_mult_default=2000, num_i=500, num_trials=20,
    #                folder='1sweepbin_emax_long')
    # plot_composite(plot_single=True, plot_all=False)
    # plot_composite(plot_single=False, plot_all=True)
    # plot_massratio_sample()
    # plot_long_composite()
    # plot_completeness(num_ts=50)

    # emax_cfgs_short = [
    #     [0.3, 0.6, '1p3dist_2800', 100, 2800],
    #     [0.3, 0.6, '1p3dist_2000', 100, 2000],
    # ]
    # run_emax_sweep(num_i=200, num_trials=3, nthreads=32,
    #                run_cfgs=emax_cfgs_short)

    # emax_cfgs_other = [
    #     [0.3, 0.6, '1p3dist_gr0', 100, 3600, dict(l=0)],
    #     # [0.4, 0.6, '1p4dist_gr0', 100, 3600, dict(l=0)],
    # ]
    # run_emax_sweep(nthreads=6, run_cfgs=emax_cfgs_other)
    # plot_emax_dq(I0=93, fn='q_sweep93')
    # plot_emax_dq(I0=93.5, fn='q_sweep_935')
    # plot_emax_dq(I0=95, fn='q_sweep_95')
    # plot_emax_dq(I0=96.2, fn='q_sweep_962')
    # plot_emax_dq(I0=97, fn='q_sweep_97')
    # plot_emax_dq(I0=99, fn='q_sweep_99')

    # emax_omega_sweep()
    # k_sweep()

    # m1, m2, m3, a, a2, e2 = 25, 25, 30, 100, 4500, 0.6
    # eps = get_eps(50 / 3, 100 / 3, m3, a, a2, e2)
    # print(eps[2])
    # print('eta', eps[3])
    # tlk0 = get_tlk0(m1, m2, m3, a, a2 * (1 - 0.9**2)**(1/2))
    # print('tlk', tlk0)
    # m12 = m1 + m2
    # m123 = m12 + m3
    # n = np.sqrt(G * m12 / a**3)
    # print('Pin', 1 / n * S_PER_UNIT / S_PER_YR)
    # Pout = 1 / np.sqrt(G * m123 / a2**3) * S_PER_UNIT / S_PER_YR
    # print('Pout', Pout)
    # print('1 - emax, DA', 1 - np.sqrt(1 - (Pout / tlk0)**2))
    # elim = get_elim(eps[3], eps[1])
    # print('1 - elim', 1 - elim)

    # a2effs = [3600, 5500]
    # plot = True
    # for a2eff in a2effs:
    #     frac = pop_synth(a2eff=a2eff, base_fn='a2eff%d' % a2eff, to_plot=plot)
    # a2effs2 = [2800, 4500]
    # for a2eff in a2effs2:
    #     frac = pop_synth(a2eff=a2eff, base_fn='a2eff%d' % a2eff, to_plot=plot,
    #                      n_qs=7, ntrials=10500)

    # tot_frac = []
    # a2effs_nogwonly = [3600, 2800, 4500, 5500, 7000]
    # a2effs_nogwonly = [3600, 5500]
    # for a2eff in a2effs_nogwonly:
    #     ret = pop_synth_nogw(a2eff=a2eff, base_fn='a2eff_nogw%d' % a2eff,
    #                          to_plot=plot, n_qs=31, q_min=0.2,
    #                          n_e2s=17, n_I0s=41, stride=1, reps=1, nthreads=64)
    #     frac = ret[0]
    #     tot_frac.append(frac)
    # pop_synth_nogw(a2eff=3600, base_fn='a2eff_nogw_lowq3600',
    #                n_e2s=17, n_I0s=41, reps=2, stride=7, n_qs=11,
    #                to_plot=True, q_min=1e-5, q_max=0.1, q_spread=np.geomspace,
    #                nthreads=64)
    # plt.plot(a2effs_nogwonly, tot_frac, 'ko')
    # plt.xlabel(r'$a_{\rm out, eff}$')
    # plt.ylabel(r'Merger Fraction (\%)')
    # plt.savefig('1popsynth/total', dpi=300)
    # plt.close()

    # 0.456 Gyr = 500 Tk
    # run_laetitia(nthreads=4, offsets=np.arange(10), e2=0.1, base_fn='e2_1')
    # run_laetitia(nthreads=4, offsets=np.arange(10), e2=0.3, base_fn='e2_3')
    # run_laetitia(nthreads=4, offsets=np.arange(10), e2=0.5, base_fn='e2_5')
    # run_laetitia(nthreads=4, offsets=np.arange(10), e2=0.8, base_fn='e2_8')
    # run_laetitia(nthreads=2, offsets=np.arange(10), e2=0.9, base_fn='e2_9')
    # run_laetitia(nthreads=10, offsets=np.arange(0, 10, 2), M3=1, a2=500,
    #              e2=0.1, base_fn='e2_1tp')
    # run_laetitia(nthreads=10, offsets=np.arange(0, 10, 2), M3=1, a2=500,
    #              e2=0.3, base_fn='e2_3tp')
    # run_laetitia(nthreads=10, offsets=np.arange(0, 10, 2), M3=1, a2=500,
    #              e2=0.5, base_fn='e2_5tp')
    # run_laetitia(nthreads=11, offsets=np.arange(0, 10, 2), M3=1, a2=500,
    #              e2=0.8, base_fn='e2_8tp')
    # run_laetitia(nthreads=5, offsets=np.arange(0, 10, 2), M3=1, a2=500,
    #              e2=0.9, base_fn='e2_9tp')
    # run_laetitia(nthreads=4, offsets=[0], M3=1, a2=500,
    #              e2=0.6, base_fn='e2_6tp_w0',
    #              w1=0, w2=0, W=0,
    #              ntrials=1)
    # run_laetitia(nthreads=4, offsets=np.arange(10), e2=0.3, base_fn='e2_3_gr0',
    #              l=0, k2=0, R2=0)
    # run_laetitia(nthreads=4, offsets=np.arange(10), e2=0.6, base_fn='e2_6')
    # run_laetitia(nthreads=11, offsets=np.arange(0, 10, 2), M3=1, a2=500,
    #              e2=0.6, base_fn='e2_6tp')
    # for m3_mult in [2, 3, 5, 8, 10, 30]:
    #     run_laetitia(nthreads=4, offsets=np.arange(0, 10), M3=1e-3 * m3_mult,
    #                  a2=50 * (m3_mult**(1/3)), e2=0.6,
    #                  base_fn='e2_6_m%d' % m3_mult)
    #     run_laetitia(nthreads=4, offsets=np.arange(0, 10), M3=1e-3 * m3_mult,
    #                  a2=50 * (m3_mult**(1/3)), e2=0.8,
    #                  base_fn='e2_8_m%d' % m3_mult)
    plot_laetitia()
    # run_laetitia(nthreads=8, e2=0.6, base_fn='e2_6_quad', ntrials=1, mm=0)

    # make_nogw_plots()

    # plot_emaxgrid(nthreads=32)
    # plot_total_fracs_simple(a2eff=3600, suffix='outer', nthreads=48)
    # plot_total_fracs_simple(nthreads=48)
    pass

# PARAMETERS eps_oct eta
# Saving 1sweepbin/composite_1p2dist 0.013888888888888893 0.0545607084248879
# Saving 1sweepbin/composite_1p3dist 0.011217948717948716 0.06973439656672067
# Saving 1sweepbin/composite_1p4dist 0.00892857142857143 0.08017083686922306
# Saving 1sweepbin/composite_1p5dist 0.006944444444444447 0.08729713347982068
# Saving 1sweepbin/composite_1p7dist 0.0036764705882352967 0.09515085483094642
# Saving 1sweepbin/composite_1equaldist 0.0 0.09820927516479827
# Saving 1sweepbin/composite_e81p2dist 0.02469135802469137 0.06300127939257143
# Saving 1sweepbin/composite_e81p3dist 0.019943019943019946 0.08052234525914459
# Saving 1sweepbin/composite_e81p4dist 0.01587301587301588 0.09257330849520702
# Saving 1sweepbin/composite_e81p5dist 0.012345679012345685 0.10080204702811432
# Saving 1sweepbin/composite_e81p7dist 0.006535947712418305 0.10987074330053984
# Saving 1sweepbin/composite_e81equaldist 0.0 0.11340230290662862
# Saving 1sweepbin/composite_e91p2dist 0.03823595564509364 0.07391568293882321
# Saving 1sweepbin/composite_e91p3dist 0.030882887251806393 0.09447211547210546
# Saving 1sweepbin/composite_e91p4dist 0.024580257200417337 0.1086107994203117
# Saving 1sweepbin/composite_e91p5dist 0.01911797782254682 0.11826509270211719
# Saving 1sweepbin/composite_e91p7dist 0.010121282376642437 0.12890485882756716
# Saving 1sweepbin/composite_e91equaldist 0.0 0.13304822928988183
# Saving 1sweepbin/composite_tp 0.02042079207920792 0.0038509665783667485
# Saving 1sweepbin/bindist 0.02900107411385608 0.11797414551348571
