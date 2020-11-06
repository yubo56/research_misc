from collections import defaultdict
import time
from utils import *
import numpy as np
from cython_utils import *
from multiprocessing import Pool

import os
import pickle

from scipy.integrate import solve_ivp

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
TOL = 1e-11

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
def sweeper_bin(idx, q, t_final, a0, a2, e0, e2, I0):
    np.random.seed(idx + int(time.time()))
    M1 = 50 / (1 + q)
    AF = 0.5 / a0
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
        TOL=1e-9,
        AF=AF,
        w1=np.random.rand() * 2 * np.pi,
        w2=np.random.rand() * 2 * np.pi,
        W=np.random.rand() * 2 * np.pi,
    )
    tf = ret.t[-1]
    print(idx, q, t_final / 1e9, a0, a2, e0, e2, np.degrees(I0))
    return tf

# def sweep(num_trials=20, num_trials_purequad=4, num_i=200, t_hubb_gyr=10,
def sweep(num_trials=3, num_trials_purequad=1, num_i=200, t_hubb_gyr=10,
          folder='1sweepbin', nthreads=60):
    mkdirp(folder)
    m12, m3, e0 = 50, 30, 1e-3

    to_plot = plt is not None
    # to_plot = False

    # q, e2, filename, ilow, ihigh, a0, a2eff
    run_cfgs = [
        # Bin's weird case
        # [0.4, 0.9, 'bindist', 70, 110, 10, 700 * np.sqrt(1 - 0.9**2)],

        # exploratory, find the right inclination range to restrict to
        # [0.2, 0.6, 'explore_1p2dist', 50, 130, 100, 3600],
        # [0.3, 0.6, 'explore_1p3dist', 50, 130, 100, 3600],
        # [0.4, 0.6, 'explore_1p4dist', 50, 130, 100, 3600],
        # [0.5, 0.6, 'explore_1p5dist', 50, 130, 100, 3600],
        # [0.7, 0.6, 'explore_1p7dist', 50, 130, 100, 3600],
        [1.0, 0.6, 'explore_1equaldist', 50, 130, 100, 3600],

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

        # a2 = 4500, e2 = 0.6
        # [0.2, 0.6, '1p2dist', 89.5, 105, 100, 3600],
        # [0.3, 0.6, '1p3dist', 90.5, 100, 100, 3600],
        # [0.4, 0.6, '1p4dist', 90.5, 98, 100, 3600],
        # [0.5, 0.6, '1p5dist', 91, 98, 100, 3600],
        # [0.7, 0.6, '1p7dist', 91, 95, 100, 3600],
        # [1.0, 0.6, '1equaldist', 92.1, 93.5, 100, 3600],
        # [0.2, 0.6, '1p2distp2', 66, 87, 100, 3600],
        # [0.3, 0.6, '1p3distp2', 73, 86, 100, 3600],

        # [0.2, 0.8, 'e81p2dist', 89, 107, 100, 3600],
        # [0.2, 0.8, 'e81p2distp2', 57, 86.5, 100, 3600],
        # [0.3, 0.8, 'e81p3dist', 90.5, 100, 100, 3600],
        # [0.4, 0.8, 'e81p4dist', 90.5, 98, 100, 3600],
        # [0.5, 0.8, 'e81p5dist', 91, 98, 100, 3600],
        # [0.7, 0.8, 'e81p7dist', 91, 95, 100, 3600],
        # [1.0, 0.8, 'e81equaldist', 92.1, 93.5, 100, 3600],

        # [0.2, 0.9, 'e91p2dist', 89.5, 105, 100, 3600],
        # [0.3, 0.9, 'e91p3dist', 90.5, 100, 100, 3600],
        # [0.4, 0.9, 'e91p4dist', 90.5, 98, 100, 3600],
        # [0.5, 0.9, 'e91p5dist', 91, 98, 100, 3600],
        # [0.7, 0.9, 'e91p7dist', 91, 95, 100, 3600],
        # [1.0, 0.9, 'e91equaldist', 92.1, 93.5, 100, 3600],
    ]
    total_merge_fracs = []
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

        tmerges = np.array(tmerges)
        merged = np.where(tmerges < 9.9e9)[0]
        nmerged = np.where(tmerges > 9.9e9)[0]

        # calculate histogram of merged vs not merged
        merge_probs = []
        for I in I0s:
            merge_probs.append(
                len(np.where(np.abs(I_plots[merged] - I) < 1e-6)[0]) /
                len(np.where(np.abs(I_plots - I) < 1e-6)[0]))
        total_merge_fracs.append(
            (ihigh - ilow) / 180 * np.sum(merge_probs) / len(I0s)
        )
        if to_plot:
            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=(9, 12),
                gridspec_kw={'height_ratios': [1, 2]},
                sharex=True)
            ax1.set_title(r'$q = %.1f$' % q)
            ax1.plot(np.degrees(I0s), merge_probs, 'k')
            ax1.set_ylabel(r'Merge Prob')

            # plot actual merger times
            ax2.semilogy(np.degrees(I_plots[merged]), tmerges[merged], 'go', ms=1)
            ax2.semilogy(np.degrees(I_plots[nmerged]), tmerges[nmerged], 'b^', ms=1)
            ax2.set_xlabel(r'$I_0$')
            ax2.set_ylabel(r'$T_m$ (yr)')
            plt.tight_layout()

            fig.subplots_adjust(wspace=0.03)
            plt.savefig(fn, dpi=300)
            plt.close()
    if not to_plot:
        return
    q_vals = np.array([cfg[0] for cfg in run_cfgs])
    total_merge_fracs = np.array(total_merge_fracs)
    eps_oct0 = get_eps(0, m12, m3, a0, a2, e2)[2]
    eps_oct = (1 - q_vals) / (1 + q_vals) * eps_oct0
    plt.plot(eps_oct * 100, total_merge_fracs * 100, 'bo')
    plt.xlabel(r'$100\epsilon_{\rm oct}$')
    plt.ylabel(r'$f_{\rm merger}$ [\%]')

    # plot quadrupole merger fraction
    plt.ylim(bottom=0)
    # I have the explicit formula in my notes...
    f_merge_quad = 1.231 / 180
    plt.axhline(f_merge_quad * 100, c='r', ls=':')

    # plot the naive fit using MLL16's fit
    # awful fit...
    #
    # ilims = np.arccos(np.sqrt(
    #     0.26 * (eps_oct / 0.1)
    #     - 0.536 * (eps_oct / 0.1)**2
    #     + 12.05 * (eps_oct / 0.1)**3
    #     -16.78 * (eps_oct / 0.1)**4
    # ))
    # delta_ilims = np.pi / 2 - ilims
    # f_merge_mll = (delta_ilims / np.pi) / 2
    # plt.plot(eps_oct * 100, f_merge_mll * 100, 'g')

    # linear fiducial
    # plt.plot([0, 100 * eps_oct[0]],
    #          [0, total_merge_fracs[0] * 100],
    #          'k', lw=0.5)

    plt.title(r'$e_{\rm out} = 0.6$')

    plt.tight_layout()

    curr_ax = plt.gca()
    ax3 = curr_ax.twiny()
    ax3.set_xlim(curr_ax.get_xlim())
    ax3.set_xticks(eps_oct * 100)
    ax3.set_xticklabels(['%.1f' % q for q in q_vals])
    ax3.set_ylabel(r'$q$')

    plt.savefig('1sweepbin/total_merger_fracs', dpi=300)
    plt.close()

def plot_emax_sweep(num_trials=4, num_trials_purequad=1, num_i=2000,
                    folder='1sweepbin_emax', nthreads=1):
    mkdirp(folder)
    m12, m3, e0 = 50, 30, 1e-3

    # to_plot = plt is not None
    to_plot = False

    # q, e2, filename, ilow, ihigh, a0, a2eff
    run_cfgs = [
        # a2 = 4500, e2 = 0.6
        [0.2, 0.6, '1p2dist', 100, 3600],
        [0.3, 0.6, '1p3dist', 100, 3600],
        [0.4, 0.6, '1p4dist', 100, 3600],
        [0.5, 0.6, '1p5dist', 100, 3600],
        [0.7, 0.6, '1p7dist', 100, 3600],
        [1.0, 0.6, '1equaldist', 100, 3600],

        [0.2, 0.8, 'e81p2dist', 100, 3600],
        [0.3, 0.8, 'e81p3dist', 100, 3600],
        [0.4, 0.8, 'e81p4dist', 100, 3600],
        [0.5, 0.8, 'e81p5dist', 100, 3600],
        [0.7, 0.8, 'e81p7dist', 100, 3600],
        [1.0, 0.8, 'e81equaldist', 100, 3600],

        [0.2, 0.9, 'e91p2dist', 100, 3600],
        [0.3, 0.9, 'e91p3dist', 100, 3600],
        [0.4, 0.9, 'e91p4dist', 100, 3600],
        [0.5, 0.9, 'e91p5dist', 100, 3600],
        [0.7, 0.9, 'e91p7dist', 100, 3600],
        [1.0, 0.9, 'e91equaldist', 100, 3600],

        # Bin's weird case
        [0.4, 0.9, 'bindist', 70, 110, 10, 700 * np.sqrt(1 - 0.9**2)],
    ]
    for cfg in run_cfgs:
        q, e2, base_fn, a0, a2eff = cfg
        a2 = a2eff / np.sqrt(1 - e2**2)

        I0s = np.radians(np.linspace(50, 130, num_i))
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
            (idx, q, I0, None, dict(
                a0=a0, a2=a2, e2=e2,
            ))
            for idx, I0 in enumerate(I_plots)
        ]
        if not os.path.exists(pkl_fn):
            print('Running %s' % pkl_fn)
            p = Pool(nthreads)
            rets = p.starmap(get_emax_series, args)
            with open(pkl_fn, 'wb') as f:
                pickle.dump(rets, f)
        else:
            with open(pkl_fn, 'rb') as f:
                print('Loading %s' % pkl_fn)
                rets = pickle.load(f)
        I_vals = []
        emaxes = []
        for arg_lst, ret in zip(args, rets):
            I = np.degrees(arg_lst[2])
            emaxes = np.max(ret[1])

        if to_plot:
            plt.semilogy(I_vals, 1 - emaxes, 'bo', ms=1)
            plt.xlabel(r'$I_0$')
            plt.ylabel(r'$1 - e_{\max}$')
            plt.savefig(fn, dpi=300)
            plt.close()

# default tf is 500 tk, if tf == None
def get_emax_series(idx, q, I0, tf, inits={}):
    np.random.seed(idx + int(time.time()))
    M1 = 50 / (1 + q)
    M2 = 50 - M1
    M3 = 30
    ain = inits.get('a0', 100)
    a2 = inits.get('a2', 4500)
    E2 = inits.get('e2', 0.6)
    n1 = np.sqrt((k*(M1 + M2))/ain ** 3)
    tk = 1/n1*((M1 + M2)/M3)*(a2/ain)**3*(1 - E2**2)**(3.0/2)

    if tf == None:
        tf = 500 * tk

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
        w1=inits.get('w1', np.random.rand() * 2 * np.pi),
        w2=inits.get('w2', np.random.rand() * 2 * np.pi),
        W=inits.get('W', np.random.rand() * 2 * np.pi),
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
    t_idxs = np.where(abs(demag) < 1e-5)[0] + 10

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
    elim = get_elim(eps[3] / np.sqrt(1 - 0.6**2),
                    eps[1] * (1 - 0.6**2)**(3/2))

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

def run_nogw_vec(fn='1nogw_vec', **kwargs):
    a2 = 4500
    Itot = kwargs.get('Itot', 93.5)
    eps = get_eps(20, 30, 30, 100, a2, 0.6)
    eta_ecc = eps[3] / np.sqrt(1 - 0.6**2)

    ret = run_vec(a2=a2, **kwargs)
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
    eta = eps[3] / np.sqrt(1 - eoutvec_mags**2)
    K = (
        np.sqrt(1 - evec_mags**2) * np.cos(np.radians(I + Iout))
        - eta * evec_mags**2/2
    )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2,
        figsize=(12, 9),
        sharex=True)

    ax1.plot(ret.t / 1e8, eoutvec_mags)
    ax1.set_ylabel(r'$e_{\rm out}$')
    ax2.semilogy(ret.t / 1e8, 1 - evec_mags)
    ax2.set_ylabel(r'$1 - e$')
    ax3.plot(ret.t / 1e8, I + Iout)
    ax3.set_ylabel(r'$I$ (Deg)')
    ax3.set_xlabel(r'Time $(10^8 \;\mathrm{yr})$')
    ax4.plot(ret.t / 1e8, K)
    ax4.axhline(-eta[0] / 2, c='k', ls=':', lw=2)
    ax4.set_ylabel(r'$K = j\cos(I) - \eta e^2/2$')
    ax4.set_xlabel(r'Time $(10^8 \;\mathrm{yr})$')

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
    plt.savefig(fn, dpi=300)
    plt.clf()

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
    eta_ecc = eps[3] / np.sqrt(1 - 0.6**2)
    eta = eps[3] / np.sqrt(1 - eoutvec_mags**2)
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
    plt.legend(loc='upper left', ncol=2, fontsize=14)
    plt.xlabel(r'$I_0$')
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
    plt.legend(loc='upper left', ncol=2, fontsize=14)
    plt.xlabel(r'$\epsilon_{\rm oct} / \epsilon_{\rm oct}(q = 0)$')
    plt.ylabel(r'$\Delta K$')
    plt.savefig(fn + 'vsI', dpi=300)
    plt.close()

if __name__ == '__main__':
    # UNUSED
    # timing_tests()

    # testing elim calculation
    # emaxes = get_emax_series(0, 1, 92.8146, 2e7)[1]
    # print(1 - np.mean(emaxes))

    # sweep(folder='1sweepbin', nthreads=12)
    # plot_emax_sweep(nthreads=4)

    # plot_emax_dq(I0=93, fn='q_sweep93')
    # plot_emax_dq(I0=93.5, fn='q_sweep_935')
    # plot_emax_dq(I0=95, fn='q_sweep_95')
    # plot_emax_dq(I0=96.2, fn='q_sweep_962')
    # plot_emax_dq(I0=97, fn='q_sweep_97')
    # plot_emax_dq(I0=99, fn='q_sweep_99')

    # run_nogw_vec(ll=0, T=3e9, method='Radau', TOL=1e-9)
    # run_nogw_vec(ll=0, T=3e9, method='Radau', TOL=1e-9, fn='1nogw_vec95',
    #              Itot=96)
    # emax_omega_sweep()
    # k_sweep()
    pass

    # I think width ~ 1/epsoct^p, probably random walk timescale?

    # exo3 is running bindist
    # exo2a is running all the rest of the explores
    # exo15c is running emax_sweeps
    # exo4 is running 1p2distp2, 1p3distp2, e81p2dist, e81p2distp2

