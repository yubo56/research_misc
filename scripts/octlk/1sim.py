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
    axs[2].set_ylabel(r'$I_{\rm tot}$')
    axs[2].set_xlabel(r'Time $(10^8 \;\mathrm{yr})$')

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.03)
    plt.savefig('1fiducial', dpi=300)
    plt.clf()

def test_vec():
    ret = run_vec()
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

def sweeper(idx, q, t_final, tlk0, a0, e0, I0, e2, eps):
    def a_term_event(_t, y, *_args):
        return y[0] - AF
    a_term_event.terminal = True

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

# manually codify a0, a, M12
def sweeper_bin(idx, q, t_final, _tlk0, a0, e0, I0, e2, _eps):
    np.random.seed(idx + int(time.time()))
    M1 = 50 / (1 + q)
    ret = run_vec(
        T=1e10,
        M1=M1,
        M2=50 - M1,
        Itot=np.degrees(I0),
        INTain=100,
        a2=4500,
        E10=e0,
        E20=e2,
        w1=np.random.rand() * 2 * np.pi,
        w2=np.random.rand() * 2 * np.pi,
        W=np.random.rand() * 2 * np.pi,
    )
    tf = ret.t[-1]
    print(idx, q, np.degrees(I0), tf / 1e9)
    return tf

def sweep(num_trials=20, num_i=200, t_hubb_gyr=10,
          folder='1sweep', func=sweeper, nthreads=10):
    mkdirp(folder)
    a0 = 1
    m12, m3, a, a2, e0, e2 = 50, 30, 100, 4500, 1e-3, 0.6
    p = Pool(nthreads)

    for q, base_fn, ilow, ihigh in [
            [0.2, '1p2dist', 89.5, 105],
            [0.3, '1p3dist', 90.5, 100],
            [0.4, '1p4dist', 90.5, 98],
            [0.5, '1p5dist', 91, 98],
            [0.7, '1p7dist', 91, 95],
            [1.0, '1equaldist', 92, 93.5],
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
                (idx, q, t_final, tlk0, a0, e0, I0, e2, eps)
                for idx, I0 in enumerate(I_plots)
            ]
            tmerges = p.starmap(func, args)
            with open(pkl_fn, 'wb') as f:
                pickle.dump((I_plots, tmerges), f)
        else:
            with open(pkl_fn, 'rb') as f:
                print('Loading %s' % pkl_fn)
                I_plots, tmerges = pickle.load(f)

        tmerges = np.array(tmerges)
        merged = np.where(tmerges < 9.9e9)[0]
        nmerged = np.where(tmerges > 9.9e9)[0]
        if plt is not None:
            plt.semilogy(np.degrees(I_plots[merged]), tmerges[merged], 'go', ms=1)
            plt.semilogy(np.degrees(I_plots[nmerged]), tmerges[nmerged], 'b^', ms=1)
            plt.xlabel(r'$I_{\rm tot, 0}$')
            plt.ylabel(r'$T_m$ (Gyr)')
            plt.savefig(fn, dpi=300)
            plt.close()

def sweeper_comp(nthreads=1, nruns=1000):
    mkdirp('1sweep')

    m1, m2, m3, a, a2, e0, e2 = 50/3, 100/3, 30, 100, 4500, 1e-3, 0.6
    a0 = 1
    q = m1 / m2
    p = Pool(nthreads)
    eps = get_eps(m1, m2, m3, a, a2, e2)
    tlk0 = get_tlk0(m1, m2, m3, a, a2) / 1e9
    t_final = 10 / tlk0
    I0 = np.radians(93.2)

    args = [[idx, q, t_final, tlk0, a0, e0, I0, e2, eps] for idx in range(nruns)]
    pkl_fn = '1sweep/sweeper_comp.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        start = time.time()
        my_tmerges = p.starmap(sweeper, args)
        print('Orbs used', time.time() - start)
        start = time.time()
        bin_tmerges = p.starmap(sweeper_bin, args)
        print('Bin used', time.time() - start)
        with open(pkl_fn, 'wb') as f:
            pickle.dump((my_tmerges, bin_tmerges), f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            my_tmerges, bin_tmerges = pickle.load(f)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 5),
        gridspec_kw={'height_ratios': [1, 1]},
        sharex=True)
    ax1.hist(np.log10(my_tmerges) + 9, bins=30)
    ax2.hist(np.log10(bin_tmerges), bins=30)
    ax1.set_ylabel('Orb. El. Eqs')
    ax2.set_ylabel('Vec. Eqs')
    ax2.set_xlabel(r'$\log_{10}$ Merger Time')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    plt.tight_layout()

    fig.subplots_adjust(hspace=0.03)
    plt.savefig('1sweep/sweeper_comp', dpi=300)
    plt.close()

def get_emax_series(idx, q, I0, tf):
    np.random.seed(idx + int(time.time()))
    M1 = 50 / (1 + q)
    M2 = 50 - M1
    M3 = 30
    ain = 100
    a2 = 4500
    E2 = 0.6
    n1 = np.sqrt((k*(M1 + M2))/ain ** 3)
    tk = 1/n1*((M1 + M2)/M3)*(a2/ain)**3*(1 - E2**2)**(3.0/2)

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
        w1=0,
        w2=0,
        W=0
        # w1=np.random.rand() * 2 * np.pi,
        # w2=np.random.rand() * 2 * np.pi,
        # W=np.random.rand() * 2 * np.pi,
    )
    evec = ret.y[3:6, :]
    evec_mags = np.sqrt(np.sum(evec**2, axis=0))

    # extract emaxes by looking in windows where de/dt is small. assume emaxes
    # are well separated by > 0.1 * tk
    ts = []
    emaxes = []
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
        if emax < 0.3: # eliminate minima from calculation
            continue
        ts.append(ret.t[emax_idx])
        emaxes.append(emax)
        blockstartidx = next_t_idx
    print('Ran for', idx, q)
    return np.array(ts), np.array(emaxes)

def plot_emax_dq(I0=93.5, tf=1e9, num_reps=10,
                 fn='q_sweep_935'):
    folder = '1emax_q'
    mkdirp(folder)
    p = Pool(10)

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
    q_arr = list(axmap.keys())

    eps = get_eps(25, 25, 30, 100, 4500, 0.6)
    # my epsilons above assume e2 = 0
    elim = get_elim(eps[3] / np.sqrt(1 - 0.6**2),
                    eps[1] * (1 - 0.6**2)**(3/2))

    filename = folder + '/' + fn
    pkl_fn = filename + '.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)

        # just run 1.0 once
        args = [(q, I0, tf) for q in q_arr[ :-1]]
        args_full = [(idx, *args[idx // num_reps])
                     for idx in range(num_reps * len(args))]
        args_full.append((-1, 1.0, I0, tf))
        ret = p.starmap(get_emax_series, args_full)

        q_full = np.repeat(q_arr, num_reps)
        ts = [k[0] for k in ret]
        emax_arr = [k[1] for k in ret]
        dat = (q_full, ts, emax_arr)
        with open(pkl_fn, 'wb') as f:
            pickle.dump(dat, f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            dat = pickle.load(f)

    colors = ['k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan']
    num_plotted = {q:0 for q in q_arr}
    for q, t, emaxes in zip(*dat):
        ax = axmap[q]
        idx = num_plotted[q]
        if q == 1.0:
            if idx == 0:
                emax_quad = np.mean(emaxes)
            elif idx > 0:
                continue # do not plot q=1.0 more than once
        num_plotted[q] += 1
        c = colors[idx % len(colors)]
        ax.semilogy(t / tf * 10, 1 - emaxes,
                    c=c, marker='o', ms=2.5, lw=0, ls='')
        ax.plot(t / tf * 10, 1 - emaxes,
                c=c, ls=':', lw=0.7)
    for q, ax in axmap.items():
        ax.axhline(1 - elim, c='k', ls='--', lw=1.0)
        ax.axhline(1 - emax_quad, c='b', ls='--', lw=1.0)
    # text last, after lims are set
    for q, ax in axmap.items():
        ax.text(ax.get_xlim()[0] + 0.2, ax.get_ylim()[1] / 3, 'q=%.1f' % q)
    axs[0].set_ylabel(r'$1 - e_{\max}$')
    axs[2].set_ylabel(r'$1 - e_{\max}$')
    axs[4].set_ylabel(r'$1 - e_{\max}$')
    axs[4].set_xlabel(r'$t$ ($10^{%d}$ Gyr)' % (np.round(np.log10(tf)) - 1))
    axs[5].set_xlabel(r'$t$ ($10^{%d}$ Gyr)' % (np.round(np.log10(tf)) - 1))

    plt.suptitle(r'$I_{\rm tot} = %.1f$' % I0, fontsize=20)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.03, wspace=0.03)
    plt.savefig(filename, dpi=300)
    plt.close()

    # do it again, but hist the emaxes

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
    for q, t, emaxes in zip(*dat):
        ax = axmap[q]
        idx = num_plotted[q]
        if q == 1.0 and len(hist_vals[q]) > 0:
            continue
        hist_vals[q].extend(np.log10(1 - emaxes))

    # use global hist bins
    _, bin_edges = np.histogram([v for x in hist_vals.values() for v in x],
                                bins=100)
    for q, ax in axmap.items():
        ax.hist(hist_vals[q], bins=bin_edges)
        ax.axvline(np.log10(1 - elim), c='k', ls='--', lw=1.0)
        ax.axvline(np.log10(1 - emax_quad), c='b', ls='--', lw=1.0)
    for q, ax in axmap.items():
        ax.text(ax.get_xlim()[1] - 0.45, ax.get_ylim()[1] * 0.9, 'q=%.1f' % q)
    axs[4].set_xlabel(r'$\log_{10}(1 - e_{\max})$')
    axs[5].set_xlabel(r'$\log_{10}(1 - e_{\max})$')
    plt.suptitle(r'$I_{\rm tot} = %.1f$' % I0, fontsize=20)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.03, wspace=0.03)
    plt.savefig(filename + 'hist', dpi=300)
    plt.close()

if __name__ == '__main__':
    # timing_tests()

    # sweep(folder='1sweepbin', func=sweeper_bin, nthreads=50)
    # sweep(nthreads=50)

    # sweeper_comp(nthreads=4, nruns=10000)
    plot_emax_dq(I0=93, fn='q_sweep93')
    plot_emax_dq()
    plot_emax_dq(I0=95, fn='q_sweep_95')
    plot_emax_dq(I0=96.5, fn='q_sweep_965')
    plot_emax_dq(I0=97, fn='q_sweep_97')
    plot_emax_dq(I0=99, fn='q_sweep_99')

    # testing elim calculation
    # emaxes = get_emax_series(0, 1, 92.8146, 2e7)[1]
    # print(1 - np.mean(emaxes))
    pass
