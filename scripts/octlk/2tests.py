'''
a few test model explorations:

* check whether Hamiltonian can give merger boundaries

* make plot of "single shot merger" region of parameter space
    - overlay the DA/SA transition as well
    - overlay the effective merger reion?
'''
from utils import *
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=3.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

import scipy.optimize as opt

# For H, we take equations from Naoz et al 2013, with C2 = 1, C3 = eps_naoz
def H_quad(e1, w1, Itot):
    return (
        (2 + 3 * e1**2) * (3 * np.cos(Itot)**2 - 1)
        + 15 * e1**2 * np.sin(Itot)**2 * np.cos(2 * w1)
    )
def H_oct(e1, w1, e2, w2, Itot):
    cosphi = -np.cos(w1) * np.cos(w2) - np.cos(Itot) * np.sin(w1) * np.sin(w2)
    B = 2 + 5 * e1**2 - 7 * e1**2 * np.cos(2 * w1)
    A = 4 + 3 * e1**2 - 5 * B * np.sin(Itot)**2 / 2
    return e1 * (
        A * cosphi
        + 10 * np.cos(Itot) * np.sin(Itot)**2
            * (1 - e1**2) * np.sin(w1) * np.sin(w2)
    )
def H_full(e1, w1, I1, e2, w2, I2, eps_naoz, eps_quad=1):
    Itot = I1 + I2
    return eps_quad * H_quad(e1, w1, Itot) + eps_naoz * H_oct(e1, w1, e2, w2, Itot)

def eval_H(e1, w1, I1, w2, eta0, ltot_i, eps_naoz, eps_quad=1):
    e2, I2 = get_eI2(e1, I1, eta0, ltot_i)
    return H_full(e1, w1, I1, e2, w2, I2, eps_naoz, eps_quad)

def get_e1(Itot, eta, e1base, Ilim):
    ''' find an e1 such that K(Itot, e1) = Kbase '''
    def get_K(e, I):
        return (
            np.sqrt(1 - e**2) * np.cos(I)
                - eta * e**2 / 2
        )
    opt_func = lambda e1: get_K(e1, Itot) - get_K(e1base, Ilim)
    e1 = opt.brenth(opt_func, 1e-3, 1)
    return e1

def get_H(e20, Itot0, eta0, e1=1e-3, **kwargs):
    eta = eta0 / np.sqrt(1 - e20**2)
    I1 = np.radians(get_I1(Itot0, eta))
    ltot_i = ltot(e1, I1, e20, eta0)
    return eval_H(e1, 0, I1, 0, eta0, ltot_i, **kwargs)

def calculate_elim_regions():
    # LML15, fig 7
    m1 = 0
    m2 = 1
    m3 = 0.04
    a0 = 6
    a2 = 100
    e10 = 1e-3

    e2_vals = np.linspace(0, 0.57, 20)
    Icrits = []
    for idx, e2 in enumerate(e2_vals):
        eps_gw, eps_gr, eps_oct, eta0 = get_eps_eta0(0, 1, 0.04, 6, 100, e2)
        eta = eta0 / np.sqrt(1 - e2**2)
        Ilim = np.radians(get_Ilim(eta, eps_gr)) # starting e of 0
        Hlim = get_H(e2, Ilim, eta0, e1=0, eps_naoz=0)
        _, hoct_max = minimize_hoct(e2)
        eps_oct = a0 / a2 * e2 / (1 - e2**2)
        print(e2, eps_oct, hoct_max)
        def opt_func(I_test):
            return (
                get_H(e2, I_test, eta0, e1=0, eps_naoz=0) - Hlim
                - hoct_max * 15 / 4 * eps_oct
            )
        Icrit = opt.brenth(opt_func, np.radians(50), np.radians(90))
        Icrits.append(Icrit)

        fig = plt.figure(figsize=(6, 6))
        I_plot = np.radians(np.linspace(50, 130, 100))
        H_plot = []
        for I in I_plot:
            H_plot.append(get_H(e2, I, eta0, e1=0, eps_naoz=0))
            # H_plot.append(get_H(e2, I, eta0, e1=1e-2, eps_quad=0, eps_naoz=1))
        plt.plot(np.degrees(I_plot), H_plot)
        # plt.axhline(Hlim + hoct_max * 15 / 4 * eps_oct)
        plt.savefig('/tmp/foo' + str(idx))
        plt.close()

    # fig = plt.figure(figsize=(6, 6))
    # eps_oct_plot = a0 / a2 * e2_vals / (1 - e2_vals**2)
    # plt.plot(eps_oct_plot, np.degrees(Icrits), label='Me')
    # plt.plot(eps_oct_plot, np.degrees(np.arccos(np.sqrt(
    #     0.26 * (eps_oct_plot / 0.1)
    #     - 0.536 * (eps_oct_plot / 0.1)**2
    #     + 12.05 * (eps_oct_plot / 0.1)**3
    #     -16.78 * (eps_oct_plot / 0.1)**4
    # ))), label='MLL16')
    # plt.legend(fontsize=14)
    # plt.xlabel(r'$\epsilon_{\rm oct}$')
    # plt.ylabel(r'$I_{0, \lim}$ (Deg)')
    # plt.savefig('2_ilim_eta0', dpi=300)
    # plt.close()

def minimize_hoct(e2=0.6, to_print=False):
    '''
    sanity check, minimum and maximum at:
    '''
    # -sin^3(I) + 2 * sin * cos^2(I) = 0
    # 2 * cos^2(I) - sin^2(I) = 0
    def objective_func(x, sign=+1):
        w1, w2, Itot = x
        return sign * H_oct(1, w1, e2, w2, Itot)
    Itotmin = np.arccos(np.sqrt(3/5))
    ret = opt.minimize(
        objective_func,
        (0.1, 0.5, np.pi / 3),
        bounds=[
            (0, 2 * np.pi),
            (0, 2 * np.pi),
            (Itotmin, np.pi / 2),
        ])
    ret_neg = opt.minimize(
        objective_func,
        (0.1, 0.5, np.pi / 3),
        args=(-1),
        bounds=[
            (0, 2 * np.pi),
            (0, 2 * np.pi),
            (Itotmin, np.pi / 2),
        ])
    if to_print:
        print(ret.x, ret.fun)
        print(ret_neg.x, -ret_neg.fun)
    return ret.fun, -ret_neg.fun # minimum/maximum

def run_one_cycle(q, M12, plot=False, num_periods=1, mm=1e-5, **kwargs):
    ''' run one cycle, eps_oct = eps_gr = 0 '''
    M1 = M12 / (1 + q)
    M2 = M12 - M1
    M3 = kwargs.get('M3', 30)
    ain = kwargs.get('a0', 100)
    a2 = kwargs.get('a2', 4500)
    E2 = kwargs.get('e2', 0.6)
    n1 = np.sqrt((k*(M1 + M2))/ain ** 3)
    tk = 1/n1*((M1 + M2)/M3)*(a2/ain)**3*(1 - E2**2)**(3.0/2)
    T = 100 * tk
    kwargs['ll'] = 0 # eps_gw
    kwargs['mm'] = mm # eps_oct
    ret = run_vec(M1=M1, M2=M2, T=T, **kwargs)

    Lin_vec = ret.y[ :3]
    ein_vec = ret.y[3:6]
    Lout_vec = ret.y[6:9]
    eout_vec = ret.y[9:12]
    Lin_mag = np.sqrt(np.sum(Lin_vec**2, axis=0))
    ein = np.sqrt(np.sum(ein_vec**2, axis=0))
    Lout_mag = np.sqrt(np.sum(Lout_vec**2, axis=0))
    eout = np.sqrt(np.sum(eout_vec**2, axis=0))
    I = np.arccos(Lin_vec[2] / Lin_mag)
    Iout = np.arccos(Lout_vec[2] / Lout_mag)
    w = np.arcsin(reg(ein_vec[2] / (ein * np.sin(I))))
    Linxy_x = Lin_vec[0] / (Lin_mag * np.sin(I))
    Linxy_y = Lin_vec[1] / (Lin_mag * np.sin(I))
    W = np.unwrap(np.arctan2(Linxy_y, Linxy_x))

    Ie = np.arccos(ein_vec[2] / ein)
    einxy_x = ein_vec[0] / (ein * np.sin(Ie))
    einxy_y = ein_vec[1] / (ein * np.sin(Ie))
    We = np.unwrap(np.arctan2(einxy_y, einxy_x))

    de = np.gradient(ein)
    emax_idxs = np.where(np.logical_and(
        ein > 0.9 * np.max(ein),
        abs(de) < 1e-4
    ))[0]
    dt = np.diff(ret.t[emax_idxs])
    gaps = np.where(dt > 0.1 * tk)[0]
    emax_right_tentative = emax_idxs[gaps[num_periods - 1] + 1]
    emax_left = np.argmax(ein[ :emax_right_tentative])
    right_search = (
        emax_idxs[-1] if len(gaps) <= num_periods
        else emax_idxs[gaps[num_periods] + 1]
    )
    emax_right = (
        np.argmax(ein[emax_right_tentative:right_search])
        + emax_right_tentative
    )

    eta = Lin_mag[0] / (Lout_mag * np.sqrt(1 - ein[0]**2))
    K = np.sqrt(1 - ein**2) * np.cos(I + Iout) - eta * ein**2 / 2

    # relative angles
    n2hat = Lout_vec / Lout_mag
    u2hat = eout_vec / eout
    v2hat = ts_cross(n2hat, u2hat)
    We_rel = np.arctan2(ts_dot(ein_vec, v2hat), ts_dot(ein_vec, u2hat))

    if plot:
        plot_slice = np.s_[emax_left:emax_right]
        # plot_slice = np.s_[::]
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(
            3, 3,
            figsize=(9, 9),
            sharex=True)
        ax1.plot(ret.t[plot_slice] / tk,
                 np.degrees(w[plot_slice]))
        ax2.semilogy(ret.t[plot_slice] / tk,
                     1 - ein[plot_slice])
        ax3.plot(ret.t[plot_slice] / tk,
                     np.degrees(I[plot_slice]))
        ax4.plot(ret.t[plot_slice] / tk,
                 np.degrees(W[plot_slice]))
        ax5.plot(ret.t[plot_slice] / tk,
                 np.degrees(We[plot_slice]))
        ax6.plot(ret.t[plot_slice] / tk,
                 np.degrees(We_rel[plot_slice]))
        ax7.plot(ret.t[plot_slice] / tk,
                 K[plot_slice])
        ax1.set_ylabel(r'$\omega$')
        ax2.set_ylabel(r'$1 - e$')
        ax3.set_ylabel(r'$I$')
        ax4.set_ylabel(r'$\Omega$')
        ax5.set_ylabel(r'$\Omega_e$')
        ax6.set_ylabel(r'$\Omega_e$ (Rel)')
        ax7.set_ylabel(r'$K$')
        plt.tight_layout()
        plt.savefig('/tmp/foo')
        plt.close()

    dW = np.degrees(W[emax_right] - W[emax_left])
    dw = np.degrees(w[emax_right] - w[emax_left])
    dWe = np.degrees(We[emax_right] - We[emax_left])
    dWe_rel = np.degrees(We_rel[emax_right] - We_rel[emax_left])
    dK = K[emax_right] - K[emax_left]
    return dw, dW, dWe, dWe_rel, dK

def re_angle(arr):
    return (arr + 540) % 360 - 180

def run_sweeps(q=0.2, M12=50, base_fn='2_dWsweeps6_2', e2=0.6, Ivert=90, N=1000,
               mm=1e-5):
    folder = '2dW_sweeps'
    mkdirp(folder)
    I0ds = np.linspace(80, 100, N)
    pkl_fn = '%s/%s.pkl' % (folder, base_fn)
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        lib_vals = []
        circ_vals = []
        for I0d in I0ds:
            circ_val = run_one_cycle(q, M12, w1=0, mm=mm, Itot=I0d, E20=e2,
                                     num_periods=1)
            circ_vals.append(circ_val[1: ])
            lib_val = run_one_cycle(q, M12, w1=np.pi / 2, mm=mm,
                                    Itot=I0d, E20=e2, num_periods=1)
            lib_vals.append(lib_val[1: ])
            if mm == 0 and not abs(abs(circ_val[0]) - 180) < 30:
                print('Circ', I0d, circ_val[0])
            if mm == 0 and not abs(lib_val[0]) < 30:
                print('Lib', I0d, lib_val[0])
        circ_vals = np.array(circ_vals)
        lib_vals = np.array(lib_vals)
        with open(pkl_fn, 'wb') as f:
            pickle.dump((circ_vals, lib_vals), f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            circ_vals, lib_vals = pickle.load(f)
    dWs_circ, dWes_circ, dWe_rels_circ, dKs_circ = circ_vals.T
    dWs_lib, dWes_lib, dWe_rels_lib, dKs_lib = lib_vals.T
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2,
        figsize=(8, 8),
        sharex=True)

    circ_plot_idxs = np.where(np.logical_and(
        dWs_circ != -1,
        dKs_circ > -0.5,
    ))[0]
    lib_plot_idxs = np.where(np.logical_and(
        dWs_lib != -1,
        dKs_lib > -0.5,
    ))[0]
    ax1.plot(I0ds[lib_plot_idxs], re_angle(dWs_lib[lib_plot_idxs]), 'go',
             label=r'$\omega_{1,0} = \pi / 2$', ms=0.8)
    ax1.plot(I0ds[circ_plot_idxs], re_angle(dWs_circ[circ_plot_idxs]), 'ro',
             label=r'$\omega_{1,0} = 0$', ms=0.8)
    ax2.plot(I0ds[lib_plot_idxs], re_angle(dWes_lib[lib_plot_idxs]), 'go',
             ms=0.8)
    ax2.plot(I0ds[circ_plot_idxs], re_angle(dWes_circ[circ_plot_idxs]), 'ro',
             ms=0.8)

    ax3.plot(I0ds[lib_plot_idxs], re_angle(dWe_rels_lib[lib_plot_idxs]),
             'go', ms=0.8)
    ax3.plot(I0ds[circ_plot_idxs], re_angle(dWe_rels_circ[circ_plot_idxs]),
             'ro', ms=0.8)
    # eps_oct scaled by 1e-5
    if mm == 0:
        mm = 1
    ax4.plot(I0ds[lib_plot_idxs], 100 / mm * dKs_lib[lib_plot_idxs], 'go',
             ms=0.8, label=r'$\omega_{1,0} = \pi / 2$')
    ax4.plot(I0ds[circ_plot_idxs], 100 / mm * dKs_circ[circ_plot_idxs], 'ro',
             ms=0.8, label=r'$\omega_{1,0} = 0$')

    ax4.legend(fontsize=12)

    tic_locs = [-180, 0, 180]
    for ax in [ax2, ax3]:
        ax.set_yticks(tic_locs)
        ax.set_yticklabels([str(t) for t in tic_locs])
        for loc in tic_locs:
            ax.axhline(loc, c='k', lw=0.7)
    ax4.axhline(0, c='k', lw=0.7)

    ax1.axvline(Ivert, c='k', lw=0.7)
    ax2.axvline(Ivert, c='k', lw=0.7)
    ax3.axvline(Ivert, c='k', lw=0.7)
    ax4.axvline(Ivert, c='k', lw=0.7)

    ax3.set_xlabel(r'$I_0$ (Deg)')
    ax4.set_xlabel(r'$I_0$ (Deg)')
    ax1.set_ylabel(r'$\Delta \Omega$')
    ax2.set_ylabel(r'$\Delta \Omega_e$')
    ax3.set_ylabel(r'$\Delta \Omega_e$ (Rel)')
    ax4.set_ylabel(r'$100 \times K$')
    if mm == 1e-5:
        ax4.set_ylim(-0.1, 0.1)
    else:
        ax4.set_ylim(-1, 1)

    ax2.yaxis.set_label_position('right')
    ax4.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax4.yaxis.tick_right()

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.02, wspace=0.02)
    plt.savefig('%s/%s' % (folder, base_fn), dpi=300)
    plt.close()

def plot_dW_sweeps(q, base_fn, Ivert, N=1000):
    folder = '2dW_sweeps'
    I0ds = np.linspace(80, 100, N)
    pkl_fn = '%s/%s.pkl' % (folder, base_fn)
    pkl_fn1 = '%s/%s_1.pkl' % (folder, base_fn)
    with open(pkl_fn, 'rb') as f:
        circ_vals_eps0, lib_vals_eps0 = pickle.load(f)
    with open(pkl_fn1, 'rb') as f:
        circ_vals_eps1, lib_vals_eps1 = pickle.load(f)
    dWes_circ0 = re_angle(circ_vals_eps0[ :,1])
    dWes_lib0 = re_angle(lib_vals_eps0[ :,1])
    dWes_circ1 = re_angle(circ_vals_eps1[ :,1])
    dWes_lib1 = re_angle(lib_vals_eps1[ :,1])
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(7, 7),
        sharex=True, sharey=True
    )
    ax1.plot(I0ds, dWes_circ0, 'go', label=r'$\omega_0 = 0$',
             ms=1.0)
    ax1.plot(I0ds, dWes_lib0, 'ro', label=r'$\omega_0 = \pi / 2$',
             ms=1.0)
    ax2.plot(I0ds, dWes_circ1, 'go', label=r'$\omega_0 = 0$',
             ms=1.0)
    ax2.plot(I0ds, dWes_lib1, 'ro', label=r'$\omega_0 = \pi / 2$',
             ms=1.0)
    ax1.legend(fontsize=16)
    ax1.axvline(Ivert, c='k', lw=2.0)
    ax2.axvline(Ivert, c='k', lw=2.0)
    _, eps_gr, _, eta = get_eps(10, 40, 50, 100, 4500, 0.6)
    Ilimd = get_Ilim(eta, eps_gr)
    ax1.axvline(Ilimd, c='k', lw=2.0)
    ax2.axvline(Ilimd, c='k', lw=2.0)
    ax1.set_yticks([-180, 0, 180])
    ax1.set_yticklabels(['-180', '0', '180'])
    ax1.set_ylabel(r'$\Delta \Omega_e$ (mod $360^\circ$)')
    ax2.set_xlabel(r'$I_0$ (Deg)')
    ax1.grid(True, axis='y')
    ax2.grid(True, axis='y')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    plt_fn = '%s/%s_dual' % (folder, base_fn)
    print('Saving', plt_fn)
    plt.savefig(plt_fn, dpi=300)
    plt.close()

def plot_H(eta=0.1):
    K0 = np.cos(np.radians(88))

    fig = plt.figure(figsize=(8, 8))
    n_pts = 100
    emax = 1 - 1e-3
    log_neg_e = np.linspace(0, np.log10(1 - emax) * 1.05, n_pts)
    omega = np.linspace(0, np.pi, n_pts)
    log_neg_e_grid = np.outer(log_neg_e, np.ones_like(omega))
    omega_grid = np.outer(np.ones_like(log_neg_e), omega)
    e_grid = 1 - (10**log_neg_e_grid)
    # e_grid = np.outer(np.linspace(0, 0.01, n_pts), np.ones_like(omega))

    # K = j * np.cos(I) - eta * e**2 / 2
    # cos(I) = (K0 + eta * e**2 / 2) / j
    I_grid = np.arccos(reg((K0 + eta * e_grid**2 / 2)/ np.sqrt(1 - e_grid**2)))
    H_vals = H_quad(e_grid, omega_grid, I_grid)
    H0 = H_quad(0, 0, np.arccos(K0))
    plt.contour(omega_grid, log_neg_e_grid, H_vals, cmap='RdBu_r',
                levels=10, linewidths=1.0)
    plt.contour(omega_grid, log_neg_e_grid, H_vals, colors='k',
                levels=[H0], linewidths=2.0)
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\log_{10} (1 - e)$')
    plt.savefig('2H_contour', dpi=200)
    plt.close()

if __name__ == '__main__':
    # calculate_elim_regions()
    # minimize_hoct(0.8, to_print=True)
    # for w1 in np.linspace(0, np.pi, 10):
    #     dw = run_one_cycle(2/3, 50, w1=w1)[0]
    #     print(w1, dw)
    # print(run_one_cycle(2/3, 50, plot=True, w1=np.pi))

    # plot_H()

    N = 1000
    run_sweeps(q=0.2, base_fn='2_dWsweeps6_2', Ivert=89.7997997997998, mm=0,
               N=N)
    run_sweeps(q=0.2, base_fn='2_dWsweeps6_2_1', Ivert=89.7997997997998, mm=1,
               N=N)
    plot_dW_sweeps(q=0.2, base_fn='2_dWsweeps6_2', Ivert=88.31663, N=N)

    # run_sweeps(q=0.3, base_fn='2_dWsweeps6_3', Ivert=88.1981981981982, mm=0)
    # run_sweeps(q=0.3, base_fn='2_dWsweeps6_3_1', Ivert=88.1981981981982, mm=1)
    # plot_dW_sweeps(q=0.3, base_fn='2_dWsweeps6_3', Ivert=88.1981981981982)

    # run_sweeps(q=0.4, base_fn='2_dWsweeps6_4', Ivert=87.87787787787788, mm=0)
    # run_sweeps(q=0.4, base_fn='2_dWsweeps6_4_1', Ivert=87.87787787787788, mm=1)
    # plot_dW_sweeps(q=0.4, base_fn='2_dWsweeps6_4', Ivert=87.87787787787788)

    # run_sweeps(q=0.5, base_fn='2_dWsweeps6_5', Ivert=87.47747747747748, mm=0)
    # run_sweeps(q=0.5, base_fn='2_dWsweeps6_5_1', Ivert=87.47747747747748, mm=1)
    # plot_dW_sweeps(q=0.5, base_fn='2_dWsweeps6_5', Ivert=87.47747747747748)

    # run_sweeps(q=0.7, base_fn='2_dWsweeps6_7', Ivert=87.31731731731732, mm=0)
    # run_sweeps(q=0.7, base_fn='2_dWsweeps6_7_1', Ivert=87.31731731731732, mm=1)
    # plot_dW_sweeps(q=0.7, base_fn='2_dWsweeps6_7', Ivert=87.31731731731732)
    pass
