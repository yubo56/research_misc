'''
does Kozai simulations in orbital elements, then computes spin evolution after
the fact w/ a given trajectory
'''
import pickle
import os
from collections import defaultdict

from multiprocessing import Pool
import numpy as np
import matplotlib
try:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    plt.rc('lines', lw=1.5)
    plt.rc('xtick', direction='in', top=True, bottom=True)
    plt.rc('ytick', direction='in', left=True, right=True)
except: # let it fail later
    pass
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brenth
from utils import *

N_THREADS = 40
m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)

def get_fn_I(I_deg):
    return ('%.3f' % I_deg).replace('.', '_')

def get_kozai(folder, I_deg, getter_kwargs,
              a0=1, e0=1e-3, W0=0, w0=0, tf=np.inf, af=0,
              pkl_template='4sim_lk_%s.pkl', save=True, **kwargs):
    mkdirp(folder)
    I0 = np.radians(I_deg)
    pkl_fn = folder + pkl_template % get_fn_I(I_deg)

    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        eps_gr = getter_kwargs['eps_gr']
        eps_gw = getter_kwargs['eps_gw']
        def dydt(t, y):
            a, e, W, I, w = y
            x = 1 - e**2
            dadt =  (
                -eps_gw * (64 * (1 + 73 * e**2 / 24 + 37 * e**4 / 96)) / (
                    5 * a**3 * x**(7/2))
            )
            dedt = (
                15 * a**(3/2) * e * np.sqrt(x) * np.sin(2 * w)
                        * np.sin(I)**2 / 8
                    - eps_gw * 304 * e * (1 + 121 * e**2 / 304)
                        / (15 * a**4 * x**(5/2))
            )
            dWdt = (
                3 * a**(3/2) * np.cos(I) *
                        (5 * e**2 * np.cos(w)**2 - 4 * e**2 - 1)
                    / (4 * np.sqrt(x))
            )
            dIdt = (
                -15 * a**(3/2) * e**2 * np.sin(2 * w)
                    * np.sin(2 * I) / (16 * np.sqrt(x))
            )
            dwdt = (
                3 * a**(3/2)
                    * (2 * x + 5 * np.sin(w)**2 * (e**2 - np.sin(I)**2))
                    / (4 * np.sqrt(x))
                + eps_gr / (a**(5/2) * x)
            )
            return (dadt, dedt, dWdt, dIdt, dwdt)
        y0 = (a0, e0, W0, I0, w0)

        peak_event = lambda t, y: (y[4] % np.pi) - (np.pi / 2)
        peak_event.direction = +1 # only when w is increasing
        a_term_event = lambda t, y: y[0] - af
        a_term_event.terminal = True
        events = [peak_event, a_term_event]
        ret = solve_ivp(dydt, (0, tf), y0, events=events, **kwargs)
        t, y, t_events = ret.t, ret.y, ret.t_events
        print('Finished for I=%.3f, took %d steps, t_f %.3f (%d cycles)' %
              (I_deg, len(t), t[-1], len(t_events[0])))

        if save:
            with open(pkl_fn, 'wb') as f:
                pickle.dump((t, y, t_events), f)
    else:
        print('Loading %s' % pkl_fn)
        with open(pkl_fn, 'rb') as f:
            t, y, t_events = pickle.load(f)
    return t, y, t_events

def get_spins_inertial(folder, I_deg, ret_lk, getter_kwargs,
                       q_sl0=0, # NB: for backcompat, but q_sb0 = I - q_sl0
                       phi_sb=0,
                       pkl_template='4sim_s_%s.pkl',
                       save=True,
                       **kwargs):
    ''' uses the same times as ret_lk '''
    mkdirp(folder)
    pkl_fn = folder + pkl_template % get_fn_I(I_deg)
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        t_lk, y, _ = ret_lk
        a_arr, e_arr, W, I, _ = y
        eps_sl = getter_kwargs['eps_sl']
        Lx = interp1d(t_lk, np.sin(I) * np.cos(W))
        Ly = interp1d(t_lk, np.sin(I) * np.sin(W))
        Lz = interp1d(t_lk, np.cos(I))
        a = interp1d(t_lk, a_arr)
        e = interp1d(t_lk, e_arr)
        def dydt(t, s):
            # apparently not guaranteed, see
            # https://github.com/scipy/scipy/issues/9198
            if t > t_lk[-1]:
                return None
            Lhat = [Lx(t), Ly(t), Lz(t)]
            return eps_sl * np.cross(Lhat, s) / (
                a(t)**(5/2) * (1 - e(t)**2))
        t0 = t_lk[0]

        # initial spin (for q_sl0=0, phi_sb0=0, equals L[0])
        q_sb0 = np.radians(I_deg) - q_sl0
        s_rot = [np.sin(q_sb0) * np.cos(phi_sb),
                 np.sin(q_sb0) * np.sin(phi_sb),
                 np.cos(q_sb0)]

        ret = solve_ivp(dydt, (t0, t_lk[-1]), s_rot, dense_output=True, **kwargs)
        y = ret.sol(t_lk)
        print('Finished spins for I=%.3f, took %d steps' %
              (I_deg, len(ret.t)))

        if save:
            with open(pkl_fn, 'wb') as f:
                pickle.dump(y, f)
    else:
        print('Loading %s' % pkl_fn)
        with open(pkl_fn, 'rb') as f:
            y = pickle.load(f)
    return y

def plot_all(folder, ret_lk, s_vec, getter_kwargs,
             fn_template='4sim_%s', time_slice=np.s_[::],
             **kwargs):
    mkdirp(folder)
    lk_t, lk_y, lk_events = ret_lk
    if s_vec is None:
        s_vec = np.array(
            [np.zeros_like(lk_t),
             np.zeros_like(lk_t),
             np.ones_like(lk_t)])
    s_vec = s_vec[:, time_slice]
    sx, sy, sz = s_vec
    fig, axs_orig = plt.subplots(3, 5, figsize=(16, 9))
    axs = np.reshape(axs_orig, np.size(axs_orig))
    a, e, W, I, w = lk_y[:, time_slice]
    t = lk_t[time_slice]
    I0 = np.degrees(I[0])

    K = np.sqrt(1 - e**2) * np.cos(I)
    Lhat = get_hat(W, I)
    Lout_hat = get_hat(0 * I, 0 * I)
    q_sl = np.arccos(ts_dot(Lhat, s_vec))
    q_sb = np.arccos(reg(sz))
    Wsl = getter_kwargs['eps_sl'] / (a**(5/2) * (1 - e**2))
    Wdot_eff = (
        3 * a**(3/2) / 4 * np.cos(I) * (4 * e**2 + 1) / np.sqrt(1 - e**2))
    Wdot = (
        3 * a**(3/2) / 4 * np.cos(I) * (
            5 * e**2 * np.cos(w)**2 - 4 * e**2 - 1) / np.sqrt(1 - e**2))
    Idot = (-15 * a**(3/2) * e**2 * np.sin(2 * w)
                    * np.sin(2 * I) / (16 * np.sqrt(1 - e**2)))
    # A = np.abs(Wsl / Wdot)
    # A_eff = np.abs(Wsl / Wdot_eff)

    # use the averaged Wdot, else too fluctuating
    W_eff_bin = Wsl * Lhat + Wdot_eff * Lout_hat
    W_eff_bin /= np.sqrt(np.sum(W_eff_bin**2, axis=0))
    q_eff_bin = np.degrees(np.arccos(ts_dot(W_eff_bin, s_vec)))

    # oh boy... (edit: same as Weff)
    # def get_opt_func(Wsl, Wdot, I):
    #     return lambda Io: Wsl * np.sin(I - Io) - Wdot * np.sin(Io)
    # Iouts = []
    # for Wsl_val, Wdot_val, I_val in zip(Wsl, Wdot_eff, I):
    #     Iout_val = brenth(
    #         get_opt_func(Wsl_val, Wdot_val, I_val),
    #         0,
    #         np.pi)
    #     Iouts.append(Iout_val)
    # Iouts = np.array(Iouts)
    # W_eff_me = get_hat(W, Iouts)
    # q_eff_me = np.degrees(np.arccos(ts_dot(W_eff_me, s_vec)))
    cross = ts_cross(Lout_hat, Lhat)
    W_eff_tot = Wsl * Lhat - Wdot * Lout_hat - Idot * cross / np.sin(I)
    W_eff_hat = W_eff_tot / np.sqrt(np.sum(W_eff_tot**2, axis=0))
    q_eff_me = np.degrees(np.arccos(ts_dot(W_eff_hat, s_vec)))

    # computing two resonant angles
    phi_idxs = np.where(abs(np.sin(q_sl)) > 0)[0]
    yhat = cross / np.sin(I)
    sin_phisl = (
        ts_dot(s_vec[:, phi_idxs], yhat[:, phi_idxs]) / np.sin(q_sl[phi_idxs])
    )
    xhat = ts_cross(yhat, Lhat)
    cos_phisl = (
        ts_dot(s_vec[:, phi_idxs], xhat[:, phi_idxs]) / np.sin(q_sl[phi_idxs])
    )
    phi_sl = np.arctan2(sin_phisl, cos_phisl)
    phi_sb = np.arctan2(sy, sx)
    dphi_sb_dt = np.gradient(phi_sb) / np.gradient(t)
    lk_period = np.gradient(lk_events[0])
    lk_p_interp = interp1d(lk_events[0], lk_period)

    # computing effective angle to N = 0 axis
    _, dWsl, dWdot, t_lkmids, dWslx, dWslz = get_dWs(ret_lk, getter_kwargs)
    eff_idx = np.where(np.logical_and(t < t_lkmids[-1], t > t_lkmids[0]))[0]
    t_eff = t[eff_idx]
    Wslx = interp1d(t_lkmids, dWslx)(t_eff)
    Wslz = interp1d(t_lkmids, dWslz)(t_eff)
    Wdot = interp1d(t_lkmids, dWdot)(t_eff)
    Lhat_xy = get_hat(W[eff_idx], I[eff_idx])
    Lhat_xy[2] *= 0
    Lhat_xy /= np.sqrt(np.sum(Lhat_xy**2, axis=0))
    Weff = np.outer(np.array([0, 0, 1]), -Wdot + Wslz) + Wslx * Lhat_xy
    Weff_hat = Weff / np.sqrt(np.sum(Weff**2, axis=0))
    _q_eff0 = np.arccos(ts_dot(s_vec[:, eff_idx], Weff_hat))
    q_eff0 = np.degrees(_q_eff0)
    # predict q_eff final by averaging over an early Kozai cycle (more interp1d)
    q_eff0_interp = interp1d(t_eff, q_eff0)
    q_eff_pred = np.mean(q_eff0_interp(
        np.linspace(t_lkmids[2], t_lkmids[3], 1000)))
    # print(Weff_hat[:, 0], Weff_hat[:, -1],
    #       get_hat(W[eff_idx[-1]], I[eff_idx[-1]]))
    # compute phi as well
    yhat = ts_cross(Weff_hat, Lout_hat[:, eff_idx])
    xhat = ts_cross(yhat, Weff_hat)
    sin_phiWeff = (ts_dot(s_vec[:, eff_idx], yhat) / np.sin(_q_eff0))
    cos_phiWeff = (ts_dot(s_vec[:, eff_idx], xhat) / np.sin(_q_eff0))
    phi_Weff = np.arctan2(sin_phiWeff, cos_phiWeff)
    # f = lambda x: np.degrees(np.arccos(x))
    # print(f(np.dot(s_vec[:, 0], Weff_hat[:, 0])),
    #       f(np.dot([0, s_vec[0, 0], s_vec[2, 0]], Weff_hat[:, 0])),
    #       f(np.dot([-s_vec[0, 0], 0, s_vec[2, 0]], Weff_hat[:, 0])),
    #       )
    A = np.abs(dWsl / dWdot)

    # Plot averaged I, Iouts
    I_avg = np.arccos(Wslz / np.sqrt(Wslx**2 + Wslz**2))
    def get_Iout(W, Wsl, I):
        def Iout_constr(_Iout):
            return -W * np.sin(_Iout) + Wsl * np.sin(I + _Iout)

        if I0 > 90:
            return brenth(Iout_constr, 0, np.pi - I)
        return brenth(Iout_constr, -I, 0)
    Wsl = np.sqrt(Wslx**2 + Wslz**2)
    Iouts = [get_Iout(*args) for args in zip(Wdot, Wsl, I_avg)]

    alf=0.7
    axs[0].semilogy(t, a, 'r', alpha=alf)
    axs[0].set_ylabel(r'$a$')
    axs[1].semilogy(t, 1 - e, 'r', alpha=alf)
    axs[1].set_ylabel(r'$1 - e$')
    axs[2].plot(t, W % (2 * np.pi), 'r,', alpha=alf)
    axs[2].set_ylabel(r'$\Omega$')
    axs[3].plot(t, np.degrees(I), 'r', alpha=alf)
    axs[3].set_ylabel(r'$I$')
    axs[4].plot(t, w % (2 * np.pi), 'r,', alpha=alf)
    axs[4].set_ylabel(r'$w$')
    # axs[5].set_ylim([2 * K[0], 0])
    # axs[5].plot(t, K, 'r', alpha=alf)
    # axs[5].set_ylabel(r'$K$')
    axs[5].semilogy(t_lkmids, A, 'r', alpha=alf)
    axs[5].set_ylabel(r'$\Omega_{\rm SL,N=0} / \dot{\Omega}_{\rm N=0}$')
    axs[5].axhline(1, c='k', lw=1)
    axs[6].plot(t, np.degrees(q_sl), 'r', alpha=alf)
    axs[6].set_ylabel(r'$\theta_{\rm sl}$ ($\theta_{\rm sl,f} = %.2f$)'
                      % np.degrees(q_sl)[-1])
    axs[7].plot(t[phi_idxs], phi_sl % (2 * np.pi), 'r,', alpha=alf)
    axs[7].set_ylabel(r'$\phi_{\rm sl}$')
    # axs[8].plot(t, q_eff_me, 'r', alpha=alf)
    # axs[8].plot(t, q_eff_bin, 'k', alpha=alf)
    # axs[8].set_ylabel(r'$\left<\theta_{\rm eff, S}\right>$' +
    #                   r'($\left<\theta_{\rm eff, S, f}\right> = %.2f$)'
    #                   % q_eff_bin[-1])
    axs[8].plot(t, np.degrees(q_sb), 'r', alpha=alf)
    axs[8].set_ylabel(r'$\theta_{\rm sb}$ ($\theta_{\rm sb,i} = %.2f$)'
                      % np.degrees(q_sb)[0])
    # axs[9].plot(t[phi_idxs], (phi_sl - 2 * w[phi_idxs]) % (2 * np.pi), 'r,', alpha=alf)
    # axs[9].set_ylabel(r'$\phi_{\rm sl} - 2 \omega$')
    # interp_idx = np.where(np.logical_and(
    #     t > min(lk_events[0]), A < 1))[0]
    # axs[9].plot(t[interp_idx],
    #              dphi_sb_dt[interp_idx] / lk_p_interp(t[interp_idx]),
    #              'r',
    #              alpha=alf,
    #              lw=0.5
    #              )
    # axs[9].set_ylabel(r'$\dot{\phi}_{\rm sb} / t_{LK}$')
    # axs[9].set_ylim((-5, 1))
    axs[9].plot(t, phi_sb, 'r,', alpha=alf)
    axs[9].set_ylabel(r'$\phi_{\rm sb}$')

    axs[10].plot(t_eff, q_eff0, 'k', alpha=alf)
    axs[10].set_ylabel(r'$\theta_{N=0}$ [$%.2f (%.2f)$--$%.2f$]' %
                       (q_eff0[0], q_eff_pred, q_eff0[-1]))
    axs[10].plot(t, q_eff_bin, 'r', alpha=0.3)
    axs[11].plot(t_eff, phi_Weff, 'r,', alpha=alf)
    axs[11].set_ylabel(r'$\phi_{N=0}$')
    lk_axf = 11 # so it's hard to forget lol

    axs[12].plot(t_eff, np.degrees(I_avg), 'r')
    axs[12].set_ylabel(r'$\langle I \rangle_{LK}$')
    axs[13].plot(t_eff, np.degrees(Iouts), 'r')
    twinIout_ax = axs[13].twinx()
    Iout_dot = [(Iouts[i + 4] - Iouts[i]) / (t_eff[i + 4] - t_eff[i])
                for i in range(len(Iouts) - 4)]
    twinIout_ax.plot(t_eff[2:-2], Iout_dot, 'k', alpha=0.2)
    twinIout_ax.set_yticks([])
    print('max Iout_dot', np.max(Iout_dot))
    axs[13].set_ylabel(r'$I_{\rm out}$')
    axs[14].semilogy(t_lkmids, dWsl, 'g')
    axs[14].semilogy(t_lkmids, dWdot, 'k')
    axs[14].set_ylabel(r'$d\phi / dt$')
    # axs[14].loglog(t_lkmids[-1] - t_lkmids, dWsl * np.diff(lk_events[0]), 'g')
    # axs[14].loglog(t_lkmids[-1] - t_lkmids, dWdot * np.diff(lk_events[0]), 'k')
    # axs[14].set_ylabel(r'$\Delta \phi$')
    axs[14].set_ylim([0.01, 100])
    # NB: dtheta theory for 90.2 ~ 6.12 degrees
    # for dphi in np.linspace(0, 2 * np.pi, 10):
    #     print('Idot_num_integ', np.degrees(np.sum(
    #         Iout_dot * np.cos(dphi + phi_Weff[2:-2]) * np.gradient(t_eff[2:-2]))))
    # for ax in axs[lk_axf + 1: ]:
    #     ax.set_xlim(left=0.9 * t[-1])
    #     ax.set_xlim(220, 225)

    # scatter plots in LK phase space
    # final_idx = np.where(t < t[-1] * 0.7)[0][-1] # cut off plotting of end
    # sl = np.s_[ :final_idx]
    # axs[lk_axf + 1].scatter(w[sl] % (2 * np.pi), 1 - e[sl]**2, c=t[sl],
    #                         marker=',', alpha=0.5)
    # axs[lk_axf + 1].set_yscale('log')
    # axs[lk_axf + 1].set_ylim(min(1 - e**2), max(1 - e**2))
    # axs[lk_axf + 1].set_xlabel(r'$\omega$')
    # axs[lk_axf + 1].set_ylabel(r'$1 - e^2$')
    # axs[lk_axf + 2].scatter(w[sl] % (2 * np.pi), np.degrees(I[sl]), c=t[sl],
    #                         marker=',', alpha=0.5)
    # axs[lk_axf + 2].set_xlabel(r'$\omega$')
    # axs[lk_axf + 2].set_ylabel(r'$I$')

    # set effectively for axs[0-9], label in last
    xticks = axs[lk_axf].get_xticks()[1:-1]
    for i in range(lk_axf):
        axs[i].set_xticks(xticks)
        axs[i].set_xticklabels([])
    axs[lk_axf].set_xlabel(r'$t / t_{LK,0}$')

    plt.tight_layout()
    plt.savefig(folder + fn_template % get_fn_I(I0), dpi=200)
    plt.close()

def plot_good(folder, ret_lk, s_vec, getter_kwargs,
             fn_template='4sim_%s', time_slice=np.s_[::],
             **kwargs):
    mkdirp(folder)
    alf = 0.7

    lk_t, lk_y, lk_events = ret_lk
    if s_vec is None:
        s_vec = np.array(
            [np.zeros_like(lk_t),
             np.zeros_like(lk_t),
             np.ones_like(lk_t)])
    s_vec = s_vec[:, time_slice]
    sx, sy, sz = s_vec
    a, e, W, I, w = lk_y[:, time_slice]
    t = lk_t[time_slice]
    I0 = np.degrees(I[0])

    Lhat = get_hat(W, I)
    Lout_hat = get_hat(0 * I, 0 * I)
    Wsl = getter_kwargs['eps_sl'] / (a**(5/2) * (1 - e**2))
    Wdot_eff = (
        3 * a**(3/2) / 4 * np.cos(I) * (4 * e**2 + 1) / np.sqrt(1 - e**2))
    Idot = (-15 * a**(3/2) * e**2 * np.sin(2 * w)
                    * np.sin(2 * I) / (16 * np.sqrt(1 - e**2)))

    # use the averaged Wdot, else too fluctuating
    W_eff_bin = Wsl * Lhat + Wdot_eff * Lout_hat
    W_eff_bin /= np.sqrt(np.sum(W_eff_bin**2, axis=0))
    q_eff_bin = np.degrees(np.arccos(ts_dot(W_eff_bin, s_vec)))

    # computing effective angle to N = 0 axis
    dWtot, dWsl, dWdot, t_lkmids, dWslx, dWslz = get_dWs(ret_lk, getter_kwargs)
    eff_idx = np.where(np.logical_and(t < t_lkmids[-1], t > t_lkmids[0]))[0]
    t_eff = t[eff_idx]
    Wslx = interp1d(t_lkmids, dWslx)(t_eff)
    Wslz = interp1d(t_lkmids, dWslz)(t_eff)
    Wdot = interp1d(t_lkmids, dWdot)(t_eff)
    Lhat_xy = get_hat(W[eff_idx], I[eff_idx])
    Lhat_xy[2] *= 0
    Lhat_xy /= np.sqrt(np.sum(Lhat_xy**2, axis=0))
    Weff = np.outer(np.array([0, 0, 1]), -Wdot + Wslz) + Wslx * Lhat_xy
    Weff_hat = Weff / np.sqrt(np.sum(Weff**2, axis=0))
    _q_eff0 = np.arccos(ts_dot(s_vec[:, eff_idx], Weff_hat))
    q_eff0 = np.degrees(_q_eff0)
    # predict q_eff final by averaging over an early Kozai cycle (more interp1d)
    q_eff0_interp = interp1d(t_eff, q_eff0)
    q_eff_pred = np.mean(q_eff0_interp(
        np.linspace(t_lkmids[2], t_lkmids[3], 1000)))
    # compute phi as well
    yhat = ts_cross(Weff_hat, Lout_hat[:, eff_idx])
    xhat = ts_cross(yhat, Weff_hat)
    sin_phiWeff = (ts_dot(s_vec[:, eff_idx], yhat) / np.sin(_q_eff0))
    cos_phiWeff = (ts_dot(s_vec[:, eff_idx], xhat) / np.sin(_q_eff0))
    phi_Weff = np.arctan2(sin_phiWeff, cos_phiWeff)

    # Plot averaged I, Iouts
    I_avg = np.arccos(Wslz / np.sqrt(Wslx**2 + Wslz**2))
    def get_Iout(W, Wsl, I):
        def Iout_constr(_Iout):
            return -W * np.sin(_Iout) + Wsl * np.sin(I + _Iout)

        if I0 > 90:
            return brenth(Iout_constr, 0, np.pi - I)
        return brenth(Iout_constr, -I, 0)
    Wsl = np.sqrt(Wslx**2 + Wslz**2)
    Iouts = [get_Iout(*args) for args in zip(Wdot, Wsl, I_avg)]

    # plots
    plt.plot(t_eff, q_eff0, 'k', alpha=alf)
    plt.ylabel(r'$\theta_{N=0}$ [$\langle %.2f \rangle$--$%.2f$]' %
                   (q_eff_pred, q_eff0[-1]))
    plt.plot(t, q_eff_bin, 'r', alpha=0.3)
    plt.xlabel(r'$t / t_{\rm LK, 0}$')
    plt.savefig(folder + (fn_template % get_fn_I(I0)) + '_qN0', dpi=200)
    plt.close()

    # plt.plot(t_eff, np.degrees(I_avg), 'r')
    # plt.ylabel(r'$\langle I \rangle_{LK}$')
    plt.plot(t_eff, np.degrees(Iouts), 'r')
    plt.gca().set_ylabel(r'$I_{\rm out}$')
    plt.gca().set_xlabel(r'$t / t_{\rm LK, 0}$')
    twinIout_ax = plt.gca().twinx()
    Iout_dot = [(Iouts[i + 4] - Iouts[i]) / (t_eff[i + 4] - t_eff[i])
                for i in range(len(Iouts) - 4)]
    twinIout_ax.plot(t_eff[2:-2], np.degrees(Iout_dot), 'k', alpha=0.3)
    twinIout_ax.set_ylabel(r'$\dot{I}_{\rm out}$')
    plt.savefig(folder + (fn_template % get_fn_I(I0)) + '_Iout', dpi=200)
    plt.close()

    plt.semilogy(t_lkmids, dWsl, 'g')
    plt.semilogy(t_lkmids, dWdot, 'r')
    plt.semilogy(t_lkmids, dWtot, 'k')
    plt.ylabel(r'$d\phi / dt$')
    plt.ylim([0.01, 100])
    plt.xlabel(r'$t / t_{\rm LK, 0}$')
    plt.savefig(folder + (fn_template % get_fn_I(I0)) + '_phidots', dpi=200)
    plt.close()

def run_for_Ideg(folder, I_deg, af=5e-3,
                 atol=1e-8, rtol=1e-8, short=False, plotter=plot_all, **kwargs):
    mkdirp(folder)
    ret_lk = get_kozai(folder, I_deg, getter_kwargs,
                       af=af, atol=atol, rtol=rtol)
    s_vec = get_spins_inertial(folder, I_deg, ret_lk, getter_kwargs,
                               atol=atol,
                               rtol=rtol,
                               )
    plotter(folder, ret_lk, s_vec, getter_kwargs)
    if short:
        return

    # try with q_sl0
    s_vec = get_spins_inertial(folder, I_deg, ret_lk, getter_kwargs,
                               q_sl0=np.radians(20),
                               atol=atol,
                               rtol=rtol,
                               pkl_template='4sim_qsl20_%s.pkl')
    plotter(folder, ret_lk, s_vec, getter_kwargs,
             fn_template='4sim_qsl20_%s')

    s_vec = get_spins_inertial(folder, I_deg, ret_lk, getter_kwargs,
                               q_sl0=np.radians(40),
                               atol=atol,
                               rtol=rtol,
                               pkl_template='4sim_qsl40_%s.pkl')
    plotter(folder, ret_lk, s_vec, getter_kwargs,
             fn_template='4sim_qsl40_%s')

def get_qslfs_base(folder, I_deg, q_sl_arr, af, phi_sbs,
                   atol=1e-8, rtol=1e-8, save_pkl=True, save_png=False, **kwargs):
    ''' this is the ensemble simulation, default do not save '''
    mkdirp(folder)
    ret_lk = get_kozai(folder, I_deg, getter_kwargs,
                       af=af, atol=atol, rtol=rtol, save=save_pkl)

    qslfs_tot = []
    for phi_sb in phi_sbs:
        qslfs = []
        for q_sl0 in q_sl_arr:
            if phi_sb == 0:
                pkl_fn = '4sim_qsl' + ('%d' % np.degrees(q_sl0)) + '_%s.pkl'
            else:
                pkl_fn = '4sim_qsl' + ('%d' % np.degrees(q_sl0)) + \
                    ('_phi_sb%d' % np.degrees(phi_sb)) + '_%s.pkl'
            s_vec = get_spins_inertial(
                folder, I_deg, ret_lk, getter_kwargs,
                q_sl0=q_sl0,
                atol=atol,
                rtol=rtol,
                save=save_pkl,
                phi_sb=phi_sb,
                pkl_template=pkl_fn,
                )
            if save_png:
                fn_template = '4sim_qsl' + ('%d' % np.degrees(q_sl0)) + '_%s'
                plot_all(folder, ret_lk, s_vec, getter_kwargs,
                         fn_template=fn_template)
            _, _, W, I, _ = ret_lk[1][:, -1]
            sf = s_vec[:, -1]

            Lhat = get_hat(W, I)
            qslfs.append(np.degrees(np.arccos(np.dot(Lhat, sf))))
        qslfs_tot.append(qslfs)

    return I_deg, ret_lk[0][-1], q_sl_arr, qslfs_tot

def get_qslfs(folder, I_deg, q_sl_arr, af, **kwargs):
    I_deg, t_merge, q_sl_arr, qslfs_tot = get_qslfs_base(
        folder, I_deg, q_sl_arr, af, [0], **kwargs)
    return I_deg, t_merge, q_sl_arr, qslfs_tot[0]

def run_ensemble(folder, I_vals=np.arange(90.01, 90.4001, 0.001),
                 af=3e-3, save_fn='ensemble.pkl'):
    pkl_fn = folder + save_fn
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        mkdirp(folder)
        q_sl_arr = np.radians([
            -40, -30, -20, -10, -5, -3, -2, -1, -0.5,
            0,
            40, 30, 20, 10, 5, 3, 2, 1, 0.5,
        ])
        args = [(folder, I_deg, q_sl_arr, af) for I_deg in I_vals[::-1]]
        with Pool(N_THREADS) as p:
            res = p.starmap(get_qslfs, args)
        with open(folder + save_fn, 'wb') as f:
            pickle.dump(res, f)
    else:
        with open(folder + save_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            res = pickle.load(f)
    return res

def plot_ensemble(folder, ensemble_dat, ensemble_dat2):
    # fig, axs = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), sharex=True)
    axs = [ax]
    I_degs = []
    t_mergers = []
    qsl_dats = defaultdict(list)
    for I_deg, t_merger, q_sl_inits, q_sl_finals in ensemble_dat:
        I_degs.append(I_deg)
        t_mergers.append(t_merger)

        qsb_is_deg = I_deg - np.degrees(np.array(q_sl_inits))
        dqs_deg = 180 - np.array(q_sl_finals) - qsb_is_deg
        for qsl_init, dq_deg in zip(q_sl_inits, dqs_deg):
            qsl_dats[qsl_init].append(dq_deg)
    for I_deg, t_merger, q_sl_inits, q_sl_finals in ensemble_dat2:
        I_degs.append(I_deg)
        t_mergers.append(t_merger)

        qsb_is_deg = I_deg - np.degrees(np.array(q_sl_inits))
        dqs_deg = np.array(q_sl_finals) - qsb_is_deg
        for qsl_init, dq_deg in zip(q_sl_inits, dqs_deg):
            qsl_dats[qsl_init].append(dq_deg)

    # axs[1].semilogy(I_degs, t_mergers)
    # axs[1].set_ylabel(r'$T_m / t_{\rm LK,0}$')
    axs[0].set_xlabel(r'$I^0$')

    qsldat_keys = sorted(qsl_dats.keys(), reverse=True)
    for qsl_init in qsldat_keys:
        axs[0].plot(I_degs, qsl_dats[qsl_init],
                    marker='.', linestyle='', markersize=1.5,
                    label=r'$%d^\circ$' % (90 - np.degrees(qsl_init)))
    axs[0].legend(fontsize=10, ncol=4, loc='lower right')
    axs[0].set_ylabel(r'$\theta_{\rm sl}^{f} - \theta_{\rm sl, th}^f$')
    axs[-1].set_xlabel(r'$I_0$ (Deg)')
    plt.tight_layout()
    plt.savefig(folder + 'ensemble', dpi=200)
    plt.close()

def run_ensemble_phase(folder, I_vals=np.arange(90.01, 90.4001, 0.001),
                       phi_sbs=[0],
                       af=3e-3, save_fn='ensemble_phase.pkl'):
    pkl_fn = folder + save_fn
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        mkdirp(folder)
        q_sl_arr = np.radians([-30, 0, 30])
        args = [(folder, I_deg, q_sl_arr, af, phi_sbs) for I_deg in I_vals[::-1]]
        with Pool(N_THREADS) as p:
            res = p.starmap(get_qslfs_base, args)
        with open(folder + save_fn, 'wb') as f:
            pickle.dump(res, f)
    else:
        with open(folder + save_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            res = pickle.load(f)
    return res

def plot_ensemble_phase(folder, ensemble_phase, phi_sbs):
    I_degs = []
    # key first by qsl_i, then phi_sb
    qsl_phase_dats = defaultdict(lambda: defaultdict(list))
    for I_deg, _, q_sl_inits, qslf_arr in ensemble_phase:
        I_degs.append(I_deg)
        for q_slf, phi_sb in zip(qslf_arr, phi_sbs):
            qsb_is_deg = I_deg - np.degrees(np.array(q_sl_inits))
            dqs_deg = 180 - np.array(q_slf) - qsb_is_deg
            for qsl_init, dq_deg in zip(q_sl_inits, dqs_deg):
                qsl_phase_dats[qsl_init][phi_sb].append(dq_deg)

    plt.xlabel(r'$I^0$')

    qsldat_keys = sorted(qsl_phase_dats.keys(), reverse=True)
    colors = ['r', 'b', 'g']
    styles = ['-', ':', '-.', '--']
    for qsl_init, c in zip(qsldat_keys, colors):
        for idx, (phi_sb, qslf) in enumerate(qsl_phase_dats[qsl_init].items()):
            ls = '-' if idx == 0 else styles[idx - 1]
            lw = 1.5 if idx == 0 else 0.7
            plt.plot(I_degs, qslf, linestyle=ls, linewidth=lw, color=c,
                     label=r'$%d^\circ, %d^\circ$' % (
                         (90 - np.degrees(qsl_init)), np.degrees(phi_sb)))
    plt.legend(fontsize=10, ncol=3, loc='lower right')
    plt.ylabel(r'$\theta_{\rm sl}^{f} - \theta_{\rm sl, th}^f$')
    plt.xlabel(r'$I_0$ (Deg)')
    plt.tight_layout()
    plt.savefig(folder + 'ensemble_phase', dpi=200)
    plt.close()

def plot_deviations(folder, ensemble_dat):
    ''' plot deviations from analytic prediction due to fast merger limit '''
    I_degs = []
    qsl_dats = defaultdict(list)
    for I_deg, _, q_sl_inits, q_sl_finals in ensemble_dat:
        I_degs.append(I_deg)

        qsb_is_deg = I_deg - np.degrees(np.array(q_sl_inits))
        dqs_deg = 180 - np.array(q_sl_finals) - qsb_is_deg
        for qsl_init, dq_deg in zip(q_sl_inits, dqs_deg):
            qsl_dats[qsl_init].append(dq_deg)

    I_degs = np.array(I_degs)
    qsldat_keys = sorted(qsl_dats.keys(), reverse=True)
    for qsl_init in qsldat_keys:
        plt.loglog(I_degs - 90, np.abs(qsl_dats[qsl_init]),
                    marker='.', linestyle='', markersize=1.5,
                    label=r'$%d^\circ$' % (90 - np.degrees(qsl_init)))

    # fit using the exact integral, two halves
    cosd = lambda x: np.cos(np.radians(x))
    # (i) exp part, say I ~ 120 (avg value) (don't think is relevant)
    # fit_line = 120 * np.exp(
    #     -(np.radians(120) * 100**2) /
    #     (np.pi * (25 * 4.63 * cosd(90.3)**6 / cosd(I_degs)**6)))
    # (ii) phi_dot ~ 100, Adot ~ 4.63 @ I = 90.3
    # fit_line2 = np.degrees(np.sqrt(
    #     (np.pi * 1.723 * cosd(90.3)**6)
    #      / (cosd(I_degs)**6)
    # ) / 3)
    # plt.plot(I_degs - 90, fit_line2, 'k', lw=3)

    # plt.plot(0.2, 6, 'ko', ms=3) # hand-curated from integrating 90.2

    plt.xlabel(r'$I^0$')
    plt.ylabel(r'$\left|\theta_{\rm sl}^{f} - \theta_{\rm sl, th}^f\right|$ (Deg)')
    plt.xlim(left=0.1, right=0.4)
    plt.ylim(bottom=1, top=100)
    plt.tight_layout()
    plt.savefig(folder + 'deviations', dpi=200)
    plt.close()

# make the prediction for theta_sl_f using N=0 prediction (re-process from pkl)
def plot_deviations_good(folder, I_vals=np.arange(90.01, 90.4001, 0.001)):
    atol=1e-8
    rtol=1e-8
    af=3e-3

    deltas_deg = []
    a_vals = []
    for I_deg in I_vals:
        ret_lk = get_kozai(folder, I_deg, getter_kwargs)
        pkl_fn = '4sim_qsl0_%s.pkl'
        s_vec = get_spins_inertial(
            folder, I_deg, ret_lk, getter_kwargs,
            q_sl0=0,
            pkl_template=pkl_fn,
            )
        _, _, Wf, If, _ = ret_lk[1][:, -1]
        sf = s_vec[:, -1]

        Lhat = get_hat(Wf, If)
        qslf = np.degrees(np.arccos(np.dot(Lhat, sf)))

        # Use shortened ret_lk to pass in just the first few Kozai cycles
        t0, y0, [events0, _] = ret_lk
        num_cycles = 5
        t_idx = np.where(t0 < events0[num_cycles])[0][-1]
        t = t0[ :t_idx + 1]
        y = y0[:, :t_idx + 1]
        events = events0[ :num_cycles]
        ret_lk_new = (t, y, [events, None])
        _, (a, e, W, I, w), [t_lks, _] = ret_lk_new

        _, _, dWdot, t_lkmids, dWslx, dWslz = get_dWs(ret_lk_new, getter_kwargs)
        eff_idx = np.where(np.logical_and(t < t_lkmids[-1], t > t_lkmids[0]))[0]
        t_eff = t[eff_idx]
        Wslx = interp1d(t_lkmids, dWslx)(t_eff)
        Wslz = interp1d(t_lkmids, dWslz)(t_eff)
        Wdot = interp1d(t_lkmids, dWdot)(t_eff)
        Lhat_xy = get_hat(W[eff_idx], I[eff_idx])
        Lhat_xy[2] *= 0
        Lhat_xy /= np.sqrt(np.sum(Lhat_xy**2, axis=0))
        Weff = np.outer(np.array([0, 0, 1]), -Wdot + Wslz) + Wslx * Lhat_xy
        Weff_hat = Weff / np.sqrt(np.sum(Weff**2, axis=0))
        _q_eff0 = np.arccos(ts_dot(s_vec[:, eff_idx], Weff_hat))
        q_eff0 = np.degrees(_q_eff0)
        # predict q_eff final by averaging over an early Kozai cycle (more interp1d)
        q_eff0_interp = interp1d(t_eff, q_eff0)
        q_eff_pred = np.mean(q_eff0_interp(
            np.linspace(t_lkmids[1], t_lkmids[2], 1000)))
        t_lk2 = np.where(t < t_lkmids[2])[0][-1]
        a_vals.append(y0[0, t_lk2])

        deltas_deg.append(q_eff_pred - qslf)
    plt.loglog(I_vals - 90, deltas_deg, 'ko', ms=0.5)
    plt.loglog(I_vals - 90, -np.array(deltas_deg), 'ro', ms=0.5)
    ylims = plt.ylim()
    plt.plot(I_vals - 90,
             5 * (
                 np.cos(np.radians(90.2))
                 / np.cos(np.radians(I_vals))
             )**(9), 'b', lw=2)
    plt.ylim(ylims)
    plt.xlabel(r'$I - 90^\circ$ (Deg)')
    plt.ylabel(r'$\theta_{\rm i} - \theta_{\rm sl,f}$ (Deg)')
    plt.xlim(left=0.1)
    plt.ylim(bottom=1e-3)
    plt.plot(0.2, 6.14, 'kx') # checked from 90.2 simulation by hand-integrating

    a_ax = plt.gca().twinx()
    a_ax.loglog(I_vals - 90, a_vals, 'g--', lw=2)
    a_ax.set_ylim(0.01, 1)
    a_ax.set_ylabel(r'$a$ at Time of Averaging')
    plt.tight_layout()
    plt.savefig(folder + 'deviations_one', dpi=200)
    plt.close()

def run_close_in():
    '''
    run spins after running ret_lk to see which times are useful to exmaine
    '''
    m1, m2, m3, a0, a2, e2 = 30, 30, 30, 0.1, 3, 0
    getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    I_deg = 80
    getter_kwargs['eps_gw'] *= 3
    folder = '4inner/'

    # ret_lk = get_kozai(folder, I_deg, getter_kwargs, atol=1e-7, rtol=1e-7,
    #                    af=0.9)
    # s_vec = get_spins_inertial(folder, I_deg, ret_lk, getter_kwargs,
    #                            atol=1e-6, rtol=1e-6)
    # plot_all(folder, ret_lk, None, getter_kwargs,
    #          time_slice=np.s_[::1000])

    ret_lk = get_kozai(folder, 70, getter_kwargs, atol=1e-7, rtol=1e-7,
                       af=0.9)
    # plot_all(folder, ret_lk, None, getter_kwargs,
    #          time_slice=np.s_[::1000])

if __name__ == '__main__':
    # I_deg = 90.5
    # folder = './'
    # ret_lk = get_kozai(folder, I_deg, getter_kwargs, af=5e-3, atol=1e-9,
    #                    rtol=1e-9, pkl_template='4shorto_%s.pkl')
    # s_vec = get_spins_inertial(folder, I_deg, ret_lk, getter_kwargs,
    #                            pkl_template='4shorto_s_%s.pkl')
    # plot_all(folder, ret_lk, s_vec, getter_kwargs, fn_template='4shorto_%s')
    # s_vec = get_spins_inertial(folder, I_deg, ret_lk, getter_kwargs,
    #                            atol=1e-8, rtol=1e-8,
    #                            pkl_template='4shorto_good_s_%s.pkl')
    # plot_all(folder, ret_lk, s_vec, getter_kwargs, fn_template='4shorto_good_%s')
    # s_vec = get_spins_inertial(folder, I_deg, ret_lk, getter_kwargs,
    #                            atol=1e-8, rtol=1e-8,
    #                            q_sl0=np.radians(-0.5),
    #                            pkl_template='4shorto_tilt_s_%s.pkl')
    # plot_all(folder, ret_lk, s_vec, getter_kwargs, fn_template='4shorto_tilt_%s')

    # for I_deg in np.arange(90.1, 90.51, 0.05):
    #     run_for_Ideg('4sims/', I_deg, short=True)
    # run_for_Ideg('4sims/', 90.475, short=True)
    # run_for_Ideg('4sims/', 90.3, short=True, plotter=plot_good)

    # ensemble_dat = run_ensemble('4sims_ensemble/')
    # ensemble_dat2 = run_ensemble('4sims_ensemble/',
    #                              I_vals=np.arange(89.99, 89.5999, 0.001),
    #                              save_fn='ensemble2.pkl')
    # plot_ensemble('4sims_ensemble/', ensemble_dat, ensemble_dat2)

    # phi_sbs = np.radians([0, 45, 90, 180, 270])
    # ensemble_phase = run_ensemble_phase('4sims_ensemble/', phi_sbs=phi_sbs)
    # plot_ensemble_phase('4sims_ensemble/', ensemble_phase, phi_sbs)

    run_close_in()

    # ensemble_dat = run_ensemble('4sims_ensemble/')
    # plot_deviations('4sims_ensemble/', ensemble_dat)

    # plot_deviations_good('4sims_ensemble/')
