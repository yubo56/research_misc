'''
does Kozai simulations in orbital elements, then computes spin evolution after
the fact w/ a given trajectory
'''
import pickle
import os
from collections import defaultdict

from multiprocessing import Pool
import numpy as np
try:
    import matplotlib
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

N_THREADS = 64
# N_THREADS = 35
m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)

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
                       q_sb0=None,
                       phi_sb=0,
                       pkl_template='4sim_s_%s.pkl',
                       save=True,
                       t_final=None,
                       **kwargs):
    ''' uses the same times as ret_lk '''
    mkdirp(folder)
    pkl_fn = folder + pkl_template % get_fn_I(I_deg)
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        t_lk, y, _ = ret_lk
        if t_final is not None:
            print('Orig length', len(t_lk), t_lk[-1])
            where_idx = np.where(t_lk < t_final)[0]
            t_lk = t_lk[where_idx]
            y = y[:, where_idx]
            print('New length', len(t_lk), t_lk[-1])
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
            # if t > t_lk[-1]:
            #     return None
            Lhat = [Lx(t), Ly(t), Lz(t)]
            return eps_sl * np.cross(Lhat, s) / (
                a(t)**(5/2) * (1 - e(t)**2))
        t0 = t_lk[0]

        # initial spin (for q_sl0=0, phi_sb0=0, equals L[0])
        if q_sb0 is None:
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
    valid_idxs = np.where(np.logical_and(
        lk_events[0] < t[-1], lk_events[0] > t[0]))[0]
    lk_events_sliced = [lk_events[0][valid_idxs], lk_events[1]]
    ret_lk_sliced = [t, [a, e, W, I, w], lk_events_sliced]

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
    _, dWsl, dWdot, t_lkmids, dWslx, dWslz = get_dWs(ret_lk_sliced, getter_kwargs)
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
    axs[4].set_ylabel(r'$\omega$')
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

def get_plot_good_quants(ret_lk, s_vec, getter_kwargs, time_slice=np.s_[::]):
    lk_t, lk_y, lk_events = ret_lk
    if s_vec is None:
        s_vec = np.array(
            [np.zeros_like(lk_t),
             np.zeros_like(lk_t),
             np.ones_like(lk_t)])
    # print(np.shape(s_vec))
    s_vec = s_vec[:, time_slice]
    sx, sy, sz = s_vec
    a, e, W, I, w = lk_y[:, time_slice]
    t = lk_t[time_slice]
    I0 = np.degrees(lk_y[3, 0])
    valid_idxs = np.where(np.logical_and(
        lk_events[0] < t[-1], lk_events[0] > t[0]))[0]
    lk_events_sliced = [lk_events[0][valid_idxs], lk_events[1]]
    ret_lk_sliced = [t, [a, e, W, I, w], lk_events_sliced]

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
    dWtot, dWsl, dWdot, t_lkmids, dWslx, dWslz = get_dWs(ret_lk_sliced, getter_kwargs)
    eff_idx = np.where(np.logical_and(t < t_lkmids[-1], t > t_lkmids[0]))[0]
    t_eff = t[eff_idx]
    Wslx = interp1d(t_lkmids, dWslx)(t_eff)
    Wslz = interp1d(t_lkmids, dWslz)(t_eff)
    Wdot = interp1d(t_lkmids, dWdot)(t_eff)
    Lhat_xy = get_hat(W[eff_idx], I[eff_idx])
    Lhat_xy[2] *= 0
    Lhat_xy /= np.sqrt(np.sum(Lhat_xy**2, axis=0))
    Weff = np.outer(np.array([0, 0, 1]), -Wdot + Wslz) + Wslx * Lhat_xy
    Weffmag = np.sqrt(np.sum(Weff**2, axis=0))
    Weff_hat = Weff / Weffmag
    _q_eff0 = np.arccos(ts_dot(s_vec[:, eff_idx], Weff_hat))
    q_eff0 = np.degrees(_q_eff0)
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
    Iout_dot = [(Iouts[i + 4] - Iouts[i]) / (t_eff[i + 4] - t_eff[i])
                for i in range(len(Iouts) - 4)]

    # define gaussian estimate to Iout_dot
    idxmax = np.argmax(Iout_dot) # teff[2 + idxmax] is the maximum time
    Iout_dot_max = np.max(Iout_dot)
    Iout_sigm2 = ((Iouts[-1] - Iouts[0]) / Iout_dot_max)**2 / (2 * np.pi) # sigma^2
    Iout_dot_gauss = Iout_dot_max * (
            np.exp(-(t_eff[2:-2] - t_eff[2 + idxmax])**2 / (2 * Iout_sigm2)))

    dqeff_max = -1
    dqeff_gauss = -1
    for phi_offset in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        dts = np.diff(t_eff[1:-2])
        dqeff = np.sum(np.degrees(Iout_dot)
                       * np.cos(Weffmag[2:-2] * t_eff[2:-2] + phi_offset)
                       * dts)
        dqeff_gauss_curr = np.sum(
            np.degrees(Iout_dot_gauss)
            * np.cos(2 * Weffmag[2:-2] * t_eff[2:-2] + phi_offset)
            * dts)
        dqeff_max = max(dqeff_max, abs(dqeff))
        dqeff_gauss = max(dqeff_gauss, abs(dqeff_gauss_curr))
    print('I0, dqeff_max, dqeff_gauss, Weffmag[2 + idxmax], Weffmag[tleftidx]')
    tleftidx = np.where(t_eff > t_eff[2 + idxmax] - np.sqrt(Iout_sigm2))[0][0]
    print(I0, dqeff_max, dqeff_gauss, Weffmag[2 + idxmax], Weffmag[tleftidx])
    return t_lkmids, t_eff, q_eff0, Iouts, Iout_dot, Weffmag, Weff_hat,\
        dWsl, dWdot, dWtot, dqeff_max, dqeff_gauss, Iout_dot_gauss

def plot_good(folder, ret_lk, s_vec, getter_kwargs,
              fn_template='4sim_%s', time_slice=np.s_[::], ylimdy=None,
              ylim3=None, **kwargs):
    mkdirp(folder)
    alf = 0.7
    I0 = np.degrees(ret_lk[1][3, 0])
    t_lkmids, t_eff, q_eff0, Iouts, Iout_dot, Weffmag, Weff_hat,\
        dWsl, dWdot, dWtot, dqeff, _, Iout_dot_gauss = \
        get_plot_good_quants(ret_lk, s_vec, getter_kwargs, time_slice)

    # plots
    if time_slice == np.s_[::]:
        # if default time slice, plotting full simulation, use a very long view
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 5), sharex=True)
    plt.plot(t_eff, q_eff0, 'k', alpha=0.3, label=r'$\theta_{\rm eff}$')
    end_qeffs = []
    averaged_qeff = []
    plotted_ts = []
    qeffinterp = interp1d(t_eff, q_eff0)
    # pick an intermediate piece b/c two ends are outside of interp range
    for start, end in zip(t_lkmids[10:-11], t_lkmids[11:-10]):
        interpt = np.linspace(start, end, 1000)
        qeff_interval = qeffinterp(interpt)
        averaged_qeff.append(np.mean(qeff_interval))
        end_qeffs.append(qeff_interval[len(interpt) // 2])
        plotted_ts.append((start + end) / 2)
    plt.plot(plotted_ts, averaged_qeff, 'ro', ms=1, alpha=0.7,
             label='LK-avg')
    plt.plot(plotted_ts, end_qeffs, 'bo', ms=1, alpha=0.7,
             label='LK-start')
    # predict q_eff final by averaging over an early Kozai cycle (more interp1d)
    try:
        q_eff0_interp = interp1d(t_eff, q_eff0)
        q_eff_pred = np.mean(q_eff0_interp(
            np.linspace(t_lkmids[2], t_lkmids[3], 1000)))
        plt.ylabel(r'[$\langle %.2f \rangle$--$%.2f$]' %
                       (q_eff_pred, q_eff0[-1]))
        qeffdiff = np.abs(q_eff_pred - q_eff0[-1])
    except: # time_slice can screw up the above
        qeffdiff = None
        pass
    # plt.plot(t, q_eff_bin, 'r', alpha=0.3)
    plt.xlabel(r'$t / t_{\rm LK, 0}$')
    # estimated amplitude of oscillations: Iout_dot / W
    # / Weff
    plt.plot(t_eff[2:-2],
             q_eff0[-1] + np.degrees(Iout_dot) / Weffmag[2:-2],
             'g', lw=1.5, label=r'$N = 0$ Integ.')
    plt.plot(t_eff[2:-2],
             q_eff0[-1] - np.degrees(Iout_dot) / Weffmag[2:-2],
             'g', lw=1.5)
    plt.legend(loc='upper left')
    if ylimdy is not None:
        curr_ylim = plt.ylim()
        ylim_cent = np.sum(curr_ylim) / 2
        plt.ylim(bottom=ylim_cent - ylimdy, top=ylim_cent + ylimdy)
    plt.tight_layout()
    plt.savefig(folder + (fn_template % get_fn_I(I0)) + '_qN0', dpi=200)
    plt.close()

    # plt.plot(t_eff, np.degrees(I_avg), 'r')
    # plt.ylabel(r'$\langle I \rangle_{LK}$')
    l1 = plt.semilogy(t_lkmids, dWsl, 'g', label=r'$\Omega_{\rm SL}$',
                      lw=0.7, alpha=0.5)
    l2 = plt.semilogy(t_lkmids, dWdot, 'r', label=r'$\dot{\Omega}$',
                      lw=0.7, alpha=0.5)
    l3 = plt.semilogy(t_lkmids, dWtot, 'k',
                      label=r'$\langle\Omega_{\rm e}\rangle$', lw=2)
    l5 = plt.semilogy(t_eff[2:-2], np.degrees(Iout_dot), 'b:',
                      lw=0.7, alpha=0.3, label=r'$\dot{I}_{\rm e}$')
    ylims = plt.ylim()
    l6 = plt.semilogy(t_eff[2:-2], np.degrees(Iout_dot_gauss), 'r:',
                      lw=0.7, alpha=0.3, label=r'$\dot{I}_{\rm e,gauss}$')
    plt.ylim(ylims)
    plt.ylabel(r'Frequency ($t_{\rm LK, 0}^{-1}$)')
    plt.xlabel(r'$t / t_{\rm LK, 0}$')
    twinIout_ax = plt.gca().twinx()
    l4 = twinIout_ax.plot(t_eff, np.degrees(Iouts), 'b', label=r'$I_{\rm e}$')
    twinIout_ax.set_ylabel(r'$I_{\rm e}$')
    lns = l1 + l2 + l3 + l4 + l5 + l6
    plt.legend(lns, [l.get_label() for l in lns], fontsize=10)
    plt.savefig(folder + (fn_template % get_fn_I(I0)) + '_Iout', dpi=200)
    plt.close()

    # plt.semilogy(t_lkmids, dWsl, 'g', label=r'$\Omega_{\rm SL}$',
    #              lw=0.7, alpha=0.5)
    # plt.semilogy(t_lkmids, dWdot, 'r', label=r'$\dot{\Omega}$',
    #              lw=0.7, alpha=0.5)
    # plt.semilogy(t_lkmids, dWtot, 'k',
    #              label=r'$\langle\Omega_{\rm e}\rangle$', lw=2)
    # plt.semilogy(t_eff[2:-2], np.degrees(Iout_dot), 'b',
    #              lw=2, label=r'$\dot{I}_{\rm e}$')
    # plt.ylabel(r'Frequency ($t_{\rm LK, 0}^{-1}$)')
    # plt.xlabel(r'$t / t_{\rm LK, 0}$')
    # plt.legend()
    # if ylim3 is not None:
    #     plt.ylim(ylim3)
    # else:
    #     plt.ylim([0.01, 100])
    # if qeffdiff is not None:
    #     plt.title(r'$I_0 = %.2f^\circ, \Delta \theta_{\rm eff, 0} = (%s)^\circ$'
    #               % (I0, get_scinot(qeffdiff)))
    # plt.tight_layout()
    # plt.savefig(folder + (fn_template % get_fn_I(I0)) + '_phidots', dpi=200)
    # plt.close()

def run_for_Ideg(folder, I_deg, af=5e-3,
                 atol=1e-8, rtol=1e-8, short=False, plotter=plot_all, **kwargs):
    mkdirp(folder)
    ret_lk = get_kozai(folder, I_deg, getter_kwargs,
                       af=af, atol=atol, rtol=rtol)
    s_vec = get_spins_inertial(folder, I_deg, ret_lk, getter_kwargs,
                               atol=atol,
                               rtol=rtol,
                               )
    plotter(folder, ret_lk, s_vec, getter_kwargs, **kwargs)
    if short:
        return

    # try with q_sl0
    s_vec = get_spins_inertial(folder, I_deg, ret_lk, getter_kwargs,
                               q_sl0=np.radians(20),
                               atol=atol,
                               rtol=rtol,
                               pkl_template='4sim_qsl20_%s.pkl')
    plotter(folder, ret_lk, s_vec, getter_kwargs,
             fn_template='4sim_qsl20_%s', **kwargs)

    s_vec = get_spins_inertial(folder, I_deg, ret_lk, getter_kwargs,
                               q_sl0=np.radians(40),
                               atol=atol,
                               rtol=rtol,
                               pkl_template='4sim_qsl40_%s.pkl')
    plotter(folder, ret_lk, s_vec, getter_kwargs,
             fn_template='4sim_qsl40_%s', **kwargs)

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

    pkl_fn = folder + 'deviations_good.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        deltas_deg = []
        a_vals = []
        dqeff_maxes = []
        dqeff_maxes_g = []
        for I_deg in I_vals:
            deltas_per_I = []
            ret_lk = get_kozai(folder, I_deg, getter_kwargs)
            # all the plots run in run_ensemble + run_ensemble_phase
            for q_sl0, phi_sb in zip(
                    np.radians([
                        -40, -30, -20, -10, -5, -3, -2, -1,
                        0,
                        40, 30, 20, 10, 5, 3, 2, 1,
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0,

                    ]),
                    np.radians([
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        45, 90, 180, 270,
                        45, 90, 180, 270,
                        45, 90, 180, 270,
                    ])
            ):
                if phi_sb == 0:
                    pkl_template = '4sim_qsl' + ('%d' % np.degrees(q_sl0)) + '_%s.pkl'
                else:
                    pkl_template = '4sim_qsl' + ('%d' % np.degrees(q_sl0)) + \
                        ('_phi_sb%d' % np.degrees(phi_sb)) + '_%s.pkl'
                s_vec = get_spins_inertial(
                    folder, I_deg, ret_lk, getter_kwargs,
                    q_sl0=q_sl0,
                    phi_sb=phi_sb,
                    pkl_template=pkl_template,
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
                # predict q_eff final by averaging over an early Kozai cycle (more
                # interp1d)
                q_eff0_interp = interp1d(t_eff, q_eff0)
                q_eff_pred = np.mean(q_eff0_interp(
                    np.linspace(t_lkmids[1], t_lkmids[2], 1000)))
                t_lk2 = np.where(t < t_lkmids[2])[0][-1]

                deltas_per_I.append(q_eff_pred - qslf)

                if q_sl0 == 0:
                    a_vals.append(y0[0, t_lk2])
                    good_quants = get_plot_good_quants(ret_lk, s_vec, getter_kwargs)
                    dqeff_maxes.append(good_quants[-3])
                    dqeff_maxes_g.append(good_quants[-2]) # gaussest
            deltas_deg.append(deltas_per_I)
        with open(pkl_fn, 'wb') as f:
            pickle.dump((deltas_deg, a_vals, dqeff_maxes, dqeff_maxes_g), f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            deltas_deg, a_vals, dqeff_maxes, dqeff_maxes_g = pickle.load(f)
    for I, deltas_per_I in zip(I_vals, deltas_deg):
        plt.loglog(np.full_like(deltas_per_I, I - 90), deltas_per_I,
                   'ko', ms=0.3)
        plt.loglog(np.full_like(deltas_per_I, I - 90), -np.array(deltas_per_I),
                   'ro', ms=0.3)
    ylims = plt.ylim()
    # plt.plot(I_vals - 90,
    #          5 * (
    #              np.cos(np.radians(90.2))
    #              / np.cos(np.radians(I_vals))
    #          )**(9), 'b', lw=2)
    plt.ylim(ylims)
    plt.xlabel(r'$I - 90^\circ$ (Deg)')
    plt.ylabel(r'$\theta_{\rm i} - \theta_{\rm sl,f}$ (Deg)')
    plt.xlim(left=0.1)
    plt.ylim(bottom=1e-3)
    plt.plot(I_vals - 90, dqeff_maxes, 'gx', ms=0.5)
    plt.plot(I_vals - 90, dqeff_maxes_g, 'bx', ms=0.5)

    Adot = (
        200 * getter_kwargs['eps_gw'] / getter_kwargs['eps_sl']
             * (8 * 5 * cosd(I_vals)**2 / 3)**(-3)) * 2
    dqeff_th = 120 * np.exp(
        -(30**2 * np.radians(115)**2) / (np.pi * adot**2))
    print(i_vals[-1], dqeff_th[len(i_vals) // 2], adot[-1])
    return
    plt.plot(I_vals, dqeff_th, 'b', lw=2)

    # a_ax = plt.gca().twinx()
    # a_ax.loglog(I_vals - 90, a_vals, 'g--', lw=2)
    # a_ax.set_ylim(0.01, 1)
    # a_ax.set_ylabel(r'$a$ at Time of Averaging')
    plt.tight_layout()
    plt.savefig(folder + 'deviations_one', dpi=200)
    plt.close()

def run_close_in(I_deg=80, t_final_mult=0.5, time_slice=None, plotter=plot_all):
    '''
    run spins after running ret_lk to see which times are useful to exmaine
    '''
    m1, m2, m3, a0, a2, e2 = 30, 30, 30, 0.1, 3, 0
    getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    # getter_kwargs['eps_gw'] *= 3
    folder = '4inner/'

    ret_lk = get_kozai(folder, I_deg, getter_kwargs, atol=1e-8, rtol=1e-8,
                       af=0.9)
    t_final = t_final_mult * ret_lk[0][-1]
    s_vec = get_spins_inertial(folder, I_deg, ret_lk, getter_kwargs,
                               atol=1e-8, rtol=1e-8, t_final=t_final)
    idx_f = np.where(ret_lk[0] < t_final)[0][-1]
    if time_slice is None:
        time_slice = np.s_[:idx_f:100]
    plotter(folder, ret_lk, s_vec, getter_kwargs,
             time_slice=time_slice)


def run_for_grid(newfolder, I_deg, ret_lk, getter_kwargs, atol, rtol, q, phi,
                 t_eff, Weff_hat, t_lkmids, Lhat_f, eff_idx):
    print('Running for', q, phi)
    pkl_template = '4sim_qsl' + ('%d' % np.degrees(q)) + \
        ('_phi_sb%d' % np.degrees(phi)) + '_%s.pkl'
    s_vec = get_spins_inertial(
        newfolder, I_deg, ret_lk, getter_kwargs, atol=atol,
        rtol=rtol, q_sb0=q, phi_sb=phi,
        pkl_template=pkl_template)
    _q_eff0 = np.arccos(ts_dot(s_vec[:, eff_idx], Weff_hat))
    q_eff0 = np.degrees(_q_eff0)
    q_eff0_interp = interp1d(t_eff, q_eff0)
    q_eff_pred = np.mean(q_eff0_interp(
        np.linspace(t_lkmids[2], t_lkmids[3], 1000)))

    dot_slf = np.dot(Lhat_f, s_vec[:,-1])
    q_slf = np.degrees(np.arccos(dot_slf))
    print('Ran for', q, phi, 'final angs are', q_eff_pred, q_slf)
    return q_eff_pred - q_slf

# try 90.5 simulation over an isotropic grid of ICs
def run_905_grid(I_deg=90.5, newfolder='4sims905/', af=5e-3, n_pts=20,
                 atol=1e-7, rtol=1e-7, getter_kwargs=getter_kwargs,
                 orig_folder='4sims/'):
    mkdirp(newfolder)

    pkl_fn = newfolder + 'devgrid.pkl'
    mus_edges = np.linspace(-1, 1, n_pts + 1)
    phis_edges = np.linspace(0, 2 * np.pi, n_pts + 1)
    mus = (mus_edges[ :-1] + mus_edges[1: ]) / 2
    phis = (phis_edges[ :-1] + phis_edges[1: ]) / 2

    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        ret_lk = get_kozai(orig_folder, I_deg, getter_kwargs,
                           af=af, atol=atol, rtol=rtol)
        t, (a, e, W, I, w), _ = ret_lk
        # compute Weff_hat orientation
        _, dWsl, dWdot, t_lkmids, dWslx, dWslz = get_dWs(ret_lk, getter_kwargs)
        eff_idx = np.where(np.logical_and(t < t_lkmids[-1], t > t_lkmids[0]))[0]
        t_eff = t[eff_idx]
        Wslx = interp1d(t_lkmids, dWslx)(t_eff)
        Wslz = interp1d(t_lkmids, dWslz)(t_eff)
        Wdot = interp1d(t_lkmids, dWdot)(t_eff)
        Lhat_f = get_hat(W[-1], I[-1])
        Lhat_xy = get_hat(W[eff_idx], I[eff_idx])
        Lhat_xy[2] *= 0
        Lhat_xy /= np.sqrt(np.sum(Lhat_xy**2, axis=0))
        Weff = np.outer(np.array([0, 0, 1]), -Wdot + Wslz) + Wslx * Lhat_xy
        Weff_hat = Weff / np.sqrt(np.sum(Weff**2, axis=0))

        args = []
        for q in np.arccos(mus):
            for phi in phis:
                args.append((newfolder, I_deg, ret_lk, getter_kwargs, atol, rtol,
                             q, phi, t_eff, Weff_hat, t_lkmids, Lhat_f, eff_idx))
        with Pool(N_THREADS) as p:
            res = p.starmap(run_for_grid, args)
        res_nparr = np.reshape(np.array(res), (n_pts, n_pts)).T
        with open(pkl_fn, 'wb') as f:
            pickle.dump(res_nparr, f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            res_nparr = pickle.load(f)

    phis_grid = np.outer(phis_edges, np.ones_like(mus_edges))
    mus_grid = np.outer(np.ones_like(phis_edges), mus_edges)
    # figure out bounds by dropping outliers
    sorted_res_nparr = np.sort(res_nparr.flatten())
    plt.pcolormesh(phis_grid, mus_grid, res_nparr, cmap='viridis',
                   vmin=sorted_res_nparr[1], vmax=sorted_res_nparr[-2])
    plt.colorbar()
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\cos \theta$')
    plt.title(r'$\Delta \theta_{\rm eff}$')
    plt.tight_layout()
    plt.savefig(newfolder + 'devsgrid', dpi=200)
    plt.close()

    plt.hist(sorted_res_nparr[1:-1], bins=30)
    plt.xlabel(r'$\Delta \theta_{\rm eff}$')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(newfolder + 'devshist', dpi=200)
    plt.close()

def plot_good_quants():
    '''
    Using ensemble data, plot: dWeff(A=1), Adot(A = 1), max(Ioutdot), Iout
    width

    Figure out whether can de-noise Ioutdot?
    '''
    I_degs = np.concatenate((np.arange(90.1, 90.51, 0.05), [90.475]))
    I_degs_ensemble = np.arange(90.1, 90.4001, 0.005)
    dirs = ['4sims/'] * len(I_degs)
    dirs_ensemble = ['4sims_ensemble/'] * len(I_degs_ensemble)
    all_Is = np.concatenate((I_degs, I_degs_ensemble))

    pkl_fn = '4sims/good_quants.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        dWeff_mids = []
        dWeff_Imaxes = []
        Iout_dot_maxes = []
        Ioutdot_max_gaussest = []
        dqeff_num = []
        dqeff_gauss = []
        tmerges = []
        for folder, I_deg in zip(dirs + dirs_ensemble, all_Is):
            ret_lk = get_kozai(folder, I_deg, getter_kwargs)
            _, t_eff, _, Iouts, Iout_dot, Weffmag, _, dWsl, dWdot, _,\
                dqeff, _, _= get_plot_good_quants(ret_lk, None, getter_kwargs)
            if folder == '4sims/':
                tmerges.append(ret_lk[0][-1])# don't get too many merger times
            Iout_min = np.min(Iouts)
            Iout_max = np.max(Iouts)
            Iout_mid = (Iout_max + Iout_min) / 2
            mid_idx = (
                np.where(Iouts < Iout_mid)[0][-1] +
                np.where(Iouts > Iout_mid)[0][0]
            ) // 2 # estimate where transition occurs
            # TODO do Gaussian fit to Idotout_width
            dWeff_mids.append(Weffmag[mid_idx])
            dWeff_Imaxes.append(Weffmag[np.argmax(Iouts)])
            Iout_dot_maxes.append(Iout_dot[2 + mid_idx])

            # more robust estimate of Iout_dot (34% to 68%, one sigma)
            left_idx = np.where(Iouts > (0.68 * Iout_min + 0.34 * Iout_max))[0][0]
            right_idx = np.where(Iouts < (0.34 * Iout_min + 0.68 * Iout_max))[0][-1]
            sigm_t = (t_eff[right_idx] - t_eff[left_idx]) / 2
            # Iout_dot ~ I / sqrt(2pi sigma^2) * exp
            Ioutdot_gaussmax = (Iout_max - Iout_min) / (sigm_t * np.sqrt(2 * np.pi))
            Ioutdot_max_gaussest.append(Ioutdot_gaussmax)

            dqeff_num.append(dqeff)

            dqeff_max = -1
            for phi_offset in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                dts = np.diff(t_eff[1:-2])
                Ioutdot_gauss = Ioutdot_gaussmax * np.exp(
                    -(t_eff - t_eff[mid_idx])**2 / (2 * sigm_t**2))
                dqeff = np.sum(np.degrees(Ioutdot_gauss[2:-2])
                               * np.cos(Weffmag[2:-2] * t_eff[2:-2] + phi_offset)
                               * dts)
                dqeff_max = max(dqeff_max, abs(dqeff))
            dqeff_gauss.append(dqeff_max)

        with open(pkl_fn, 'wb') as f:
            pickle.dump(
                (dWeff_mids, dWeff_Imaxes, Iout_dot_maxes, Ioutdot_max_gaussest,
                 dqeff_num, dqeff_gauss, tmerges), f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            dWeff_mids, dWeff_Imaxes, Iout_dot_maxes, Ioutdot_max_gaussest,\
                dqeff_num, dqeff_gauss, tmerges = pickle.load(f)

    # try estimating Iout_dot scaling
    f_jmin = 2.2 # (j_cross) / (j_min) [parameterized]
    I_LK = np.radians(125) # LK-averaged I?
    jmin = np.sqrt(5 * cosd(all_Is)**2 / 3)
    Iout_dot_th = (
        200 * getter_kwargs['eps_gw'] / getter_kwargs['eps_sl']
             * (f_jmin * jmin)**(-6)) * (-np.tan(I_LK + (np.pi - I_LK) / 2) / 2)
    Weff_th = (getter_kwargs['eps_sl']**(3/8) / (f_jmin * jmin)**(11/8)) \
        * 2 * (-np.cos(I_LK))
    as_idx = np.argsort(all_Is)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax1.semilogy(all_Is, dWeff_mids, 'ko', label=r'$\Omega_{\rm eff}$', ms=1)
    # ax1.semilogy(all_Is, dWeff_Imaxes, 'bo',
    #              label=r'$\Omega_{\rm eff}$ (at $\dot{I}_{\rm eff}$ max)')
    ax1.semilogy(all_Is, Iout_dot_maxes, 'go',
                 label=r'$\dot{I}_{\rm eff}$', ms=1)
    ax1.semilogy(all_Is[as_idx], Weff_th[as_idx], 'g', alpha=0.5,
                 label=r'$\Omega_{\rm eff}$ (Th)')
    ax1.semilogy(all_Is[as_idx], Iout_dot_th[as_idx], 'r', alpha=0.5,
                 label=r'$\dot{I}_{\rm eff}$ (Th)')
    # ax1.semilogy(all_Is, Ioutdot_max_gaussest, 'ro',
    #              label=r'$\dot{I}_{\rm eff}$ (Bounds)')
    ax1.legend(fontsize=10)
    ax1.set_ylabel(r'Frequency ($t_{\rm LK, 0}^{-1}$)')
    ax1.set_yticks([0.01, 0.1, 1, 10, 100])
    ax1.set_yticklabels([r'$0.01$', r'$0.1$', r'$1$', r'$10$', r'$100$'])

    tmerge_ax = ax1.twiny()
    tmerge_ax.set_xlim(ax1.get_xlim())
    tmerge_ax.set_xticks(I_degs[::2])
    tmerge_ax.set_xticklabels([
        '%.1f' % np.log10(tm) for tm in tmerges[::2]
    ])
    plt.setp(tmerge_ax.get_xticklabels(), rotation=45)
    tmerge_ax.set_xlabel(r'$\log_{10} (T_{\rm m} / t_{\rm LK,0})$')

    ax2.semilogy(all_Is, dqeff_num, 'bo', label='Num', ms=1)
    # ax2.semilogy(all_Is, dqeff_gauss, 'go', label=r'Gaussian Fit')
    # ax2.legend(fontsize=12)
    ax2.set_ylabel(r'$\max \Delta \theta_{\rm eff}$')
    ax2.set_xlabel(r'$I$ (Deg)')
    plt.tight_layout()
    plt.savefig('4sims/good_quants', dpi=200)
    plt.close()

    pass

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
    #     run_for_Ideg('4sims/', I_deg, short=True, plotter=plot_good)
    # run_for_Ideg('4sims/', 90.475, short=True, plotter=plot_good)
    # run_for_Ideg('4sims/', 90.5, plotter=plot_good, short=True,
    #              ylimdy=2)
    # run_for_Ideg('4sims/', 90.35, plotter=plot_good, short=True,
    #              ylimdy=2, time_slice=np.s_[5500:9000])
    # run_for_Ideg('4sims/', 90.5, plotter=plot_good, short=True,
    #              time_slice=np.s_[-45000:-20000])
    plot_good_quants()

    # compare plot_good for 90.5 for a few different angular locations
    # s_fns = ['4sim_qsl87_phi_sb189_%s',
    #          '4sim_qsl31_phi_sb333_%s',
    #          '4sim_qsl56_phi_sb63_%s']
    # ret_lk = get_kozai('4sims/', 90.5, getter_kwargs, af=5e-3)
    # for s_fn in s_fns:
    #     pkl_fn = s_fn + '.pkl'
    #     s_vec = get_spins_inertial('4sims905/', 90.5, ret_lk, getter_kwargs,
    #                                pkl_template=pkl_fn)
    #     plot_good('4sims905/', ret_lk, s_vec, getter_kwargs, fn_template=s_fn,
    #               time_slice=np.s_[-45000:-20000])

    # ensemble_dat = run_ensemble('4sims_ensemble/')
    # ensemble_dat2 = run_ensemble('4sims_ensemble/',
    #                              I_vals=np.arange(89.99, 89.5999, 0.001),
    #                              save_fn='ensemble2.pkl')
    # plot_ensemble('4sims_ensemble/', ensemble_dat, ensemble_dat2)

    # phi_sbs = np.radians([0, 45, 90, 180, 270])
    # ensemble_phase = run_ensemble_phase('4sims_ensemble/', phi_sbs=phi_sbs)
    # plot_ensemble_phase('4sims_ensemble/', ensemble_phase, phi_sbs)

    # run_close_in(t_final_mult=0.5)
    # run_close_in(t_final_mult=0.5, I_deg=80.01)

    # ensemble_dat = run_ensemble('4sims_ensemble/')
    # plot_deviations('4sims_ensemble/', ensemble_dat)

    # plot_deviations_good('4sims_ensemble/')

    # run_905_grid()
    # run_905_grid(newfolder='4sims905_htol/', orig_folder='4sims905_htol/',
    #              atol=1e-10, rtol=1e-10, n_pts=30)
    # getter_kwargs_in = get_eps(30, 30, 30, 0.1, 3, 0)
    # run_905_grid(I_deg=80, newfolder='4sims80/', af=0.5, n_pts=10,
    #              atol=1e-7, rtol=1e-7, getter_kwargs=getter_kwargs_in,
    #              orig_folder='4inner/')

    # one of the deviant cases!
    # ret_lk = get_kozai('4sims/', 90.5, getter_kwargs)
    # fn_template = '4sim_qsl87_phi_sb333_%s'
    # s_vec = get_spins_inertial(
    #     '4sims905/', 90.5, ret_lk, getter_kwargs,
    #     q_sb0=np.radians(87), phi_sb=np.radians(333),
    #     pkl_template=fn_template + '.pkl')
    # plot_good('4sims905/', ret_lk, s_vec, getter_kwargs,
    #           fn_template=fn_template)
    pass
