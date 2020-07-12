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
    plt.rc('font', family='serif', size=20)
    plt.rc('lines', lw=1.5)
    plt.rc('xtick', direction='in', top=True, bottom=True)
    plt.rc('ytick', direction='in', left=True, right=True)
except: # let it fail later
    pass
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import brenth
from utils import *

N_THREADS = 4
# N_THREADS = 35
m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)

f_jmin = 2.2 # (j_cross) / (j_min) [parameterized]
I_LK = np.radians(125) # LK-averaged I?

def get_kozai(folder, I_deg, getter_kwargs,
              a0=1, e0=1e-3, W0=0, w0=0, tf=np.inf, af=0,
              pkl_template='4sim_lk_%s.pkl', save=True, method='Radau',
              **kwargs):
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
        ret = solve_ivp(dydt, (0, tf), y0, events=events,
                        method=method, **kwargs)
        t, y, t_events = ret.t, ret.y, ret.t_events
        print('Finished for I=%.3f, took %d steps, t_f %.3f (%d cycles)' %
              (I_deg, len(t), t[-1], len(t_events[0])))

        if save:
            with open(pkl_fn, 'wb') as f:
                pickle.dump((t, y, t_events), f)
        else:
            return ret
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
                       load=True,
                       method='Radau',
                       **kwargs):
    ''' uses the same times as ret_lk '''
    mkdirp(folder)
    pkl_fn = folder + pkl_template % get_fn_I(I_deg)
    if not os.path.exists(pkl_fn):
        if load == False:
            raise ValueError('%s does not exist' % pkl_fn)
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

        max_step = 0.01 * t_lk[-1]
        ret = solve_ivp(dydt, (t0, t_lk[-1]), s_rot, dense_output=True,
                        first_step=max_step, max_step=max_step,
                        method=method, **kwargs)
        y = ret.sol(t_lk)
        print('Finished spins for I=%.3f, took %d steps' %
              (I_deg, len(ret.t)))

        if save:
            with open(pkl_fn, 'wb') as f:
                pickle.dump(y, f)
        else:
            return ret
    else:
        print('Loading %s' % pkl_fn)
        with open(pkl_fn, 'rb') as f:
            y = pickle.load(f)
    return y

def plot_all(folder, ret_lk, s_vec, getter_kwargs,
             fn_template='4sim_%s', time_slice=np.s_[::],
             **kwargs):
    alf=0.8
    mkdirp(folder)

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
    I0 = np.degrees(lk_y[3, 0])
    valid_idxs = np.where(np.logical_and(
        lk_events[0] < t[-1], lk_events[0] > t[0]))[0]
    lk_events_sliced = [lk_events[0][valid_idxs], lk_events[1]]
    ret_lk_sliced = [t, [a, e, W, I, w], lk_events_sliced]
    lk_times = lk_events_sliced[0]

    a_interp = interp1d(t, a)
    e_interp = interp1d(t, e)
    I_interp = interp1d(t, I)
    W_interp = interp1d(t, W)
    sinterp = interp1d(t, s_vec)

    K = np.sqrt(1 - e**2) * np.cos(I)
    Lhat = get_hat(W, I)
    Lout_hat = get_hat(0 * I, 0 * I)
    q_sl = np.arccos(ts_dot(Lhat, s_vec))
    q_sb = np.arccos(reg(sz))
    Wsl = getter_kwargs['eps_sl'] / (a**(5/2) * (1 - e**2))
    Wdot_eff = (
        3 * a**(3/2) / 4 * np.cos(I) * (4 * e**2 + 1) / np.sqrt(1 - e**2))
    Wdot_exact = (
        3 * a**(3/2) / 4 * np.cos(I) * (
            5 * e**2 * np.cos(w)**2 - 4 * e**2 - 1) / np.sqrt(1 - e**2))
    Idot = (-15 * a**(3/2) * e**2 * np.sin(2 * w)
                    * np.sin(2 * I) / (16 * np.sqrt(1 - e**2)))
    # A = np.abs(Wsl / Wdot_exact)
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
    # W_eff_tot = Wsl * Lhat - Wdot * Lout_hat - Idot * cross / np.sin(I)
    # W_eff_hat = W_eff_tot / np.sqrt(np.sum(W_eff_tot**2, axis=0))
    # q_eff_me = np.degrees(np.arccos(ts_dot(W_eff_hat, s_vec)))

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

    Lhat_lks = get_hat(W_interp(lk_times), I_interp(lk_times))
    svec_lks = sinterp(lk_times)
    q_sl_lks = np.arccos(ts_dot(Lhat_lks, svec_lks))

    # computing effective angle to N = 0 axis
    dWtot, dWsl, dWdot, t_lkmids, dWslx, dWslz, dW_comps\
        = get_dWs(ret_lk_sliced, getter_kwargs, get_comps=True)
    eff_idx = np.where(np.logical_and(t < t_lkmids[-1], t > t_lkmids[0]))[0]
    t_eff = t[eff_idx]
    Wslx = interp1d(t_lkmids, dWslx)(t_eff)
    Wslz = interp1d(t_lkmids, dWslz)(t_eff)
    Wdot = interp1d(t_lkmids, dWdot)(t_eff)
    Lhat_xy = get_hat(W[eff_idx], np.full_like(W[eff_idx], np.pi / 2))
    Weff = np.outer(np.array([0, 0, 1]), -Wdot + Wslz) + Wslx * Lhat_xy
    Weffmags = np.sqrt(np.sum(Weff**2, axis=0))

    Weff_hat = Weff / Weffmags
    mu_eff0 = ts_dot(s_vec[:, eff_idx], Weff_hat)
    _q_eff0 = np.arccos(mu_eff0)
    q_eff0 = np.degrees(_q_eff0)

    yhat = ts_cross(Weff_hat, Lout_hat[:, eff_idx])
    xhat = ts_cross(yhat, Weff_hat)
    sin_phiWeff = (ts_dot(s_vec[:, eff_idx], yhat) / np.sin(_q_eff0))
    cos_phiWeff = (ts_dot(s_vec[:, eff_idx], xhat) / np.sin(_q_eff0))
    phi_Weff = np.arctan2(sin_phiWeff, cos_phiWeff)
    A = np.abs(dWsl / dWdot)

    # predict q_eff final by averaging over an early Kozai cycle (more interp1d)
    q_eff0_interp = interp1d(t_eff, q_eff0)
    q_eff_pred = np.mean(q_eff0_interp(
        np.linspace(t_lkmids[2], t_lkmids[4], 1000)))

    dWsl = np.sqrt(dWslx**2 + dWslz**2)
    dI_avg = np.arccos(dWslz / np.sqrt(dWslx**2 + dWslz**2))
    I_avg = interp1d(t_lkmids, dI_avg)(t_eff)

    _Iouts = []
    I1s = []
    W1mags = []
    for dW_comp in dW_comps:
        xdct, zdct = dW_comp
        _Iouts.append(np.arctan2(xdct[0], -zdct[0]))
        I1s.append(np.arctan2(xdct[1], -zdct[1]))
        W1mags.append(np.sqrt(xdct[1]**2 + zdct[1]**2))

    W_eff_exact = Wsl * Lhat - Wdot_exact * Lout_hat
    Iouts_exact = np.arcsin(
        np.sqrt(np.sum(W_eff_exact[ :2]**2, axis=0)) /
        np.sqrt(np.sum(W_eff_exact**2, axis=0)))

    t_Iout_smoothed = np.linspace(t_eff.min(), t_eff.max(), len(t_eff))
    def get_smoothed(vec): # over t_eff
        sm_len = min(101, len(t_eff) // 100)
        sm_len = 2 * (sm_len // 2) + 1 # ensure odd
        return smooth(interp1d(t_lkmids, vec)(t_Iout_smoothed), sm_len)
    Iout_smoothed = get_smoothed(_Iouts)
    I1_smoothed = get_smoothed(I1s)
    offset_idx = 1
    Iout_dot_smoothed = [(Iout_smoothed[i + 1] - Iout_smoothed[i]) /
                             (t_Iout_smoothed[i + 1] - t_Iout_smoothed[i])
                         for i in range(offset_idx, len(Iout_smoothed) -
                                        offset_idx)]

    # plot averaged theta_e
    averaged_qeff = []
    plot_ts = []
    qeffinterp = interp1d(t_eff, q_eff0)
    # pick an intermediate piece b/c two ends are outside of interp range
    avg_len = 1
    for start, end in zip(t_lkmids[offset_idx:-offset_idx - avg_len],
                          t_lkmids[offset_idx + avg_len:-offset_idx]):
        interpt = np.linspace(start, end, 1000)
        qeff_interval = qeffinterp(interpt)
        averaged_qeff.append(np.mean(qeff_interval))
    Weffmag_smootheds = interp1d(t_eff, Weffmags)(t_Iout_smoothed)

    # get bounds for zoomed-in plot
    idxmax = np.argmax(Iout_dot_smoothed) # t_Iout_smoothed[2 + idxmax]
    dIout_tot = Iout_smoothed[-1] - Iout_smoothed[0]
    jmin = np.sqrt(5 * cosd(I0)**2 / 3)
    Iout_dot_th = 1.5 * (
        200 * getter_kwargs['eps_gw'] / getter_kwargs['eps_sl']
             * (f_jmin * jmin)**(-6)) * (-np.tan(I_LK + (np.pi - I_LK) / 2) / 2)
    sigm_iout = dIout_tot / (Iout_dot_th * np.sqrt(2 * np.pi))
    try:
        tleftidx = np.where(t > t_Iout_smoothed[offset_idx + idxmax]
                            - 8 * sigm_iout)[0][0]
        trightidx = np.where(t < t_Iout_smoothed[offset_idx + idxmax]
                             + 5 * sigm_iout)[0][-1]
    except:
        tleftidx = 0
        trightidx = len(t) - 1
    xlim_idxs = [tleftidx, min(trightidx, len(t) - 1)]

    fig, axs_orig = plt.subplots(2, 3, figsize=(16, 9))
    axs = np.reshape(axs_orig, np.size(axs_orig))

    dWtot, dWsl, dWdot, t_lkmids, dWslx, dWslz = get_dWs(ret_lk_sliced, getter_kwargs)
    axs[0].semilogy(t, a * a0, 'k')
    axs[0].set_ylabel(r'$a$ (AU)')

    axs[1].semilogy(t, 1 - e, 'k')
    axs[1].set_ylabel(r'$1 - e$')

    l1 = axs[2].plot(t, np.degrees(I), 'k', alpha=0.3, lw=0.7, label=r'$I$')
    l2 = axs[2].plot(t_eff, np.degrees(I_avg), 'r', alpha=1, lw=2,
                     label=r'$\bar{I}$')
    axs[2].set_ylim(bottom=82) # make some space for legend below plot
    axs[2].set_ylabel(r'$I$')
    axs[2].legend(fontsize=14, ncol=2)

    axs[3].plot(t, np.degrees(q_sl), 'k')
    axs[3].set_ylabel(r'$\theta_{\rm sl}$')

    axs[4].plot(t_lkmids[offset_idx + avg_len // 2:
                         -offset_idx - (avg_len + 1) // 2],
                averaged_qeff,
                'ro', ms=1.5 if len(averaged_qeff) > 100 else 3)
    axs[4].plot(t_eff, q_eff0, 'k', alpha=0.3)
    axs[4].set_ylabel(r'$\theta_{\rm e}$')

    axs[5].semilogy(t_lkmids, dWsl, 'g', label=r'$\overline{\Omega}_{\rm SL}$',
                    lw=1.5, alpha=0.5)
    axs[5].semilogy(t_lkmids, dWdot, 'r', label=r'$-\overline{\Omega}_{\rm L}$',
                    lw=1.5, alpha=0.5)
    axs[5].semilogy(t_lkmids, dWtot, 'k',
                    label=r'$\overline{\Omega}_{\rm e}$', lw=4)
    # axs[5].semilogy(lk_times[ :-1], 2 * np.pi / (np.diff(lk_times)), 'k--',
    #                 label=r'$2\pi / T_{\rm LK}$', lw=2)
    ylims = axs[5].get_ylim()
    axs[5].semilogy(t_Iout_smoothed[offset_idx:-offset_idx],
                    Iout_dot_smoothed, 'b',
                    lw=4, label=r'$-\dot{I}_{\rm e}$')
    # axs[5].semilogy(t_eff[offset_idx:-offset_idx], np.degrees(Iout_dot_gauss), 'r:',
    #                 lw=0.7, alpha=0.3, label=r'$\dot{I}_{\rm e,gauss}$')
    axs[5].legend(fontsize=14, ncol=2, loc='upper left')
    axs[5].set_ylim(ylims)
    axs[5].set_ylim(bottom=0.03)
    axs[5].set_ylabel(r'Frequency ($t_{\rm LK, 0}^{-1}$)')

    lk_axf = len(axs) - 1
    axs[lk_axf].set_xlabel(r'$t / t_{\rm LK,0}$')
    axs[lk_axf].set_xlim(left=t[0], right=t[-1])
    xticks = axs[lk_axf].get_xticks()[1:-1]
    for ax in axs[ :lk_axf]:
        ax.set_xticks(xticks)
        ax.set_xticklabels([])
    print('Saving', folder + fn_template % get_fn_I(I0))
    plt.tight_layout()
    plt.savefig(folder + fn_template % get_fn_I(I0), dpi=200)

    # zoomed in plots are fancier
    axs[3].clear()
    axs[4].clear()

    twin2 = axs[2].twinx()
    l3 = twin2.plot(t_Iout_smoothed, np.degrees(Iout_smoothed), 'b',
                    label=r'$-I_{\rm e}$')
    l4 = twin2.plot(t_Iout_smoothed, np.degrees(I1_smoothed), 'g',
                    label=r'$-I_{\rm 1}$')

    lns = l1 + l2 + l3 + l4
    twin2.set_ylim(bottom=-7)
    axs[2].legend(lns, [l.get_label() for l in lns], fontsize=12,
                  loc='lower left', ncol=4)
    twin2.set_ylabel(r'$-I_{\rm e}$')

    axs[2].plot(t, np.degrees(I), 'k', alpha=0.3, lw=0.7, label=r'$I$')
    axs[2].plot(t_eff, np.degrees(I_avg), 'r', alpha=1, lw=2,
                label=r'$\bar{I}$')

    W_lk_ratio = dWtot[ :-1] * np.diff(t_lkmids) / (2 * np.pi)
    # there's a spike in ratio at late times, probably not physical, don't plot
    last_idx = np.where(t_lkmids > t[xlim_idxs[1]])[0][0]
    axs[3].plot(t_lkmids[ :last_idx],
                W_lk_ratio[ :last_idx],
                'k', label=r'$\overline{\Omega}_{\rm e} / \Omega$')
    axs[3].plot(t_lkmids[ :-1], W1mags[ :-1] * np.diff(t_lkmids) / (2 * np.pi),
                'b', label=r'$\Omega_{\rm e1} / \Omega$')
    axs[3].legend(fontsize=14)
    # axs[3].set_ylabel(r'$\Omega_{\rm e} / \Omega$')

    axs[4].plot(t_lkmids[offset_idx + avg_len // 2:
                         -offset_idx - (avg_len + 1) // 2],
                np.abs(averaged_qeff - q_eff_pred), 'ro', ms=0.7, alpha=0.7,
                label=r'$\left|\Delta \theta_{\rm e}\right|$')
    axs[4].plot(t_Iout_smoothed[offset_idx:-offset_idx],
                np.degrees(Iout_dot_smoothed) /
                Weffmag_smootheds[offset_idx:-offset_idx] / 2,
                'g', lw=1.5,
                label=r'$\dot{I}_{\rm e} / \overline{\Omega}_{\rm e}$')

    I0_lkmids = interp1d(t_Iout_smoothed, Iout_smoothed)(t_lkmids[1:last_idx])
    I1_lkmids = interp1d(t_Iout_smoothed, I1_smoothed)(t_lkmids[1:last_idx])
    pert1_strength = W1mags[ :-1] / np.abs(
        (2 * np.pi) / np.diff(t_lkmids) - dWtot[ :-1])
    axs[4].plot(t_lkmids[1:last_idx],
                np.degrees(np.sin(abs(I1_lkmids - I0_lkmids))
                           * pert1_strength[1:last_idx] / 2), 'b',
                label='Harmonic')
    axs[4].set_ylabel('Degrees')
    axs[4].legend(fontsize=12, ncol=2)
    axs[4].set_yscale('log')
    axs[4].set_ylim(bottom=1e-3)

    axs[lk_axf].set_xlim(left=t[xlim_idxs[0]], right=t[xlim_idxs[1]])
    xticks = axs[lk_axf].get_xticks()[1:-1]
    for ax in axs[ :lk_axf]:
        ax.set_xticks(xticks)
        ax.set_xticklabels([])
        ax.set_xlim([t[xlim_idxs[0]], t[xlim_idxs[1]]])
    plt.tight_layout()
    plt.savefig(folder + fn_template % get_fn_I(I0) + '_zoom', dpi=200)
    plt.close()

def get_plot_good_quants(ret_lk, s_vec, getter_kwargs, time_slice=np.s_[::]):
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
    Lhat_xy = get_hat(W[eff_idx], np.full_like(W[eff_idx], np.pi / 2))
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
    dts = np.diff(t_eff[1:-2])
    # phi_lks = []
    # t_smoothed = []
    # for t in t_lkmids:
    #     try:
    #         idx = np.where(t_eff > t)[0][0]
    #         phi_lks.append(np.unwrap(phi_Weff)[idx])
    #         t_smoothed.append(t_eff[idx])
    #     except:
    #         break
    # print(np.array(phi_lks[1: ]) - np.array(phi_lks[ :-1]))
    # phi_Weffsmooth = interp1d(t_smoothed, phi_lks)(t_eff[2:-202])
    for phi_offset in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        dqeff = np.sum(np.degrees(Iout_dot_gauss)
                       * np.cos(Weffmag[2:-2] * t_eff[2:-2] + phi_offset)
                       # * np.cos(phi_Weff[2:-2] + phi_offset)
                       * dts)
        dqeff_gauss_curr = np.sum(
            np.degrees(Iout_dot_gauss)
            * np.cos(phi_Weff[2:-2] + phi_offset)
            * dts)
        # dqeff_gauss_curr = np.sum(
        #     np.degrees(Iout_dot_gauss[:-200])
        #     * np.cos(phi_Weffsmooth + phi_offset)
        #     * dts[:-200])
        dqeff_max = max(dqeff_max, abs(dqeff))
        # if abs(dqeff_gauss_curr) > dqeff_gauss:
        #     phi_offset_max = phi_offset
        dqeff_gauss = max(dqeff_gauss, abs(dqeff_gauss_curr))

    # debug frequencies: conclusion, Weffmag is too noisy for this integral
    # Weff_const = np.full_like(t_eff[2:-2], Weffmag[2 + idxmax])
    # Weff_changing = np.full_like(t_eff[2:-2], Weffmag[2 + idxmax])\
    #     + 0.003 * (t_eff[2:-2] - t_eff[2 + idxmax])
    # # plt.semilogy(t_eff[2:-2], Weff_changing * 1.01, 'g')
    # plt.plot(t_eff, Weffmag, 'b')
    # # plt.semilogy(t_eff[1: ], np.diff(np.unwrap(phi_Weff)) / np.diff(t_eff), 'r')
    # plt.plot(t_eff[3:-202], np.diff(np.unwrap(phi_Weffsmooth)) /
    #              np.diff(t_eff[2:-202]), 'r')
    # plt.ylim(bottom=0.01, top=1000)

    # twinx = plt.gca().twinx()
    # twinx.plot(t_eff[2:-2], np.cumsum(
    #     Iout_dot_gauss * np.cos(phi_Weff[2:-2] + phi_offset_max) * dts),
    #            'bo', ms=0.3, alpha=0.7)
    # ylims = twinx.get_ylim()
    # # cumsum_th = np.cumsum(
    # #     Iout_dot_gauss[:-200]
    # #     * np.cos(phi_Weffsmooth + phi_offset_max)
    # #     * dts[:-200])
    # # print(np.degrees(Iouts[-1] - Iouts[0]))
    # # print(cumsum_th[-1], dqeff, dqeff_gauss, (Iouts[-1] - Iouts[0])
    # #       * np.exp(-(Weff_const[0]**2 * Iout_sigm2) / 2))
    # twinx.plot(t_eff[2:-202], np.cumsum(
    #     Iout_dot_gauss[:-200]
    #     * np.cos(phi_Weffsmooth + phi_offset_max)
    #     * dts[:-200]),
    #            'ko', ms=0.3, alpha=0.7)
    # # twinx.plot(t_eff, _q_eff0 - _q_eff0[-1], 'go',
    # #                        ms=0.3, alpha=0.7)
    # twinx.set_yscale('symlog', linthreshy=1e-4)
    # twinx.set_ylim(ylims)
    # plt.savefig('/tmp/plots', dpi=200)
    # plt.close()
    # raise ValueError('foo')

    # print('I0, dqeff_max, dqeff_gauss, Weffmag[2 + idxmax], Weffmag[tleftidx]')
    # tleftidx = np.where(t_eff > t_eff[2 + idxmax] - np.sqrt(Iout_sigm2))[0][0]
    # print(I0, dqeff_max, dqeff_gauss, Weffmag[2 + idxmax], Weffmag[tleftidx])
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
            np.linspace(t_lkmids[2], t_lkmids[4], 1000)))
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
                      label=r'$\overline{\Omega}_{\rm e}$', lw=2)
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
    #              label=r'$\overline{\Omega}_{\rm e}$', lw=2)
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

def plot_good_quants():
    '''
    Using ensemble data, plot: dWeff(A=1), Adot(A = 1), max(Ioutdot), Iout
    width

    Figure out whether can de-noise Ioutdot?
    '''
    # just use intermediate values, otherwise dupes
    I_degs = np.arange(90.175, 90.51, 0.05)
    I_degs_ensemble = np.arange(90.15, 90.5001, 0.01)
    dirs = ['4sims/'] * len(I_degs)
    dirs_ensemble = ['4sims_scan/'] * len(I_degs_ensemble)
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
        dq_dl_maxes = []
        tmerges = []
        for folder, I_deg in zip(dirs + dirs_ensemble, all_Is):
            ret_lk = get_kozai(folder, I_deg, getter_kwargs)
            _, t_eff, _, Iouts, Iout_dot, Weffmag, _, dWsl, dWdot, _,\
                dqeff, _, _= get_plot_good_quants(ret_lk, None, getter_kwargs)
            tmerges.append(ret_lk[0][-1])
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
            Iout_dot_maxes.append(Iout_dot[mid_idx - 2])

            dq_dl_maxes.append(np.max(np.abs(Iout_dot) / np.abs(Weffmag[2:-2])))

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
                 dqeff_num, dqeff_gauss, tmerges, dq_dl_maxes), f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            dWeff_mids, dWeff_Imaxes, Iout_dot_maxes, Ioutdot_max_gaussest,\
                dqeff_num, dqeff_gauss, tmerges, dq_dl_maxes = pickle.load(f)

    # try estimating Iout_dot scaling
    f_jmin = 2.65
    jmin = np.sqrt(5 * cosd(all_Is)**2 / 3)
    Iout_dot_th = 1.5 * (
        200 * getter_kwargs['eps_gw'] / getter_kwargs['eps_sl']
             * (f_jmin * jmin)**(-6)) * (-np.tan(I_LK + (np.pi - I_LK) / 2) / 2)
    Weff_th = (1.5)**(5/8) * (
        getter_kwargs['eps_sl']**(3/8) / (f_jmin * jmin)**(11/8))\
        * 2 * (np.cos(I_LK / 2))
    as_idx = np.argsort(all_Is)

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    # ax2.semilogy(all_Is, dqeff_num, 'bo', label='Num', ms=1)
    # ax2.semilogy(all_Is, dqeff_gauss, 'go', label=r'Gaussian Fit')
    # ax2.legend(fontsize=12)
    # ax2.set_ylabel(r'$\max \Delta \theta_{\rm eff}$')
    # ax2.set_xlabel(r'$I$ (Deg)')

    # fig, ax1 = plt.subplots(1, 1, figsize=(6, 6), sharex=True)
    # ax1.semilogy(all_Is, dWeff_mids, 'ko', label=r'$\Omega_{\rm e}$', ms=1)
    # ax1.semilogy(all_Is, dWeff_Imaxes, 'bo',
    #              label=r'$\Omega_{\rm e}$ (at $\dot{I}_{\rm e}$ max)')
    # ax1.semilogy(all_Is, Iout_dot_maxes, 'go',
    #              label=r'$\dot{I}_{\rm e}$', ms=1)
    # ax1.semilogy(all_Is[as_idx], Weff_th[as_idx], 'g', alpha=0.5,
    #              label=r'$\Omega_{\rm e}$ (Th)')
    # ax1.semilogy(all_Is[as_idx], Iout_dot_th[as_idx], 'r', alpha=0.5,
    #              label=r'$\dot{I}_{\rm e}$ (Th)')
    # ax1.semilogy(all_Is, Ioutdot_max_gaussest, 'ro',
    #              label=r'$\dot{I}_{\rm e}$ (Bounds)')
    # ax1.legend(fontsize=10)
    # ax1.set_ylabel(r'Frequency ($t_{\rm LK, 0}^{-1}$)')
    # ax1.set_yticks([0.01, 0.1, 1, 10, 100])
    # ax1.set_yticklabels([r'$0.01$', r'$0.1$', r'$1$', r'$10$', r'$100$'])

    ax1 = plt.gca()
    ax1.semilogy(all_Is,
                 np.degrees(np.array(Iout_dot_maxes)) / np.array(dWeff_mids),
                 'ko', label='Numeric')
    ax1.plot(all_Is[as_idx], np.degrees(Iout_dot_th[as_idx]) / Weff_th[as_idx],
             'r', label='Analytic')
    ax1.set_ylabel(r'$\left|\dot{I}_{\rm e} / \overline{\Omega}_{\rm e}\right|'
                   r'_{\max}$ (Deg)')
    ax1.set_xlabel(r'$I_0$')
    ax1.legend(fontsize=14)

    tmerge_ax = ax1.twiny()
    tmerge_ax.set_xlim(ax1.get_xlim())
    tmerge_ax.set_xticks(I_degs[::2])
    tmerge_ax.set_xticklabels([
        '%.1f' % np.log10(tm) for tm in tmerges[::2]
    ])
    plt.setp(tmerge_ax.get_xticklabels(), rotation=45)
    tmerge_ax.set_xlabel(r'$\log_{10} (T_{\rm m} / t_{\rm LK,0})$')

    plt.tight_layout()
    plt.savefig('4sims/good_quants', dpi=200)
    plt.close()

    pass

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

def get_qslfs_base(folder, I_deg, q_sb0, af, phi_sb, **kwargs):
    ''' this is the ensemble simulation, default do not save '''
    mkdirp(folder)
    ret_lk = get_kozai(folder, I_deg, getter_kwargs)

    if phi_sb == 0:
        pkl_fn = '4sim_qsl' + ('%d' % np.degrees(q_sb0)) + '_%s.pkl'
    else:
        pkl_fn = '4sim_qsl' + ('%d' % np.degrees(q_sb0)) + \
            ('_phi_sb%d' % np.degrees(phi_sb)) + '_%s.pkl'
    s_vec = get_spins_inertial(
        folder, I_deg, ret_lk, getter_kwargs,
        q_sb0=q_sb0,
        phi_sb=phi_sb,
        pkl_template=pkl_fn,
        save=True,
        atol=1e-9,
        rtol=1e-9,
        method='BDF',
        )

# for run_ensemble, idk how to starmap w/ kwargs
def get_kozai_kwargs(folder, I_deg, getter_kwargs):
    return get_kozai(folder, I_deg, getter_kwargs,
                     af=3e-3, atol=1e-9, rtol=1e-9, method='LSODA')

def run_ensemble(folder, I_vals=np.arange(90.15, 90.5001, 0.005),
                 af=3e-3):
    mkdirp(folder)

    n_pts = 10
    mus_edges = np.linspace(-1, 1, n_pts + 1)
    phis_edges = np.linspace(0, 2 * np.pi, n_pts + 1)
    mus = (mus_edges[ :-1] + mus_edges[1: ]) / 2
    qsb_arr = np.arccos(mus)
    phis = (phis_edges[ :-1] + phis_edges[1: ]) / 2

    with Pool(32) as p:
        p.starmap(
            get_kozai_kwargs,
            [(folder, I_deg, getter_kwargs) for I_deg in I_vals])

    args = []
    for I_deg in I_vals[::-1]:
        for q_sb0 in qsb_arr:
            for phi_sb in phis:
                args.append((folder, I_deg, q_sb0, af, phi_sb))
    with Pool(32) as p:
        p.starmap(get_qslfs_base, args)

# make the prediction for theta_sl_f using N=0 prediction (re-process from pkl)
def plot_deviations_good(folder):
    pkl_fn = folder + 'deviations_good.pkl'
    I_vals=np.arange(90.15, 90.5001, 0.005)

    n_pts = 10
    mus_edges = np.linspace(-1, 1, n_pts + 1)
    phis_edges = np.linspace(0, 2 * np.pi, n_pts + 1)
    mus = (mus_edges[ :-1] + mus_edges[1: ]) / 2
    qsb_arr = np.arccos(mus)
    phis = (phis_edges[ :-1] + phis_edges[1: ]) / 2

    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        deltas_deg = []
        dqeff_maxes = []
        dqeff_maxes_g = []
        for I_deg in I_vals:
            deltas_per_I = []
            ret_lk = get_kozai(folder, I_deg, getter_kwargs)
            # all the plots run in run_ensemble + run_ensemble_phase
            for q_sb0 in qsb_arr:
                for phi_sb in phis:
                    if phi_sb == 0:
                        pkl_template = '4sim_qsl'\
                            + ('%d' % np.degrees(q_sb0)) + '_%s.pkl'
                    else:
                        pkl_template = '4sim_qsl' + ('%d' % np.degrees(q_sb0))\
                            + ('_phi_sb%d' % np.degrees(phi_sb)) + '_%s.pkl'
                    try:
                        s_vec = get_spins_inertial(
                            folder, I_deg, ret_lk, getter_kwargs,
                            q_sb0=q_sb0,
                            phi_sb=phi_sb,
                            pkl_template=pkl_template,
                            load=False
                            )
                    except ValueError:
                        continue
                    _, _, Wf, If, _ = ret_lk[1][:, -1]
                    sf = s_vec[:, -1]

                    Lhat = get_hat(Wf, If)
                    qslf = np.degrees(np.arccos(np.dot(Lhat, sf)))

                    # predict q_eff final from locally nondissipative sim
                    getter_kwargs_nondisp = dict(getter_kwargs)
                    getter_kwargs_nondisp['eps_gw'] = 0
                    tf = 10 * ret_lk[2][0][0] # initial half-period
                    ret_nd = get_kozai('nosave', I_deg, getter_kwargs_nondisp,
                                       save=False, tf=tf,
                                       atol=1e-9, rtol=1e-9, method='LSODA')
                    ret_lk_nd = ret_nd.t, ret_nd.y, ret_nd.t_events
                    svec_ret = get_spins_inertial(
                        'nosave', I_deg, ret_lk_nd, getter_kwargs_nondisp,
                        q_sb0=q_sb0,
                        phi_sb=phi_sb,
                        save=False,
                        atol=1e-9,
                        rtol=1e-9,
                        method='BDF',
                        )
                    s_vec_nd = svec_ret.sol(ret_lk_nd[0])

                    # Use shortened ret_lk to pass in just the first few Kozai
                    # cycles
                    _, _, dWdot, t_lkmids, dWslx, dWslz = get_dWs(
                        ret_lk_nd, getter_kwargs)
                    eff_idx = np.where(np.logical_and(
                        ret_lk_nd[0] < t_lkmids[-1], ret_lk_nd[0] > t_lkmids[0]))[0]
                    t_eff = ret_lk_nd[0][eff_idx]
                    Wslx = interp1d(t_lkmids, dWslx)(t_eff)
                    Wslz = interp1d(t_lkmids, dWslz)(t_eff)
                    Wdot = interp1d(t_lkmids, dWdot)(t_eff)
                    W = ret_lk_nd[1][2]
                    Lhat_xy = get_hat(W[eff_idx],
                                      np.full_like(W[eff_idx], np.pi / 2))
                    Weff = np.outer(np.array([0, 0, 1]),
                                    -Wdot + Wslz) + Wslx * Lhat_xy
                    Weff_hat = Weff / np.sqrt(np.sum(Weff**2, axis=0))
                    _q_eff0 = np.arccos(ts_dot(s_vec_nd[:, eff_idx], Weff_hat))
                    q_eff0 = np.degrees(_q_eff0)

                    q_eff0_interp = interp1d(t_eff, q_eff0)
                    q_eff_pred = np.mean(q_eff0_interp(
                        np.linspace(ret_lk_nd[2][0][1], ret_lk_nd[2][0][2], 10000)))

                    deltas_per_I.append(q_eff_pred - qslf)

            deltas_deg.append(deltas_per_I)
        with open(pkl_fn, 'wb') as f:
            pickle.dump(deltas_deg, f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            deltas_deg = pickle.load(f)
    for I, deltas_per_I in zip(I_vals, deltas_deg):
        if I == I_vals[0]:
            plt.loglog(np.full_like(deltas_per_I, I - 90), np.abs(deltas_per_I),
                       'ko', ms=0.3, label='Sim')
        else:
            plt.loglog(np.full_like(deltas_per_I, I - 90), np.abs(deltas_per_I),
                       'ko', ms=0.3)
    plt.xlabel(r'$I - 90^\circ$ (Deg)')
    plt.ylabel(r'$\left|\Delta \theta_{\rm e}^{(f)}\right|$ (Deg)')
    plt.xlim(left=0.15)
    plt.ylim(bottom=1e-3, top=10)

    jmin = np.sqrt(5 * cosd(I_vals)**2 / 3)
    Iout_dot_th = (
        200 * getter_kwargs['eps_gw'] / getter_kwargs['eps_sl']
             * (f_jmin * jmin)**(-6)) * (-np.tan(I_LK + (np.pi - I_LK) / 2) / 2)
    Weff_th = (getter_kwargs['eps_sl']**(3/8) / (f_jmin * jmin)**(11/8)) \
        * 2 * (-np.cos(I_LK))
    dIout_tot = (np.pi - I_LK - np.radians(10)) # Iout(t=0) ~ 10 deg
    sigm_iout = dIout_tot / (Iout_dot_th * np.sqrt(2 * np.pi))
    dqeff_th = np.degrees(dIout_tot) * np.exp(
        -(Weff_th**2 * sigm_iout**2) / 2)
    # plt.plot(I_vals - 90, dqeff_th, 'b', lw=2,
    #          label=r'$\Omega_{\rm e}$ Constant')
    plt.plot(I_vals - 90, np.degrees(Iout_dot_th / Weff_th), 'r',
             label=r'$[\dot{I}_{\rm e} / \overline{\Omega}_{\rm e}]_{\max}$')

    # try second scaling?
    Weff_ratio = 1/8
    dqeff_new = np.degrees(dIout_tot) / (
        np.sqrt(2 * sigm_iout * Weff_th * Weff_ratio)) * np.exp(
            -1 / (8 * Weff_ratio**2))
    # plt.plot(I_vals - 90, dqeff_new, 'c', lw=2, label='Linear')

    plt.legend(fontsize=14, loc='lower left')
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
                 t_eff, Weff_hat, t_lkmids, Lhat_f, eff_idx, avg_periods):
    print('Running for', q, phi)
    pkl_template = '4sim_qsl' + ('%d' % np.degrees(q)) + \
        ('_phi_sb%d' % np.degrees(phi)) + '_%s.pkl'
    s_vec = get_spins_inertial(
        newfolder, I_deg, ret_lk, getter_kwargs, atol=atol,
        rtol=rtol, q_sb0=q, phi_sb=phi,
        pkl_template=pkl_template)

    mu_eff0 = ts_dot(s_vec[:, eff_idx], Weff_hat)
    _q_eff0 = np.arccos(mu_eff0)
    q_eff0 = np.degrees(_q_eff0)
    q_eff0_interp = interp1d(t_eff, q_eff0)
    t_avg = np.linspace(t_lkmids[2], t_lkmids[2 + avg_periods], 1000)
    q_eff_pred = np.mean(q_eff0_interp(t_avg))

    Lout_hat = get_hat(0 * t_eff, 0 * t_eff)
    yhat = ts_cross(Weff_hat, Lout_hat)
    xhat = ts_cross(yhat, Weff_hat)
    sin_phiWeff = (ts_dot(s_vec[:, eff_idx], yhat) / np.sin(_q_eff0))
    cos_phiWeff = (ts_dot(s_vec[:, eff_idx], xhat) / np.sin(_q_eff0))
    phi_Weff = np.unwrap(np.arctan2(sin_phiWeff, cos_phiWeff))

    # mu_eff0_avg = np.sum(mu_eff0 * np.gradient(phi_Weff)) / (
    #     phi_Weff[-1] - phi_Weff[0])

    # predict q_eff final by averaging over an early Kozai cycle
    # min_dt = np.min(np.diff(phi_Weff))
    # pred_t_vals = np.arange(t_lkmids[2], t_lkmids[2 + avg_periods], min_dt)
    # mu_eff0_interp = interp1d(t_eff, mu_eff0)(pred_t_vals)
    # phi_Weffinterp = interp1d(t_eff, np.unwrap(phi_Weff))(pred_t_vals)
    # mu_eff0_avg = np.sum(mu_eff0_interp * np.gradient(phi_Weffinterp)) / (
    #     phi_Weff[-1] - phi_Weff[0])
    # q_eff_pred = np.degrees(np.arccos(mu_eff0_avg))

    # leftidx = np.where(phi_Weff > phi_Weff[0] + 2 * np.pi)[0][0]
    # rightidx = np.where(phi_Weff > phi_Weff[0] + 4 * np.pi)[0][0]
    # t_avg = np.linspace(t_eff[leftidx], t_eff[rightidx], 1000)
    # q_eff_pred = np.mean(q_eff0_interp(t_avg))

    dot_slf = np.dot(Lhat_f, s_vec[:,-1])
    q_slf = np.degrees(np.arccos(dot_slf))
    print('Ran for', q, phi, 'final angs are', q_eff_pred, q_slf)
    return q_eff_pred - q_slf

# try 90.5 simulation over an isotropic grid of ICs
def run_905_grid(I_deg=90.5, newfolder='4sims905/', af=5e-3, n_pts=20,
                 atol=1e-7, rtol=1e-7, getter_kwargs=getter_kwargs,
                 orig_folder='4sims/', suffix='', avg_periods=1):
    mkdirp(newfolder)

    pkl_fn = newfolder + ('devgrid%s.pkl' % suffix)
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
        Lhat_xy = get_hat(W[eff_idx], np.full_like(W[eff_idx], np.pi / 2))
        Weff = np.outer(np.array([0, 0, 1]), -Wdot + Wslz) + Wslx * Lhat_xy
        Weff_hat = Weff / np.sqrt(np.sum(Weff**2, axis=0))

        args = []
        for q in np.arccos(mus):
            for phi in phis:
                args.append((newfolder, I_deg, ret_lk, getter_kwargs, atol, rtol,
                             q, phi, t_eff, Weff_hat, t_lkmids, Lhat_f, eff_idx,
                             avg_periods))
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
    plt.pcolormesh(phis_grid, mus_grid, res_nparr, cmap='viridis')
    plt.colorbar()
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\cos \theta$')
    plt.title(r'$\Delta \theta_{\rm eff}$')
    plt.tight_layout()
    plt.savefig(newfolder + 'devsgrid' + suffix, dpi=200)
    plt.close()

    plt.hist(sorted_res_nparr[1:-1], bins=30)
    plt.xlabel(r'$\Delta \theta_{\rm eff}$')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(newfolder + 'devshist' + suffix, dpi=200)
    plt.close()

def bifurcation(m_t=60, num_ratios=10, num_cycles=50, I_deg=70,
                folder='4_bifurcation/', tol=1e-10):
    mkdirp(folder)
    mass_ratios = np.linspace(0, 1, num_ratios + 2)[1:-1]
    # mass_ratios = [0.5]
    pkl_fn = folder + 'bifurcation.pkl'
    m3, a0, a2, e2 = 30, 0.1, 3, 0
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        qsl_arrs = []
        qeff_arrs = []
        for mass_ratio in mass_ratios:
            m1 = mass_ratio * m_t
            m2 = (1 - mass_ratio) * m_t
            getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
            getter_kwargs['eps_gw'] = 0
            # just run to fixed tf, idk how to count periods
            tf = num_cycles * 10 # probably approximately right idk
            ret_lk = get_kozai(folder, I_deg, getter_kwargs, tf=tf,
                               atol=tol, rtol=tol, dense_output=True,
                               save=False)
            t, y, lk_events = ret_lk.t, ret_lk.y, ret_lk.t_events
            ret_s = get_spins_inertial(folder, I_deg, [t, y, lk_events],
                                       getter_kwargs, save=False,
                                       atol=tol, rtol=tol)

            # evaluate qsl, qeff
            a, e, W, I, w = ret_lk.sol(lk_events[0])
            Lhat_arr = get_hat(W, I)
            s_arr = ret_s.sol(lk_events[0])
            qsl = np.degrees(np.arccos(ts_dot(Lhat_arr, s_arr)))
            print(mass_ratio, qsl.min(), qsl.max())
            qsl_arrs.append(qsl)
        with open(pkl_fn, 'wb') as f:
            pickle.dump((qsl_arrs, qeff_arrs), f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            qsl_arrs, qeff_arrs = pickle.load(f)
    for mass_ratio, qsl_arr in zip(mass_ratios, qsl_arrs):
        plt.plot(np.full_like(qsl_arr, mass_ratio), qsl_arr, 'bo', ms=1)
    plt.xlabel(r'$m_1 / m_{12}$')
    plt.ylabel(r'$\theta_{\rm SL}$ (eccentricity maxima)')
    plt.title(r'$(m_{12}, m_3, a_0, a_2, e_0, e_2) ='
              '(%d M_{\odot}, %d M_{\odot}, %.1f \;\mathrm{AU},'
              '%d\;\mathrm{AU}, %.3f, %d)$'
              % (m_t, m3, a0, a2, 1e-3, e2), fontsize=12)
    plt.savefig(folder + 'bifurcation', dpi=150)
    plt.close()

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

    # for I_deg in np.arange(90.15, 90.51, 0.05):
    #     run_for_Ideg('4sims/', I_deg, short=True, plotter=plot_good)
    # run_for_Ideg('4sims/', 90.475, short=True, plotter=plot_good)
    # run_for_Ideg('4sims/', 90.5, plotter=plot_good, short=True,
    #              ylimdy=2)
    # run_for_Ideg('4sims/', 90.2, plotter=plot_good, short=True,
    #              ylimdy=2, time_slice=np.s_[300::])
    # run_for_Ideg('4sims/', 90.35, plotter=plot_good, short=True,
    #              ylimdy=2, time_slice=np.s_[5500:10500])
    # run_for_Ideg('4sims/', 90.5, plotter=plot_good, short=True,
    #              time_slice=np.s_[-45000:-20000])

    # for I_deg in np.arange(90.15, 90.51, 0.025)[::-1]:
    # for I_deg in [90.2, 90.35]:
    #     run_for_Ideg('4sims/', I_deg, plotter=plot_all, short=True,
    #                  atol=1e-10, rtol=1e-10)
    # plot_good_quants()

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

    run_ensemble('4sims_scan/')
    plot_deviations_good('4sims_scan/')

    # run_905_grid()
    # run_905_grid(newfolder='4sims905_htol/', orig_folder='4sims905_htol/',
    #              atol=1e-10, rtol=1e-10, n_pts=30, avg_periods=2)
    # run_905_grid(newfolder='4sims905_htol/', orig_folder='4sims905_htol/',
    #              atol=1e-10, rtol=1e-10, n_pts=30, avg_periods=3,
    #              suffix='_avg1')
    # getter_kwargs_in = get_eps(30, 30, 30, 0.1, 3, 0)
    # run_905_grid(I_deg=80, newfolder='4sims80/', af=0.5, n_pts=10,
    #              atol=1e-7, rtol=1e-7, getter_kwargs=getter_kwargs_in,
    #              orig_folder='4inner/')

    # bifurcation(num_cycles=200, num_ratios=10)
    # bifurcation(num_cycles=200, num_ratios=50, I_deg=70)
    pass
