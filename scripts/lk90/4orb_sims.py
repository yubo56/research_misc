'''
does Kozai simulations in orbital elements, then computes spin evolution after
the fact w/ a given trajectory
'''
import pickle
import os

from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('lines', lw=1.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
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
                       q_sl0=0,
                       f_sl0=0, # not supported yet
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
                a(t) * (1 - e(t)**2))
        t0 = t_lk[0]
        s0 = [Lx(t0), Ly(t0), Lz(t0)]

        # q_sl0 rotation
        xy_mag = np.sqrt(s0[0]**2 + s0[1]**2)
        new_zmag = s0[2] * np.cos(q_sl0) + xy_mag * np.sin(q_sl0)
        new_xy = -s0[2] * np.sin(q_sl0) + xy_mag * np.cos(q_sl0)
        s_new = [s0[0] * new_xy / xy_mag,
                 s0[1] * new_xy / xy_mag,
                 new_zmag]

        ret = solve_ivp(dydt, (t0, t_lk[-1]), s_new, dense_output=True, **kwargs)
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
    A = np.abs(Wsl / Wdot)
    A_eff = np.abs(Wsl / Wdot_eff)

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
    interp_idx = np.where(np.logical_and(
        t > min(lk_events[0]), A_eff < 1))[0]

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
    axs[5].set_ylim([2 * K[0], 0])
    axs[5].plot(t, K, 'r', alpha=alf)
    axs[5].set_ylabel(r'$K$')
    axs[6].semilogy(t, A, 'r', alpha=alf)
    axs[6].semilogy(t, A_eff, 'k', alpha=alf)
    axs[6].set_ylabel(r'$\Omega_{\rm SL} / \dot{\Omega}$')
    axs[6].axhline(1, c='k', lw=1)
    axs[7].plot(t, np.degrees(q_sl), 'r', alpha=alf)
    axs[7].set_ylabel(r'$\theta_{\rm sl}$ ($\theta_{\rm sl,f} = %.2f$)'
                      % np.degrees(q_sl)[-1])
    axs[8].plot(t, np.degrees(q_sb), 'r', alpha=alf)
    axs[8].set_ylabel(r'$\theta_{\rm sb}$ ($\theta_{\rm sb,i} = %.2f$)'
                      % np.degrees(q_sb)[0])
    axs[9].plot(t, q_eff_me, 'r', alpha=alf)
    axs[9].plot(t, q_eff_bin, 'k', alpha=alf)
    axs[9].set_ylabel(r'$\left<\theta_{\rm eff, S}\right>$' +
                      r'($\left<\theta_{\rm eff, S, f}\right> = %.2f$)'
                      % q_eff_bin[-1])
    axs[10].plot(t[phi_idxs], phi_sl % (2 * np.pi), 'r,', alpha=alf)
    axs[10].set_ylabel(r'$\phi_{\rm sl}$')
    # axs[11].plot(t[interp_idx],
    #              dphi_sb_dt[interp_idx] / lk_p_interp(t[interp_idx]),
    #              'r',
    #              alpha=alf,
    #              lw=0.5
    #              )
    # axs[11].set_ylabel(r'$\dot{\phi}_{\rm sb} / t_{LK}$')
    # axs[11].set_ylim((-5, 1))
    axs[11].plot(t, phi_sb, 'r', alpha=alf)
    axs[11].set_ylabel(r'$\phi_{\rm sb}$')
    lk_axf = 11 # so it's hard to forget lol

    # scatter plots in LK phase space
    final_idx = np.where(t < t[-1] * 0.7)[0][-1] # cut off plotting of end
    sl = np.s_[ :final_idx]
    axs[lk_axf + 1].scatter(w[sl] % (2 * np.pi), 1 - e[sl]**2, c=t[sl],
                            marker=',', alpha=0.5)
    axs[lk_axf + 1].set_yscale('log')
    axs[lk_axf + 1].set_ylim(min(1 - e**2), max(1 - e**2))
    axs[lk_axf + 1].set_xlabel(r'$\omega$')
    axs[lk_axf + 1].set_ylabel(r'$1 - e^2$')
    axs[lk_axf + 2].scatter(w[sl] % (2 * np.pi), np.degrees(I[sl]), c=t[sl],
                            marker=',', alpha=0.5)
    axs[lk_axf + 2].set_xlabel(r'$\omega$')
    axs[lk_axf + 2].set_ylabel(r'$I$')

    # set effectively for axs[0-9], label in last
    xticks = axs[lk_axf].get_xticks()[1:-1]
    for i in range(lk_axf):
        axs[i].set_xticks(xticks)
        axs[i].set_xticklabels([])
    axs[lk_axf].set_xlabel(r'$t / t_{LK,0}$')

    plt.tight_layout()
    plt.savefig(folder + fn_template % get_fn_I(I0), dpi=200)
    plt.close()

def run_for_Ideg(folder, I_deg, af=1e-4,
                 atol=1e-10, rtol=1e-10, **kwargs):
    mkdirp(folder)
    ret_lk = get_kozai(folder, I_deg, getter_kwargs,
                       af=af, atol=atol, rtol=rtol)
    s_vec = get_spins_inertial(folder, I_deg, ret_lk, getter_kwargs,
                               atol=atol,
                               rtol=rtol,
                               )
    plot_all(folder, ret_lk, s_vec, getter_kwargs)
    # return

    # try with q_sl0
    s_vec = get_spins_inertial(folder, I_deg, ret_lk, getter_kwargs,
                               q_sl0=np.radians(20),
                               atol=atol,
                               rtol=rtol,
                               pkl_template='4sim_qsl20_%s.pkl')
    plot_all(folder, ret_lk, s_vec, getter_kwargs,
             fn_template='4sim_qsl20_%s')

    s_vec = get_spins_inertial(folder, I_deg, ret_lk, getter_kwargs,
                               q_sl0=np.radians(40),
                               atol=atol,
                               rtol=rtol,
                               pkl_template='4sim_qsl40_%s.pkl')
    plot_all(folder, ret_lk, s_vec, getter_kwargs,
             fn_template='4sim_qsl40_%s')

def get_qslfs(folder, I_deg, q_sl_arr, af,
              atol=1e-10, rtol=1e-10,  save=False, **kwargs):
    ''' this is the ensemble simulation, default do not save '''
    mkdirp(folder)
    ret_lk = get_kozai(folder, I_deg, getter_kwargs,
                       af=af, atol=atol, rtol=rtol, save=save)
    qslfs = []
    t_mergers = []
    for q_sl0 in q_sl_arr:
        pkl_fn = '4sim_qsl' + ('%d' % np.degrees(q_sl0)) + '_%s.pkl'
        s_vec = get_spins_inertial(
            folder, I_deg, ret_lk, getter_kwargs,
            q_sl0=q_sl0,
            atol=atol,
            rtol=rtol,
            save=save,
            pkl_template=pkl_fn,
            )
        if save:
            fn_template = '4sim_qsl' + ('%d' % np.degrees(q_sl0)) + '_%s'
            plot_all(folder, ret_lk, s_vec, getter_kwargs,
                     fn_template=fn_template)
        _, _, W, I, _ = ret_lk[1][:, -1]
        sf = s_vec[:, -1]

        Lhat = get_hat(W, I)
        qslfs.append(np.degrees(np.arccos(np.dot(Lhat, sf))))

    return I_deg, ret_lk[0][-1], q_sl_arr, qslfs

def run_ensemble(folder, I_vals=np.arange(90.01, 90.5001, 0.002),
                 save_fn='ensemble.pkl'):
    if not os.path.exists(folder + save_fn):
        af = 3e-3
        mkdirp(folder)
        q_sl_arr = np.arange(-20, 21, 4)
        args = [(folder, I_deg, q_sl_arr, af) for I_deg in I_vals[::-1]]
        with Pool(N_THREADS) as p:
            res = p.starmap(get_qslfs, args)
        with open(folder + save_fn, 'wb') as f:
            pickle.dump(res, f)
    else:
        with open(folder + save_fn, 'rb') as f:
            res = pickle.load(f)
    return res

if __name__ == '__main__':
    # I_deg = 90.5
    # folder = ''
    # getter_kwargs['eps_gr'] *= 1
    # ret_lk = get_kozai(folder, I_deg, getter_kwargs, af=1e-3, atol=1e-10,
    #                    rtol=1e-10, pkl_template='4sim_gr0_lk_%s')
    # s_vec = get_spins_inertial(folder, I_deg, ret_lk, getter_kwargs,
    #                            pkl_template='4sim_gr0_s_%s')
    # plot_all(folder, ret_lk, s_vec, getter_kwargs, fn_template='4sim_gr0_%s')

    # for I_deg in np.arange(90.1, 90.51, 0.05)[-1: ]:
    # for I_deg in np.arange(90.1, 90.51, 0.05):
    #     run_for_Ideg('4sims/', I_deg)
    # run_for_Ideg('4sims/', 90.475)

    run_ensemble('4sims_ensemble/')
