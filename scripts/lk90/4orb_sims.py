'''
does Kozai simulations in orbital elements, then computes spin evolution after
the fact w/ a given trajectory
'''
import pickle
import os

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

def get_fn_I(I_deg):
    return ('%.3f' % I_deg).replace('.', '_')

def get_kozai(I_deg, getter_kwargs,
              a0=1, e0=1e-3, W0=0, w0=0, tf=np.inf, af=0,
              pkl_template='4sim_lk_%s.pkl', **kwargs):
    I0 = np.radians(I_deg)
    pkl_fn = '4sims/' + pkl_template % get_fn_I(I_deg)

    if not os.path.exists(pkl_fn):
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
                    * (2 * (1 - e**2) + 5 * np.sin(w)**2 * (e**2 - np.sin(I)**2))
                    / (4 * np.sqrt(x))
                + eps_gr / (a**(5/2) * (1 - e**2))
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
        print('Finished for I=%.1f, took %d steps, t_f %.3f (%d cycles)' %
              (I_deg, len(t), t[-1], len(t_events[0])))

        with open(pkl_fn, 'wb') as f:
            pickle.dump((t, y, t_events), f)
    else:
        print('Loading %s' % pkl_fn)
        with open(pkl_fn, 'rb') as f:
            t, y, t_events = pickle.load(f)
    return t, y, t_events

def get_spins_inertial(I_deg, ret_lk, getter_kwargs,
                       q_sl0=0,
                       f_sl0=0, # not supported yet
                       pkl_template='4sim_s_%s.pkl',
                       **kwargs):
    ''' uses the same times as ret_lk '''
    pkl_fn = '4sims/' + pkl_template % get_fn_I(I_deg)
    if not os.path.exists(pkl_fn):
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
        print('Finished spins for I=%.1f, took %d steps' %
              (I_deg, len(ret.t)))

        with open(pkl_fn, 'wb') as f:
            pickle.dump(y, f)
    else:
        print('Loading %s' % pkl_fn)
        with open(pkl_fn, 'rb') as f:
            y = pickle.load(f)
    return y

def plot_all(ret_lk, s_vec, getter_kwargs,
             fn_template='4sim_%s', time_slice=np.s_[::],
             **kwargs):
    lk_t, lk_y, _ = ret_lk
    sx, sy, sz = s_vec
    fig, axs_orig = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
    axs = np.reshape(axs_orig, np.size(axs_orig))
    a, e, W, I, w = lk_y[:, time_slice]
    t = lk_t[time_slice]
    I0 = np.degrees(I[0])

    K = np.sqrt(1 - e**2) * np.cos(I)
    def get_hat(phi, theta):
        return np.array([np.cos(phi) * np.sin(theta),
                         np.sin(phi) * np.sin(theta),
                         np.cos(theta)])
    Lhat = get_hat(W, I)
    q_sl = np.degrees(np.arccos(ts_dot(Lhat, s_vec)))
    q_sb = np.degrees(np.arccos(sz))
    Wsl = getter_kwargs['eps_sl'] / (a**(5/2) * (1 - e**2))
    Wdot_eff = (
        3 * a**(3/2) / 4 * np.cos(I) * (4 * e**2 + 1) / np.sqrt(1 - e**2))
    Wdot = (
        3 * a**(3/2) / 4 * np.cos(I) * (
            5 * e**2 * np.cos(w)**2 - 4 * e**2 - 1) / np.sqrt(1 - e**2))
    A = np.abs(Wsl / Wdot)

    # use the averaged Wdot, else too fluctuating
    Wdot_eff = (
        3 * a**(3/2) / 4 * np.cos(I) * (4 * e**2 + 1) / np.sqrt(1 - e**2))
    W_eff_bin = Wsl * Lhat + Wdot_eff * get_hat(0 * I, 0 * I)
    W_eff_bin /= np.sqrt(np.sum(W_eff_bin**2, axis=0))
    q_eff_bin = np.degrees(np.arccos(ts_dot(W_eff_bin, s_vec)))

    # oh boy...
    def get_opt_func(Wsl, Wdot, I):
        return lambda Io: Wsl * np.sin(I - Io) - Wdot * np.sin(Io)
    Iouts = []
    for Wsl_val, Wdot_val, I_val in zip(Wsl, Wdot_eff, I):
        Iout_val = brenth(
            get_opt_func(Wsl_val, Wdot_val, I_val),
            0,
            np.pi)
        Iouts.append(Iout_val)
    Iouts = np.array(Iouts)
    W_eff_me = get_hat(W, Iouts)
    q_eff_me = np.degrees(np.arccos(ts_dot(W_eff_me, s_vec)))

    axs[0].semilogy(t, a, 'r')
    axs[0].set_ylabel(r'$a$')
    axs[1].semilogy(t, 1 - e, 'r')
    axs[1].set_ylabel(r'$1 - e$')
    axs[2].plot(t, W % (2 * np.pi), 'ro', ms=0.5)
    axs[2].set_ylabel(r'$\Omega$')
    axs[3].plot(t, np.degrees(I), 'r')
    axs[3].set_ylabel(r'$I$')
    axs[4].plot(t, w % (2 * np.pi), 'ro', ms=0.5)
    axs[4].set_ylabel(r'$w$')
    axs[5].plot(t, K, 'r')
    axs[5].set_ylabel(r'$K$')
    axs[6].plot(t, q_sl, 'r')
    axs[6].set_ylabel(r'$\theta_{\rm sl}$ ($\theta_{\rm sl,f} = %.2f$)'
                      % q_sl[-1])
    axs[7].plot(t, q_sb, 'r')
    axs[7].set_ylabel(r'$\theta_{\rm sb}$ ($\theta_{\rm sb,i} = %.2f$)'
                      % q_sb[0])
    axs[8].semilogy(t, A, 'r')
    axs[8].set_ylabel(r'$\Omega_{\rm SL} / \Omega_{\rm GR}$')
    axs[9].plot(t, q_eff_bin, 'r')
    axs[9].set_ylabel(r'$\theta_{\rm eff, S}$ ($\theta_{\rm eff, S} = %.2f$)'
                      % q_eff_bin[0])
    axs[10].plot(t, q_eff_me, 'r')
    axs[10].set_ylabel(r'$\theta_{\rm YS}$ ($\theta_{\rm YS} = %.2f$)'
                      % q_eff_me[0])
    axs[11].plot(t, np.degrees(Iouts), 'r')
    axs[11].plot(t, np.degrees(I), 'k')
    axs[11].set_ylabel(r'$I_o$')

    axs_orig[-1][0].set_xlabel(r'$t / t_{LK,0}$')

    plt.tight_layout()
    plt.savefig('4sims/' + fn_template % get_fn_I(I0), dpi=200)
    plt.clf()

def run_for_Ideg(I_deg, af=1e-4):
    m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
    getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)

    ret_lk = get_kozai(I_deg, getter_kwargs, af=af, atol=1e-10, rtol=1e-10)
    s_vec = get_spins_inertial(I_deg, ret_lk, getter_kwargs,
                               atol=1e-10,
                               rtol=1e-10)
    plot_all(ret_lk, s_vec, getter_kwargs)

    # try with q_sl0
    s_vec = get_spins_inertial(I_deg, ret_lk, getter_kwargs,
                               q_sl0=np.radians(20),
                               atol=1e-10,
                               rtol=1e-10,
                               pkl_template='4sim_qsl20_%s.pkl')
    plot_all(ret_lk, s_vec, getter_kwargs,
             fn_template='4sim_qsl20_%s')

    s_vec = get_spins_inertial(I_deg, ret_lk, getter_kwargs,
                               q_sl0=np.radians(40),
                               atol=1e-10,
                               rtol=1e-10,
                               pkl_template='4sim_qsl40_%s.pkl')
    plot_all(ret_lk, s_vec, getter_kwargs,
             fn_template='4sim_qsl40_%s')
if __name__ == '__main__':
    # I_deg = 90.35
    # m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
    # getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    # ret_lk = get_kozai(I_deg, getter_kwargs, af=1e-3, atol=1e-10, rtol=1e-10)
    # s_vec = get_spins_inertial(I_deg, ret_lk, getter_kwargs)
    # plot_all(ret_lk, s_vec, getter_kwargs)

    for I_deg in np.arange(90.1, 90.51, 0.05):
        run_for_Ideg(I_deg)
