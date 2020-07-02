''' use t_LK = 1 throughout '''
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('lines', lw=3.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brenth
from scipy.interpolate import interp1d
import pickle
funcs4 = __import__('4orb_sims')
from utils import *
from scipy.fftpack import fft

def check_anal_dWs(e0=0.01, I0=np.radians(80), eps_gr=0, eps_gw=0, eps_sl=1,
                   **kwargs):
    dW, dWSL = get_dW(e0, I0, eps_gr, eps_gw, eps_sl)
    dW_anal, dWSL_anal = get_dW_anal(e0, I0, eps_sl=eps_sl)

    print(dWSL_anal, dWSL)
    print(dW_anal, dW)

def plot_dWs(fn='5_dWs', num_Is=200, **kwargs):
    ''' useful when eps_gr not << 1, significant corrections '''
    I0_max = np.pi - np.arccos(np.sqrt(3/5))
    I0s = np.linspace(np.pi / 2 + 0.001, I0_max, num_Is)
    I0s_d = np.degrees(I0s)

    e0_labels = ['1e-3', '0.003', '0.03', '0.2', '0.9']
    e0s = [1e-3, 0.01, 0.1, 0.3, 0.9]
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

    pkl_fn = fn + '.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        dWs_lst = []
        dWs_anal_lst = []
        for e0 in e0s:
            dWs = []
            dWs_anal = []
            for I0 in I0s:
                print(e0, I0)
                dW_num, (dWSLz_num, dWSLx_num) = get_dW(e0, I0, **kwargs)
                dWs.append(np.sqrt((abs(dW_num) + dWSLz_num)**2 + dWSLx_num**2))

                dW_anl, (dWSLz_anl, dWSLx_anl) = get_dW_anal(e0, I0)
                dWs_anal.append(np.sqrt(
                    (abs(dW_anl) + dWSLz_anl)**2 + dWSLx_anl**2))

            dWs_lst.append(dWs)
            dWs_anal_lst.append(dWs_anal)
        with open(pkl_fn, 'wb') as f:
            pickle.dump((dWs_lst, dWs_anal_lst), f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            (dWs_lst, dWs_anal_lst) = pickle.load(f)
    I0s_d_flip = 180 - I0s_d
    for e0, lbl, c, dWs, dWs_anal in\
            zip(e0s, e0_labels, colors, dWs_lst, dWs_anal_lst):
        plt.plot(I0s_d, np.array(dWs) / (2 * np.pi),
                 ls='', marker='o', c=c, label=lbl, ms=1.0)
        plt.plot(I0s_d_flip, np.array(dWs) / (2 * np.pi),
                 ls='', marker='o', c=c, ms=1.0)
        # plt.plot(I0s_d, dWs_anal / (2 * np.pi),
        #          c=c, ls=':', lw=1.0)
    plt.xlabel(r'$I_0$')
    plt.ylabel(r'$\bar{\Omega}_{\rm e} / \Omega$')
    plt.ylim(bottom=0, top=1.5)
    plt.legend(fontsize=10, ncol=3)
    plt.tight_layout()
    plt.savefig(fn, dpi=200)
    plt.close()

def get_I_avg_traj(folder, pkl_head):
    with open(folder + pkl_head + '.pkl', 'rb') as f:
        t, (a, e, W, I, w), t_events = pickle.load(f)
    a_int = interp1d(t, a)
    e_int = interp1d(t, e)
    I_int = interp1d(t, I)
    W_int = interp1d(t, W)
    w_int = interp1d(t, w)
    num_periods = int(W[-1] // (2 * np.pi))
    start = 0
    I_avgs = [] # average Is over each W period, W weighted
    t_mids = [] # t during the middle of each W period
    for i in range(num_periods - 1):
        end = brenth(lambda t: W_int(t) - (i + 1) * (2 * np.pi), start, t[-1])

        # def I_dWdt(t):
        #     dWdt = (3 * a_int(t)**(3/2) * np.cos(I_int(t)) *
        #             (5 * e_int(t)**2 * np.cos(w_int(t))**2
        #              - 4 * e_int(t)**2 - 1)
        #         / (4 * np.sqrt(1 - e_int(t)**2)))
        #     return I_int(t) * dWdt
        # I_avg2 = quad(I_dWdt, start, end, limit=100)[0] / (2 * np.pi)

        # other, lazier integration
        dt = min(np.diff(t[np.where(np.logical_and(t <= end, t >= start))[0]]))
        ts = np.arange(start, end, dt / 4)
        Is = I_int(ts)
        dWs = np.gradient(W_int(ts))
        I_avg = np.sum(Is * dWs) / (2 * np.pi)
        # print(I_avg, I_avg2, I[-1])

        I_avgs.append(np.degrees(I_avg))
        t_mids.append((end + start) / 2)
        # print('Processed between', start, end)
        start = end
    plt.plot(t_mids, I_avgs, 'ko', ms=1.0)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\left\langle I \right \rangle$ (Deg)')
    plt.axhline(np.degrees(I[-1]), c='r')
    ax1 = plt.gca()
    ax3 = ax1.twinx()
    ax3.set_yticks([np.degrees(I[-1])])
    ax3.set_yticklabels([r'$%.2f^\circ$' % np.degrees(I[-1])])
    ax3.set_ylim(ax1.get_ylim())
    plt.tight_layout()
    plt.savefig('5I_avg_' + pkl_head, dpi=200)
    plt.close()

def plot_Wdot_ft(folder, pkl_head):
    '''
    we want the Fourier components of the fast-varying component in the
    Hamliltonian, so the components of <(W_sl * cos(I)) + Wdot>, <W_sl * sin(I)>

    Let's plot the shape of these coefficients (within one LK cycle) evolving
    over time (or the cycle number...)
    '''
    m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
    getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    with open(folder + pkl_head + '.pkl', 'rb') as f:
        t, (a, e, W, I, w), [t_lks, _] = pickle.load(f)
    a_int = interp1d(t, a)
    e_int = interp1d(t, e)
    I_int = interp1d(t, I)
    W_int = interp1d(t, W)
    w_int = interp1d(t, w)
    ffts_x = []
    ffts_z = []
    N_maxs = []
    times = []
    num_lk = len(t_lks) // 16
    size = 20000
    for start, end in list(zip(t_lks[ :-1], t_lks[1: ]))[ :num_lk:num_lk // 8]:
        ts = np.linspace(start, end, size)
        e_t = e_int(ts)
        dWdt = (3 * a_int(ts)**(3/2) * np.cos(I_int(ts)) *
                (5 * e_t**2 * np.cos(w_int(ts))**2
                 - 4 * e_t**2 - 1)
            / (4 * np.sqrt(1 - e_t**2)))
        W_sl = getter_kwargs['eps_sl'] / (a_int(ts)**(5/2) * (1 - e_t**2))
        fftz = 2 * np.real(fft(W_sl * np.cos(I_int(ts)) + dWdt)[1:size//2])
        fftx = 2 * np.real(fft(W_sl * np.sin(I_int(ts)))[1:size//2])
        # fftz = np.real(fft(W_sl * np.cos(I_int(ts)) + dWdt))
        # fftx = np.real(fft(W_sl * np.sin(I_int(ts))))
        ffts_x.append(fftx / size)
        ffts_z.append(fftz / size)
        N_maxs.append(int(1 / np.sqrt(1 - max(e_t)**2)))
        times.append((end + start) / 2)

    idxs = np.arange(1, size // 2)
    # idxs = np.arange(size) - (size // 2)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] + \
        ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    for fftz, nmax, time, c in zip(ffts_z, N_maxs, times, colors):
        plt.semilogy(idxs, fftz, c=c, label=r'$%.1f$' % time, alpha=0.7)
        plt.semilogy(idxs, -fftz, c=c, ls=':', alpha=0.7)
        plt.xscale('log')
    plt.xlim(right=size // 4)
    # plt.xlim((-size // 20, size // 20))
    plt.ylim(bottom=1e-5)
    plt.xlabel(r'$N$')
    plt.legend(fontsize=10, ncol=2, loc='upper right')
    plt.tight_layout()
    plt.savefig('5ffts_' + pkl_head, dpi=200)
    plt.close()

def plot_dWeff_mags(folder, pkl_head, stride=1, size=50):
    '''
    we want the Fourier components of the fast-varying component in the
    Hamliltonian, so the components of <(W_sl * cos(I)) + Wdot>, <W_sl * sin(I)>

    Let's plot the shape of these coefficients (within one LK cycle) evolving
    over time (or the cycle number...)
    '''
    # m1, m2, m3, a0, a2, e2 = 30, 30, 30, 0.1, 3, 0
    m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0

    getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    with open(folder + pkl_head + '.pkl', 'rb') as f:
        ret_lk = pickle.load(f)
    t, (a, e, W, I, w), [t_lks, _] = ret_lk
    t_lk0, _, _, _ = get_vals(m1, m2, m3, a0, a2, e2, I[0])
    # print(t_lk0)
    dWeff_mags, dWsl_mags, dWdot_mags, times, _, _ = get_dWs(
        ret_lk, getter_kwargs, stride, size)
    times *= t_lk0
    if I[0] < np.radians(90):
        dWdot_mags *= -1

    plt.loglog(times[-1] - times, dWeff_mags, 'k')
    plt.loglog(times[-1] - times, dWsl_mags, 'r')
    plt.loglog(times[-1] - times, dWdot_mags, 'g')
    # idx = np.where(dWdot_mags - dWsl_mags < 0)[0][0]
    # print('dWs', dWsl_mags[idx], dWdot_mags[idx])
    # print('ddot(phi)',
    #       dWeff_mags[idx + 1] - dWeff_mags[idx - 1] /
    #       (times[idx + 1] - times[idx - 1]))
    # print('dA/dt', (
    #     np.log(dWsl_mags[idx + 1] / dWdot_mags[idx + 1])
    #     - np.log(dWsl_mags[idx - 1] / dWdot_mags[idx - 1])
    # ) / (times[idx + 1] - times[idx - 1]) * t_lk0)
    # plt.ylim(-5, 5)
    plt.xlabel(r'$t_{\rm f} - t$')
    plt.savefig('5dWeffs_' + pkl_head, dpi=200)
    plt.close()

    delta_ts = np.diff(t_lks)[::stride]
    plt.semilogx(times[-1] - times, dWeff_mags * delta_ts, 'k')
    plt.semilogx(times[-1] - times, dWsl_mags * delta_ts, 'r')
    plt.semilogx(times[-1] - times, dWdot_mags * delta_ts, 'g')
    # plt.ylim(0.01, 10)
    plt.axhline(np.pi)
    plt.xlabel(r'$t_{\rm f} - t$')
    plt.savefig('5dWeffs_' + pkl_head + '_dphi', dpi=200)
    plt.close()

def resonance_sim_single(_W0, psi, tf, q0, tol, eps, freq_mult):
    ''' inertial frame '''
    W0 = np.array([0, 0, _W0])
    W1hat = np.array([np.sin(psi), 0, np.cos(psi)])
    def dydt(t, y):
        return np.cross(W0, y) + eps * (
            np.sin(freq_mult * _W0 * t) * np.cross(W1hat, y))
    return solve_ivp(dydt, (0, tf), [0, np.sin(q0), np.cos(q0)],
                     atol=tol, rtol=tol)

def resonance_sim_model1(_W0, psi, tf, q0, tol, eps, freq_mult):
    '''
    instead of using a sinusoidal component, use a rotating component
    indeed, resonance at freq_mult = 0.5 *vanishes* compared to sim_single
    '''
    W0 = np.array([0, 0, _W0])
    def dydt(t, y):
        W1hat = np.array([np.sin(psi) * np.sin(freq_mult * _W0 * t),
                          np.sin(psi) * np.cos(freq_mult * _W0 * t),
                          np.cos(psi)])
        return np.cross(W0, y) + eps * np.cross(W1hat, y)
    return solve_ivp(dydt, (0, tf), [0, np.sin(q0), np.cos(q0)],
                     atol=tol, rtol=tol)

def resonance_sim(_W0=0.5, psi=np.radians(60), tf=500, q0=np.radians(20),
                  tol=1e-8, num_freqs=250, freq_max=2.5,
                  fn='5_resonance_heights'):
    '''
    simulate the linear resonance effect across perturbation frequency +
    strength
    '''
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    eps_arr = [0.01, 0.1, 0.3, 1]
    freq_mults = np.linspace(0.01, 2.5, num_freqs)
    # freq_mults = [0.95, 1, 1.05]
    d_omega_cutoff = 1e-4
    # NB: code's eps * _W0 = eps in notes

    pkl_fn = '%s.pkl' % fn
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        q_amps_tot = []
        q_amps_th_tot = []
        for eps, c in zip(eps_arr, colors):
        # for eps, c in zip([0.1], colors):
            q_amps = []
            q_amps_th = []
            for freq_mult in freq_mults:
                ret = resonance_sim_single(_W0, psi, tf, q0, tol, eps, freq_mult)

                q_amps.append(ret.y[2, :].max() - ret.y[2, :].min())
                print('Finished for', eps, freq_mult, q_amps[-1])
                dW = freq_mult - 1
                if abs(dW) < d_omega_cutoff: # regularize div-by-zero
                    dW = d_omega_cutoff
                psi_p = np.arctan(eps * np.sin(psi) / dW)
                q_amps_th.append(abs(np.cos(q0) - np.cos(2 * psi_p - q0)))
            q_amps_tot.append(q_amps)
            q_amps_th_tot.append(q_amps_th)
        with open(pkl_fn, 'wb') as f:
            pickle.dump((q_amps_tot, q_amps_th_tot), f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            q_amps_tot, q_amps_th_tot = pickle.load(f)
    for q_amps, q_amps_th, c, eps in\
            zip(q_amps_tot, q_amps_th_tot, colors, eps_arr):
        plt.plot(freq_mults, q_amps, '%so' % c, ms=1, label=eps)
        # plt.plot(freq_mults, q_amps_th, '%s:' % c, lw=1)
    plt.xlabel(r'$\Omega_0 / \omega$')
    plt.ylabel(r'$\max \cos \theta - \min \cos \theta$')
    plt.legend(fontsize=14)
    plt.savefig(fn, dpi=200)
    plt.close()

def plot_resonance_rates():
    '''
    plot time growth rate of the N = 1/2 parametric resonances (versus eps) and
    the maximum amplitude
    '''
    pass

if __name__ == '__main__':
    m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
    getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    plot_dWs(**getter_kwargs, intg_pts=int(3e5))

    m1, m2, m3, a0, a2, e2 = 30, 30, 30, 0.1, 3, 0
    getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    plot_dWs(fn='5_dWs_inner', intg_pts=int(3e5),
             **getter_kwargs)

    # print('No GR limit')
    # check_anal_dWs()
    # m1, m2, m3, a0, a2, e2 = 30, 30, 30, 0.1, 3, 0
    # getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    # print('Compact Params')
    # check_anal_dWs(**getter_kwargs)
    # print('Compact Params (I = 100)')
    # check_anal_dWs(I0=np.radians(100), **getter_kwargs)
    # m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
    # getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    # print('Outer Params')
    # check_anal_dWs(**getter_kwargs)

    # get_I_avg_traj('4sims/', '4sim_lk_90_500')
    # get_I_avg_traj('4sims/', '4sim_lk_90_400')
    # get_I_avg_traj('4sims/', '4sim_lk_90_250')

    # test get_I_avg w/o pericenter precession
    # m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
    # getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    # getter_kwargs['eps_gr'] = 0
    # folder = '4sims/'
    # I_deg = 90.4
    # ret_lk = funcs4.get_kozai(folder, I_deg, getter_kwargs, af=5e-3, atol=1e-9,
    #                           rtol=1e-9, pkl_template='4sim_nogr_%s.pkl')
    # s_vec = funcs4.get_spins_inertial(folder, I_deg, ret_lk, getter_kwargs,
    #                                   atol=1e-8, rtol=1e-8,
    #                                   pkl_template='4sim_nogr_s_%s.pkl')
    # funcs4.plot_all(folder, ret_lk, s_vec, getter_kwargs,
    #                 fn_template='4sim_nogr_%s')
    # get_I_avg_traj(folder, '4sim_nogr_90_400')

    # plot_Wdot_ft('4sims/', '4sim_lk_90_500')
    # plot_dWeff_mags('4inner/', '4sim_lk_80_000', stride=100)

    # plot_dWeff_mags('4sims/', '4sim_lk_90_200')
    # plot_dWeff_mags('4sims/', '4sim_lk_90_300')
    # plot_dWeff_mags('4sims/', '4sim_lk_90_350')
    # plot_dWeff_mags('4sims/', '4sim_lk_90_500')

    # ret = resonance_sim_single(1, np.radians(0.01), 1000,
    #                            np.radians(20), 1e-8, 0.1, 1)
    # plt.plot(ret.t, ret.y[2, :])
    # plt.savefig('/tmp/foo')

    # resonance_sim(tf=5000, freq_max=2, num_freqs=500, tol=1e-6)
    # resonance_sim(tf=5000, freq_max=2, num_freqs=500, tol=1e-6,
    #               psi=np.radians(90), fn='5_resonance_sims_90')
    # resonance_sim(tf=5000, freq_max=2, num_freqs=500, tol=1e-6,
    #               psi=np.radians(5), fn='5_resonance_sims_5')
    # resonance_sim(tf=5000, freq_max=2, num_freqs=500, tol=1e-6,
    #               psi=np.radians(45), fn='5_resonance_sims_45')

    pass
