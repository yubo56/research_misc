''' use t_LK = 1 throughout '''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('lines', lw=3.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
import scipy.special as spe
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brenth
from scipy.interpolate import interp1d
import pickle
funcs4 = __import__('4orb_sims')
from utils import *
from scipy.fftpack import fft

def get_dW(e0, I0, eps_gr=0, eps_gw=0, eps_sl=1,
           atol=1e-9, rtol=1e-9, intg_pts=int(1e5)):
    '''
    total delta Omega over an LK cycle, integrate the LK equations w/ eps_gr and
    eps_gw

    wdot = 3 * sqrt(h) / 4 * (1 - 2 * (x0 - h) / (x - h))
    '''
    a0 = 1
    W0 = 0
    w0 = 0

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
    def term_event(t, y):
        return y[4] - np.pi
    term_event.terminal = True
    ret = solve_ivp(dydt, (0, np.inf), [a0, e0, W0, I0, w0],
                    events=[term_event], atol=atol, rtol=rtol,
                    dense_output=True)
    times = np.linspace(0, ret.t[-1], intg_pts)
    a_arr, e_arr, W_arr, I_arr, w_arr = ret.sol(times)
    dWsl_z = np.sum(eps_sl / a_arr**(5/2) * np.cos(I_arr) / (1 - e_arr**2)
                    * ret.t[-1] / len(times))
    dWsl_x = np.sum(eps_sl / a_arr**(5/2) * np.sin(I_arr) / (1 - e_arr**2)
                    * ret.t[-1] / len(times))
    return ret.y[2, -1], (dWsl_z, dWsl_x)

def get_dW_anal(e0, I0, intg_pts=int(1e5), eps_sl=0, **kwargs):
    # calculate n_e...
    x0 = 1 - e0**2
    h = x0 * np.cos(I0)**2
    b = -(5 + 5 * h - 2 * x0) / 3
    c = 5 * h / 3
    x1 = (-b - np.sqrt(b**2 - 4 * c)) / 2
    x2 = (-b + np.sqrt(b**2 - 4 * c)) / 2
    k_sq = (x0 - x1) / (x2 - x1)
    K = spe.ellipk(k_sq)
    ne = 6 * np.pi * np.sqrt(6) / (8 * K) * np.sqrt(x2 - x1)

    def solve_Wsl():
        phi = np.linspace(0, np.pi, intg_pts)
        intg_tot = np.pi / (
            (x0 + (x1 - x0) * np.cos(phi)**2)
                * np.sqrt(1 - k_sq * np.sin(phi)**2)
                * ne * K)
        intg_z = np.pi * np.sqrt(h) / (
            (x0 + (x1 - x0) * np.cos(phi)**2)**(3/2)
                * np.sqrt(1 - k_sq * np.sin(phi)**2)
                * ne * K)
        intg_x = np.pi * np.sqrt(1 - h / (x0 + (x1 - x0) * np.cos(phi)**2)) / (
            (x0 + (x1 - x0) * np.cos(phi)**2)
                * np.sqrt(1 - k_sq * np.sin(phi)**2)
                * ne * K)
        # wsl_tot = np.sum(intg_tot * phi[-1] / len(phi))
        wsl_z = np.sum(intg_z * phi[-1] / len(phi))
        wsl_x = np.sum(intg_x * phi[-1] / len(phi))
        return eps_sl * wsl_z, eps_sl * wsl_x
    def solve_dot():
        phi = np.linspace(0, np.pi, intg_pts)
        intg = (np.sin(I0)**2 / np.sqrt(1 - k_sq * np.sin(phi)**2)) / (
            np.sin(I0)**2 + (x1 / x0 - 1) * np.cos(phi)**2)
        int_res = np.sum(intg * phi[-1] / len(phi))
        return (3 * np.sqrt(h) * 2 * np.pi) / (4 * ne) * (1 - 1 / K * int_res)
    return solve_dot(), solve_Wsl()
def check_anal_dWs(e0=0.01, I0=np.radians(80), eps_gr=0, eps_gw=0, eps_sl=1,
                   **kwargs):
    dW, dWSL = get_dW(e0, I0, eps_gr, eps_gw, eps_sl)
    dW_anal, dWSL_anal = get_dW_anal(e0, I0, eps_sl=eps_sl)

    print(dWSL_anal, dWSL)
    print(dW_anal, dW)

def plot_dWs(fn='5_dWs', num_Is=100, **kwargs):
    ''' useful when eps_gr not << 1, significant corrections '''
    I0_max = np.pi - np.arccos(np.sqrt(3/5))
    I0s = np.linspace(np.pi / 2 + 0.001, I0_max, num_Is)
    I0s_d = np.degrees(I0s)

    e0_labels = ['1e-3', '0.01', '0.1', '0.3', '0.9']
    e0s = [1e-3, 0.01, 0.1, 0.3, 0.9]
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

    for e0, lbl, c in zip(e0s, e0_labels, colors):
        dWs = []
        dWs_anal = []
        for I0 in I0s:
            print(e0, I0)
            dW_num, (dWSLz_num, dWSLx_num) = get_dW(e0, I0, **kwargs)
            dWs.append(np.sqrt((abs(dW_num) + dWSLz_num)**2 + dWSLx_num**2))

            dW_anl, (dWSLz_anl, dWSLx_anl) = get_dW_anal(e0, I0)
            dWs_anal.append(np.sqrt(
                (abs(dW_anl) + dWSLz_anl)**2 + dWSLx_anl**2))
        plt.plot(I0s_d, dWs, ls='', marker='o', c=c, label=lbl, ms=1.0)
        plt.plot(I0s_d, dWs_anal, c=c, ls=':', lw=1.0)
    plt.xlabel(r'$I_0$')
    plt.ylabel(r'$\Delta \Omega$')
    plt.ylim(bottom=0, top=2 * np.pi)
    plt.axhline(np.pi, c='k', ls='-')
    plt.legend(fontsize=10, ncol=3)
    plt.tight_layout()
    plt.savefig(fn, dpi=200)

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

if __name__ == '__main__':
    m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
    getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    plot_dWs(**getter_kwargs)

    m1, m2, m3, a0, a2, e2 = 30, 30, 30, 0.1, 3, 0
    getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    plot_dWs(fn='5_dWs_inner', **getter_kwargs)

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
