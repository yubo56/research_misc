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

def get_dW(e0, I0):
    '''
    total delta Omega over an LK cycle

    wdot = 3 * sqrt(h) / 4 * (1 - 2 * (x0 - h) / (x - h))
    '''
    x0 = 1 - e0**2
    h = x0 * np.cos(I0)**2

    # quadratic for x1/x2
    b = -(5 + 5 * h - 2 * x0) / 3
    c = 5 * h / 3
    x1 = (-b - np.sqrt(b**2 - 4 * c)) / 2
    x2 = (-b + np.sqrt(b**2 - 4 * c)) / 2

    k_sq = (x0 - x1) / (x2 - x1)
    K = spe.ellipk(k_sq)
    ne = 6 * np.pi * np.sqrt(6) / (8 * K) * np.sqrt(x2 - x1)
    def dWdt(t, _):
        q = K / np.pi * (ne * t + np.pi)
        x = x0 + (x1 - x0) * spe.ellipj(q, k_sq)[1]**2
        return 3 * np.sqrt(h) / 4 * (
            1 - 2 * (x0 - h) / x - h)
    ret = solve_ivp(dWdt, (0, 2 * np.pi / ne), [0],
                    atol=1e-9, rtol=1e-9)
    return ret.y[0, -1]

def plot_dWs():
    I0_max = np.pi - np.arccos(np.sqrt(3/5))
    I0s = np.linspace(np.pi / 2 + 0.001, I0_max, 50)
    e0_labels = ['1e-3', '0.01', '0.1', '0.3', '0.9']
    e0s = [1e-3, 0.01, 0.1, 0.3, 0.9]
    plt.axhline(-np.pi, c='k', ls='-')
    for e0, lbl in zip(e0s, e0_labels):
        dWs = []
        for I0 in I0s:
            dWs.append(get_dW(e0, I0))
        plt.plot(np.degrees(I0s), dWs, ls='', marker='o', label=lbl, ms=1.0)
    plt.xlabel(r'$I_0$')
    plt.ylabel(r'$\Delta \Omega$')
    plt.ylim(bottom=-2 * np.pi, top=0)
    plt.legend(fontsize=10, ncol=3)
    plt.tight_layout()
    plt.savefig('5_dWs', dpi=200)

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
    I_avgs = [] # average Is over each LK period, W weighted
    t_mids = [] # t during the middle of each LK period
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

if __name__ == '__main__':
    # plot_dWs()
    # get_I_avg_traj('4sims/', '4sim_lk_90_500')
    # get_I_avg_traj('4sims/', '4sim_lk_90_400')
    # get_I_avg_traj('4sims/', '4sim_lk_90_250')

    m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
    getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    getter_kwargs['eps_gr'] = 0
    folder = '4sims/'
    I_deg = 90.4
    ret_lk = funcs4.get_kozai(folder, I_deg, getter_kwargs, af=5e-3, atol=1e-9,
                              rtol=1e-9, pkl_template='4sim_nogr_%s.pkl')
    s_vec = funcs4.get_spins_inertial(folder, I_deg, ret_lk, getter_kwargs,
                                      atol=1e-8, rtol=1e-8,
                                      pkl_template='4sim_nogr_s_%s.pkl')
    funcs4.plot_all(folder, ret_lk, s_vec, getter_kwargs,
                    fn_template='4sim_nogr_%s')
    get_I_avg_traj(folder, '4sim_nogr_90_400')
