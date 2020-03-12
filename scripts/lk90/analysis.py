import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('lines', lw=3.5)

from utils import *

FN = '1sim_90_36.pkl'
def plot_vec(ax, v, c, l):
    ''' plots arrow for vec, using x-z coords, y = alpha '''
    try:
        ax.annotate('', (v[0], v[2]), xytext=(0, 0),
                    arrowprops={'fill': v[1] > 0,
                                'alpha': np.abs(v[1]),
                                'width': 0.7,
                                'color': c})
        deg = np.radians(10)
        rot = lambda x, y: (np.cos(deg) * x - np.sin(deg) * y,
                            np.sin(deg) * x + np.cos(deg) * y)
        ax.text(*rot(v[0], v[2]), l, color=c, fontsize=14)
    except:
        print('Error', v)

def plot_time(ret, idx, filename='dir/analysis'):
    L, e, s = to_vars(ret.y)
    ax = plt.gca()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    L_idx = L[:, idx]
    e_idx = e[:, idx]
    Lhat = L_idx / np.sqrt(np.sum(L_idx**2))
    ehat = e_idx / np.sqrt(np.sum(e_idx**2))
    plot_vec(ax, Lhat, 'r', r'$\vec{L}$')
    plot_vec(ax, ehat, 'k', r'$\vec{e}$')
    plot_vec(ax, s[:, idx], 'g', r'$\vec{s}$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$z$')
    mu_sl = np.dot(Lhat, s[:, idx])
    ax.text(-0.9, 0.9, r'$e = %.4f, t = %.3f, \cos \theta_{SL} = %.3f$' %
            (np.sqrt(np.sum(e_idx**2)), ret.t[idx], mu_sl))
    plt.savefig(filename + '_%09d' % idx)
    plt.clf()

def plot_lk_phase():
    e0 = 0.001
    I0 = np.radians(95)
    n_pts = 100
    emax = np.sqrt(1 - 5 * np.cos(I0)**2 / 3)
    log_neg_e = np.linspace(0, np.log10(1 - emax) * 1.05, n_pts)
    omega = np.linspace(0, 2 * np.pi, n_pts)
    log_neg_e_grid = np.outer(log_neg_e, np.ones_like(omega))
    omega_grid = np.outer(np.ones_like(log_neg_e), omega)
    e_grid = 1 - (10**log_neg_e_grid)
    I_grid = np.arccos(np.sqrt(1 - e0**2) * np.cos(I0) /
                       np.sqrt(1 - e_grid**2))
    H_vals = (
        (2 + 3 * e_grid**2) * (3 * np.cos(I_grid)**2 - 1)
         + 15 * e_grid**2 * np.sin(I_grid)**2 * np.cos(omega_grid * 2))
    plt.contour(omega_grid, log_neg_e_grid, H_vals, cmap='RdBu_r')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\log_{10} (1 - e)$')
    plt.savefig('analysis_H')
    plt.clf()

    plt.contour(omega_grid, np.degrees(I_grid), H_vals, cmap='RdBu_r')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$I$')
    plt.savefig('analysis_H_I')
    plt.clf()

if __name__ == '__main__':
    plot_lk_phase()
    # with open(FN, 'rb') as f:
    #     ret = pickle.load(f)

        # plot particular idxs of the simulation
        # idxs = np.where(np.logical_and(ret.t < 24, ret.t > 26))[0][::200]
        # idxs = np.arange(len(ret.t))[::00]
        # idxs = np.where(ret.t < 25.5)[0][::2]
        # for idx in idxs:
        #     plot_time(ret, idx)

        # try plotting L at [non-]peak-e moments of oscillation
        # min_e = 0.9
        # max_e = 0.998

        # e = np.sqrt(np.sum(ret.y[3:6]**2, axis=0))
        # left_idx = 0
        # guess_idx = np.where(e > max_e)[0][0]
        # while ret.t[guess_idx] < 27:
        #     right_idx = np.where(e[guess_idx : ] < min_e)[0][0] + guess_idx
        #     plot_idx = np.argmax(e[guess_idx: right_idx]) + guess_idx
        #     plot_time(ret, plot_idx)
        #     print(plot_idx, ret.t[plot_idx])

        #     plot_idx = np.argmin(e[left_idx: guess_idx]) + left_idx
        #     plot_time(ret, plot_idx)
        #     print(plot_idx, ret.t[plot_idx])

        #     left_idx = right_idx
        #     guess_idx = np.where(e[left_idx : ] > max_e)[0][0] + left_idx
