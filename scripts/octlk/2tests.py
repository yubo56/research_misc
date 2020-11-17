'''
a few test model explorations:

* check whether Hamiltonian can give merger boundaries

* make plot of "single shot merger" region of parameter space
    - overlay the DA/SA transition as well
    - overlay the effective merger reion?
'''
from utils import *

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
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

    e2_vals = np.linspace(0, 0.57, 300)
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
        Icrit = brenth(opt_func, np.radians(50), np.radians(90))
        Icrits.append(Icrit)

        # fig = plt.figure(figsize=(6, 6))
        # I_plot = np.radians(np.linspace(89, 90, 20))
        # H_plot = []
        # for I in I_plot:
        #     H_plot.append(get_H(e2, I, eta0, e1=0, eps_naoz=0))
        # plt.plot(np.degrees(I_plot), H_plot)
        # plt.axhline(Hlim + hoct_max * 15 / 4 * eps_oct)
        # plt.savefig('/tmp/foo' + str(idx))
        # plt.close()

    fig = plt.figure(figsize=(6, 6))
    eps_oct_plot = a0 / a2 * e2_vals / (1 - e2_vals**2)
    plt.plot(eps_oct_plot, np.degrees(Icrits), label='Me')
    plt.plot(eps_oct_plot, np.degrees(np.arccos(np.sqrt(
        0.26 * (eps_oct_plot / 0.1)
        - 0.536 * (eps_oct_plot / 0.1)**2
        + 12.05 * (eps_oct_plot / 0.1)**3
        -16.78 * (eps_oct_plot / 0.1)**4
    ))), label='MLL16')
    plt.legend(fontsize=14)
    plt.xlabel(r'$\epsilon_{\rm oct}$')
    plt.ylabel(r'$I_{0, \lim}$ (Deg)')
    plt.savefig('2_ilim_eta0', dpi=300)
    plt.close()

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

if __name__ == '__main__':
    calculate_elim_regions()
    # minimize_hoct(0.8, to_print=True)
    pass
