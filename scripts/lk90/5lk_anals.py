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
from scipy.integrate import solve_ivp

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

if __name__ == '__main__':
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
