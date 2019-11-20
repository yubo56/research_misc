import numpy as np

from scipy.integrate import quad
from scipy.stats import linregress
from scipy.fftpack import ifft
from scipy.optimize import bisect

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

ms = 2
def f(E, e):
    ''' returns true anomaly for given eccentric anomaly '''
    return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))

def hansen_integrand(N, m, e):
    '''
    gets the Hansen coefficient integrand for F_{Nm} including the 1/2pi
    normalization, so F_{Nm}(e = 0) = \delta_{Nm}
    '''
    def integrand(E):
        return (
            np.cos(N * (E - e * np.sin(E)) - m * f(E, e)) /
            (1 - e * np.cos(E))**2) / np.pi
    return integrand

def get_hansen(N, m, e):
    '''
    just evaluates the hansen integral (so I don't have to keep typing the
    bounds everywhere). We need about 2N limits I think.
    '''
    integrand = hansen_integrand(N, m, e)
    return quad(integrand, 0, np.pi, limit=max(100, int(2 * N)))

def get_coeffs(nmax, m, e):
    n_vals = np.arange(nmax)
    coeffs = np.zeros_like(n_vals, dtype=np.float64)
    coeffs2 = np.zeros_like(n_vals, dtype=np.float64)
    for idx, N in enumerate(n_vals):
        y, abserr = get_hansen(N, m, e)
        coeffs[idx] = y
        y2, abserr2 = get_hansen(N, -m, e)
        coeffs2[idx] = y2
        # print('%03d, %.8f, %.8f, %.8e' % (N, y, y2, abserr))
    return n_vals, coeffs, coeffs2

def get_coeffs_fft(nmax, m, e):
    n_used = 4 * nmax # nyquist not enough??
    def f(E):
        return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    def E(M):
        return bisect(lambda E_v: E_v - e * np.sin(E_v) - M, 0, 2 * np.pi)
    m_vals = 2 * np.pi * np.arange(n_used) / n_used
    f_vals = f(np.array([E(M) for M in m_vals]))
    func_vals = ((1 + e * np.cos(f_vals)) / (1 - e**2))**3\
        * np.exp(-1j * 2 * f_vals)
    func_vals2 = ((1 + e * np.cos(f_vals)) / (1 - e**2))**3\
        * np.exp(1j * 2 * f_vals)
    FN2_ifft = np.real(ifft(func_vals))
    FN2_ifft2 = np.real(ifft(func_vals2))
    return np.arange(nmax), FN2_ifft[ :nmax], FN2_ifft2[ :nmax]

def plot_hansens(m, e, coeff_getter=get_coeffs):
    N_peak = (1 - e)**(-3/2)
    print('Ansatz N', N_peak)

    nmax = int(20 * N_peak)
    # nmax = 1000
    n_vals, coeffs, coeffs2 = coeff_getter(nmax, m, e)

    max_n = np.argmax(np.abs(coeffs))
    max_c = np.max(np.abs(coeffs))
    pos_idx = np.where(coeffs > 0)[0]
    neg_idx = np.where(coeffs < 0)[0]
    plt.loglog(n_vals[pos_idx], np.abs(coeffs[pos_idx]) / max_c,
               'ko', ms=ms, label=r'$F_{N2} > 0$')
    plt.loglog(n_vals[neg_idx], np.abs(coeffs[neg_idx]) / max_c,
               'ro', ms=ms, label=r'$F_{N2} < 0$')
    max_n2 = np.argmax(np.abs(coeffs2))
    max_c2 = np.max(np.abs(coeffs2))
    pos_idx2 = np.where(coeffs2 > 0)[0]
    neg_idx2 = np.where(coeffs2 < 0)[0]
    plt.loglog(n_vals[pos_idx2], np.abs(coeffs2[pos_idx2]) / max_c2,
               'go', ms=ms, label=r'$F_{N-2} > 0$')
    plt.loglog(n_vals[neg_idx2], np.abs(coeffs2[neg_idx2]) / max_c2,
               'bo', ms=ms, label=r'$F_{N-2} < 0$')
    plt.xlabel('N')
    plt.ylabel(r'$F_{N2} / F_{N2,\max}$')
    plt.axvline(max_n, c='k', linewidth=1)
    plt.axvline(max_n2, c='g', linewidth=1)
    plt.axvline(N_peak, c='b')
    plt.title(r'$e = %.2f$' % e)
    plt.legend()
    plt.tight_layout()
    plt.savefig('hansens', dpi=400)
    plt.clf()

    plt.loglog(n_vals[pos_idx],
               (n_vals**(8/3) * np.abs(coeffs) / max_c)[pos_idx],
               'kx', ms=ms, label=r'$F_{N2}N^{8/3} > 0$')
    plt.loglog(n_vals[neg_idx],
               (n_vals**(8/3) * np.abs(coeffs) / max_c)[neg_idx],
               'rx', ms=ms, label=r'$F_{N2}N^{8/3} < 0$')
    plt.loglog(n_vals[pos_idx2],
               (n_vals**(8/3) * np.abs(coeffs2) / max_c2)[pos_idx2],
               'go', ms=ms, label=r'$F_{N-2}N^{8/3} > 0$')
    plt.loglog(n_vals[neg_idx2],
               (n_vals**(8/3) * np.abs(coeffs2) / max_c2)[neg_idx2],
               'bo', ms=ms, label=r'$F_{N-2}N^{8/3} < 0$')
    plt.xlabel('N')
    plt.ylabel(r'$F_{N2} N^{8/3} / F_{N2,\max}$')
    plt.axvline(max_n, c='k', linewidth=1)
    plt.axvline(max_n2, c='g', linewidth=1)
    plt.axvline(N_peak, c='b')
    plt.title(r'$e = %.2f$' % e)
    plt.legend()
    plt.tight_layout()
    plt.savefig('hansens_83', dpi=400)
    plt.clf()

def plot_maxes(m=2):
    '''
    plot maximum of argmax_N F_{Nm} as function of e
    scales N_max ~ (1-e)^(-3/2) as expected
    '''
    nmax = 1000
    e_vals = np.arange(0.5, 0.975, 0.01)
    maxes2 = []
    maxesm2 = []
    for e in e_vals:
        print('running for e =', e)
        N_vals, FN2, FNm2 = get_coeffs_fft(nmax, m, e)
        maxes2.append(N_vals[np.argmax(np.abs(FN2))])
        maxesm2.append(N_vals[np.argmax(np.abs(FNm2))])
    plt.loglog(1 - e_vals, maxes2, label=r'$m = 2$')
    plt.loglog(1 - e_vals, maxesm2, label=r'$m = -2$')
    m, b, _, _, _ = linregress(np.log(1 - e_vals), np.log(maxes2))
    m2, b2, _, _, _ = linregress(np.log(1 - e_vals), np.log(maxesm2))
    plt.title(r'$%.2f (1 - e)^{%.2f}, %.2f (1 - e)^{%.2f}$'
              % (np.exp(b), m, np.exp(b2), m2))
    plt.loglog(1 - e_vals, np.exp(b) * (1 - e_vals)**(m), 'r:', label='Fit')
    plt.loglog(1 - e_vals, np.exp(b2) * (1 - e_vals)**(m2), 'r:', label='Fit2')
    plt.xlabel(r'$1 - e$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('hansen_maxes', dpi=400)
    plt.clf()

if __name__ == '__main__':
    m = 2
    e = 0.9
    # plot_hansens(m, e, coeff_getter=get_coeffs_fft)
    plot_maxes()
