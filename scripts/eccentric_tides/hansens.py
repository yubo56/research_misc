import numpy as np

from scipy.integrate import quad
from scipy.stats import linregress
from scipy.fftpack import ifft
from scipy.optimize import bisect, curve_fit
from scipy.special import gamma

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
    '''
    coeffs, coeffs2 share a DC bin, = F_{0 +-2}, which are the same
    Recall F_{-N+m} = F_{+N-m}, so we have F_{N2} for all N in +- inf
    '''
    n_used = 4 * nmax # nyquist not enough??
    def f(E):
        return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    def E(M):
        return bisect(lambda E_v: E_v - e * np.sin(E_v) - M, 0, 2 * np.pi)
    m_vals = 2 * np.pi * np.arange(n_used) / n_used
    f_vals = f(np.array([E(M) for M in m_vals]))
    func_vals = ((1 + e * np.cos(f_vals)) / (1 - e**2))**3\
        * np.exp(-1j * m * f_vals)
    func_vals2 = ((1 + e * np.cos(f_vals)) / (1 - e**2))**3\
        * np.exp(1j * m * f_vals)
    FN2_ifft = np.real(ifft(func_vals))
    FN2_ifft2 = np.real(ifft(func_vals2))
    return np.arange(nmax), FN2_ifft[ :nmax], FN2_ifft2[ :nmax]

def powerlaw(n, C, p, a):
    return C * n**p * np.exp(-n / a)

def fit_powerlaw_hansens(N, coeffs, p_exact=2, use_p2=True):
    ''' if use_p2, force p = 2 '''
    if use_p2:
        def fit_func(n, C, a):
            return powerlaw(n, C, p_exact, a)
        params, _ = curve_fit(fit_func, N, coeffs, p0=(1, 40),
                              bounds=((0, 0.01), (np.inf, np.inf)))
        return params[0], p_exact, params[1]

    else:
        params, _ = curve_fit(powerlaw, N, coeffs, p0=(1, 2, 40),
                              bounds=((0, 1, 0.01), (np.inf, 4, np.inf)))
        return params[0], params[1], params[2]

def plot_hansens_0(e, m=0):
    N_peak = np.sqrt(1 + e) * (1 - e**2)**(-3/2)
    nmax = int(10 * N_peak)
    n_vals, coeffs, coeffs2 = get_coeffs_fft(nmax, m, e)

    n_tot = np.concatenate((-n_vals[1: ][::-1], n_vals))
    coeffs_tot = np.concatenate((coeffs2[1: ][::-1], coeffs))
    plt.semilogy(n_tot, coeffs_tot, 'k', alpha=0.7,
                 label=r'$F_{N0}$')

    # fit_func
    amp = coeffs[0]
    def f(n, a):
        return amp * np.exp(-np.abs(n) / (a * N_peak))
    [beta], _ = curve_fit(f, n_tot, coeffs_tot, p0=(2))
    f3 = 1 + 15*e**2/4 + 15*e**4/8 + 5*e**6/64
    f5 = 1 + 3 * e**2 + 3 * e**4 / 8
    eta0 = np.sqrt(9 * e**2 * f3 / ((1 - e**2)**3 * f5))
    c0 = np.sqrt(f5 / (eta0 * (1 - e**2)**(9/2)))
    plt.semilogy(n_tot, c0 * np.exp(-np.abs(n_tot) / eta0), 'g',
                 alpha=0.7, label='Parameterization')

    plt.xlabel(r'$N$')
    plt.xlim([-6 * N_peak, 6 * N_peak])
    plt.legend(fontsize=12, loc='lower center', ncol=2)
    plt.tight_layout()
    plt.savefig('hansens/hansens%s' % ('%.2f' % e).replace('.', '_'), dpi=400)
    plt.close()

def plot_fitted_hansens(m, e, coeff_getter=get_coeffs, fn='hansens'):
    N_peak = np.sqrt(1 + e) * (1 - e**2)**(-3/2)
    # just plot +2, no 8/3's laws for now
    nmax = 4 * int(max(N_peak, 150))
    n_vals, coeffs, coeffs2 = coeff_getter(nmax, m, e)
    n_vals = n_vals[1: ] # drop the 0 bin, throws off loglog plotting
    coeffs = coeffs[1: ]

    max_n = np.argmax(np.abs(coeffs))
    max_c = np.max(np.abs(coeffs))
    pos_idx = np.where(coeffs > 0)[0]
    neg_idx = np.where(coeffs < 0)[0]
    plt.loglog(n_vals[pos_idx], np.abs(coeffs[pos_idx]),
               'k', ms=ms, label=r'$F_{N2}$', alpha=0.7)
    # plt.loglog(n_vals, np.abs(coeffs2[1: ]),
    #            'r', ms=ms, label=r'$F_{(-N)2}$', alpha=0.7)
    plt.loglog(n_vals[neg_idx], np.abs(coeffs[neg_idx]),
               'k:', ms=ms, alpha=0.7)
    # params = fit_powerlaw_hansens(n_vals[100: ], coeffs[100: ])
    # fit = powerlaw(n_vals, params[0], params[1], params[2])
    f2 = 1 + 15*e**2/2 + 45*e**4/8 + 5*e**6/16
    f5 = 1 + 3 * e**2 + 3 * e**4 / 8
    eta2 = 4 * f2 / (5 * f5 * (1 - e**2)**(3/2))
    c2 = np.sqrt(32 * f5 / (24 * eta2**5 * (1 - e**2)**(9/2)))
    fit = powerlaw(n_vals, c2, 2, eta2)
    plt.loglog(n_vals, fit, 'g', label='Parameterization', alpha=0.7)

    plt.xlabel('$N$')
    plt.ylabel(r'$F_{N2}$')
    # plt.axvline(max_n, c='k', linewidth=1)
    # plt.axvline(N_peak, c='b')
    print('N_peri, N_max', N_peak, max_n)

    plt.ylim(bottom=abs(coeffs[0]) / 100)
    # plt.text(
    #     np.sqrt(plt.xlim()[0] * plt.xlim()[1]), max(abs(coeffs)) * 1.5,
    #     r'$(F_{N2} = %.3fN^{%d}e^{-N/%.3f})$' % tuple(params),
    #     color='r',
    #     size=12)
    plt.legend(fontsize=12, ncol=3, loc='lower center')
    plt.tight_layout()
    plt.savefig(fn, dpi=400)
    plt.close()

def plot_maxes(m=2):
    '''
    plot maximum of argmax_N F_{Nm} as function of e
    scales N_max ~ (1-e)^(-3/2) as expected
    '''
    nmax = 1000
    e_vals = np.arange(0.51, 0.975, 0.02)
    maxes2 = []
    maxesn2 = []
    maxes83_2 = []
    maxes83_n2 = []
    for e in e_vals:
        print('running for e =', e)
        N_vals, FN2, FNn2 = get_coeffs_fft(nmax, m, e)
        maxes2.append(N_vals[np.argmax(np.abs(FN2))])
        maxesn2.append(N_vals[np.argmax(np.abs(FNn2))])
        maxes83_2.append(N_vals[np.argmax(N_vals**(8/3) * np.abs(FN2))])
        maxes83_n2.append(N_vals[np.argmax(N_vals**(8/3) * np.abs(FNn2))])
    plt.loglog(1 - e_vals, maxes2, label=r'$m = 2$')
    maxesn2 = np.array(maxesn2)
    plt.loglog(1 - e_vals, maxesn2, label=r'$m = -2$')
    m, b, _, _, _ = linregress(np.log(1 - e_vals), np.log(maxes2))
    m2_idxs = np.where(1 - e_vals < 0.1)[0] # fit only good parts of data
    m2, b2, _, _, _ = linregress(np.log(1 - e_vals)[m2_idxs],
                                 np.log(maxesn2)[m2_idxs])
    plt.title(r'$%.2f (1 - e)^{%.2f}, %.2f (1 - e)^{%.2f}$'
              % (np.exp(b), m, np.exp(b2), m2))
    plt.loglog(1 - e_vals, np.exp(b) * (1 - e_vals)**(m), 'r:', label='Fit')
    plt.loglog(1 - e_vals, np.exp(b2) * (1 - e_vals)**(m2), 'r:', label='Fit2')
    plt.xlabel(r'$1 - e$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('hansen_maxes', dpi=400)
    plt.close()

    plt.loglog(1 - e_vals, maxes83_2, label=r'$m = 2$')
    plt.loglog(1 - e_vals, maxes83_n2, label=r'$m = -2$')
    m, b, _, _, _ = linregress(np.log(1 - e_vals), np.log(maxes83_2))
    m2, b2, _, _, _ = linregress(np.log(1 - e_vals), np.log(maxes83_n2))
    plt.title(r'$%.2f (1 - e)^{%.2f}, %.2f (1 - e)^{%.2f}$'
              % (np.exp(b), m, np.exp(b2), m2))
    plt.loglog(1 - e_vals, np.exp(b) * (1 - e_vals)**(m), 'r:', label='Fit')
    plt.loglog(1 - e_vals, np.exp(b2) * (1 - e_vals)**(m2), 'r:', label='Fit2')
    plt.xlabel(r'$1 - e$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('hansen_maxes83', dpi=400)
    plt.close()

def plot_fit_scalings(m=2):
    '''
    plot maximum of argmax_N F_{Nm} as function of e
    scales N_max ~ (1-e)^(-3/2) as expected
    '''
    nmax = 1000
    e_vals = np.concatenate((
        np.arange(0.51, 0.91, 0.02),
        np.arange(0.91, 0.975, 0.005)
    ))
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    c1s = np.zeros_like(e_vals)
    # c2s = np.zeros_like(e_vals)
    p1s = np.zeros_like(e_vals)
    # p2s = np.zeros_like(e_vals)
    a1s = np.zeros_like(e_vals)
    # a2s = np.zeros_like(e_vals)
    for idx, e in enumerate(e_vals):
        print('fitting for e =', e)
        N_vals, FN2, FNn2 = get_coeffs_fft(nmax, m, e)
        c1s[idx], p1s[idx], a1s[idx] = fit_powerlaw_hansens(N_vals, FN2)
        # c2s[idx], p2s[idx], a2s[idx] = fit_powerlaw_hansens(N_vals, FNn2)
    ax1.plot(e_vals, c1s, 'bo', ms=ms)
    # ax1.plot(e_vals, c2s, 'r')
    ax3.plot(e_vals, a1s, 'bo', label=r'$m = +2$', ms=ms)
    # ax3.plot(e_vals, a2s, 'r', label=r'$m = -2$')

    # try plotting guesses now
    p0 = 2 # always p0
    a0 = np.sqrt(2 * (1 + e_vals)) / (p0 * (1 - e_vals)**(3/2))
    ax3.plot(e_vals, a0, 'r:', label=r'Th.')
    c0 = np.sqrt(
        (1 + 3 * e_vals**2 + 3 * e_vals**4 / 8)
        / (1 - e_vals**2)**(9/2)
        / ((a0 / 2)**(2 * p0 + 1) * gamma(2 * p0 + 1)))
    ax1.plot(e_vals, c0, 'r:')

    ax1.set_yscale('log')
    ax3.set_yscale('log')
    ax1.set_ylabel(r'$C$')
    ax3.set_ylabel(r'$a$')
    ax3.set_xlabel(r'$e$')
    ax3.legend(fontsize=12, ncol=2)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.savefig('hansen_params', dpi=400)
    plt.close()

def plot_naked_hansens(m=2, e=0.9):
    nmax = 4 * int(max((1 + e) * (1 - e)**(-3/2), 600))
    n_vals, coeffs, coeffs2 = get_coeffs_fft(nmax, m, e)

    pos_idx = np.where(coeffs > 0)[0]
    neg_idx = np.where(coeffs < 0)[0]
    plt.semilogy(n_vals[pos_idx], np.abs(coeffs[pos_idx]),
                'ko', ms=ms, label=r'$F_{N2} > 0$')
    plt.semilogy(n_vals[neg_idx], np.abs(coeffs[neg_idx]),
                'ro', ms=ms, label=r'$F_{N2} < 0$')

    pos_idx2 = np.where(coeffs2 > 0)[0]
    neg_idx2 = np.where(coeffs2 < 0)[0]
    plt.semilogy(-n_vals[pos_idx2], np.abs(coeffs2[pos_idx2]),
                'ko', ms=ms)
    plt.semilogy(-n_vals[neg_idx2], np.abs(coeffs2[neg_idx2]),
                'ro', ms=ms)
    plt.xlabel(r'$N$')
    plt.title(r'$e = %.2f$' % e)
    plt.legend()
    plt.ylim(bottom=1e-4)
    plt.xscale('symlog')
    plt.savefig('hansens_plain', dpi=400)
    plt.close()

    plt.loglog(n_vals[pos_idx], np.abs(coeffs[pos_idx]),
                'ko', ms=ms, label=r'$F_{N2} > 0$')
    plt.loglog(n_vals[neg_idx], np.abs(coeffs[neg_idx]),
                'ro', ms=ms, label=r'$F_{N2} < 0$')
    plt.xlabel(r'$N$')
    plt.title(r'$e = %.2f$' % e)
    plt.legend()
    plt.ylim(bottom=1e-4)
    plt.savefig('hansens_plain_right', dpi=400)
    plt.close()

if __name__ == '__main__':
    m = 2
    e = 0.9
    plot_fitted_hansens(m, e, coeff_getter=get_coeffs_fft)
    # plot_fitted_hansens(m, 0.98, coeff_getter=get_coeffs_fft, fn='hansens99')
    # plot_maxes()
    # plot_fit_scalings()

    # energy terms
    # for e_val in np.arange(0.5, 0.96, 0.05):
    #     plot_hansens_0(e_val)
    # plot_hansens_0(0.9)

    # plot_naked_hansens(e=0.9)
