import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

def f(E, e):
    ''' returns true anomaly for given eccentric anomaly '''
    return 2 * np.arctan((1 + e) / (1 - e) * np.tan(E / 2))

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
    return quad(integrand, 0, np.pi, limit=max(100, 2 * N))

if __name__ == '__main__':
    ms = 2
    m = 2
    e = 0.9

    N_peak = (1 - e)**(-3/2)
    print('Ansatz N', N_peak)

    n_vals = np.arange(1, int(30 * N_peak))
    coeffs = np.zeros_like(n_vals, dtype=np.float64)
    for idx, N in enumerate(n_vals):
        y, abserr = get_hansen(N, m, e)
        coeffs[idx] = y
        print('%03d, %.8f, %.8e' % (N, y, abserr))

    max_n = np.argmax(np.abs(coeffs))
    max_c = np.max(np.abs(coeffs))
    pos_idx = np.where(coeffs > 0)[0]
    neg_idx = np.where(coeffs < 0)[0]
    plt.loglog(n_vals[pos_idx], np.abs(coeffs[pos_idx]) / max_c,
               'ko', ms=ms, label=r'$F_{Nm}$')
    plt.loglog(n_vals[neg_idx], np.abs(coeffs[neg_idx]) / max_c,
               'ro', ms=ms)
    plt.xlabel('N')
    plt.ylabel(r'$F_{Nm} / F_{Nm,\max}$')
    plt.axvline(max_n, c='k', linewidth=1)
    plt.axvline(N_peak, c='b')
    plt.title(r'$e = %.2f$' % e)
    plt.tight_layout()
    plt.savefig('hansens', dpi=400)
    plt.clf()

    plt.loglog(n_vals[pos_idx],
               (n_vals**(8/3) * np.abs(coeffs) / max_c)[pos_idx],
               'kx', ms=ms, label=r'$F_{Nm}N^{8/3}$')
    plt.loglog(n_vals[neg_idx],
               (n_vals**(8/3) * np.abs(coeffs) / max_c)[neg_idx],
               'rx', ms=ms)
    plt.xlabel('N')
    plt.ylabel(r'$F_{Nm} N^{8/3} / F_{Nm,\max}$')
    plt.axvline(max_n, c='k', linewidth=1)
    plt.axvline(N_peak, c='b')
    plt.title(r'$e = %.2f$' % e)
    plt.tight_layout()
    plt.savefig('hansens_83', dpi=400)
