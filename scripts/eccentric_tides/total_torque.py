'''
investigate total torque and approximations
'''
from scipy.optimize import bisect
from scipy.fftpack import ifft
from scipy.special import gamma
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

ms = 3
def get_coeffs_fft(nmax, m, e):
    '''
    returns coeffs for N \in [-nmax + 1, nmax - 1]
    '''
    n_used = 4 * nmax # nyquist not enough?? oh well, still fast
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
    # these two share a DC bin
    ret = np.concatenate((FN2_ifft2[ :nmax][::-1], FN2_ifft[1:nmax]))
    return ret

def get_torques(nmax, m, e, w_s):
    ''' really just gets hat{tau} '''
    n_vals = np.arange(-nmax + 1, nmax)
    return np.sign(n_vals - 2 * w_s) * np.abs(n_vals - 2 * w_s)**(8/3)

def plot_ecc(w_s=0, m=2, nmax=1000):
    ''' plots behavior of total torque with varying eccentricity '''
    e_vals = np.arange(0.5, 0.96, 0.01)
    totals = []
    for e in e_vals:
        coeffs = get_coeffs_fft(nmax, m, e)
        torques = get_torques(nmax, m, e, w_s)
        totals.append(np.sum(coeffs**2 * torques))
        print('Ran for e =', e)
    plt.semilogy(e_vals, totals, 'bo', ms=3)
    plt.xlabel(r'$e$')
    plt.ylabel(r'$\hat{\tau}_n$')
    plt.title(r'$\frac{\Omega_s}{\Omega} = %d$' % w_s)
    plt.tight_layout()
    plt.savefig('totals_ecc_%d' % w_s, dpi=400)
    plt.clf()

def plot_spin(e=0.9, m=2, nmax=1000):
    ''' plots behavior of total torque with varying omega_spin '''
    Nmax = np.sqrt(1 + e) / (1 - e)**(3/2)
    w_vals = np.linspace(0, 10 * Nmax, 200)
    totals = []
    for w_s in w_vals:
        coeffs = get_coeffs_fft(nmax, m, e)
        torques = get_torques(nmax, m, e, w_s)
        totals.append(np.sum(coeffs**2 * torques))
        print('Ran for w_s =', w_s)
    totals = np.array(totals)

    plt.xlabel(r'$\frac{\Omega_s}{\Omega}$')
    plt.ylabel(r'$\hat{\tau}_n$')
    plt.title(r'$e = %.1f, N_{\max} \equiv \frac{\sqrt{1 + e}}{(1 - e)^{3/2}} \simeq %d$'
              % (e, Nmax))
    pos_idxs = np.where(totals > 0)[0]
    neg_idxs = np.where(totals < 0)[0]
    plt.semilogy(w_vals[pos_idxs], totals[pos_idxs], 'bo',
                 ms=ms, label=r'$\tau > 0$')
    plt.semilogy(w_vals[neg_idxs], -totals[neg_idxs], 'ro',
                 ms=ms, label=r'$\tau < 0$')
    ylims = plt.ylim() # store ylims from here

    # prediction time w/ hard coded fit values
    cp, pp, ap, cm, pm, am = 0.013, 2.13, 27.7, 0.042, 0.75, 19.2
    torque_low_spin = (
        cp**2 * (ap / 2)**(2 * pp + 11/3) * gamma(2 * pp + 11/3) -
        cm**2 * (am / 2)**(2 * pm + 11/3) * gamma(2 * pm + 11/3)
    )
    w_hs = w_vals[np.where(2 * w_vals > Nmax)[0]] # high spin case only
    torque_high_spin = -(2 * w_hs - Nmax)**(8/3) * (
        cp**2 * (ap / 2)**(2 * pp + 1) * gamma(2 * pp + 1) +
        cm**2 * (am / 2)**(2 * pm + 1) * gamma(2 * pm + 1)
    )
    plt.axhline(torque_low_spin, c='k', lw=1)
    plt.semilogy(w_hs, -torque_high_spin, 'r:')
    plt.ylim(ylims)

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('totals_s_%s' % ('%.1f' % e).replace('.', '_'), dpi=400)
    plt.clf()

if __name__ == '__main__':
    # plot_ecc()
    plot_spin()
