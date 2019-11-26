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

ms = 2
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

def get_cpa(e):
    p0 = 2 # always p0
    a0 = np.sqrt(2 * (1 + e)) / (p0 * (1 - e)**(3/2))
    c0 = np.sqrt(
        (1 + 3 * e**2 + 3 * e**4 / 8)
        / (1 - e**2)**(9/2)
        / ((a0 / 2)**(2 * p0 + 1) * gamma(2 * p0 + 1)))
    return c0, p0, a0

def get_torques(nmax, w_s):
    ''' really just gets hat{tau}_n (no hansen coeff)'''
    n_vals = np.arange(-nmax + 1, nmax)
    return np.sign(n_vals - 2 * w_s) * np.abs(n_vals - 2 * w_s)**(8/3)

def plot_ecc(w_s=0, m=2, nmax=1000):
    ''' plots behavior of total torque with varying eccentricity '''
    e_vals = np.arange(0.5, 0.96, 0.01)
    N_peri_max = np.sqrt(1 + max(e_vals)) / (1 - max(e_vals))**(3/2)
    totals = []
    for e in e_vals:
        coeffs = get_coeffs_fft(nmax, m, e)
        torques = get_torques(nmax, w_s)
        totals.append(np.sum(coeffs**2 * torques))
        print('Ran for e =', e)
    totals = np.array(totals)

    cp, pp, ap = get_cpa(e_vals)
    if w_s < N_peri_max:
        torque_low_spin = (
            cp**2 * (ap / 2)**(2 * pp + 11/3) * gamma(2 * pp + 11/3)
        )
        plt.semilogy(e_vals, torque_low_spin, 'b:')
        plt.semilogy(e_vals, totals, 'bo', ms=3)
        plt.title(r'$\frac{\Omega_s}{\Omega} = %d \ll N_{\rm peri}$' % w_s)
    else:
        N_peri = np.sqrt(1 + e_vals) / (1 - e_vals)**(3/2)
        Nmax = np.sqrt(2) * N_peri
        torque_high_spin = -(2 * w_s - Nmax)**(8/3) * (
            cp**2 * (ap / 2)**(2 * pp + 1) * gamma(2 * pp + 1)
        )
        plt.semilogy(e_vals, -torque_high_spin, 'b:')
        plt.semilogy(e_vals, -totals, 'bo', ms=3)
        plt.title(r'$\frac{\Omega_s}{\Omega} = %d \gg N_{\rm peri}$' % w_s)
    plt.xlabel(r'$e$')
    plt.ylabel(r'$\tau / \hat{T}$')
    plt.tight_layout()
    plt.savefig('totals_ecc_%d' % w_s, dpi=400)
    plt.clf()

def plot_spin(e=0.9, m=2, nmax=1000):
    ''' plots behavior of total torque with varying omega_spin '''
    N_peri = np.sqrt(1 + e) / (1 - e)**(3/2)
    Nmax = np.sqrt(2) * N_peri
    w_vals = np.linspace(0, 2 * Nmax, 70)
    totals = []
    for w_s in w_vals:
        coeffs = get_coeffs_fft(nmax, m, e)
        torques = get_torques(nmax, m, e, w_s)
        totals.append(np.sum(coeffs**2 * torques))
        print('Ran for w_s =', w_s)
    totals = np.array(totals)

    plt.xlabel(r'$\frac{\Omega_s}{\Omega}$')
    plt.ylabel(r'$\tau / \hat{T}$')
    plt.title(r'$e = %.1f, N_{\rm peri} \equiv \frac{\sqrt{1 + e}}{(1 - e)^{3/2}} \simeq %d$'
              % (e, N_peri))
    pos_idxs = np.where(totals > 0)[0]
    neg_idxs = np.where(totals < 0)[0]
    plt.semilogy(w_vals[pos_idxs], totals[pos_idxs], 'bo',
                 ms=ms, label=r'$\tau > 0$')
    plt.semilogy(w_vals[neg_idxs], -totals[neg_idxs], 'ro',
                 ms=ms, label=r'$\tau < 0$')
    ylims = plt.ylim() # store ylims from here
    plt.axvline(N_peri, c='k')

    # predictions
    cp, pp, ap = get_cpa(e)
    w_ls = w_vals[np.where(w_vals < N_peri)[0]] # low spin case only
    torque_low_spin = (
        cp**2 * (ap / 2)**(2 * pp + 11/3) * gamma(2 * pp + 11/3)
    ) * (1 - w_ls / N_peri)**(8/3)
    w_hs = w_vals[np.where(2 * w_vals > Nmax)[0]] # high spin case only
    torque_high_spin = -(2 * w_hs - Nmax)**(8/3) * (
        cp**2 * (ap / 2)**(2 * pp + 1) * gamma(2 * pp + 1)
    )
    plt.semilogy(w_ls, torque_low_spin, 'k:')
    plt.semilogy(w_hs, -torque_high_spin, 'r:')
    plt.ylim(ylims)

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('totals_s_%s' % ('%.1f' % e).replace('.', '_'), dpi=400)
    plt.clf()

def get_energies(nmax, m, e, w_s):
    ''' gets E/hat{T}W '''
    coeffs0 = get_coeffs_fft(nmax, 0, e)
    coeffs2 = get_coeffs_fft(nmax, 2, e)
    n_vals = np.arange(-nmax + 1, nmax)
    return 0.5 * (
        n_vals * coeffs2**2 * np.sign(n_vals - 2 * w_s)
            * np.abs(n_vals - 2 * w_s)**(8/3)
        + 2/3 * coeffs0**2 * np.abs(n_vals)**(11/3)
    )

def plot_energy(w_s=0, m=2, nmax=1000):
    ''' plots behavior of heating with varying eccentricity '''
    e_vals = np.arange(0.5, 0.96, 0.01)
    N_peri_max = np.sqrt(1 + max(e_vals)) / (1 - max(e_vals))**(3/2)
    totals = []
    for e in e_vals:
        coeffs = get_coeffs_fft(nmax, m, e)
        energies = get_energies(nmax, m, e, w_s)
        totals.append(np.sum(energies))
        print('Ran for e =', e)
    totals = np.array(totals)
    plt.xlabel(r'$\frac{\Omega_s}{\Omega}$')

    # anal fits
    cp, pp, ap = get_cpa(e_vals)
    N_peri = np.sqrt(1 + e_vals) / (1 - e_vals)**(3/2)
    c0 = np.sqrt(
        (1 + 3 * e_vals**2 + 3 * e_vals**4 / 8)
        / (1 - e_vals**2)**(9/2)
        / (2 * N_peri * np.sqrt(2)))
    disp_2 = 2/3 * c0**2 * (N_peri / (2 * np.sqrt(2)))**(14/3) * gamma(14 / 3)

    if w_s < N_peri_max:
        disp_1 = 0.5 * cp**2 * (ap / 2)**(26 / 3) * gamma(26 / 3)
        plt.title(r'$\frac{\Omega_s}{\Omega} = %d \ll N_{\rm peri}$' % w_s)
        plt.semilogy(e_vals, totals, 'bo', ms=3)
        plt.ylabel(r'$\dot{E}_{in} / \hat{T}\Omega$')
    else:
        N_peri = np.sqrt(1 + e) / (1 - e)**(3/2)
        Nmax = np.sqrt(2) * N_peri
        disp_1 = 0.5 * (2 * w_s - Nmax)**(8/3) * (
            cp**2 * (ap / 2)**(2 * pp + 2) * gamma(2 * pp + 2)
        )
        plt.title(r'$\frac{\Omega_s}{\Omega} = %d \gg N_{\rm peri}$' % w_s)
        plt.semilogy(e_vals, -totals, 'bo', ms=3)
        plt.ylabel(r'$-\dot{E}_{in} / \hat{T}\Omega$')
    plt.semilogy(e_vals, disp_1 + disp_2, 'r:')

    plt.savefig('totals_e_%d' % w_s, dpi=400)
    plt.clf()

if __name__ == '__main__':
    # plot_ecc()
    # plot_ecc(400)
    # plot_spin()
    # plot_energy()
    plot_energy(400)
