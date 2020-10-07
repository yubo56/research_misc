'''
investigate total torque and approximations

units of Omega = 1
'''
from scipy.optimize import bisect, brenth
from scipy.fftpack import ifft
from scipy.special import gamma
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

ms = 4
lw = 1.5
af = 0.6
def get_coeffs_fft(num_pts, m, e):
    '''
    returns coeffs for N \in [-num_pts + 1, num_pts - 1]
    '''
    n_used = 4 * num_pts # nyquist not enough?? oh well, still fast
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
    # these two share a DC bin
    ret = np.concatenate((FN2_ifft2[ :num_pts][::-1], FN2_ifft[1:num_pts]))
    return ret

def get_torques(num_pts, w_s):
    ''' really just gets hat{tau}_n (no hansen coeff)'''
    n_vals = np.arange(-num_pts + 1, num_pts)
    return np.sign(n_vals - 2 * w_s) * np.abs(n_vals - 2 * w_s)**(8/3)

def plot_ecc(w_s=0, num_pts=1000):
    ''' plots behavior of total torque with varying eccentricity '''
    e_vals = np.arange(0.5, 0.96, 0.01)
    totals = []
    totals_integ = []
    for e in e_vals:
        coeffs = get_coeffs_fft(num_pts, 2, e)
        torques = get_torques(num_pts, w_s)
        totals.append(np.sum(coeffs**2 * torques))
        totals_integ.append(get_integral(w_s, e))
        print('Ran for e =', e)
    totals = np.array(totals)
    totals_integ = np.array(totals_integ)

    f2 = 1 + 15*e_vals**2/2 + 45*e_vals**4/8 + 5*e_vals**6/16
    f5 = 1 + 3 * e_vals**2 + 3 * e_vals**4 / 8
    eta2 = 4 * f2 / (5 * f5 * (1 - e_vals**2)**(3/2))
    prefactor = f5 * eta2**(8/3) / (1 - e_vals**2)**(9/2)

    thresh = np.max(eta2) / 0.691
    if w_s < thresh:
        torque = prefactor * (
            (1 - 0.691 * w_s / eta2)**(8/3) * gamma(23 / 3) / gamma(5)
            / 2**(8/3))
        sign = +1
        # plt.title(r'$\frac{\Omega_s}{\Omega} = %d \ll N_{\rm peri}$' % w_s)
        plt.ylabel(r'$\tau / \hat{\tau}$')
    else:
        torque = -prefactor * ((w_s / eta2 - 1)**(8/3) * 2**(8/3))
        sign = -1
        # plt.title(r'$\frac{\Omega_s}{\Omega} = %d \gg N_{\rm peri}$' % w_s)
        plt.ylabel(r'$-\tau / \hat{\tau}$')
    plt.semilogy(e_vals, sign * totals, 'k+', ms=ms, alpha=af,
                 label='Sum')
    plt.semilogy(e_vals, sign * totals_integ, 'bx', ms=ms, alpha=af,
                 label='Integral')
    plt.semilogy(e_vals, sign * torque, 'g:', lw=lw, alpha=af,
                 label='Asymptotic')
    plt.xlabel(r'$e$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('1totals_ecc_%d' % w_s, dpi=400)
    plt.clf()

def get_explicit(w_s, e, Nmax=500):
    coeffs = get_coeffs_fft(num_pts, 2, e)
    torques = get_torques(num_pts, w_s)
    totals.append(np.sum(coeffs**2 * torques))

def get_integral(w_s, e, Nmax=500, dN=0.01):
    # numerically evaluates the integral for arbitrary spin
    f2 = 1 + 15*e**2/2 + 45*e**4/8 + 5*e**6/16
    f5 = 1 + 3 * e**2 + 3 * e**4 / 8
    eta2 = 4 * f2 / (5 * f5 * (1 - e**2)**(3/2))
    c2sq = 32 * f5 / (24 * eta2**5 * (1 - e**2)**(9/2))

    N_vals = np.arange(0, Nmax, dN)
    integrand = c2sq * N_vals**4 * np.exp(-2 * N_vals / eta2) \
        * np.sign(N_vals - 2 * w_s) * np.abs(N_vals - 2 * w_s)**(8/3)
    return np.sum(integrand * dN)

def plot_spin(e=0.9, num_pts=1000):
    ''' plots behavior of total torque with varying omega_spin '''
    w_vals = np.linspace(-120, 200, 100)
    totals = []
    totals_integ = []
    for w_s in w_vals:
        coeffs = get_coeffs_fft(num_pts, 2, e)
        torques = get_torques(num_pts, w_s)
        totals.append(np.sum(coeffs**2 * torques))
        totals_integ.append(get_integral(w_s, e))
        print('Ran for w_s =', w_s)
    totals = np.array(totals)
    totals_integ = np.array(totals_integ)

    plt.xlabel(r'$\Omega_s / \Omega$')
    plt.ylabel(r'$\tau / \hat{\tau}$')
    plt.title(r'$e = %.1f$' % e)
    pos_idxs = np.where(totals > 0)[0]
    neg_idxs = np.where(totals < 0)[0]
    plt.semilogy(w_vals[pos_idxs], totals[pos_idxs], 'b+',
                 ms=ms, label=r'Sum', alpha=af)
    plt.semilogy(w_vals[neg_idxs], -totals[neg_idxs], 'r+', ms=ms, alpha=af)
    plt.semilogy(w_vals[pos_idxs], totals_integ[pos_idxs], 'bx',
                 ms=ms, label=r'Integral', alpha=af)
    plt.semilogy(w_vals[neg_idxs], -totals_integ[neg_idxs], 'rx', ms=ms,
                 alpha=af)
    ylims = plt.ylim() # store ylims from here

    # predictions
    f2 = 1 + 15*e**2/2 + 45*e**4/8 + 5*e**6/16
    f5 = 1 + 3 * e**2 + 3 * e**4 / 8
    eta2 = 4 * f2 / (5 * f5 * (1 - e**2)**(3/2))
    prefactor = f5 * eta2**(8/3) / (1 - e**2)**(9/2)

    w_ls = w_vals[np.where(w_vals * 0.691 / eta2 < 1)[0]]
    torque_low_spin = prefactor * (
        (1 - 0.691 * w_ls / eta2)**(8/3) * gamma(23 / 3) / gamma(5)
        / 2**(8/3))

    w_hs = w_vals[np.where(w_vals * 0.691 / eta2 > 1)[0]]
    # torque_high_spin = -prefactor * ((w_hs / eta2 - 1)**(8/3) * 2**(8/3))
    torque_high_spin = -prefactor * (
        (0.691 * w_hs / eta2 - 1)**(8/3) * gamma(23 / 3) / gamma(5)
        / 2**(8/3))

    plt.semilogy(w_ls, torque_low_spin, 'b',
                 label='Asymptotic', lw=lw, alpha=af)
    plt.semilogy(w_hs, -torque_high_spin, 'r', lw=lw, alpha=af)
    plt.ylim(ylims)

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('1totals_s_%s' % ('%.1f' % e).replace('.', '_'), dpi=400)
    plt.clf()

def get_energies(num_pts, e, w_s):
    ''' gets E/\hat{tau}W '''
    coeffs0 = get_coeffs_fft(num_pts, 0, e)
    coeffs2 = get_coeffs_fft(num_pts, 2, e)
    n_vals = np.arange(-num_pts + 1, num_pts)
    return 0.5 * (
        n_vals * coeffs2**2 * np.sign(n_vals - 2 * w_s)
            * np.abs(n_vals - 2 * w_s)**(8/3)
        + 2 / 3 * coeffs0**2 * np.abs(n_vals)**(11/3)
    )

def get_energies_integ(w_s, e, Nmax=300, dN=0.01):
    '''
    numerically evaluates the energy integral for arbitrary spin
    returns Edot / (\hat{tau}\Omega)
    '''
    f2 = 1 + 15*e**2/2 + 45*e**4/8 + 5*e**6/16
    f3 = 1 + 15*e**2/4 + 15*e**4/8 + 5*e**6/64
    f5 = 1 + 3 * e**2 + 3 * e**4 / 8
    eta2 = 4 * f2 / (5 * f5 * (1 - e**2)**(3/2))
    c2sq = 32 * f5 / (24 * eta2**5 * (1 - e**2)**(9/2))

    eta0 = np.sqrt(9 * e**2 * f3 / (f5 * (1 - e**2)**3))
    c0sq = f5 / (eta0 * (1 - e**2)**(9/2))

    N_vals = np.arange(0, Nmax, dN)
    integrand = 0.5 * (
        c2sq * N_vals**5 * np.exp(-2 * N_vals / eta2)
            * np.sign(N_vals - 2 * w_s) * np.abs(N_vals - 2 * w_s)**(8/3)
        + 4 / 3 * c0sq * np.exp(-2 * N_vals / eta0) * N_vals**(11/3)
    )
    return np.sum(integrand * dN)

def plot_energy(w_s=0, num_pts=1000):
    ''' plots behavior of heating with varying eccentricity '''
    e_vals = np.arange(0.5, 0.96, 0.01)
    totals = []
    totals_integ = []
    for e in e_vals:
        energies = get_energies(num_pts, e, w_s)
        totals.append(np.sum(energies))
        totals_integ.append(get_energies_integ(w_s, e, Nmax=1000))
        print('Ran for e =', e)
    totals = np.array(totals)
    totals_integ = np.array(totals_integ)
    plt.xlabel(r'$e$')

    f2 = 1 + 15*e_vals**2/2 + 45*e_vals**4/8 + 5*e_vals**6/16
    f3 = 1 + 15*e_vals**2/4 + 15*e_vals**4/8 + 5*e_vals**6/64
    f5 = 1 + 3 * e_vals**2 + 3 * e_vals**4 / 8
    eta2 = 4 * f2 / (5 * f5 * (1 - e_vals**2)**(3/2))

    # anal fits
    if w_s < np.max(eta2):
        # plt.title(r'$\frac{\Omega_s}{\Omega} = %d \ll N_{\rm peri}$' % w_s)
        plt.ylabel(r'$\dot{E}_{in} / \hat{\tau}\Omega$')
        sign = +1
    else:
        # plt.title(r'$\frac{\Omega_s}{\Omega} = %d \gg N_{\rm peri}$' % w_s)
        plt.ylabel(r'$-\dot{E}_{in} / \hat{\tau}\Omega$')
        sign = -1

    term0 = (
        f5 * gamma(14 / 3) * (3/2)**(8/3) / (1 - e_vals**2)**10
            * (e_vals**2 * f3 / (2 * f5))**(11/6))
    term2 = sign * (
        np.abs(1 - 0.5886 * w_s / eta2)**(8/3)
            * f5 / (1 - e_vals**2)**(9/2)
            * gamma(26/3) / gamma(5)
            * (eta2 / 2)**(11/3))

    plt.semilogy(e_vals, sign * totals, 'k+', ms=ms, alpha=af, label='Sum')
    plt.semilogy(e_vals, sign * totals_integ, 'bx', ms=ms, alpha=af,
                 label='Integral')
    plt.semilogy(e_vals, sign * (term0 + term2) / 2, 'g:', ms=ms, alpha=af,
                 label='Asymptotic')

    plt.legend()
    plt.tight_layout()
    plt.savefig('1totals_e_%d' % w_s, dpi=400)
    plt.clf()

def plot_spin_energy(e=0.9, num_pts=300):
    ''' plots behavior of heating with varying spin '''
    w_vals = np.linspace(-120, 200, 100)
    totals = []
    totals_integ = []
    for w_s in w_vals:
        energies = get_energies(num_pts, e, w_s)
        totals.append(np.sum(energies))
        totals_integ.append(get_energies_integ(w_s, e, Nmax=num_pts))
        print('Ran for w_s =', w_s)
    totals = np.array(totals)
    totals_integ = np.array(totals_integ)

    plt.xlabel(r'$\Omega_s / \Omega$')
    plt.ylabel(r'$\dot{E}_{in} / \hat{\tau}\Omega$')
    plt.title(r'$e = %.1f$' % e)

    pos_idxs = np.where(totals > 0)[0]
    neg_idxs = np.where(totals < 0)[0]
    plt.semilogy(w_vals[pos_idxs], totals[pos_idxs], 'b+',
                 ms=ms, alpha=af, label='Sum')
    plt.semilogy(w_vals[neg_idxs], -totals[neg_idxs], 'r+', ms=ms, alpha=af)

    posi_idxs = np.where(totals_integ > 0)[0]
    negi_idxs = np.where(totals_integ < 0)[0]
    plt.semilogy(w_vals[posi_idxs], totals_integ[posi_idxs],
                 'bx', ms=ms, alpha=af, label='Integral')
    plt.semilogy(w_vals[negi_idxs], -totals_integ[negi_idxs],
                 'rx', ms=ms, alpha=af)
    ylims = plt.ylim()

    # anal fits
    f2 = 1 + 15*e**2/2 + 45*e**4/8 + 5*e**6/16
    f3 = 1 + 15*e**2/4 + 15*e**4/8 + 5*e**6/64
    f5 = 1 + 3 * e**2 + 3 * e**4 / 8
    eta2 = 4 * f2 / (5 * f5 * (1 - e**2)**(3/2))

    w_ls = w_vals[np.where(w_vals * 0.5886 / eta2 < 1)[0]]
    w_hs = w_vals[np.where(w_vals * 0.5886 / eta2 > 1)[0]]

    term0 = (
        f5 * gamma(14 / 3) * (3/2)**(8/3) / (1 - e**2)**10
            * (e**2 * f3 / f5)**(11/6))
    term2_ls = (
        np.abs(1 - 0.5886 * w_ls / eta2)**(8/3)
            * f5 / (1 - e**2)**(9/2)
            * gamma(26/3) / gamma(5)
            * (eta2 / 2)**(11/3))
    term2_hs = - (
        np.abs(1 - 0.5886 * w_hs / eta2)**(8/3)
            * f5 / (1 - e**2)**(9/2)
            * gamma(26/3) / gamma(5)
            * (eta2 / 2)**(11/3))

    blue = (np.concatenate([term2_ls, term2_hs]) + term0) / 2
    red = -(np.concatenate([term2_ls, term2_hs]) + term0) / 2
    blue_idx = np.where(blue > 0)[0]
    red_idx = np.where(red > 0)[0]
    plt.semilogy(w_vals[blue_idx], blue[blue_idx], 'b', ms=ms, alpha=af,
                 label='Asymptotic')
    plt.semilogy(w_vals[red_idx], red[red_idx], 'r', ms=ms, alpha=af)
    plt.ylim(ylims)
    plt.ylim(bottom=1e9)

    plt.legend()
    plt.tight_layout()
    plt.savefig('1totals_NRG_e_%s' % ('%.1f' % e).replace('.', '_'), dpi=400)
    plt.clf()

def plot_pseudo():
    ''' plots pseudosynchronous frequency vs e '''
    e_vals = np.linspace(0.5, 0.99, 100)

    f2 = 1 + 15*e_vals**2/2 + 45*e_vals**4/8 + 5*e_vals**6/16
    f5 = 1 + 3 * e_vals**2 + 3 * e_vals**4 / 8
    eta2 = 4 * f2 / (5 * f5 * (1 - e_vals**2)**(3/2))

    w_syncs = []
    w_syncs_sum = []
    for e in e_vals:
        Nmax_max = 20 * np.sqrt(1 + e) / (1 - e**2)**(3/2)
        def opt_func(w):
            return get_integral(w, e, Nmax=int(Nmax_max))
        def opt_func_sum(w):
            return get_energies_integ(w, e, Nmax=int(Nmax_max))
        res = brenth(opt_func, 0, Nmax_max)
        res2 = brenth(opt_func_sum, 0, Nmax_max)
        print(e, res, res2)
        w_syncs.append(res)
        w_syncs_sum.append(res2)
    # plt.loglog(1 - e_vals**2, w_syncs, label='Integral')
    plt.loglog(1 - e_vals**2, w_syncs_sum, label='Sum')
    plt.loglog(1 - e_vals**2, eta2 / 0.691, label=r'$\eta_2 / 0.691$')
    plt.loglog(1 - e_vals**2, np.sqrt(1 + e_vals) / (1 - e_vals**2)**(3/2),
               label=r'$N_{\rm peri}$')
    plt.xlabel(r'$1 - e^2$')
    plt.ylabel(r'$\Omega_{\rm s, sync} / \Omega$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('1pseudosynchronous', dpi=200)

# don't think this plot makes sense, we don't know the breakup frequency
# independently of r_c
def plot_7319(e=0.808, obs_disp=1.536e11, breakup=428.822, prefix='7319'):
    num_pts = max(4 * int(np.sqrt(1 + e) * (1 - e**2)**(-3/2)), 150)
    def get_disp_spin(w_s):
        return np.sum(get_energies(num_pts, e, w_s))
    def get_torque(w_s):
        coeffs = get_coeffs_fft(num_pts, 2, e)
        torques = get_torques(num_pts, w_s)
        return np.sum(coeffs**2 * torques)
    spins = np.linspace(-breakup, breakup, 50)
    disp_spins = np.array([get_disp_spin(w_s) for w_s in spins])
    torque_spins = np.array([get_torque(w_s) for w_s in spins])

    plt.plot(spins, disp_spins)
    plt.axhline(-obs_disp, c='r', ls='dashed', lw=1.5)
    plt.axhline(obs_disp, c='r', ls='dashed', lw=1.5)
    plt.xlabel(r'$\Omega_s / \Omega_o$')
    plt.ylabel(r'$\dot{E}_{\rm in} / (\hat{\tau} \Omega)$')
    plt.savefig('1%s_disps' % prefix)
    plt.clf()

    retrospin_idx = np.argmin(np.abs(disp_spins - obs_disp))
    prospin_idx = np.argmin(np.abs(disp_spins + obs_disp))
    print('Naive, retro spin', spins[retrospin_idx])
    print('Naive, pro spin', spins[prospin_idx])

    edot_rot = disp_spins - spins * torque_spins
    print('edot_rot, retro', '%.3e' % edot_rot[retrospin_idx])
    print('edot_rot, pro', '%.3e' % edot_rot[prospin_idx])
    plt.plot(spins, edot_rot)
    plt.xlabel(r'$\Omega_s / \Omega_o$')
    plt.ylabel(r'$\dot{E}_{\rm in} / (\hat{\tau} \Omega)$')
    plt.savefig('1%s_heating' % prefix)
    plt.clf()

if __name__ == '__main__':
    # plot_ecc()
    # plot_ecc(400)
    # plot_spin()
    # plot_energy()
    # plot_energy(400)
    # plot_spin_energy(e=0.9)
    plot_pseudo()
    # plot_7319()
    # plot_7319(obs_disp = 2.81e13, breakup=1077.76, prefix='7319_mesa_10_8')
    pass
