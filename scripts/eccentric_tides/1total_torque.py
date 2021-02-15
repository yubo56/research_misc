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
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.0)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

ms = 5.3
lw = 2
af = 1
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

def plot_ecc(w_s=0, num_pts=2000, emax=0.99):
    ''' plots behavior of total torque with varying eccentricity '''
    e_vals = np.linspace(0, emax**2, 99)**(1/2)
    totals = []
    totals_integ = []

    Nmax_max = 15 * np.sqrt(1 + emax) / (1 - emax)**(3/2)
    for e in e_vals:
        num_pts_e = int(max(Nmax_max * (
            (np.sqrt(1 + e) / (1 - e)**(3/2))
            / (np.sqrt(1 + emax) / (1 - emax)**(3/2))
        ), 30))
        coeffs = get_coeffs_fft(num_pts_e, 2, e)
        torques = get_torques(num_pts_e, w_s)
        totals.append(np.sum(coeffs**2 * torques))
        totals_integ.append(get_integral(w_s, e, Nmax=num_pts_e))
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
        plt.ylabel(r'$T / T_0$')
    else:
        torque = -prefactor * ((w_s / eta2 - 1)**(8/3) * 2**(8/3))
        sign = -1
        # plt.title(r'$\frac{\Omega_s}{\Omega} = %d \gg N_{\rm peri}$' % w_s)
        plt.ylabel(r'$-T / T_0$')
    plt.semilogy(e_vals, sign * totals, ls='', marker='o', mfc='none', mec='k',
                 ms=ms, label='Sum')
    plt.semilogy(e_vals, sign * totals_integ, 'kx', ms=ms, label='Integral')
    plt.semilogy(e_vals, sign * torque, 'k', lw=lw, alpha=af,
                 label='Analytic')
    plt.xlabel(r'$e$')
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.text(0.72, (plt.ylim()[1] / plt.ylim()[0])**(0.03) * plt.ylim()[0],
             r'$\Omega_{\rm s} = %s$'
                % ('0' if w_s == 0 else r'%d \Omega' % w_s))
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

    w_p = np.sqrt(1 + e) / (1 - e)**(3/2)
    plt.xlabel(r'$\Omega_{\rm s} / \Omega_{\rm p}$')
    plt.ylabel(r'$T / T_0$')
    # plt.title(r'$e = %.1f$' % e)
    plt.plot(w_vals / w_p, totals, ls='', marker='o', mfc='none', mec='k',
                ms=ms, label=r'Sum', alpha=af)
    plt.plot(w_vals / w_p, totals_integ, 'kx',
                 ms=ms, label=r'Integral', alpha=af)
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

    torques = np.concatenate((torque_low_spin, torque_high_spin))
    plt.plot(w_vals / w_p, torques, 'k',
                 label='Analytic', lw=lw, alpha=af)
    plt.ylim(ylims)
    plt.yscale('symlog', linthresh=1e8)

    plt.legend(fontsize=14)
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

def plot_energy(w_s=0, num_pts=2000, emax=0.99):
    ''' plots behavior of heating with varying eccentricity '''
    e_vals = np.linspace(0, emax**2, 99)**(1/2)
    totals = []
    totals_integ = []
    Nmax_max = 10 * np.sqrt(1 + emax) / (1 - emax)**(3/2)
    for e in e_vals:
        num_pts_e = int(max(Nmax_max * (
            (np.sqrt(1 + e) / (1 - e)**(3/2))
            / (np.sqrt(1 + emax) / (1 - emax)**(3/2))
        ), 30))
        energies = get_energies(num_pts_e, e, w_s)
        totals.append(np.sum(energies))
        totals_integ.append(get_energies_integ(w_s, e, Nmax=num_pts_e))
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
        plt.ylabel(r'$\dot{E}_{\rm in} / T_0\Omega$')
        sign = +1
    else:
        # plt.title(r'$\frac{\Omega_s}{\Omega} = %d \gg N_{\rm peri}$' % w_s)
        plt.ylabel(r'$-\dot{E}_{\rm in} / T_0\Omega$')
        sign = -1

    term0 = (
        f5 * gamma(14 / 3) * (3/2)**(8/3) / (1 - e_vals**2)**10
            * (e_vals**2 * f3 / (2 * f5))**(11/6))
    term2 = sign * (
        np.abs(1 - 0.5886 * w_s / eta2)**(8/3)
            * f5 / (1 - e_vals**2)**(9/2)
            * gamma(26/3) / gamma(5)
            * (eta2 / 2)**(11/3))

    plt.semilogy(e_vals, sign * totals, ls='', marker='o', mfc='none', mec='k',
                 ms=ms, label='Sum')
    plt.semilogy(e_vals, sign * totals_integ, 'kx', ms=ms, label='Integral')
    plt.semilogy(e_vals, sign * (term0 + term2) / 2, 'k', lw=lw, alpha=af,
                 label='Analytic')

    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.text(0.72, (plt.ylim()[1] / plt.ylim()[0])**(0.03) * plt.ylim()[0],
             r'$\Omega_{\rm s} = %s$'
                % ('0' if w_s == 0 else r'%d \Omega' % w_s))
    plt.savefig('1totals_e_%d' % w_s, dpi=400)
    plt.clf()

def plot_spin_energy(e=0.9, num_pts=300):
    ''' plots behavior of heating with varying spin '''
    w_p = np.sqrt(1 + e) / (1 - e)**(3/2)
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

    plt.xlabel(r'$\Omega_{\rm s} / \Omega_{\rm p}$')
    plt.ylabel(r'$\dot{E}_{\rm in} / T_0\Omega$')
    # plt.title(r'$e = %.1f$' % e)

    plt.plot(w_vals / w_p, totals, ls='', marker='o', mfc='none', mec='k',
             ms=ms, alpha=af, label='Sum')

    plt.plot(w_vals / w_p, totals_integ,
                 'kx', ms=ms, alpha=af, label='Integral')
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

    edotin = (np.concatenate([term2_ls, term2_hs]) + term0) / 2
    plt.plot(w_vals / w_p, edotin, 'k', ms=ms, alpha=af, label='Analytic')
    plt.yscale('symlog', linthresh=1e10)
    plt.ylim(ylims)

    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('1totals_NRG_e_%s' % ('%.1f' % e).replace('.', '_'), dpi=400)
    plt.clf()

def plot_pseudo():
    ''' plots pseudosynchronous frequency vs e '''
    e_vals = np.linspace(0.02, 0.98, 100)

    f2 = 1 + 15*e_vals**2/2 + 45*e_vals**4/8 + 5*e_vals**6/16
    f5 = 1 + 3 * e_vals**2 + 3 * e_vals**4 / 8
    eta2 = 4 * f2 / (5 * f5 * (1 - e_vals**2)**(3/2))

    w_syncs = []
    w_syncs_sum = []
    Nmax_max_0 = 10 * np.sqrt(1 + 0.98) / (1 - 0.98)**(3/2)
    for e in e_vals:
        Nmax_max =  int(max(Nmax_max_0 * (
            (np.sqrt(1 + e) / (1 - e)**(3/2))
            / (np.sqrt(1 + 0.98) / (1 - 0.98)**(3/2))
        ), 30))
        def opt_func(w):
            return get_integral(w, e, Nmax=int(Nmax_max))
        def opt_func_sum(w):
            return get_energies_integ(w, e, Nmax=int(Nmax_max))
        res = brenth(opt_func, 0, Nmax_max)
        res2 = brenth(opt_func_sum, 0, Nmax_max)
        print(e, res, res2)
        w_syncs.append(res)
        w_syncs_sum.append(res2)
    w_p = np.sqrt(1 + e_vals) / (1 - e_vals)**(3/2)
    # x = 1 - e_vals**2
    x = e_vals
    plt.plot(x, w_syncs_sum / w_p, 'k', label='Exact')
    # plt.loglog(x, w_syncs, label='Integral')
    plt.plot(x, eta2 / 0.691 / w_p, 'b',
               label=r'$\Omega_{\rm s, ps}$')
    plt.plot(x, f2 / (f5 * (1 - e_vals**2)**(3/2)) / w_p, 'r',
               label=r'$\Omega_{\rm s, ps}^{\rm (Eq)}$')
    # plt.xticks([1 - 0.99**2, 1 - 0.9**2, 1 - 0.5**2],
    #            labels=['0.99', '0.9', '0.5'])
    plt.xlabel(r'$e$')
    plt.ylabel(r'$\Omega_{\rm sync} / \Omega_{\rm p}$')
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('1pseudosynchronous', dpi=200)

# don't think this plot makes sense, we don't know the breakup frequency
# independently of r_c
def plot_7319(e=0.808, obs_pdot=-3.03e-7, mc=3, prefix='7319',
              rhoc_rat=0.76, rcs=[0.7, 1.0, 1.38, 1.5]):
    num_pts = max(4 * int(np.sqrt(1 + e) * (1 - e**2)**(-3/2)), 150)
    def get_disp_spin(w_s):
        return np.sum(get_energies(num_pts, e, w_s))
    def get_torque(w_s):
        coeffs = get_coeffs_fft(num_pts, 2, e)
        torques = get_torques(num_pts, w_s)
        return np.sum(coeffs**2 * torques)
    wperi = np.sqrt(1 + e) / (1 - e)**(3/2)

    rhoc_rat_base = 0.76 * (1 - 0.76)**2
    rat = rhoc_rat * (1 - rhoc_rat)**2 / rhoc_rat_base
    colors = ['b', 'g', 'c', 'r']
    for c, rc in zip(colors, rcs):
        rat_rc = (rc / 1.38)**9 / (mc / 3)**(8/3)
        breakup = np.sqrt(
            6.67e-11 * mc * 1.99e30 / (rc * 6.9e8)**3
        ) * (51 * 86400) / (2 * np.pi)

        spins = np.linspace(-np.sqrt(breakup), np.sqrt(breakup), 50)
        spins *= np.abs(spins)
        # = g(e, w_s) by notation of paper
        disp_spins = np.array([get_disp_spin(w_s) for w_s in spins])
        torque_spins = np.array([get_torque(w_s) for w_s in spins])

        th_pdot = -rat * rat_rc * 1.52e-18 * np.abs(spins / breakup)**(8/3) * disp_spins
        plt.plot(spins / wperi, th_pdot, c=c, label='%.2f' % rc)
    plt.axhline(obs_pdot, c='k', ls='dashed', lw=lw)

    plt.yscale('symlog', linthresh=1e-12)
    ticks = [-12, -10, -8, -6]
    plt.yticks([-10**(p) for p in ticks[::-1]]
               + [0]
               +[10**(p) for p in ticks],
               labels=[r'$-10^{%d}$' % p for p in ticks[::-1]]
               + [0]
               + [r'$10^{%d}$' % p for p in ticks])
    ylims = plt.ylim()
    breakup_star = np.sqrt(
        6.67e-11 * 8.8 * 1.99e30 / (6.4 * 6.9e8)**3
    ) * (51 * 86400) / (2 * np.pi) / wperi
    plt.fill_betweenx([-1, 1],
                      [-breakup_star, -breakup_star],
                      [breakup_star, breakup_star],
                      color='k',
                      alpha=0.1)
    plt.ylim(ylims)

    plt.legend(fontsize=14, loc='upper left')
    plt.xlabel(r'$\Omega_{\rm s} / \Omega_{\rm p}$')
    plt.ylabel(r'$\dot{P}$')
    plt.tight_layout()
    plt.savefig('1_%s_disps' % prefix)
    plt.clf()
    return

    # plot energy dissipation in rotating frame (heating)
    retrospin_idx = np.argmin(np.abs(th_pdot - obs_pdot))
    print('Naive', spins[retrospin_idx])

    edot_rot = (disp_spins - spins * torque_spins) * rat * 1.113e-11
    plt.plot(spins, edot_rot)
    print(edot_rot[retrospin_idx])
    plt.xlabel(r'$\Omega_{\rm s} / \Omega$')
    plt.ylabel(r'$\dot{E}_{\rm in}$ ($L_{\odot}$)')
    plt.tight_layout()
    plt.savefig('1_%s_heating' % prefix)
    plt.clf()

if __name__ == '__main__':
    # plot_ecc(emax=0.96)
    # plot_ecc(400, emax=0.96)
    # plot_spin()

    plot_energy(emax=0.96)
    plot_energy(400, emax=0.96)
    # plot_spin_energy(e=0.9)

    # plot_pseudo()
    # plot_7319(rhoc_rat=1/3, mc=3)
    # plot_7319(obs_disp = 2.81e13, breakup=1077.76, prefix='7319_mesa_10_8')
    pass
