'''
test mass approx
Notes:
* x = (vec{j}, vec{e}, vec{s})
* t_{lk, 0} = a0 = 1
'''
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.optimize import brenth
import matplotlib.pyplot as plt
plt.style.use('ggplot')

DEF_EPS_SL = 0
DEF_EPS_GR = 0
DEF_EPS_GW = 0

# by convention, use solar masses, AU, and set c = 1, in which case G = 9.87e-9
# NB: slight confusion here: to compute epsilon + timescales, we use AU as the
# unit of length, but during the calculation, a0 is the unit of length
G = 9.87e-9
S_PER_UNIT = 499 # 1AU / c, in seconds
S_PER_YR = 3.154e7 # seconds per year
def get_eps(m1, m2, m3, a0, a2, e2):
    m12 = m1 + m2
    mu = m1 * m2 / m12
    n = np.sqrt(G * m12 / a0**3)
    eps_gw = (1 / n) * (m12 / m3) * (a2**3 / a0**7) * G**3 * mu * m12**2
    eps_gr = (m12 / m3) * (a2**3 / a0**4) * (1 - e2**2)**(3/2) * 3 * G * m12
    eps_sl = (m12 / m3) * (a2**3 / a0**4) * (1 - e2**2)**(3/2) * (
        3 * G * (m2 + mu / 3) / 2)
    return {'eps_gw': eps_gw, 'eps_gr': eps_gr, 'eps_sl': eps_sl}

def get_vals(m1, m2, m3, a0, a2, e2, I):
    ''' calculates a bunch of physically relevant values '''
    m12 = m1 + m2
    m123 = m12 + m3
    mu = m1 * m2 / m12
    mu123 = m12 * m3 / m123

    # calculate lk time
    n = np.sqrt(G * m12 / a0**3)
    t_lk0 = (1 / n) * (m12 / m3) * (a2 / a0)**3 * (1 - e2**2)**(3/2)

    # calculate jmin
    eta = mu / mu123 * np.sqrt(m12 * a0 / (m123 * a2 * (1 - e2**2)))
    eps_gr = 3 * G * m12**2 * a2**3 * (1 - e2**2)**(3/2) / (a0**4 * m3)
    def jmin_criterion(j): # eq 42, satisfied when j = jmin
        return (
            3/8 * (j**2 - 1) / j**2 * (
                5 * (np.cos(I) + eta / 2)**2
                - (3 + 4 * eta * np.cos(I) + 9 * eta**2 / 4) * j**2
                + eta**2 * j**4)
            + eps_gr * (1 - 1 / j))
    def jmin_eta0(j): # set eta to zero, corresponds to test mass approx
        return (
            3/8 * (j**2 - 1) / j**2 * (
                5 * np.cos(I)**2
                - 3 * j**2)
            + eps_gr * (1 - 1 / j))
    jmin = brenth(jmin_criterion, 1e-15, 1 - 1e-15)
    jmin_eta0 = brenth(jmin_eta0, 1e-15, 1 - 1e-15)
    jmin_naive = np.sqrt(5 * np.cos(I)**2 / 3)
    emax = np.sqrt(1 - jmin**2)
    emax_eta0 = np.sqrt(1 - jmin_eta0**2)
    emax_naive = np.sqrt(1 - jmin_naive**2)

    return (t_lk0 * S_PER_UNIT) / S_PER_YR, emax, emax_eta0, emax_naive

def get_adiab(getter_kwargs, a, e, I):
    return 8 * getter_kwargs.get('eps_sl', DEF_EPS_SL) / (
           3 * a**4 * np.sqrt(1 - e**2) * (1 + 4 * e**2) *
           np.abs(np.sin(2 * I)))

def get_tmerge(m1, m2, m3, a0, a2, e2, I):
    ''' returns in units of LK0 as well as physical units (years) '''
    m12 = m1 + m2
    mu = m1 * m2 / m12
    emax = get_vals(m1, m2, m3, a0, a2, e2, I)[2]
    tm0 = 5 / (256 * G**3 * m12**2 * mu)
    tm = tm0 * (1 - emax**2)**3
    return tm, (tm * S_PER_UNIT) / S_PER_YR

def to_vars(x):
    return x[ :3], x[3:6], x[6:9]

def ts_dot(x, y):
    ''' dot product of two time series (is there a better way?) '''
    z = np.zeros(np.shape(x)[1])
    for x1, y1 in zip(x, y):
        z += x1 * y1
    return z

def plot_traj_vecs(ret, fn, *args,
                   num_pts=1000, getter_kwargs={},
                   plot_slice=np.s_[::]):
    def to_ang(vec):
        x, y, z = vec[0], vec[1], vec[2]
        r = np.sqrt(x**2 + y**2 + z**2)
        q = np.degrees(np.arccos(z / r))
        pi2 = 2 * np.pi
        phi = (np.arctan2(y / np.sin(q), x / np.sin(q)) + pi2) % pi2
        return q, phi

    t = ret.t[plot_slice]
    L, e, s = to_vars(ret.y)
    fig, ((ax1, ax2),
          (ax5, ax6)) = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    Lq, Lphi = to_ang(L[:, plot_slice])
    # eq, ephi = to_ang(e[:, plot_slice])
    sq, sphi = to_ang(s[:, plot_slice])

    ax1.plot(t, Lq)
    ax2.plot(t, Lphi)
    ax5.plot(t, sq)
    ax6.plot(t, sphi)
    ax1.set_ylabel(r'$I$')
    ax2.set_ylabel(r'$\phi_L$')
    ax5.set_ylabel(r'$\theta_{s3}$')
    ax6.set_ylabel(r'$\phi_s$')
    ax6.set_xlabel(r'$t / t_{\rm LK}$')
    plt.savefig(fn + '_vecs', dpi=300)
    plt.close()

def plot_traj(ret, fn,
              m1, m2, m3, a0, a2, e2, I0,
              num_pts=1000, getter_kwargs={},
              plot_slice=np.s_[::]):
    L, e, s = to_vars(ret.y)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3,
                                                 figsize=(14, 8),
                                                 sharex=True)
    t_lk, elim, elim_eta0, elim_naive = get_vals(m1, m2, m3, a0, a2, e2, I0)

    t_vals = ret.t[plot_slice]
    e_tot = np.sqrt(np.sum(e[:, plot_slice]**2, axis=0))
    Lnorm = np.sqrt(np.sum(L[:, plot_slice]**2, axis=0))
    Lhat = L[:, plot_slice] / Lnorm
    a = Lnorm**2 / (1 - e_tot**2)
    dot_sl = ts_dot(Lhat, s[:, plot_slice])

    # 1 - e(t)
    ax1.semilogy(t_vals, 1 - e_tot, 'r')
    ax1.set_ylabel(r'$1 - e$')
    ax1.axhline(1 - elim_eta0, c='k', ls=':', lw=0.2)
    ax1.axhline(1 - elim_naive, c='b', ls=':', lw=0.2)

    # I(t)
    I = np.arccos(Lhat[2])
    ax2.plot(t_vals, np.degrees(I), 'r')
    ax2.set_ylabel(r'$I$ (deg)')

    # a(t)
    ax3.semilogy(t_vals, a, 'r')
    ax3.set_ylabel(r'$a / a_0$')
    ax3.yaxis.tick_right()

    # q_sl
    q_sl = np.arccos(dot_sl)
    ax4.plot(t_vals, np.degrees(q_sl), 'r')
    ax4.set_ylabel(r'$\theta_{\rm sl}$')

    # $A$ Adiabaticity param
    A = get_adiab(getter_kwargs, a, e_tot, I)
    ax5.semilogy(t_vals, A, 'r')
    ax5.set_ylabel(r'$\mathcal{A}$')

    # spin-orbit coupling Hamiltonian
    # h_sl = getter_kwargs['eps_sl'] / a**(5/2) * dot_sl / (1 - e_tot**2)
    # ax6.semilogy(t_vals, h_sl, 'g')
    # ax6.semilogy(t_vals, -h_sl, 'g:')
    # ax6.set_ylabel(r'$H_{SL}$')
    # ax6.yaxis.set_label_position('right')

    # theta_s3
    ax6.plot(t_vals, np.degrees(np.arccos(s[2, plot_slice])))
    ax6.set_ylabel(r'$\theta_{S3}$')

    for ax in [ax4, ax5, ax6]:
        ax.set_xlabel(r'$t / t_{LK,0}$')
        plt.setp(ax.get_xticklabels(), rotation=45)
    plt.suptitle(r'$t_{LK,0} = %.2e\;\mathrm{yr}$' % t_lk)
    plt.savefig(fn, dpi=300)
    plt.close()

# CONVENTION: all a and epsilon dependencis are handled in the getter
def dldt_lk(j, e, e_sq, n2, a):
    return 3 / 4 * np.sqrt(a) * (
        np.dot(j, n2) * np.cross(j, n2)
            - 5 * np.dot(e, n2) * np.cross(e, n2))
def dldt_gw(L, e_sq):
    return -32 / 5 * ((1 + 7 * e_sq / 8) / (1 - e_sq)**(5/2)) * L
def dedt_lk(j, e, e_sq, n2):
    return 3 / 4 * (
        np.dot(j, n2) * np.cross(e, n2)
            - 5 * np.dot(e, n2) * np.cross(j, n2)
            + 2 * np.cross(j, e))
def dedt_gw(e, e_sq):
    return -(304 / 15) * (1 + 121 / 304 * e_sq) / (1 - e_sq)**(5/2) * e
def dedt_gr(Lhat, e, e_sq):
    return np.cross(Lhat, e) / (1 - e_sq)
def dsdt_sl(Lhat, s, e_sq):
    return np.cross(Lhat, s) / (1 - e_sq)

def get_dydt_gr(n2,
                eps_gw=DEF_EPS_GW,
                eps_gr=DEF_EPS_GR,
                eps_sl=DEF_EPS_SL,
                kozai=1, # not really physical but useful for debug?
            ):
    def dydt(t, x):
        L, e, s = to_vars(x)
        Lnorm_sq = np.sum(L**2)
        Lhat = L / np.sqrt(Lnorm_sq)
        e_sq = np.sum(e**2)
        j = Lhat * np.sqrt(1 - e_sq)
        a = Lnorm_sq / (1 - e_sq)
        dldt = (
            (kozai * a**(3/2)) * dldt_lk(j, e, e_sq, n2, a)
            + (eps_gw / a**4) * dldt_gw(L, e_sq)
        )
        dedt = (
            (kozai * a**(3/2)) * dedt_lk(j, e, e_sq, n2)
            + (eps_gw / a**4) * dedt_gw(e, e_sq)
            + (eps_gr / a**(5/2)) * dedt_gr(Lhat, e, e_sq)
        )
        dsdt = (eps_sl / a**(5/2)) * dsdt_sl(Lhat, s, e_sq)

        return np.concatenate((dldt, dedt, dsdt))
    return dydt

def solver(I, e, tf=50, atol=1e-9, rtol=1e-9,
           a_f=3e-1,
           getter_kwargs={},
           q_sl0=0,
           w0_0=False, # try allowing initial omega = 0
           **kwargs):
    ''' n2 = (0, 0, 1) by convention, choose jy(t=0) = ey(t=0) = 0 '''
    lx = -np.sin(I) * np.sqrt(1 - e**2)
    lz = np.cos(I) * np.sqrt(1 - e**2)
    ex = np.cos(I) * e
    ez = np.sin(I) * e
    sx = -np.sin(I + q_sl0)
    sz = np.cos(I + q_sl0)
    if w0_0 == True:
        y0 = np.array([lx, 0, lz, 0, e, 0, sx, 0, sz])
    else:
        y0 = np.array([lx, 0, lz, ex, 0, ez, sx, 0, sz])
    dydt = get_dydt_gr(np.array([0, 0, 1]), **getter_kwargs)

    def term_event(t, x):
        L, e, _ = to_vars(x)
        Lnorm_sq = np.sum(L**2)
        e_sq = np.sum(e**2)
        a = Lnorm_sq / (1 - e_sq)
        return a - a_f
    term_event.terminal = True
    events = [term_event]
    start = time.time()
    ret = solve_ivp(dydt, (0, tf), y0,
                    atol=atol, rtol=rtol, events=events,
                    **kwargs)
    print('Done for I0=%f, e0=%f. t_f=%f (took %.2fs)' %
          (np.degrees(I), e, ret.t[-1], time.time() - start))
    return ret
