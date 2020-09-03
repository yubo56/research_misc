'''
test mass approx
Notes:
* x = (vec{j}, vec{e}, vec{s})
* t_{lk, 0} = a0 = 1
'''
import os
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.optimize import brenth
from scipy.interpolate import interp1d
import scipy.special as spe
from scipy.fft import dct, idct, fft
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass

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

def get_hat(phi, theta):
    return np.array([np.cos(phi) * np.sin(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(theta)])

def mkdirp(path):
    if not os.path.exists(path):
        os.mkdir(path)

def to_ang(vec):
    x, y, z = vec[0], vec[1], vec[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    q = np.degrees(np.arccos(z / r))
    pi2 = 2 * np.pi
    phi = (np.arctan2(y / np.sin(q), x / np.sin(q)) + pi2) % pi2
    return q, phi

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

def reg(z):
    return np.minimum(np.maximum(z, -1), 1)

def ts_dot(x, y):
    ''' dot product of two time series (is there a better way?) '''
    z = np.zeros(np.shape(x)[1])
    for x1, y1 in zip(x, y):
        z += x1 * y1
    return reg(z)

def ts_cross(x, y):
    return np.array([
        x[1] * y[2] - x[2] * y[1],
        -x[0] * y[2] + x[2] * y[0],
        x[0] * y[1] - x[1] * y[0],
    ])

def plot_traj_vecs(ret, fn, *args,
                   num_pts=1000, getter_kwargs={},
                   plot_slice=np.s_[::]):
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
    fig, ((ax3, ax1), (ax4, ax2), (ax7, ax5)) = plt.subplots(
        3, 2, figsize=(12, 8),
        # sharex=True,
    )
    t_lk, elim, elim_eta0, elim_naive = get_vals(m1, m2, m3, a0, a2, e2, I0)

    t_vals = ret.t[plot_slice]
    e_tot = np.sqrt(np.sum(e[:, plot_slice]**2, axis=0))
    Lnorm = np.sqrt(np.sum(L[:, plot_slice]**2, axis=0))
    Lhat = L[:, plot_slice] / Lnorm
    I = np.arccos(Lhat[2])
    a = Lnorm**2 / (1 - e_tot**2)
    dot_sl = ts_dot(Lhat, s[:, plot_slice])

    # 1 - e(t)
    ax1.semilogy(t_vals, 1 - e_tot, 'r')
    ax1.set_ylabel(r'$1 - e$')
    ax1.axhline(1 - elim_eta0, c='k', ls=':', lw=0.2)
    ax1.axhline(1 - elim_naive, c='b', ls=':', lw=0.2)

    # a(t)
    ax2.semilogy(t_vals, a, 'r')
    ax2.set_ylabel(r'$a / a_0$')

    # a(1 - e)
    # ax3.loglog(1 - e_tot, a, 'r')
    # ax3.set_xlim(right=0.1)
    # ax3.yaxis.tick_right()
    # overplot adiabaticity boundaries in a(1 - e) space
    # e_vals = 1 - np.linspace(*ax3.get_xlim(), 30)
    # def get_a_adiabatic(e):
    #     I_e = np.arccos(np.cos(I0) / np.sqrt(1 - e**2))
    #     def opt_func(a):
    #         return get_adiab(getter_kwargs, a, e, I_e) - 1
    #     return brenth(opt_func, 0.05, 1)
    # a_vals = [get_a_adiabatic(e) for e in e_vals]
    # # for a, e in zip(a_vals, e_vals):
    # #     print(a, e)
    # ax3.loglog(1 - e_vals, a_vals, 'k:')

    # theta_sb
    ax3.plot(t_vals, np.degrees(np.arccos(s[2, plot_slice])))
    ax3.set_ylabel(r'$\theta_{SB}$')

    # $A$ Adiabaticity param
    A = get_adiab(getter_kwargs, a, e_tot, I)
    print('Adiab, a', A[-1], a[-1])
    ax4.semilogy(t_vals, A, 'r')
    ax4.set_ylabel(r'$\mathcal{A}$')

    # q_sl
    q_sl = np.arccos(dot_sl)
    ax5.plot(t_vals, np.degrees(q_sl), 'r')
    ax5.set_ylabel(r'$\theta_{\rm sl}$')

    # spin-orbit coupling Hamiltonian
    # h_sl = getter_kwargs['eps_sl'] / a**(5/2) * dot_sl / (1 - e_tot**2)
    # ax6.semilogy(t_vals, h_sl, 'g')
    # ax6.semilogy(t_vals, -h_sl, 'g:')
    # ax6.set_ylabel(r'$H_{SL}$')
    # ax6.yaxis.set_label_position('right')

    # theta_sb
    # ax6.plot(t_vals, np.degrees(np.arccos(s[2, plot_slice])))
    # ax6.set_ylabel(r'$\theta_{SB}$')

    # for ax in [ax4, ax5, ax6]:
    #     ax.set_xlabel(r'$t / t_{LK,0}$')
    #     plt.setp(ax.get_xticklabels(), rotation=45)
    # plt.suptitle(r'$t_{LK,0} = %.2e\;\mathrm{yr}$' % t_lk)

    # q_eff, L
    reshape_w = lambda w: np.array([w, w, w])
    w_sl = (getter_kwargs.get('eps_sl', DEF_EPS_SL) / a**(5/2)) / (1 - e_tot**2)
    w_pl = 3 * Lhat[2] * a**(3/2) / (4 * np.sqrt(1 - e_tot**2)) * (
        1 + 4 * e_tot**2)
    w_eff = reshape_w(w_sl) * Lhat +\
        reshape_w(w_pl) * np.transpose([[0, 0, 1]] * len(w_pl))
    w_eff_norm = np.sqrt(np.sum(w_eff**2, axis=0))
    dot_eff = ts_dot(s[:, plot_slice], w_eff) / w_eff_norm
    q_eff = np.arccos(dot_eff)
    ax7.plot(t_vals, np.degrees(q_eff), 'r')
    ax7.set_ylabel(r'$\theta_{\rm eff, L}$')
    ax3.set_title(r'$t_{LK,0} = %.2e\;\mathrm{yr}$' % t_lk)

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

        dadt = 2 / (1 - e_sq) * (
            np.dot(L, dldt) + a * np.dot(e, dedt))
        print(t, a)

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

# get dWs from an actual ret_lk simulation
def get_dWs(ret_lk, getter_kwargs, stride=1, size=int(1e4), get_comps=False):
    t, (a, e, W, I, w), [t_lks, _] = ret_lk
    a_int = interp1d(t, a)
    e_int = interp1d(t, e)
    I_int = interp1d(t, I)
    W_int = interp1d(t, W)
    w_int = interp1d(t, w)
    dWeff_mags = []
    dWslx = []
    dWslz = []
    dWdot_mags = []
    times = []
    comps = []

    # # test, plot Wdot/Wsl over a single "period"
    # start, end = t_lks[1000:1002]
    # ts = np.linspace(start, end, size)
    # e_t = e_int(ts)
    # dWdt = (3 * a_int(ts)**(3/2) * np.cos(I_int(ts)) *
    #         (5 * e_t**2 * np.cos(w_int(ts))**2
    #          - 4 * e_t**2 - 1)
    #     / (4 * np.sqrt(1 - e_t**2)))
    # W_sl = getter_kwargs['eps_sl'] / (a_int(ts)**(5/2) * (1 - e_t**2))
    # plt.semilogy(ts, W_sl, 'r:')
    # plt.semilogy(ts, -dWdt, 'g:')
    # plt.savefig('/tmp/Wdots', dpi=200)
    # plt.close()
    # return

    mult = 1 if I[0] < np.radians(90) else -1
    for start, end in zip(t_lks[ :-1:stride], t_lks[1::stride]):
        ts = np.linspace(start, end, size)
        e_t = e_int(ts)
        dWdt = (3 * a_int(ts)**(3/2) * np.cos(I_int(ts)) *
                (5 * e_t**2 * np.cos(w_int(ts))**2
                 - 4 * e_t**2 - 1)
            / (4 * np.sqrt(1 - e_t**2))) #* mult
        W_sl = getter_kwargs['eps_sl'] / (a_int(ts)**(5/2) * (1 - e_t**2))

        # dWdt = (W_int(end) - W_int(start)) / (end - start)# * mult
        Wtot = np.sqrt(
            (W_sl * np.cos(I_int(ts)) - dWdt)**2
            + (W_sl * np.sin(I_int(ts)))**2)
        # NB: dW = int(Wdot dt)
        dWeff_mags.append(np.mean(Wtot))
        dWslx.append(np.mean(W_sl * np.sin(I_int(ts))))
        dWslz.append(np.mean(W_sl * np.cos(I_int(ts))))
        dWdot_mags.append(np.mean(dWdt))
        times.append((end + start) / 2)
        if get_comps:
            comps.append((
                dct(W_sl * np.sin(I_int(ts)), type=1)[::2] / (2 * size),
                dct(W_sl * np.cos(I_int(ts)) - dWdt, type=1)[::2] / (2 * size),
            ))
    dWslx = np.array(dWslx)
    dWslz = np.array(dWslz)
    dWsl_mags = np.sqrt(dWslx**2 + dWslz**2)
    if get_comps:
        return (
            np.array(dWeff_mags),
            dWsl_mags,
            np.array(dWdot_mags),
            np.array(times),
            dWslx,
            dWslz,
            comps)
    return (
        np.array(dWeff_mags),
        dWsl_mags,
        np.array(dWdot_mags),
        np.array(times),
        dWslx,
        dWslz)

# get the dWs for a given parameter set (fresh simulation)
def get_dW(e0, I0, eps_gr=0, eps_gw=0, eps_sl=1,
           atol=1e-9, rtol=1e-9, intg_pts=int(1e5), **kwargs):
    '''
    total delta Omega over an LK cycle, integrate the LK equations w/ eps_gr and
    eps_gw

    wdot = 3 * sqrt(h) / 4 * (1 - 2 * (x0 - h) / (x - h))
    '''
    a0 = 1
    W0 = 0
    w0 = 0
    eps_gw = 0

    def dydt(t, y):
        a, e, W, I, w = y
        x = 1 - e**2
        dadt =  (
            -eps_gw * (64 * (1 + 73 * e**2 / 24 + 37 * e**4 / 96)) / (
                5 * a**3 * x**(7/2))
        )
        dedt = (
            15 * a**(3/2) * e * np.sqrt(x) * np.sin(2 * w)
                    * np.sin(I)**2 / 8
        )
        dWdt = (
            3 * a**(3/2) * np.cos(I) *
                    (5 * e**2 * np.cos(w)**2 - 4 * e**2 - 1)
                / (4 * np.sqrt(x))
        )
        dIdt = (
            -15 * a**(3/2) * e**2 * np.sin(2 * w)
                * np.sin(2 * I) / (16 * np.sqrt(x))
        )
        dwdt = (
            3 * a**(3/2)
                * (2 * x + 5 * np.sin(w)**2 * (e**2 - np.sin(I)**2))
                / (4 * np.sqrt(x))
            + eps_gr / (a**(5/2) * x)
        )
        return (dadt, dedt, dWdt, dIdt, dwdt)
    def term_event(t, y):
        return y[4] - np.pi
    term_event.terminal = True
    ret = solve_ivp(dydt, (0, np.inf), [a0, e0, W0, I0, w0],
                    events=[term_event], atol=atol, rtol=rtol,
                    dense_output=True, **kwargs)
    times = np.linspace(0, ret.t[-1], intg_pts)
    a_arr, e_arr, W_arr, I_arr, w_arr = ret.sol(times)
    dWsl_z = np.sum(eps_sl / a_arr**(5/2) * np.cos(I_arr) / (1 - e_arr**2)
                    * ret.t[-1] / len(times))
    dWsl_x = np.sum(eps_sl / a_arr**(5/2) * np.sin(I_arr) / (1 - e_arr**2)
                    * ret.t[-1] / len(times))
    return ret.y[2, -1], (dWsl_z, dWsl_x), ret.y[1].max()

# get the dWs with time-averaged jhat (for agreement w/ LL17)
def get_dWjhat(e0, I0, Lout_mag, L_mag, eps_gr=0, eps_gw=0, eps_sl=1,
           atol=1e-10, rtol=1e-10, intg_pts=int(3e5)):
    '''
    total delta Omega over an LK cycle, integrate the LK equations w/ eps_gr and
    eps_gw

    wdot = 3 * sqrt(h) / 4 * (1 - 2 * (x0 - h) / (x - h))
    '''
    a0 = 1
    W0 = 0
    w0 = 0

    def dydt(t, y):
        a, e, W, I, w = y
        x = 1 - e**2
        dadt =  (
            -eps_gw * (64 * (1 + 73 * e**2 / 24 + 37 * e**4 / 96)) / (
                5 * a**3 * x**(7/2))
        )
        dedt = (
            15 * a**(3/2) * e * np.sqrt(x) * np.sin(2 * w)
                    * np.sin(I)**2 / 8
        )
        dWdt = (
            3 * a**(3/2) * np.cos(I) *
                    (5 * e**2 * np.cos(w)**2 - 4 * e**2 - 1)
                / (4 * np.sqrt(x))
        )
        dIdt = (
            -15 * a**(3/2) * e**2 * np.sin(2 * w)
                * np.sin(2 * I) / (16 * np.sqrt(x))
        )
        dwdt = (
            3 * a**(3/2)
                * (2 * x + 5 * np.sin(w)**2 * (e**2 - np.sin(I)**2))
                / (4 * np.sqrt(x))
            + eps_gr / (a**(5/2) * x)
        )
        return (dadt, dedt, dWdt, dIdt, dwdt)
    def term_event(t, y):
        return y[4] - np.pi
    term_event.terminal = True
    ret = solve_ivp(dydt, (0, np.inf), [a0, e0, W0, I0, w0],
                    events=[term_event], atol=atol, rtol=rtol,
                    method='BDF', dense_output=True)
    times = np.linspace(0, ret.t[-1], intg_pts)
    a_arr, e_arr, W_arr, I_arr, w_arr = ret.sol(times)
    x_arr = 1 - e_arr**2
    _dWsl_z = eps_sl / a_arr**(5/2) * np.cos(I_arr) / x_arr
    _dWsl_x = eps_sl / a_arr**(5/2) * np.sin(I_arr) / x_arr
    dWsl_z = np.sum(_dWsl_z * ret.t[-1] / len(times))
    dWsl_x = np.sum(_dWsl_x * ret.t[-1] / len(times))
    dW = (
        3 * a_arr**(3/2) * np.cos(I_arr) *
                (5 * e_arr**2 * np.cos(w_arr)**2 - 4 * e_arr**2 - 1)
            / (4 * np.sqrt(x_arr))
    )
    Jvec = L_mag * np.array([
        np.sin(I_arr),
        np.cos(I_arr),
    ]) + Lout_mag * np.array([
        np.zeros_like(I_arr),
        np.ones_like(I_arr),
    ])
    Jvec_hatx, Jvec_hatz = Jvec / np.sqrt(np.sum(Jvec**2, axis=0))
    dWz = np.sum(Jvec_hatz * dW * ret.t[-1] / len(times))
    dWx = np.sum(Jvec_hatx * dW * ret.t[-1] / len(times))

    Weffx = _dWsl_x - Jvec_hatx * dW
    Weffz = _dWsl_z - Jvec_hatz * dW
    x_coeffs = dct(Weffx, type=1)[::2] / (2 * len(times))
    z_coeffs = dct(Weffz, type=1)[::2] / (2 * len(times))
    return (dWz, dWx), (dWsl_z, dWsl_x), (x_coeffs, z_coeffs), ret.t[-1]

def get_dW_anal(e0, I0, intg_pts=int(1e5), eps_sl=0, **kwargs):
    # calculate n_e...
    x0 = 1 - e0**2
    h = x0 * np.cos(I0)**2
    b = -(5 + 5 * h - 2 * x0) / 3
    c = 5 * h / 3
    x1 = (-b - np.sqrt(b**2 - 4 * c)) / 2
    x2 = (-b + np.sqrt(b**2 - 4 * c)) / 2
    k_sq = (x0 - x1) / (x2 - x1)
    K = spe.ellipk(k_sq)
    ne = 6 * np.pi * np.sqrt(6) / (8 * K) * np.sqrt(x2 - x1)

    def solve_Wsl():
        phi = np.linspace(0, np.pi, intg_pts)
        intg_tot = np.pi / (
            (x0 + (x1 - x0) * np.cos(phi)**2)
                * np.sqrt(1 - k_sq * np.sin(phi)**2)
                * ne * K)
        intg_z = np.pi * np.sqrt(h) / (
            (x0 + (x1 - x0) * np.cos(phi)**2)**(3/2)
                * np.sqrt(1 - k_sq * np.sin(phi)**2)
                * ne * K)
        intg_x = np.pi * np.sqrt(1 - h / (x0 + (x1 - x0) * np.cos(phi)**2)) / (
            (x0 + (x1 - x0) * np.cos(phi)**2)
                * np.sqrt(1 - k_sq * np.sin(phi)**2)
                * ne * K)
        # wsl_tot = np.sum(intg_tot * phi[-1] / len(phi))
        wsl_z = np.sum(intg_z * phi[-1] / len(phi))
        wsl_x = np.sum(intg_x * phi[-1] / len(phi))
        return eps_sl * wsl_z, eps_sl * wsl_x
    def solve_dot():
        phi = np.linspace(0, np.pi, intg_pts)
        intg = (np.sin(I0)**2 / np.sqrt(1 - k_sq * np.sin(phi)**2)) / (
            np.sin(I0)**2 + (x1 / x0 - 1) * np.cos(phi)**2)
        int_res = np.sum(intg * phi[-1] / len(phi))
        return (3 * np.sqrt(h) * 2 * np.pi) / (4 * ne) * (1 - 1 / K * int_res)
    return solve_dot(), solve_Wsl()

def plot_weff_fft(Weff_vec, times, fn='6_vecfft', plot=True):
    # cosine transform
    x_coeffs = dct(Weff_vec[0], type=1)[::2] / (2 * len(times))
    z_coeffs = dct(Weff_vec[2], type=1)[::2] / (2 * len(times))
    # print(np.mean(Weff_vec[2]), z_coeffs[0]) # are equal
    # print(np.mean(Weff_vec[0]), x_coeffs[0]) # are equal

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
        # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 7))
        # ax3.semilogy(times, -Weff_vec[2], 'k', lw=1, label=r'$-z$')
        # ax3.semilogy(times, Weff_vec[0], 'b', lw=1, label=r'$x$')
        # ax3.legend(fontsize=12)
        # ax3.set_ylabel(r'$\Omega_{\rm eff}$')

        N = np.arange(len(z_coeffs))
        ax1.semilogy(N, x_coeffs, 'bo', ms=1)
        ax1.semilogy(N, -x_coeffs, 'ro', ms=1)
        # ax2.semilogy(N, z_coeffs, 'bo', ms=1)
        ax2.semilogy(N, -z_coeffs, 'ro', ms=1)
        ax1.set_ylabel(r'$\tilde{\Omega}_{\rm eff, x, N}$ (Rad / $t_{\rm LK,0}$)')
        ax2.set_ylabel(r'$\tilde{\Omega}_{\rm eff, z, N}$ (Rad / $t_{\rm LK,0}$)')
        ax2.set_xlabel(r'$N$')

        min_fact = 1e-6
        Nmax = np.where(abs(z_coeffs) / np.max(abs(z_coeffs)) < min_fact)[0][0]
        ax1.set_ylim(bottom=np.abs(x_coeffs).max() * min_fact / 3)
        ax2.set_ylim(bottom=np.abs(z_coeffs).max() * min_fact / 3)
        ax1.set_xlim((0, Nmax))
        ax2.set_xlim((0, Nmax))

        plt.tight_layout()
        plt.savefig(TOY_FOLDER + fn, dpi=200)
        plt.clf()

    return x_coeffs, z_coeffs

TOY_FOLDER = '6toy/'
mkdirp(TOY_FOLDER)
def single_cycle_toy(getter_kwargs, e0=1e-3, I0=np.radians(95),
                     w0=0, intg_pts=int(3e4), tf=50, **_kwargs):
    '''
    for far-out system params, solve toy problem in corotating frame
    '''
    eps_sl = getter_kwargs['eps_sl']
    eps_gr = getter_kwargs['eps_gr']
    a = 1
    W0 = 0

    def dydt(t, y):
        e, W, I, w = y
        x = 1 - e**2
        dedt = (
            15 * a**(3/2) * e * np.sqrt(x) * np.sin(2 * w)
                    * np.sin(I)**2 / 8
        )
        dWdt = (
            3 * a**(3/2) * np.cos(I) *
                    (5 * e**2 * np.cos(w)**2 - 4 * e**2 - 1)
                / (4 * np.sqrt(x))
        )
        dIdt = (
            -15 * a**(3/2) * e**2 * np.sin(2 * w)
                * np.sin(2 * I) / (16 * np.sqrt(x))
        )
        dwdt = (
            3 * a**(3/2)
                * (2 * x + 5 * np.sin(w)**2 * (e**2 - np.sin(I)**2))
                / (4 * np.sqrt(x))
            + eps_gr / (a**(5/2) * x)
        )
        return (dedt, dWdt, dIdt, dwdt)
    def start_event(t, y):
        return (y[3] % np.pi) - np.pi / 2
    start_event.direction = +1

    # use implicit method to ensure symmetry about half-period
    ret = solve_ivp(dydt, (0, tf), [e0, W0, I0, w0],
                    events=[start_event],
                    atol=1e-10, rtol=1e-10,
                    method='Radau', dense_output=True)
    times = np.linspace(ret.t_events[0][0], ret.t_events[0][1], intg_pts)
    def symmetrize(x):
        return (x + np.flip(x, axis=-1)) / 2
    e_arr, W_arr, I_arr, w_arr = ret.sol(times)
    e_arr = symmetrize(e_arr)
    I_arr = symmetrize(I_arr)
    x_arr = 1 - e_arr**2

    # compute W0_vec, WSL_vec, Weff_vec
    W0_vec = np.outer(
        np.array([0, 0, 1]),
        (
            3 * a**(3/2) * np.cos(I_arr) *
                    (5 * e_arr**2 * np.cos(w_arr)**2 - 4 * e_arr**2 - 1)
                / (4 * np.sqrt(x_arr))
        ))
    WSL_vec = np.array([
        eps_sl / (a**(5/2) * x_arr) * np.sin(I_arr),
        np.zeros(intg_pts),
        eps_sl / (a**(5/2) * x_arr) * np.cos(I_arr)])
    Weff_vec = WSL_vec - W0_vec
    return Weff_vec, times

def cosd(x):
    return np.cos(np.radians(x))

def get_fn_I(I_deg):
    return ('%.3f' % I_deg).replace('.', '_')

def get_scinot(f):
    exponent = np.floor(np.log10(f))
    return r'%.1f \times 10^{%d}' % (f / 10**exponent, exponent)

def smooth(f, len_sm):
    _kernel = np.exp(-(np.arange(len_sm) - (len_sm // 2))**2 /
                     (2 * (len_sm // 4)**2))
    kernel = _kernel / sum(_kernel)
    f_padded = np.concatenate(([f[0]] * (len(kernel) // 2),
                               f,
                               [f[-1]] * (len(kernel) // 2)))
    return np.convolve(f_padded, kernel, mode='valid')

# Recall that the sign of the eigenvector is random
def get_monodromy(params, tol=1e-8, num_periods=1, **kwargs):
    getter_kwargs = get_eps(*params)
    Weff_vec, t_vals = single_cycle_toy(getter_kwargs, **kwargs)

    t0 = t_vals[0]
    tf = t_vals[-1]
    period = tf - t0
    Weff_x_interp = interp1d(t_vals, Weff_vec[0])
    Weff_z_interp = interp1d(t_vals, Weff_vec[2])
    Weff_x_mean = np.mean(Weff_vec[0])
    Weff_z_mean = np.mean(Weff_vec[2])
    def dydt(t, s):
        t_curr = (t - t0) % period + t0
        return np.cross(
            [Weff_x_interp(t_curr), 0, Weff_z_interp(t_curr)],
            s)

    sf_arr = []
    t_init = t0
    mat_init = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for s0 in mat_init:
        ret = solve_ivp(dydt, (t_init, t_init + num_periods * period),
                        s0, atol=tol, rtol=tol, method='Radau')
        sf_arr.append(ret.y[:, -1])

    mono_mat = np.array(sf_arr).T
    eigs, eigv = np.linalg.eig(np.matmul(np.linalg.inv(mat_init), mono_mat))
    one_eig_idx = np.where(abs(np.imag(eigs)) < tol)[0][0]
    mono_eig = np.real(eigv[:, one_eig_idx])
    Weff_mean = np.array([Weff_x_mean, 0, Weff_z_mean])
    Weff_hat = Weff_mean / np.sqrt(np.sum(Weff_mean**2))
    return mono_mat, mono_eig, Weff_mean

# Recall that the sign of the eigenvector is random
def get_monodromy_fast(params, tol=1e-8, num_periods=1,
                       e0=1e-3, I0=np.radians(95), **kwargs):
    getter_kwargs = get_eps(*params)
    eps_sl = getter_kwargs['eps_sl']
    eps_gr = getter_kwargs['eps_gr']
    a = 1
    w0 = 0

    def dydt(t, y):
        e, I, w, *svecs = y
        x = 1 - e**2
        dedt = (
            15 * a**(3/2) * e * np.sqrt(x) * np.sin(2 * w)
                    * np.sin(I)**2 / 8
        )
        dIdt = (
            -15 * a**(3/2) * e**2 * np.sin(2 * w)
                * np.sin(2 * I) / (16 * np.sqrt(x))
        )
        dwdt = (
            3 * a**(3/2)
                * (2 * x + 5 * np.sin(w)**2 * (e**2 - np.sin(I)**2))
                / (4 * np.sqrt(x))
            + eps_gr / (a**(5/2) * x)
        )

        # dW/dt only impacts svecs
        dWdt = (
            3 * a**(3/2) * np.cos(I) *
                (5 * e**2 * np.cos(w)**2 - 4 * e**2 - 1)
            / (4 * np.sqrt(x))
        )

        Weff_vec = np.array([
            eps_sl / (a**(5/2) * x) * np.sin(I),
            0,
            eps_sl / (a**(5/2) * x) * np.cos(I) - dWdt])
        s1, s2, s3 = np.reshape(svecs, (3, 3))
        ds_dt = np.concatenate((
            np.cross(Weff_vec, s1),
            np.cross(Weff_vec, s2),
            np.cross(Weff_vec, s3),
        ))
        return (dedt, dIdt, dwdt, *ds_dt)
    def term_event(t, y): # one Kozai cycle, starting from emax
        return y[2] - np.pi
    term_event.terminal = True
    mat_init = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # use implicit method to ensure symmetry about half-period
    ret = solve_ivp(dydt, (0, np.inf), [e0, I0, w0, *mat_init.flatten()],
                    events=[term_event], atol=tol, rtol=tol,
                    method='Radau', dense_output=True)

    sf_arr = np.reshape(ret.y[3: ,-1], (3, 3))

    mono_mat = np.array(sf_arr).T
    eigs, eigv = np.linalg.eig(np.matmul(np.linalg.inv(mat_init), mono_mat))
    one_eig_idx = np.where(abs(np.imag(eigs)) < tol)[0][0]
    mono_eig = np.real(eigv[:, one_eig_idx])

    return mono_mat, mono_eig
