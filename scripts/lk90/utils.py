'''
Notes:
* x = (vec{j}, vec{e}, vec{s})
* t_{lk, 0} = a0 = 1
'''
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def_eps_sl = 0.1
def_eps_gr = 0.1
def_eps_gw = 0.1

def to_vars(x):
    return x[ :3], x[3:6], x[6:9]

def ts_dot(x, y):
    ''' dot product of two time series (is there a better way?) '''
    z = np.zeros(np.shape(x)[1])
    for x1, y1 in zip(x, y):
        z += x1 * y1
    return z

def plot_traj(ret, fn, num_pts=1000, getter_kwargs={},
              use_stride=True, use_start=True, t_lk=None):
    L, e, s = to_vars(ret.y)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3,
                                                 figsize=(14, 8),
                                                 sharex=True)

    start_idx = len(ret.t) // 2 if use_start else 0
    stride = len(ret.t) // num_pts + 1 if use_stride else 1
    e_tot = np.sqrt(np.sum(e[:, start_idx::stride]**2, axis=0))
    Lnorm = np.sqrt(np.sum(L[:, start_idx::stride]**2, axis=0))
    Lhat = L[:, start_idx::stride] / Lnorm
    a = Lnorm**2 / (1 - e_tot**2)
    t_vals = ret.t[start_idx::stride]

    # 1 - e(t)
    ax1.semilogy(t_vals, 1 - e_tot, 'r')
    ax1.set_ylabel(r'$1 - e$')

    # I(t)
    I = np.arccos(Lhat[2])
    ax2.plot(t_vals, np.degrees(I), 'r')
    ax2.set_ylabel(r'$I$ (deg)')

    # a(t)
    ax3.semilogy(t_vals, a, 'r')
    ax3.set_ylabel(r'$a / a_0$')
    ax3.yaxis.set_label_position('right')

    # q_sl
    q_sl = np.arccos(ts_dot(Lhat, s[:, start_idx::stride]))
    ax4.plot(t_vals, np.degrees(q_sl), 'r')
    ax4.set_ylabel(r'$\theta_{\rm sl}$')

    # TODO actually code the below
    # $A$ Adiabaticity param
    # A = 8 * getter_kwargs.get('eps_sl', def_eps_sl) * np.sqrt(1 - e_tot**2) / (
    #     3 * (1 + 4 * e_tot**2) * np.abs(np.sin(2 * I))
    # ) / a[start_idx::stride]**4
    ax5.semilogy(t_vals, t_vals, 'r')
    ax5.set_ylabel(r'$\mathcal{A}$')

    # TODO decide on this?
    # e^2 + j^2 - 1
    ax6.plot(t_vals, t_vals, 'g')
    ax6.set_ylabel(r'$t$')
    ax6.yaxis.set_label_position('right')

    for ax in [ax4, ax5, ax6]:
        ax.set_xlabel(r'$t / t_{LK,0}$')
        plt.setp(ax.get_xticklabels(), rotation=45)
    if t_lk is not None:
        plt.suptitle(r'$t_{LK,0} = %.2e\;\mathrm{yr}$' % t_lk)
    plt.savefig(fn, dpi=300)
    plt.close()

# CONVENTION: all a and epsilon dependencis are handled in the getter
def dldt_lk(j, e, e_sq, n2, a):
    return 3 / 4 * np.sqrt(a) * (
        np.dot(j, n2) * np.cross(j, n2)
            - 5 * np.dot(e, n2) * np.cross(e, n2))
def dldt_gw(L, e_sq):
    return -32 / 5 * (
        (1 + 7 * e_sq / 8) / (1 - e_sq)**2
    ) * L
def dedt_lk(j, e, e_sq, n2):
    return 3 / 4 * (
        np.dot(j, n2) * np.cross(e, n2)
            - 5 * np.dot(e, n2) * np.cross(j, n2) # TODO sign error?
            + 2 * np.cross(j, e))
def dedt_gw(e, e_sq):
    return -(304 / 15) * (1 + 121 / 304 * e_sq) / (1 - e_sq)**(5/2) * e
def dedt_gr(Lhat, e, e_sq):
    return np.cross(Lhat, e) / (1 - e_sq)
def dsdt_sl(Lhat, s, e_sq):
    return np.cross(Lhat, s) / (1 - e_sq)

def get_dydt_gr(n2,
                eps_gw=def_eps_gw,
                eps_gr=def_eps_gr,
                eps_sl=def_eps_sl,
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
           **kwargs):
    ''' n2 = (0, 0, 1) by convention, choose jy(t=0) = ey(t=0) = 0 '''
    lx = -np.sin(I) * np.sqrt(1 - e**2)
    lz = np.cos(I) * np.sqrt(1 - e**2)
    ex = np.cos(I) * e
    ez = np.sin(I) * e
    # s parallel to j
    sx = -np.sin(I)
    sz = np.cos(I)
    y0 = np.array([lx, 0, lz, ex, 0, ez, sx, 0, sz])
    dydt = get_dydt_gr(np.array([0, 0, 1]), **getter_kwargs)

    def term_event(t, x):
        L, e, _ = to_vars(x)
        Lnorm_sq = np.sum(L**2)
        e_sq = np.sum(e**2)
        a = Lnorm_sq / (1 - e_sq)
        print("%d" % t, a)
        return a - a_f
    term_event.terminal = True
    events = [term_event]
    ret = solve_ivp(dydt, (0, tf), y0,
                    atol=atol, rtol=rtol, events=events,
                    **kwargs)
    print('Done for %f %f, t_f = %f' % (I, e, ret.t[-1]))
    return ret

# by convention, use solar masses, AU, and set c = 1, in which case G = 9.87e-9
# NB: slight confusion here: to compute epsilon + timescales, we use AU as the
# unit of length, but during the calculation, a0 is the unit of length
G = 9.87e-9
def get_eps(m1, m2, m3, a0, a2, e2):
    m12 = m1 + m2
    mu = m1 * m2 / m12
    n = np.sqrt(G * m12 / a0**3)
    t_lk0 = (1 / n) * (m12 / m3) * (a2 / a0)**3 * (1 - e2**2)**(3/2)
    eps_gw = (1 / n) * (m12 / m3) * (a2**3 / a0**7) * G**3 * mu * m12**2
    eps_gr = (m12 / m3) * (a2**3 / a0**4) * (1 - e2**2)**(3/2) * 3 * G * m12
    eps_sl = (m12 / m3) * (a2**3 / a0**4) * (1 - e2**2)**(3/2) * (
        3 * G * (m2 + mu / 3) / 2)

    s_per_unit = 499 # 1AU / c, in seconds
    s_per_yr = 3.154e7 # seconds per year
    # print((1 / n) * s_per_unit / s_per_yr) # orbital timescale
    return (t_lk0 * s_per_unit) / s_per_yr,\
        {"eps_gw": eps_gw, "eps_gr": eps_gr, "eps_sl": eps_sl}
