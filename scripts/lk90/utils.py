'''
Notes:
* x = (vec{j}, vec{e}, vec{s}, a)
* t_{lk, 0} = a0 = 1
'''
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def_eps_sl = 0.1
def_eps_gr = 0.1

def to_vars(x):
    return x[ :3], x[3:6], x[6:9], x[9]

def ts_dot(x, y):
    ''' dot product of two time series (is there a better way?) '''
    z = np.zeros(np.shape(x)[1])
    for x1, y1 in zip(x, y):
        z += x1 * y1
    return z

def plot_traj(ret, fn, num_pts=1000, getter_kwargs={},
              use_stride=True, use_start=True):
    j, e, s, a = to_vars(ret.y)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3,
                                                 figsize=(14, 8),
                                                 sharex=True)

    start_idx = len(ret.t) // 2 if use_start else 0
    stride = len(ret.t) // num_pts + 1 if use_stride else 1
    e_tot = np.sqrt(np.sum(e[:, start_idx::stride]**2, axis=0))
    j_tot = np.sqrt(np.sum(j[:, start_idx::stride]**2, axis=0))
    l = j[:, start_idx::stride] / j_tot

    # 1 - e(t)
    ax1.semilogy(ret.t[start_idx::stride], 1 - e_tot, 'r')
    ax1.set_ylabel(r'$1 - e$')

    # I(t)
    I = np.arccos(l[2])
    ax2.plot(ret.t[start_idx::stride], np.degrees(I), 'r')
    ax2.set_ylabel(r'$I$ (deg)')

    # a(t)
    ax3.semilogy(ret.t[start_idx::stride], a[start_idx::stride], 'r')
    ax3.set_ylabel(r'$a / a_0$')
    ax3.yaxis.set_label_position('right')

    # q_sl
    q_sl = np.arccos(ts_dot(l, s[:, start_idx::stride]))
    ax4.plot(ret.t[start_idx::stride], np.degrees(q_sl), 'r')
    ax4.set_ylabel(r'$\theta_{\rm sl}$')

    # e . j
    # edotj = ts_dot(e[:, start_idx::stride], j[:, start_idx::stride])
    # ax5.plot(ret.t[start_idx::stride], edotj, 'r')
    # ax5.set_ylabel(r'$\vec{e} \cdot \vec{j}$')

    # $A$ Adiabaticity param
    A = 8 * getter_kwargs.get('eps_sl', def_eps_sl) * np.sqrt(1 - e_tot**2) / (
        3 * (1 + 4 * e_tot**2) * np.abs(np.sin(2 * I))
    ) / a[start_idx::stride]**4
    ax5.semilogy(ret.t[start_idx::stride], A, 'r')
    ax5.set_ylabel(r'$\mathcal{A}$')

    # e^2 + j^2 - 1
    ax6.plot(ret.t[start_idx::stride], e_tot**2 + j_tot**2 - 1, 'g')
    ax6.set_ylabel(r'$j^2 + e^2 - 1$')
    ax6.yaxis.set_label_position('right')

    for ax in [ax4, ax5, ax6]:
        ax.set_xlabel(r'$t / t_{LK}$')
        plt.setp(ax.get_xticklabels(), rotation=45)
    plt.savefig(fn, dpi=300)
    plt.close()

def djdt_lk(j, e, n2):
    return 3 / 4 * (
        np.dot(j, n2) * np.cross(j, n2)
            - 5 * np.dot(e, n2) * np.cross(e, n2))
def dldt_gw(e_sq, lhat):
    return -32 / 5 * (
        (1 + 7 * e_sq / 8) / (1 - e_sq)**2
    ) * lhat
def dadt_gw(e_sq):
    return -64 / 5 * (
        (1 + 73 / 24 * e_sq + 37 / 94 * e_sq**2) / (1 - e_sq)**(7/2))
def dedt_lk(j, e, n2):
    return 3 / 4 * (
        np.dot(j, n2) * np.cross(e, n2)
            - 5 * np.dot(e, n2) * np.cross(j, n2)
            + 2 * np.cross(j, e))
def dedt_gw(e, e_sq):
    return -(304 / 15) * (1 + 121 / 304 * e_sq) / (1 - e_sq)**(5/2) * e

def get_dydt_gr(n2, eps_gr=def_eps_gr, eps_sl=def_eps_sl, kozai=1):
    def dydt(t, x):
        j, e, s, a = to_vars(x)
        e_sq = np.sum(e**2)
        j_sq = np.sum(j**2)
        lhat = j / np.sqrt(j_sq)
        djdt = (
            (kozai / a**3) * djdt_lk(j, e, n2)
            + (eps_gr / a**4) * (
                dldt_gw(e_sq, lhat) - j / 2 * dadt_gw(e_sq)))
        dedt = (
            (kozai / a**3) * dedt_lk(j, e, n2)
            + (eps_gr / a**4) * dedt_gw(e, e_sq))
        dsdt = eps_sl / a * np.cross(lhat, s)
        dadt = eps_gr / a**3 * dadt_gw(e_sq)
        return np.concatenate((djdt, dedt, dsdt, [dadt]))
    return dydt

def get_dydt_nogr(n2, eps_sl=def_eps_sl):
    ''' n2 = normal axis of perturber, see eq 3/4 in notes '''
    def dydt(_, x):
        ''' no gr effects, try to see lk oscillations '''
        j, e, s, a = to_vars(x)
        lhat = j / np.sqrt(np.sum(j**2))
        return np.concatenate((
            djdt_lk(j, e, n2),
            dedt_lk(j, e, n2),
            eps_sl / a * np.cross(lhat, s),
            [0]))
    return dydt

def solver(I, e, tf=50, atol=1e-12, rtol=1e-12,
           getter=get_dydt_nogr,
           a_f=3e-1,
           getter_kwargs={},
           **kwargs):
    ''' n2 = (0, 0, 1) by convention, choose jy(t=0) = ey(t=0) = 0 '''
    jx = -np.sin(I) * np.sqrt(1 - e**2)
    jz = np.cos(I) * np.sqrt(1 - e**2)
    ex = np.cos(I) * e
    ez = np.sin(I) * e
    # s parallel to j
    sx = -np.sin(I)
    sz = np.cos(I)
    y0 = np.array([jx, 0, jz, ex, 0, ez, sx, 0, sz, 1])
    dydt = getter(np.array([0, 0, 1]), **getter_kwargs)

    term_event = lambda _, x: x[-1] - a_f
    term_event.terminal = True
    ret = solve_ivp(dydt, (0, tf), y0,
                    atol=atol, rtol=rtol, events=[term_event],
                    **kwargs)
    print('Done for %f %f, t_f = %f' % (I, e, ret.t[-1]))
    return ret
