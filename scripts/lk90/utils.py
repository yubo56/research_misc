'''
Notes:
* x = (vec{j}, vec{e}, vec{s}, a)
* t_{lk, 0} = a0 = 1
'''
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def to_vars(x):
    return x[ :3], x[3:6], x[6:9], x[9]

def plot_traj(ret, fn, num_pts=1000):
    j, e, s, a = to_vars(ret.y)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                                                 figsize=(8, 8),
                                                 sharex=True)

    stride = len(ret.t) // num_pts
    e_tot = np.sqrt(np.sum(e[:, ::stride]**2, axis=0))
    j_tot = np.sqrt(1 - e_tot**2)
    l = j[:, ::stride] / j_tot

    # plot 1 - e(t) in top left plot
    ax1.semilogy(ret.t[::stride], 1 - e_tot, 'r')
    ax1.set_ylabel(r'$1 - e$')

    # plot I(t) in top right plot
    I = np.arccos(j[2, ::stride] / j_tot)
    ax2.plot(ret.t[::stride], np.degrees(I), 'r')
    ax2.set_ylabel(r'$I$ (deg)')
    ax2.yaxis.set_label_position('right')

    # plot a(t) in the bottom left plot
    ax3.semilogy(ret.t[::stride], a[::stride], 'r')
    ax3.set_ylabel(r'$a / a_0$')

    # plot q_sl in the bottom right plot
    q_sl = np.arccos(np.tensordot(l, s[:, ::stride], axes=[[0], [0]]))
    ax4.plot(ret.t[::stride], np.degrees(q_sl), 'r')
    ax4.set_ylabel(r'$\theta_{\rm sl}$')
    ax4.yaxis.set_label_position('right')

    ax3.set_xlabel(r'$t / t_{LK}$')
    ax4.set_xlabel(r'$t / t_{LK}$')
    plt.savefig(fn, dpi=300)
    plt.close()

def get_dydt_gr(n2, eps=0.1, delta=0.1):
    def dydt(_, x):
        j, e, s, a = to_vars(x)
        e_sq = np.sum(e**2)
        djdt = (
            3 / (4 * a**3) * (
                np.dot(j, n2) * np.cross(j, n2)
                    - 5 * np.dot(e, n2) * np.cross(e, n2))
            - (32 / 5) * (eps / a**4) * (
                (1 + 7 * e_sq / 8) / (1 - e_sq)**2
                    - (1 + 73 / 24 * e_sq + 37 / 94 * e_sq**2)
                        / (1 - e_sq)**(7/2)
            ) * j / np.sqrt(1 - e_sq))
        dedt = (
            3 / (4 * a**3) * (
                np.dot(j, n2) * np.cross(e, n2)
                    - 5 * np.dot(e, n2) * np.cross(j, n2)
                    + 2 * np.cross(j, e))
            - (304 / 15) * (eps / a**4)
                * (1 + 121 / 304 * e_sq) / (1 - e_sq)**(5/2) * e)
        dsdt = delta / a * np.cross(j, s) / np.sqrt(1 - e_sq)
        dadt = -eps / a**3 * 64 / 5 * (
            (1 + 73 / 24 * e_sq + 37 / 94 * e_sq**2) / (1 - e_sq)**(7/2))
        return np.concatenate((djdt, dedt, dsdt, [dadt]))
    return dydt

def get_dydt_nogr(n2):
    ''' n2 = normal axis of perturber, see eq 3/4 in notes '''
    def dydt(_, x):
        ''' no gr effects, try to see lk oscillations '''
        j, e, _, _ = to_vars(x)
        return 3 / 4 * np.concatenate((
            np.dot(j, n2) * np.cross(j, n2)
                - 5 * np.dot(e, n2) * np.cross(e, n2),
            np.dot(j, n2) * np.cross(e, n2)
                - 5 * np.dot(e, n2) * np.cross(j, n2)
                + 2 * np.cross(j, e),
            [0, 0, 0],
            [0]))
    return dydt

def solver(I, e, tf=50, atol=1e-12, rtol=1e-12,
           getter=get_dydt_nogr,
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
    ret = solve_ivp(dydt, (0, tf), y0, atol=atol, rtol=rtol, **kwargs)
    print('Done for %f %f %f' % (I, e, tf))
    return ret
