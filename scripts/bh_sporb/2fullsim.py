'''
I don't know what's going on, so this is just a direct implementation of R08's
Equations 3.2 + GW radiation (with a bit of acceleration)
'''
import os, pickle, lzma
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
from scipy.integrate import solve_ivp
from utils import *

# units: Msun, AU, yr
# >>> G * Msun / (AU)^3 * yr^2
# 39.483019
k = 39.4751488
# >>> G * Msun / (AU)^3 * yr^2
# 39.483019
c = 6.32397263*10**4

def dydt(t, y, m1, m2):
    '''
    Racine (3.2a-c) assuming e = 0
    '''
    s1 = y[ :3]
    s2 = y[3:6]
    lvec = y[6:9]
    stot = s1 + s2
    s0 = (1 + m2 / m1) * s1 + (1 + m1 / m2) * s2

    m12 = m1 + m2
    mu = (m1 * m2) / m12
    a = np.sum(lvec**2) / (mu**2 * k * m12)

    ret = np.zeros_like(y)
    # TODO what's the actual mass dependence here?
    l_const = np.dot(lvec, s0) / np.sum(lvec**2)
    prefix = 1/2 * k**2 * m12**2 / (c**3 * a**3)
    ret[ :3] = prefix * np.cross(
        (4 + 3 * m2 / m1 - 3 * mu / m1 * l_const) * lvec
        + s2,
        s1)
    ret[3:6] = prefix * np.cross(
        (4 + 3 * m1 / m2 - 3 * mu / m2 * l_const) * lvec
        + s1,
        s2)
    ret[6:9] = prefix * np.cross(
        stot + 3 * (1 + mu / m12 * l_const) * s0,
        # lvec) - 32/5 * mu * m12**2 / (c**5 * a**4) * lvec
        lvec) - 10 * 32/5 * mu * m12**2 / (c**5 * a**4) * lvec
    return ret

def run_example(fn='2plots/1example',
                q1=np.radians(30),
                phi1=np.radians(60),
                q2=np.radians(75),
                phi2=np.radians(200)):
    m1 = 1
    m2 = 1
    m12 = m1 + m2
    mu = (m1 * m2) / m12
    v_over_c = 0.05
    a = k * (m12) / (v_over_c**2 * c**2) # AU

    lin_mag = mu * np.sqrt(k * m12 * a)
    s1mag = k * m1**2 / c
    s2mag = k * m2**2 / c

    s1vec0 = s1mag * np.array([
        np.sin(q1) * np.cos(phi1),
        np.sin(q1) * np.sin(phi1),
        np.cos(q1),
    ])
    s2vec0 = s2mag * np.array([
        np.sin(q2) * np.cos(phi2),
        np.sin(q2) * np.sin(phi2),
        np.cos(q2),
    ])
    lvec0 = lin_mag * np.array([0, 0, 1])
    y0 = np.concatenate((s1vec0, s2vec0, lvec0))
    def term_event(t, y, m1, m2):
        lvec = y[6:9]
        m12 = m1 + m2
        mu = (m1 * m2) / m12
        a_curr = np.sum(lvec**2) / (mu**2 * k * m12)
        return a_curr / a - 0.001
    term_event.terminal = True

    pkl_fn = fn + '.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        ret = solve_ivp(dydt, (0, 100), y0, args=[m1, m2], method='DOP853',
                        atol=1e-8, rtol=1e-8, events=[term_event])
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump((ret), f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)

    s1vec = ret.y[ :3]
    s2vec = ret.y[3:6]
    lvec = ret.y[6:9]
    a = np.sum(lvec**2, axis=0) / (mu**2 * k * m12)
    jvec = s1vec + s2vec + lvec
    jvecmag = np.sqrt(np.sum(jvec**2, axis=0))
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2,
        figsize=(8, 8),
        sharex=True)
    phi1 = np.unwrap(np.arctan2(s1vec[1], s1vec[0]))
    phi2 = np.unwrap(np.arctan2(s2vec[1], s2vec[0]))

    ax1.semilogx(a, np.degrees(phi1 - phi2) % 360)
    ax1.set_ylabel(r'$\Delta \phi$')
    ax2.semilogx(a, ts_dot(s1vec, s2vec) /
             np.sqrt(np.sum(s1vec**2, axis=0) * np.sum(s2vec**2, axis=0)))
    ax2.set_ylabel(r'$\hat{s}_1 \cdot \hat{s}_2$')

    # try to find an AI
    # this one is conserved
    ax3.plot(a, ts_dot(jvec / jvecmag, s1vec + s2vec))
    # find another one?
    ax4.plot(a, ts_dot(s1vec - s2vec, s1vec + s2vec))

    ax3.set_xlabel(r'$a$ [AU]')
    ax4.set_xlabel(r'$a$ [AU]')
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axhline(0, c='k', lw=0.7, ls='--')
    plt.tight_layout()
    plt.savefig(fn, dpi=300)

if __name__ == '__main__':
    run_example()
    run_example(fn='2plots/1example_aligned',
                q1=np.radians(5), phi1=np.radians(10),
                q2=np.radians(10), phi2=np.radians(20))
    run_example(fn='2plots/1example_antialigned',
                q1=np.radians(5), phi1=np.radians(10),
                q2=np.radians(160), phi2=np.radians(200))
