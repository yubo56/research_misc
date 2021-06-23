'''
I don't know what's going on, so this is just a direct implementation of R08's
Equations 3.2 + GW radiation (with a bit of acceleration)
'''
from multiprocessing import Pool
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
        ## real GW
        # lvec) - 32/5 * mu * m12**2 / (c**5 * a**4) * lvec
        ## accelerated GW
        lvec) - 10 * 32/5 * mu * m12**2 / (c**5 * a**4) * lvec
    return ret

def run_example(fn='2plots/1example',
                q1=np.radians(30),
                phi1=np.radians(60),
                q2=np.radians(75),
                phi2=np.radians(200),
                m1=1,
                m2=1,
                s1mult=1,
                s2mult=1,
                only_ret=False,
                r_mult=1):
    m12 = m1 + m2
    mu = (m1 * m2) / m12
    v_over_c = 0.05
    a = k * (m12) / (v_over_c**2 * c**2) # AU

    lin_mag = mu * np.sqrt(k * m12 * a)
    s1mag = k * m1**2 / c * s1mult
    s2mag = k * m2**2 / c * s2mult

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

    r_sch = r_mult * k * m12 / c**2
    def term_event(t, y, m1, m2):
        ''' stop at around a ~ r_schwarzschild '''
        lvec = y[6:9]
        m12 = m1 + m2
        mu = (m1 * m2) / m12
        a_curr = np.sum(lvec**2) / (mu**2 * k * m12)
        return a_curr - r_sch
    term_event.terminal = True

    pkl_fn = fn + '.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        ret = solve_ivp(dydt, (0, 100), y0, args=[m1, m2], method='DOP853',
                        atol=1e-10, rtol=1e-10, events=[term_event])
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump((ret), f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    if only_ret:
        return ret

    s1vec = ret.y[ :3]
    s2vec = ret.y[3:6]
    lvec = ret.y[6:9]
    a = np.sum(lvec**2, axis=0) / (mu**2 * k * m12)
    jvec = s1vec + s2vec + lvec
    jvecmag = np.sqrt(np.sum(jvec**2, axis=0))
    phi1 = np.unwrap(np.arctan2(s1vec[1], s1vec[0]))
    phi2 = np.unwrap(np.arctan2(s2vec[1], s2vec[0]))
    stot = s1vec + s2vec

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2,
        figsize=(8, 8),
        sharex=True)

    ax1.semilogx(a / r_sch, np.degrees(phi1 - phi2) % 360)
    ax1.set_ylabel(r'$\Delta \phi$')
    # ax2.semilogx(a / r_sch, ts_dot_uv(s1vec, s2vec))
    # ax2.set_ylabel(r'$\hat{s}_1 \cdot \hat{s}_2$')
    ax2.semilogx(a / r_sch, ts_dot_uv(s1vec, stot))
    ax2.semilogx(a / r_sch, ts_dot_uv(s2vec, stot))

    # try to find an AI
    # this one is conserved
    # ax3.plot(a / r_sch, ts_dot_uv(jvec / jvecmag, s1vec + s2vec))
    # ax3.plot(a / r_sch, np.sum(stot**2, axis=0), alpha=0.8, lw=3.0)
    ax3.semilogy(a / r_sch, np.sum(lvec**2, axis=0), alpha=0.8, lw=3.0)
    ax3.semilogy(a / r_sch, np.sum(s1vec**2, axis=0), alpha=0.5)
    ax3.semilogy(a / r_sch, np.sum(s2vec**2, axis=0), alpha=0.5)
    # find another one?
    ax4.plot(a / r_sch, ts_dot_uv(s1vec - s2vec, s1vec + s2vec))

    ax3.set_xlabel(r'$a$ [AU]')
    ax4.set_xlabel(r'$a$ [AU]')
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axhline(0, c='k', lw=0.7, ls='--')
    plt.tight_layout()
    plt.savefig(fn, dpi=300)

def get_traj_angs(ret):
    s1vec = ret.y[ :3]
    s2vec = ret.y[3:6]
    phi1 = np.unwrap(np.arctan2(s1vec[1], s1vec[0]))
    phi2 = np.unwrap(np.arctan2(s2vec[1], s2vec[0]))
    dphibase = np.degrees(phi1 - phi2)
    dphi = dphibase % 360
    q1i = np.degrees(np.arccos(s1vec[2] / np.sqrt(np.sum(s1vec**2, axis=0)))[0])
    q2i = np.degrees(np.arccos(s2vec[2] / np.sqrt(np.sum(s2vec**2, axis=0)))[0])
    stotvechat = s1vec + s2vec
    stotvechat /= np.sqrt(np.sum(stotvechat**2, axis=0))

    dphiinterest = dphibase[int(len(dphibase) * 0.99): ]
    return (
        q1i, q2i, dphi[0], dphi[-1], np.degrees(np.arccos(stotvechat[2])[0]),
        dphiinterest.max() - dphiinterest.min()
    )

def pop(m1=1, m2=1, s1mult=1, s2mult=1, n_pts=1000, fn='2equal', r_mult=1):
    os.makedirs('2plots/{}'.format(fn), exist_ok=True)
    q1s = np.arccos(np.random.uniform(-1, 1, n_pts))
    phi1s = np.random.uniform(0, 2 * np.pi, n_pts)
    q2s = np.arccos(np.random.uniform(-1, 1, n_pts))
    phi2s = np.random.uniform(0, 2 * np.pi, n_pts)
    args = [
        ('2plots/{}/{}'.format(fn, idx), q1, phi1, q2, phi2,
         m1, m2, s1mult, s2mult, True, r_mult)
        for idx, (q1, phi1, q2, phi2) in enumerate(zip(q1s, phi1s, q2s, phi2s))]
    pkl_fn = fn + '.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        with Pool(4) as p:
            rets = p.starmap(run_example, args)
        angs = np.array([get_traj_angs(r) for r in rets]).T
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump((angs), f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            angs = pickle.load(f)
    q1is, q2is, dphi_i, dphi_f, qtotis, dphi_tot = angs
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 8),
        sharex=True)
    ax1.hist(dphi_i, bins=30)
    ax2.hist(dphi_f, bins=30)
    ax1.set_ylabel('Initial')
    ax2.set_ylabel('Final')
    ax2.set_xlabel(r'$\Delta \Phi$')
    plt.tight_layout()
    plt.savefig(fn + '_dphis', dpi=200)
    plt.close()

    plt.scatter(q1is, q2is, c=dphi_f)
    cb = plt.colorbar()
    cb.set_label(r'$\Delta \Phi_{\rm f}$')
    plt.xlabel(r'$\theta_{\rm s_1L}$')
    plt.ylabel(r'$\theta_{\rm s_2L}$')
    plt.tight_layout()
    plt.savefig(fn + '_qscat', dpi=200)
    plt.close()

    # version w/ histogram?
    # fig, (ax1, ax2) = plt.subplots(
    #     2, 1,
    #     figsize=(8, 8),
    #     sharex=True)
    # mu_totis = np.cos(np.radians(qtotis))
    # attractor_idxs = np.where(np.abs(dphi_f - 180) < 20)[0]
    # other_idxs = np.where(np.abs(dphi_f - 180) > 20)[0]
    # ax1.hist([mu_totis[attractor_idxs], mu_totis[other_idxs]], stacked=True, bins=50)
    # ax2.scatter(mu_totis, dphi_f, c='k')
    # ax2.set_xlabel(r'$\cos \theta_{\rm SL}$')
    # ax2.set_ylabel(r'$\Delta \Phi_{\rm f}$')
    # plt.scatter(mu_totis, dphi_f, c='k')
    # plt.tight_layout()
    # fig.subplots_adjust(hspace=0.02)
    # plt.savefig(fn + '_qtotscat', dpi=200)
    # plt.close()

    plt.scatter(qtotis, dphi_f, c='k')
    plt.xlabel(r'$\theta_{\rm SL}$')
    plt.ylabel(r'$\Delta \Phi_{\rm f}$')
    plt.tight_layout()
    plt.savefig(fn + '_qtotscat', dpi=200)
    plt.close()

    gtidx = np.where(dphi_tot > 730)[0]
    ltidx = np.where(dphi_tot < 730)[0]
    plt.scatter(qtotis[ltidx], dphi_tot[ltidx], c='k')
    plt.scatter(qtotis[gtidx], np.full_like(gtidx, 730), c='r', marker='^')
    plt.xlabel(r'$\theta_{\rm SL}$')
    plt.ylabel(r'$\Delta \Phi_{\max} - \Delta \Phi_{\min}$')
    plt.ylim(top=735)
    plt.tight_layout()
    plt.savefig(fn + '_deltaphi', dpi=200)
    plt.close()

if __name__ == '__main__':
    pass
    # run_example()
    # run_example(fn='2plots/1example_aligned',
    #             q1=np.radians(5), phi1=np.radians(10),
    #             q2=np.radians(10), phi2=np.radians(20))
    # run_example(fn='2plots/1example_antialigned',
    #             q1=np.radians(5), phi1=np.radians(10),
    #             q2=np.radians(160), phi2=np.radians(200))
    # run_example(fn='2plots/1_unequal', m2=0.5)
    # run_example(s1mult=0.01, s2mult=0.01,
    #             fn='2plots/1example_lowspin')
    # run_example(fn='2plots/1example_antialigned_lowspin',
    #             q1=np.radians(5), phi1=np.radians(10),
    #             q2=np.radians(160), phi2=np.radians(200),
    #             s1mult=0.01, s2mult=0.01)

    pop()
    pop(m2=0.5, fn='2half')
    pop(s1mult=0.05, s2mult=0.05, fn='2lowspin')
    pop(r_mult=0.01, fn='2limit')
