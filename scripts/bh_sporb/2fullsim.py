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

def dydt_schnitt(t, y, m1, m2):
    '''
    Schnittman 2.10, e = 0
    '''
    s1 = y[ :3]
    s2 = y[3:6]
    lvec = y[6:9]
    lvec_sq = np.sum(lvec**2)

    m12 = m1 + m2
    mu = (m1 * m2) / m12
    a = lvec_sq / (mu**2 * k * m12)

    prefix = k**2 * m12**2 / (c**3 * a**3)
    W1 = prefix * (
        (2 + 3 * m2 / m1 - 3/2 * np.dot(s2, lvec) / lvec_sq) * lvec
        + s2 / 2)
    W2 = prefix * (
        (2 + 3 * m1 / m2 - 3/2 * np.dot(s1, lvec) / lvec_sq) * lvec
        + s1 / 2)
    ret = np.zeros_like(y)
    ret[ :3] = np.cross(W1, s1)
    ret[3:6] = np.cross(W2, s2)
    ret[6:9] = (
        - (np.cross(W1, s1) + np.cross(W2, s2))
        ## real GW
        # - 32/5 * mu * m12**2 / (c**5 * a**4) * lvec
        ## accelerated GW
         - 10 * 32/5 * mu * m12**2 / (c**5 * a**4) * lvec
    )
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
                r_mult=1,
                dydt_func=dydt):
    m12 = m1 + m2
    mu = (m1 * m2) / m12
    # v_over_c = 0.05
    v_over_c = 0.1
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
        ret = solve_ivp(dydt_func, (0, 100), y0, args=[m1, m2], method='DOP853',
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

    s1hat = s1vec / np.sqrt(np.sum(s1vec**2, axis=0))
    s2hat = s2vec / np.sqrt(np.sum(s2vec**2, axis=0))
    lhat = lvec / np.sqrt(np.sum(lvec**2, axis=0))
    s1perp_hat = s1hat - ts_dot(lhat, s1hat) * lhat
    s1perp_hat /= np.sqrt(np.sum(s1perp_hat**2, axis=0))
    s2perp_hat = s2hat - ts_dot(lhat, s2hat) * lhat
    s2perp_hat /= np.sqrt(np.sum(s2perp_hat**2, axis=0))
    s2perp_x = ts_dot(s2perp_hat, s1perp_hat)
    s2perp_y_vec = s2perp_hat - s2perp_x * s1perp_hat
    s2perp_y = np.sqrt(np.sum(s2perp_y_vec**2, axis=0))
    dphibase = np.degrees(np.arctan2(s2perp_y, s2perp_x))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2,
        figsize=(8, 8),
        sharex=True)

    ax1.semilogx(a / r_sch, np.degrees(dphibase) % 360)
    ax1.set_ylabel(r'$\Delta \phi$')
    # ax2.semilogx(a / r_sch, ts_dot_uv(s1vec, s2vec))
    # ax2.set_ylabel(r'$\hat{s}_1 \cdot \hat{s}_2$')
    ax2.semilogx(a / r_sch, np.degrees(np.arccos(ts_dot_uv(s1vec, stot))),
                 label=r'$\theta_{1S}$', alpha=0.7)
    ax2.semilogx(a / r_sch, np.degrees(np.arccos(ts_dot_uv(s2vec, stot))),
                 label=r'$\theta_{2S}$', alpha=0.7)
    ax2.legend(fontsize=12)

    shat_z = stot[2] / np.sqrt(np.sum(stot**2, axis=0))
    ax3.semilogx(a / r_sch, np.degrees(np.arccos(shat_z)),
                 label=r'$\theta_{Sz}$', alpha=0.7)
    lhat_z = lvec[2] / np.sqrt(np.sum(lvec**2, axis=0))
    ax3.semilogx(a / r_sch, np.degrees(np.arccos(lhat_z)),
                 label=r'$\theta_{lz}$', alpha=0.7)
    ax3.legend(fontsize=12)

    # AI: L dot s
    ax4.semilogx(a / r_sch, np.degrees(np.arccos(ts_dot_uv(stot, lvec))),
                 label=r'$\theta_{sl}$', alpha=0.7)
    ax4.semilogx(a / r_sch, np.degrees(np.arccos(ts_dot_uv(s1vec, s2vec))),
                 label=r'$\theta_{12}$', alpha=0.7)
    ax4.legend(fontsize=12)

    # try to find an AI
    # this one is conserved
    # ax3.plot(a / r_sch, ts_dot_uv(jvec / jvecmag, s1vec + s2vec))
    # ax3.plot(a / r_sch, np.sum(stot**2, axis=0), alpha=0.8, lw=3.0)
    # ax3.semilogy(a / r_sch, np.sum(lvec**2, axis=0), alpha=0.8, lw=3.0)
    # ax3.semilogy(a / r_sch, np.sum(s1vec**2, axis=0), alpha=0.5)
    # ax3.semilogy(a / r_sch, np.sum(s2vec**2, axis=0), alpha=0.5)
    # find another one?
    # ax4.plot(a / r_sch, ts_dot_uv(s1vec - s2vec, s1vec + s2vec))

    ax3.set_xlabel(r'$a$ [$m_{12}$]')
    ax4.set_xlabel(r'$a$ [$m_{12}$]')
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axhline(0, c='k', lw=0.7, ls='--')
    plt.tight_layout()
    plt.savefig(fn, dpi=300)
    plt.close()

    # def plot_last(vec, c):
    #     vec_hat = vec / np.sqrt(np.sum(vec**2, axis=0))
    #     plt.plot([0, vec_hat[0, -1]], [0, vec_hat[2, -1]], c)
    # plot_last(s1vec, 'g')
    # plot_last(s2vec, 'c')
    # # plot_last(stot, 'k')
    # # plot_last(lvec, 'r')
    # plt.savefig('/tmp/' + fn, dpi=300)
    # plt.close()

def get_traj_angs(y):
    s1vec = y[ :3]
    s2vec = y[3:6]
    lvec = y[6:9]

    s1hat = s1vec / np.sqrt(np.sum(s1vec**2, axis=0))
    s2hat = s2vec / np.sqrt(np.sum(s2vec**2, axis=0))
    lhat = lvec / np.sqrt(np.sum(lvec**2, axis=0))
    s1perp_hat = s1hat - ts_dot(lhat, s1hat) * lhat
    s1perp_hat /= np.sqrt(np.sum(s1perp_hat**2, axis=0))
    s2perp_hat = s2hat - ts_dot(lhat, s2hat) * lhat
    s2perp_hat /= np.sqrt(np.sum(s2perp_hat**2, axis=0))
    s2perp_x = ts_dot(s2perp_hat, s1perp_hat)
    s2perp_y_vec = s2perp_hat - s2perp_x * s1perp_hat
    s2perp_y = np.sqrt(np.sum(s2perp_y_vec**2, axis=0))
    dphibase = np.degrees(np.arctan2(s2perp_y, s2perp_x))

    # phi1 = np.unwrap(np.arctan2(s1vec[1], s1vec[0]))
    # phi2 = np.unwrap(np.arctan2(s2vec[1], s2vec[0]))
    # dphibase = np.degrees(phi1 - phi2)

    dphi = dphibase % 360
    q1i = np.degrees(np.arccos(s1vec[2] / np.sqrt(np.sum(s1vec**2, axis=0)))[0])
    q2i = np.degrees(np.arccos(s2vec[2] / np.sqrt(np.sum(s2vec**2, axis=0)))[0])
    stotvec = s1vec + s2vec
    stotvechat = stotvec / np.sqrt(np.sum(stotvec**2, axis=0))

    jvec = lvec + stotvec
    qsj = np.degrees(np.arccos(ts_dot_uv(jvec, stotvec)))

    q12 = np.degrees(np.arccos(ts_dot(s1hat, s2hat)))

    dphiinterest = dphibase[int(len(dphibase) * 0.99): ]
    return (
        q1i, q2i, dphi[0], dphi[-1], np.degrees(np.arccos(stotvechat[2])[0]),
        dphiinterest.max() - dphiinterest.min(), qsj[0], qsj[-1], q12[0],
        q12[-1]
    )

def pop(m1=1, m2=1, s1mult=1, s2mult=1, n_pts=1000, fn='2equal', r_mult=1,
        q1=None, dydt_func=dydt, nthreads=4):
    os.makedirs('2plots/{}'.format(fn), exist_ok=True)
    phi1s = np.random.uniform(0, 2 * np.pi, n_pts)
    q2s = np.arccos(np.random.uniform(-1, 1, n_pts))
    phi2s = np.random.uniform(0, 2 * np.pi, n_pts)
    if q1 is None:
        q1s = np.arccos(np.random.uniform(-1, 1, n_pts))
    else:
        q1s = np.full_like(q2s, q1)
    args = [
        ('2plots/{}/{}'.format(fn, idx), q1, phi1, q2, phi2,
         m1, m2, s1mult, s2mult, True, r_mult, dydt_func)
        for idx, (q1, phi1, q2, phi2) in enumerate(zip(q1s, phi1s, q2s, phi2s))]
    pkl_fn = fn + '.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        with Pool(nthreads) as p:
            rets = p.starmap(run_example, args)
        angs = np.array([get_traj_angs(r.y) for r in rets]).T
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump((angs), f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            angs = pickle.load(f)
    q1is, q2is, dphi_i, dphi_f, qtotis, dphi_tot, qsji, qsjf, q12i, q12f = angs

    # recover theta_{1S}, theta_{2S}
    s1vec_i = m1**2 * s1mult * np.array([
        np.sin(q1s) * np.cos(phi1s),
        np.sin(q1s) * np.sin(phi1s),
        np.cos(q1s),
    ])
    s2vec_i = m2**2 * s2mult * np.array([
        np.sin(q2s) * np.cos(phi2s),
        np.sin(q2s) * np.sin(phi2s),
        np.cos(q2s),
    ])
    stot_i = s1vec_i + s2vec_i
    q1_i = np.degrees(np.arccos(ts_dot_uv(s1vec_i, stot_i)))
    q2_i = np.degrees(np.arccos(ts_dot_uv(s2vec_i, stot_i)))
    qsj_i = np.degrees(np.arccos(
        stot_i[2] / np.sqrt(np.sum(stot_i**2, axis=0)))) # approx

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

    plt.scatter(q1_i, qsjf, c=dphi_f)
    cb = plt.colorbar()
    cb.set_label(r'$\Delta \Phi_{\rm f}$')
    plt.xlabel(r'$\theta_{\rm s_1S}$')
    plt.ylabel(r'$\theta_{\rm SJ, f}$')
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

    lt_idx = np.where(np.logical_and(
        q1_i < qsj_i,
        q2_i < qsj_i,
    ))
    gt_idx = np.where(np.logical_and(
        q1_i > qsj_i,
        q2_i > qsj_i,
    ))
    plt.scatter(qsjf, dphi_f, c='k', s=4) # overwrite these w/ lt & gt
    plt.scatter(qsjf[lt_idx], dphi_f[lt_idx], c='g', s=4)
    plt.scatter(qsjf[gt_idx], dphi_f[gt_idx], c='b', s=4)
    plt.xlabel(r'$\theta_{\rm SJ, f}$')
    plt.ylabel(r'$\Delta \Phi_{\rm f}$')
    plt.tight_layout()
    plt.savefig(fn + '_qtotscat', dpi=200)
    plt.close()

    plt.scatter(dphi_i, np.cos(np.radians(q12i)), c='b')
    plt.scatter(dphi_f, np.cos(np.radians(q12f)), c='r')
    plt.xlabel(r'$\Delta \Phi$')
    plt.ylabel(r'$\cos \theta_{12}$')
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
    # run_example(q1=np.radians(5), phi1=0, q2=np.radians(170), phi2=0,
    #             fn='2plots/1example_90deg')
    # run_example(q1=np.radians(45), phi1=0, q2=np.radians(120), phi2=0,
    #             fn='2plots/1example_90deg2')
    # run_example(fn='2plots/1_unequal_schnitt', m2=0.5, dydt_func=dydt_schnitt)
    # run_example(dydt_func=dydt_schnitt, fn='2plots/1example_schnitt')
    # run_example(fn='2plots/1_55_schnitt', m1=0.55, m2=0.45,
    #             dydt_func=dydt_schnitt, q1=1e-3)
    # run_example(fn='2plots/1_55', m1=0.55, m2=0.45, q1=1e-3)

    pop()
    pop(m2=0.5, fn='2half')
    pop(s1mult=0.05, s2mult=0.05, fn='2lowspin')
    pop(r_mult=0.01, fn='2limit')
    pop(n_pts=200, fn='2_realgw')

    pop(m1=0.55, m2=0.45, fn='2half_schnitt', dydt_func=dydt_schnitt, n_pts=400,
        nthreads=8, q1=np.radians(10))
    pop(m1=0.55, m2=0.45, fn='2half_55', n_pts=400, nthreads=8,
        q1=np.radians(10))
    pop(m1=0.5, m2=0.5, fn='2_equal_5', nthreads=8, q1=np.radians(10), n_pts=400)
    pop(m1=0.55, m2=0.45, fn='2half_55_retro', n_pts=400, nthreads=8,
        q1=np.radians(170))
