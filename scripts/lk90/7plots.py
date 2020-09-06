# convert 7_3vec.png -crop 1100x1300+350+100 7_3vec_cropped.png
# convert 7_3vec_eta.png -crop 1550x1300+50+100 7_3vec_eta_cropped.png
'''
more random plots
'''
import numpy as np
import scipy.optimize as opt
from utils import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
plt.rc('text.latex', preamble=r'\usepackage{newtxmath}')
LW=3.5

def get_xy(angle, mag=1):
    return mag * np.sin(np.radians(angle)), mag * np.cos(np.radians(angle))

def plot_3vec():
    ''' plots the relative orientations of the vectors '''
    offset = 0.02 # offset for text from arrow tip
    alpha = 0.8

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.axis('off')
    ax.set_xlim(-0.1, 0.8)
    ax.set_ylim(-0.1, 1.1)

    # central dot
    ax.plot(0, 0, 'ko', ms=8, zorder=np.inf)
    arrowprops = lambda c: {'fc': c, 'alpha': alpha, 'lw': 0,
                            'width': 3, 'headwidth': 12}

    # draw arrows
    l_xy = get_xy(0)
    l_c = 'k'
    ax.annotate('', xy=l_xy, xytext=(0, 0),
                 arrowprops=arrowprops(l_c))
    ax.text(l_xy[0] - offset / 3, l_xy[1] + offset,
            r'$\hat{\mathbf{L}}_{\rm out}$',
            fontdict={'c': l_c})

    ld_q = 20
    ld_xy = get_xy(ld_q)
    ld_c = 'r'
    ax.annotate('', xy=ld_xy, xytext=(0, 0),
                 arrowprops=arrowprops(ld_c))
    ax.text(ld_xy[0] - offset / 2, ld_xy[1] + offset,
            r'$\overline{\mathbf{\Omega}}_{\rm e}$',
            fontdict={'c': ld_c})
    md_q = 30
    md_xy = get_xy(md_q)
    md_c = 'g'
    ax.annotate('', xy=md_xy, xytext=(0, 0),
                 arrowprops=arrowprops(md_c))
    ax.text(md_xy[0] - offset / 2, md_xy[1] + offset,
            r'$\mathbf{\Omega}_{\rm e1}$',
            fontdict={'c': md_c})

    s_q = 50
    s_xy = get_xy(s_q)
    s_c = 'b'
    ax.annotate('', xy=s_xy, xytext=(0, 0),
                 arrowprops=arrowprops(s_c))
    ax.text(s_xy[0] - offset, s_xy[1] + offset,
            r'$\hat{\overline{\mathbf{L}}}$',
            fontdict={'c': s_c})

    # draw arcs
    arc_lw = 3
    ld_arc = patches.Arc((0, 0), 1.1, 1.1, 0, 90 - ld_q, 90,
                         color=ld_c, lw=arc_lw, alpha=alpha,
                         label='1')
    ax.add_patch(ld_arc)

    ax.text(np.sin(np.radians(ld_q * 0.4)) * 0.55,
            np.cos(np.radians(ld_q * 0.4)) * 0.55 + 2 * offset,
            r'$\bar{I}_{\rm e}$',
            fontdict={'c': ld_c})
    xy_ld_tip = (
        0.55 * np.sin(np.radians(ld_q)),
        0.55 * np.cos(np.radians(ld_q)))
    ax.annotate('', xy=xy_ld_tip + np.array([0.003, -0.001]), xytext=xy_ld_tip,
                arrowprops=arrowprops(ld_c))

    md_arc = patches.Arc((0, 0), 1.4, 1.4, 0, 90 - md_q, 90,
                         color=md_c, lw=arc_lw, alpha=alpha,
                         label='1')
    ax.add_patch(md_arc)

    ax.text(np.sin(np.radians(md_q * 0.3)) * 0.7,
            np.cos(np.radians(md_q * 0.3)) * 0.7 + 2 * offset,
            r'$I_{\rm e1}$',
            fontdict={'c': md_c})
    xy_md_tip = (
        0.7 * np.sin(np.radians(md_q)),
        0.7 * np.cos(np.radians(md_q)))
    ax.annotate('', xy=xy_md_tip + np.array([0.003, -0.001]), xytext=xy_md_tip,
                arrowprops=arrowprops(md_c))

    s_arc = patches.Arc((0, 0), 0.8, 0.8, 0, 90 - s_q, 90,
                        color=s_c, lw=arc_lw, alpha=alpha)
    ax.add_patch(s_arc)
    ax.text(np.sin(np.radians(0.8 * s_q)) * 0.4 + offset,
            np.cos(np.radians(0.8 * s_q)) * 0.4 + 2 * offset,
            r'$\bar{I}$',
            fontdict={'c': s_c})
    xy_s_tip = (
        0.4 * np.sin(np.radians(s_q)),
        0.4 * np.cos(np.radians(s_q)))
    ax.annotate('', xy=xy_s_tip + np.array([0.003, -0.003]), xytext=xy_s_tip,
                arrowprops=arrowprops(s_c))

    # coord axis labels
    coord_xy = np.array([0.6, 0.05])
    ax.plot(*coord_xy, 'ko', ms=8, zorder=np.inf)
    ax.annotate('', xy=(coord_xy + 0.2 * np.array(ld_xy)),
                xytext=coord_xy,
                arrowprops=arrowprops('k'))
    ax.text(*(coord_xy + np.array([0, 0.01]) + 0.2 * np.array(ld_xy)),
            r'$\hat{\mathbf{z}}$',
            fontdict={'c': 'k'})
    xvec = np.array(ld_xy)[::-1]
    xvec[1] *= -1
    ax.annotate('', xy=(coord_xy + 0.2 * xvec),
                xytext=coord_xy,
                arrowprops=arrowprops('k'))
    ax.text(*(coord_xy + np.array([0.01, 0]) + 0.2 * xvec),
            r'$\hat{\mathbf{x}}$',
            fontdict={'c': 'k'})

    ax.set_aspect('equal')
    plt.savefig('7_3vec', dpi=400)
    plt.clf()

def plot_3vec_eta():
    ''' plots the relative orientations of the three vectors '''
    offset = 0.02 # offset for text from arrow tip
    alpha = 0.8

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.axis('off')
    ax.set_xlim(-0.3, 0.8)
    ax.set_ylim(-0.1, 1.1)

    # central dot
    ax.plot(0, 0, 'ko', ms=8, zorder=np.inf)
    arrowprops = lambda c: {'fc': c, 'alpha': alpha, 'lw': 0,
                            'width': 3, 'headwidth': 12}

    # draw arrows
    l_xy = get_xy(0)
    l_c = 'k'
    ax.annotate('', xy=l_xy, xytext=(0, 0),
                 arrowprops=arrowprops(l_c))
    ax.text(l_xy[0] - offset / 3, l_xy[1] + offset,
            r'$\hat{\mathbf{L}}_{\rm tot}$',
            fontdict={'c': l_c})

    ld_q = 20
    ld_xy = get_xy(ld_q)
    ld_c = 'r'
    ax.annotate('', xy=ld_xy, xytext=(0, 0),
                 arrowprops=arrowprops(ld_c))
    ax.text(ld_xy[0] - offset / 2, ld_xy[1] + offset,
            r'$\overline{\mathbf{\Omega}}_{\rm e}$',
            fontdict={'c': ld_c})
    md_q = -15
    md_xy = get_xy(md_q)
    md_c = 'g'
    ax.annotate('', xy=md_xy, xytext=(0, 0),
                 arrowprops=arrowprops(md_c))
    ax.text(md_xy[0] - offset / 2, md_xy[1] + offset,
            r'$\hat{\overline{\mathbf{L}}}_{\rm out}$',
            fontdict={'c': md_c})

    s_q = 50
    s_xy = get_xy(s_q)
    s_c = 'b'
    ax.annotate('', xy=s_xy, xytext=(0, 0),
                 arrowprops=arrowprops(s_c))
    ax.text(s_xy[0] - offset, s_xy[1] + offset,
            r'$\hat{\overline{\mathbf{L}}}$',
            fontdict={'c': s_c})

    # draw arcs
    arc_lw = 3
    ld_arc = patches.Arc((0, 0), 1.3, 1.3, 0, 90 - ld_q, 90,
                         color=ld_c, lw=arc_lw, alpha=alpha,
                         label='1')
    ax.add_patch(ld_arc)

    ax.text(np.sin(np.radians(ld_q * 0.3)) * 0.65 - 2.5 * offset,
            np.cos(np.radians(ld_q * 0.3)) * 0.65 + 2 * offset,
            r'$\bar{I}_{\rm e-tot}$',
            fontdict={'c': ld_c})
    xy_ld_tip = (
        0.65 * np.sin(np.radians(ld_q)),
        0.65 * np.cos(np.radians(ld_q)))
    ax.annotate('', xy=xy_ld_tip + np.array([0.003, -0.001]), xytext=xy_ld_tip,
                arrowprops=arrowprops(ld_c))

    md_arc = patches.Arc((0, 0), 1.8, 1.8, 0, 90 - s_q, 90 - md_q,
                         color='k', lw=arc_lw, alpha=alpha,
                         label='1')
    ax.add_patch(md_arc)

    ax.text(np.sin(np.radians(s_q * 0.9)) * 0.9 - offset,
            np.cos(np.radians(s_q * 0.9)) * 0.9 + 3 * offset,
            r'$\bar{I}$',
            fontdict={'c': 'k'})
    xy_md_tip = (
        0.9 * np.sin(np.radians(s_q)),
        0.9 * np.cos(np.radians(s_q)))
    ax.annotate('', xy=xy_md_tip + np.array([0.003, -0.003]), xytext=xy_md_tip,
                arrowprops=arrowprops('k'))

    nd_arc = patches.Arc((0, 0), 1.3, 1.3, 0, 90, 90 - md_q,
                         color=md_c, lw=arc_lw, alpha=alpha,
                         label='1')
    ax.add_patch(nd_arc)

    ax.text(np.sin(np.radians(md_q * 2.5)) * 0.65 - 2 * offset,
            np.cos(np.radians(md_q * 2.5)) * 0.65 + 3 * offset,
            r'$\bar{I}_{\rm tot, out}$',
            fontdict={'c': md_c})
    xy_md_tip = (
        0.65 * np.sin(np.radians(md_q)),
        0.65 * np.cos(np.radians(md_q)))
    ax.annotate('', xy=xy_md_tip + np.array([-0.005, -0.001]), xytext=xy_md_tip,
                arrowprops=arrowprops(md_c))

    s_arc = patches.Arc((0, 0), 0.8, 0.8, 0, 90 - s_q, 90,
                        color=s_c, lw=arc_lw, alpha=alpha)
    ax.add_patch(s_arc)
    ax.text(np.sin(np.radians(0.8 * s_q)) * 0.4 - offset,
            np.cos(np.radians(0.8 * s_q)) * 0.4 + 3 * offset,
            r'$\bar{I}_{\rm tot}$',
            fontdict={'c': s_c})
    xy_s_tip = (
        0.4 * np.sin(np.radians(s_q)),
        0.4 * np.cos(np.radians(s_q)))
    ax.annotate('', xy=xy_s_tip + np.array([0.003, -0.003]), xytext=xy_s_tip,
                arrowprops=arrowprops(s_c))

    # coord axis labels
    coord_xy = np.array([0.6, 0.05])
    ax.plot(*coord_xy, 'ko', ms=8, zorder=np.inf)
    ax.annotate('', xy=(coord_xy + 0.2 * np.array(l_xy)),
                xytext=coord_xy,
                arrowprops=arrowprops('k'))
    ax.text(*(coord_xy + np.array([0, 0.01]) + 0.2 * np.array(l_xy)),
            r'$\hat{\mathbf{Z}}$',
            fontdict={'c': 'k'})
    xvec = np.array(l_xy)[::-1]
    xvec[1] *= -1
    ax.annotate('', xy=(coord_xy + 0.2 * xvec),
                xytext=coord_xy,
                arrowprops=arrowprops('k'))
    ax.text(*(coord_xy + np.array([0.01, 0]) + 0.2 * xvec),
            r'$\hat{\mathbf{X}}$',
            fontdict={'c': 'k'})

    ax.set_aspect('equal')
    plt.savefig('7_3vec_eta', dpi=400)
    plt.clf()

def plot_bin_bifurcations():
    fns = ['M1_M2_thetasl_thetae_70.txt',
           'M1_M2_thetasl_thetae_88.txt']
    for I0, fn in zip(np.radians([70, 88]), fns):
        with open(fn) as f:
            dat = []
            for l in f.readlines():
                m1, m2, qsl, qe = [float(w) for w in l.split(' ')]
                dat.append((m1 / (m1 + m2), qsl, qe))
        ratios, qsls, qes = np.array(dat).T
        qes = np.degrees(qes)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        for ax in ax1, ax2:
            ax.set_yticks([0, 60, 120, 180])
            ax.set_yticklabels([r'$0$', r'$60$', r'$120$', r'$180$'])
        ax1.plot(ratios, qsls, 'k,')
        ax2.plot(ratios, qes, 'k,')
        ax1.set_ylabel(r'$\theta_{\rm sl}$ (Deg)')
        ax2.set_ylabel(r'$\theta_{\rm e}$ (Deg)')
        ax2.set_xlabel(r'$m_1 / m_{12}$')
        mass_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax2.set_xticks(mass_ticks)
        ax2.set_xticklabels([r'$0$', r'$0.2$', r'$0.4$', r'$0.6$',
                              r'$0.8$', r'$1.0$'])
        ax3 = ax1.twiny()
        A0s = []
        Abars = []
        for ratio in mass_ticks:
            m1 = ratio * 60
            m2 = 60 - m1
            m3 = 3e7
            a0 = 0.1
            a2 = 300
            e2 = 0
            getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
            A0s.append(r'$%0.1f$' % (getter_kwargs['eps_sl'] * 4 / 3))
            dW_num, (dWSLz_num, dWSLx_num), _ = get_dW(
                1e-3, I0, **getter_kwargs)
            Abar = np.sqrt(dWSLz_num**2 + dWSLx_num**2) / abs(dW_num)
            Abars.append(r'$%0.1f$' % Abar)
        ax3.set_xlim(ax1.get_xlim())
        ax3.set_xticks(ax1.get_xticks())
        ax3.set_xticklabels(Abars)
        ax3.set_xlabel(r'$\mathcal{A}$')

        plt.tight_layout()
        fig.subplots_adjust(hspace=0.03)
        plt.savefig(fn.replace('.txt', '.png'), dpi=300)
        plt.clf()

if __name__ == '__main__':
    # plot_3vec()
    plot_3vec_eta()
    # plot_bin_bifurcations()
