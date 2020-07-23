'''
more random plots
'''
# convert 7_3vec.png -crop 1100x1300+350+100 7_3vec_cropped.png
import numpy as np
import scipy.optimize as opt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
LW=3.5

def get_xy(angle, mag=1):
    return mag * np.sin(np.radians(angle)), mag * np.cos(np.radians(angle))

def plot_3vec():
    ''' plots the relative orientations of the three vectors '''
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

    # draw three arrows
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

    s_q = 50
    s_xy = get_xy(s_q)
    s_c = 'b'
    ax.annotate('', xy=s_xy, xytext=(0, 0),
                 arrowprops=arrowprops(s_c))
    ax.text(s_xy[0] - offset, s_xy[1] + offset,
            r'$\overline{\mathbf{\Omega}}_{\rm SL}$',
            fontdict={'c': s_c})

    # draw arcs
    # center, (dx, dy), rotation, start angle, end angle (degrees)
    arc_lw = 3
    ld_arc = patches.Arc((0, 0), 1.0, 1.0, 0, 90 - ld_q, 90,
                         color=ld_c, lw=arc_lw, alpha=alpha,
                         label='1')
    ax.add_patch(ld_arc)
    s_arc = patches.Arc((0, 0), 0.8, 0.8, 0, 90 - s_q, 90,
                        color=s_c, lw=arc_lw, alpha=alpha)
    ax.add_patch(s_arc)
    # label arcs
    ax.text(np.sin(np.radians(ld_q * 0.4)) * 0.5,
            np.cos(np.radians(ld_q * 0.4)) * 0.5 + 2 * offset,
            r'$\bar{I}_{\rm e}$',
            fontdict={'c': ld_c})
    xy_ld_tip = (
        0.5 * np.sin(np.radians(ld_q)),
        0.5 * np.cos(np.radians(ld_q)))
    ax.annotate('', xy=xy_ld_tip + np.array([0.003, -0.001]), xytext=xy_ld_tip,
                arrowprops=arrowprops(ld_c))
    ax.text(np.sin(np.radians(0.8 * s_q)) * 0.4 + offset,
            np.cos(np.radians(0.8 * s_q)) * 0.4 + 2 * offset,
            r'$\bar{I}$',
            fontdict={'c': s_c})
    xy_s_tip = (
        0.4 * np.sin(np.radians(s_q)),
        0.4 * np.cos(np.radians(s_q)))
    ax.annotate('', xy=xy_s_tip + np.array([0.003, -0.003]), xytext=xy_s_tip,
                arrowprops=arrowprops(s_c))

    ax.set_aspect('equal')

    # add coordinate axes
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


    plt.savefig('7_3vec', dpi=400)
    plt.clf()

def plot_bin_bifurcations():
    fns = ['M1_M2_thetasl_thetae_70.txt',
           'M1_M2_thetasl_thetae_88.txt']
    for fn in fns:
        with open(fn) as f:
            dat = []
            for l in f.readlines():
                m1, m2, qsl, qe = [float(w) for w in l.split(' ')]
                dat.append((m1 / (m1 + m2), qsl, qe))
        ratios, qsls, qes = np.array(dat).T

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        ax1.plot(ratios, qes, 'k,')
        ax2.plot(ratios, qsls, 'k,')
        ax1.set_ylabel(r'$\theta_{\rm e}$')
        ax2.set_ylabel(r'$\theta_{\rm sl}$')
        ax2.set_xlabel(r'$m_1 / m_{12}$')
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.03)
        plt.savefig(fn.replace('.txt', '.png'), dpi=300)
        plt.clf()

if __name__ == '__main__':
    plot_3vec()
    # plot_bin_bifurcations()
