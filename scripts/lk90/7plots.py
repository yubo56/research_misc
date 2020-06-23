'''
more random plots
'''
# convert 7_3vec.png -crop 1000x1300+350+100 7_3vec_cropped.png
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
            r'$\hat{\mathbf{L}}$',
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
            r'$I_{\rm e}$',
            fontdict={'c': ld_c})
    xy_ld_tip = (
        0.5 * np.sin(np.radians(ld_q)),
        0.5 * np.cos(np.radians(ld_q)))
    ax.annotate('', xy=xy_ld_tip + np.array([0.003, -0.001]), xytext=xy_ld_tip,
                arrowprops=arrowprops(ld_c))
    ax.text(np.sin(np.radians(0.8 * s_q)) * 0.4 + offset,
            np.cos(np.radians(0.8 * s_q)) * 0.4 + 2 * offset,
            r'$I$',
            fontdict={'c': s_c})
    xy_s_tip = (
        0.4 * np.sin(np.radians(s_q)),
        0.4 * np.cos(np.radians(s_q)))
    ax.annotate('', xy=xy_s_tip + np.array([0.003, -0.003]), xytext=xy_s_tip,
                arrowprops=arrowprops(s_c))

    ax.set_aspect('equal')
    plt.savefig('7_3vec', dpi=400)
    plt.clf()

if __name__ == '__main__':
    plot_3vec()
