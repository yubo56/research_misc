''' random plots go in here '''
from scipy.optimize import brenth
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

from utils import *

m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
def plot_emax():
    ''' plot emax vs I '''
    I_vals = np.concatenate((
        np.linspace(np.pi / 4, np.pi / 2 - 0.5, 20),
        np.linspace(np.pi / 2 - 0.5, np.pi / 2 + 0.5, 801),
        np.linspace(np.pi / 2 + 0.5, 3 * np.pi / 4, 20),
    ))
    e_maxes = []
    e_maxes_eta0 = []
    e_maxes_naive = []
    for I in I_vals:
        _, emax, emax_eta0, emax_naive = get_vals(m1, m2, m3, a0, a2, e2, I)
        e_maxes.append(emax)
        e_maxes_eta0.append(emax_eta0)
        e_maxes_naive.append(emax_naive)
    plt.semilogy(np.degrees(I_vals), 1 - np.array(e_maxes), 'k',
                 label='Full')
    plt.semilogy(np.degrees(I_vals), 1 - np.array(e_maxes_eta0), 'b:',
                 label='Test mass')
    plt.semilogy(np.degrees(I_vals), 1 - np.array(e_maxes_naive), 'r:',
                 label='No-GR')
    print('Real, eta0, naive max I',
          np.degrees(I_vals[np.argmax(e_maxes)]),
          np.degrees(I_vals[np.argmax(e_maxes_eta0)]),
          np.degrees(I_vals[np.argmax(e_maxes_naive)]))
    plt.xlabel(r'$I_0$ (Deg)')
    plt.ylabel(r'$1 - e_{\max}$')
    plt.legend()
    plt.savefig('2_emaxes', dpi=300)

def plot_tmerge():
    ''' plot merge time vs I '''
    I_vals = np.concatenate((
        np.linspace(np.pi / 4, np.pi / 2 - 0.5, 20),
        np.linspace(np.pi / 2 - 0.5, np.pi / 2 + 0.5, 801),
        np.linspace(np.pi / 2 + 0.5, 3 * np.pi / 4, 20),
    ))
    tmerges = []
    for I in I_vals:
        tmerges.append(get_tmerge(m1,  m2, m3, a0, a2, e2, I)[1])
    plt.semilogy(np.degrees(I_vals), tmerges)
    plt.xlabel(r'$I_0$ (Deg)')
    plt.ylabel(r'$T_m$ (yr)')
    plt.savefig('2_tmerges', dpi=300)

if __name__ == '__main__':
    # plot_emax()
    plot_tmerge()
