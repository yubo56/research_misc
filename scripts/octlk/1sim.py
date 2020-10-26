import time
from utils import *
import numpy as np
from cython_utils import *

from scipy.integrate import solve_ivp
from scipy.special import seterr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('font', family='serif', size=16)
plt.rc('lines', lw=1.0)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

TF = 100

def test_orbs():
    seterr(overflow='ignore')
    y0 = (1, 1e-3, np.radians(90.3), 0, 0)
    m1, m2, m3, a0, a2, e2 = 20, 30, 30, 100, 4500, 0
    eps = get_eps(m1, m2, m3, a0, a2, e2)
    def a_term_event(_t, y, *_args):
        return y[0] - 5e-3
    a_term_event.terminal = True
    ret = solve_ivp(dydt, (0, TF), y0, args=eps,
                    events=[a_term_event],
                    method='Radau', atol=1e-10, rtol=1e-10)
    plt.semilogy(ret.t, 1 - ret.y[1])
    # plt.semilogy(ret.t, ret.y[0])
    plt.tight_layout()
    plt.savefig('/tmp/foo', dpi=300)
    plt.clf()

def test_vec():
    a0, e0, I0, W0, w0 = (1, 1e-3, np.radians(90.3), 0, 0)
    j0 = np.sqrt(1 - e0**2)
    y0 = [
        a0,
        j0 * np.cos(W0) * np.sin(I0),
        j0 * np.sin(W0) * np.sin(I0),
        j0 * np.cos(I0),
        e0 * np.cos(w0) * np.cos(I0),
        e0 * np.sin(w0) * np.cos(I0),
        e0 * np.sin(I0)
    ]
    m1, m2, m3, a0, a2, e2 = 20, 30, 30, 100, 4500, 0.5
    eps = get_eps(m1, m2, m3, a0, a2, e2)
    def a_term_event(_t, y, *_args):
        return y[0] - 5e-3
    a_term_event.terminal = True
    ret = solve_ivp(dydt_vec, (0, TF), y0, args=eps,
                    events=[a_term_event],
                    method='Radau', atol=1e-12, rtol=1e-12)
    evec = ret.y[4:7, :]
    evec_mags = np.sqrt(np.sum(evec**2, axis=0))
    plt.semilogy(ret.t, 1 - evec_mags)
    # plt.semilogy(ret.t, ret.y[0])
    plt.tight_layout()
    plt.savefig('/tmp/foo_vec', dpi=300)
    plt.clf()

def timing_tests():
    '''
    To run 90.35 to coalescence, using params from SLL20, orbs = 21s, vecs = 62,
    and native python orbs ~ 50
    '''
    start = time.time()
    test_orbs()
    mid = time.time()
    print('Orbs used', mid - start)
    # test_vec()
    # print('Vecs used', time.time() - mid)

if __name__ == '__main__':
    timing_tests()
    pass
