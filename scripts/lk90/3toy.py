'''
potential toy models for 90 degree attractor?
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('lines', lw=2)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
from scipy.integrate import solve_ivp

def test_sinusoidal():
    ''' try to precess a vector around a sinusoidally oscillating vector '''
    w = 1 # oscillation frequency of the main vector
    tf = 500
    gamma = 0.005
    y0 = [0, 0, 1]

    def dydt(t, y):
        q = np.radians(10) * np.exp(-gamma * t) * (
            np.cos(w * t) * np.exp(-gamma * t))
        return np.cross([np.sin(q), 0, np.cos(q)], y)
    ret = solve_ivp(dydt, (0, tf), y0)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    ax1.plot(ret.t, ret.y[0])
    ax2.plot(ret.t, ret.y[1])
    ax3.plot(ret.t, ret.y[2])
    ax3.set_xlabel(r'$t$')
    plt.savefig('3sinosoidal')
    plt.close()

if __name__ == '__main__':
    test_sinusoidal()
