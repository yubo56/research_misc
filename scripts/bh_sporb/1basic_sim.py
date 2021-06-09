import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
from utils import *
from scipy.integrate import solve_ivp

def dydt_fixedj(t, y, eps):
    '''
    Assume jhat is really fixed, along zhat, and let alpha change
    '''
    jhat = np.array([0, 0, 1])

    s1, s2 = np.reshape(y[ :6], (2, 3))
    alpha = y[6]

    dydt = np.zeros_like(y)
    dydt[ :3] = np.cross(jhat, s1) - alpha * np.cross(s2, s1)
    dydt[3:6] = np.cross(jhat, s2) - alpha * np.cross(s1, s2)
    dydt[6] = eps * alpha
    return dydt

def H(s1s, s2s, alpha):
    return (
        -ts_dot_hat(s1s + s2s, np.array([0, 0, 1]))
        + alpha * ts_dot(s1s, s2s))

if __name__ == '__main__':
    I1 = np.radians(10)
    I2 = np.radians(20)
    y0 = [0, np.sin(I1), np.cos(I1),
          0, np.sin(I2), np.cos(I2),
          0.1]
    ret = solve_ivp(dydt_fixedj, (0, 1e2), y0, method='DOP853', atol=1e-9,
                    rtol=1e-9, args=[0.03])
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2,
        figsize=(8, 8),
        sharex=True)
    s1s = np.reshape(ret.y[ :3, :], (3, len(ret.t)))
    s2s = np.reshape(ret.y[3:6, :], (3, len(ret.t)))
    alpha = ret.y[6]
    ax1.plot(ret.t, s1s[2])
    ax1.plot(ret.t, s2s[2])
    phi1 = np.unwrap(np.arctan2(s1s[1], s1s[0]))
    phi2 = np.unwrap(np.arctan2(s2s[1], s2s[0]))
    # ax2.plot(ret.t, np.degrees(phi1 - phi2) % 180)
    ax2.plot(ret.t, ts_dot(s1s, s2s))

    # ax2.plot(ret.t, np.degrees(phi1))
    # ax2.plot(ret.t, np.degrees(phi2))
    ax3.plot(ret.t, H(s1s, s2s, alpha))
    # ax3.set_ylim(bottom=-1.2, top=-0.8)
    ax4.plot(ret.t, ts_dot_hat(s1s + s2s, np.array([0, 0, 1])))

    jhat_arr = np.array([
        np.zeros_like(s1s[2]),
        np.zeros_like(s1s[2]),
        np.ones_like(s1s[2]),
    ])
    ax4.plot(ret.t, ts_dot(s1s - s2s, jhat_arr + alpha * (s1s + s2s)))
    plt.savefig('/tmp/foo', dpi=200)
    plt.close()
