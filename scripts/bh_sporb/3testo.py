import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

def plot_pure_geom():
    '''
    what does the delta phi distribution look like if it's purely geometric? Can
    be calculable analytically, but something else is going on: this only
    matches the low-spin distribution!
    '''
    q1 = np.arccos(np.random.uniform(-1, 1, 10000))
    q2 = np.arccos(np.random.uniform(-1, 1, 10000))
    phi1 = np.random.uniform(0, 2 * np.pi, 10000)
    phi2 = np.random.uniform(0, 2 * np.pi, 10000)
    s1vec = np.array([
        np.sin(q1) * np.cos(phi1),
        np.sin(q1) * np.sin(phi1),
        np.cos(q1),
    ])
    s2vec = np.array([
        np.sin(q2) * np.cos(phi2),
        np.sin(q2) * np.sin(phi2),
        np.cos(q2),
    ])

    svectot = s1vec + s2vec
    svectot /= np.sqrt(np.sum(svectot**2, axis=0))
    qsltot = np.arccos(svectot[2])
    dphi = np.degrees((phi2 - phi1) % (2 * np.pi))

    plt.scatter(qsltot, dphi)
    plt.xlabel(r'$\theta_{\rm SL}$')
    plt.ylabel(r'$\Phi_2 - \Phi_1$')
    plt.tight_layout()
    plt.savefig('3geom', dpi=200)
    plt.close()

if __name__ == '__main__':
    plot_pure_geom()
