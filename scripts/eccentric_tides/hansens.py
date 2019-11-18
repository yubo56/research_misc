import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

def f(E, e):
    ''' returns true anomaly for given eccentric anomaly '''
    return 2 * np.arctan((1 + e) / (1 - e) * np.tan(E / 2))

def hansen_integrand(N, m, e):
    '''
    gets the Hansen coefficient integrand for F_{Nm} including the 1/2pi
    normalization, so F_{Nm}(e = 0) = \delta_{Nm}
    '''
    def integrand(E):
        return (
            np.cos(N * (E - e * np.sin(E)) - m * f(E, e)) /
            (1 - e * np.cos(E))) / (2 * np.pi)
    return integrand

if __name__ == '__main__':
    m = 2
    e = 0.9
    print('Ansatz N', (1 - e)**(-3/2))
    integrand = hansen_integrand(2, m, e)

    E = np.linspace(-np.pi, np.pi, 1000)
    plt.plot(E, integrand(E), 'r')
    plt.savefig('hansens.png')
    # for N in range(1, 100):
    #     integrand = hansen_integrand(N, m, e)
    #     y, abserr = quad(integrand, -np.pi, np.pi, limit=100)
    #     print('%03d, %.8f, %.8e' % (N, y, abserr))

