'''
check the scaling of the exact solution w/ both resonant terms
'''
import numpy as np
from scipy import special as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
plt.rc('lines', lw=0.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

def plot_for_params(fn=np.cos, filen='exact_resonance_check', eta=1):
    '''
    plots the behavior of the key function:
    int(exp(-ix - eta * sin(beta * x)), {x, 0, t}
    '''
    def make_plot(eta, beta, betastr, xf=10000):
        x_vals = np.linspace(0, xf, int(10*xf + 1))
        integrand = fn(x_vals) * np.exp(- eta * np.sin(beta * x_vals))
        # integrand = np.cos(x_vals) * (1 - eta * np.sin(beta * x_vals))
        plt.plot(x_vals, np.cumsum(integrand) * (x_vals[1] - x_vals[0]),
                 label=r'$\beta=%s$' % betastr)
    make_plot(eta, 2, r'1')
    make_plot(eta, np.pi / 4, r'\pi / 4')
    make_plot(eta, 1/2, r'1/2', xf=1000)
    make_plot(eta, 1/3, r'1/3')
    make_plot(eta, 1/4, r'1/4')
    make_plot(eta, 3/4, r'3/4')
    make_plot(eta, 2, r'2')
    make_plot(eta, 3, r'3')
    ylims = plt.ylim()
    out_x_vals = np.linspace(0, 10000)
    def get_mag(q):
        return (
            eta**q / sp.factorial(q) / 2**q
        )
    if fn == np.cos:
        for q in [2, 4]:
            plt.plot(out_x_vals, (-1)**(q // 2) * out_x_vals *
                     get_mag(q), 'k', lw=2, alpha=0.3)
    if fn == np.sin:
        q = 3
        plt.plot(out_x_vals, out_x_vals * get_mag(q), 'k', lw=2, alpha=0.3)
    plt.ylim(ylims)
    plt.xlabel(r'$x$')
    plt.legend(ncol=2, fontsize=10)
    plt.ylabel(r'$S_\perp(x)$')
    plt.savefig(filen, dpi=200)
    plt.close()

if __name__ == '__main__':
    plot_for_params()
    plot_for_params(fn=np.sin, filen='exact_resonance_sin')
