import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('lines', lw=3.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
dat = [
    [0.2, 0.6, 88.1981981981982],
    [0.3, 0.6, 87.87787787787788],
    [0.4, 0.6, 87.47747747747748],
    [0.5, 0.6, 87.31731731731732],
    [0.7, 0.6, 86.996996996997],
    [0.2, 0.8, 88.11811811811812],
    [0.3, 0.8, 87.71771771771772],
    [0.4, 0.8, 87.15715715715716],
    [0.5, 0.8, 86.996996996997],
    [0.7, 0.8, 86.67667667667668],
    [0.2, 0.9, 87.55755755755756],
    [0.3, 0.9, 87.47747747747748],
    [0.4, 0.9, 86.75675675675676],
    [0.5, 0.9, 86.51651651651652],
    [0.7, 0.9, 86.11611611611612],
]
epsocts = np.array([
    (1 - q) / (1 + q) / 36 * e2 / np.sqrt(1 - e2**2)
    for q, e2, _ in dat
])
Imins = np.array([d[2] for d in dat])

cs = ['k', 'b', 'g']
for c, q in zip(cs, [0.6, 0.8, 0.9]):
    idxs = np.where(np.array([d[1] for d in dat]) == q)[0]
    plt.plot(epsocts[idxs], np.cos(np.radians(Imins[idxs]))**2, '%so' % c,
             label=str(q))
plt.legend()
plt.ylabel(r'$\cos^2 I_{\min}$')
plt.xlabel(r'$\epsilon_{\rm oct}$')
plt.ylim(bottom=0)

plt.tight_layout()
plt.savefig('Imins', dpi=300)
