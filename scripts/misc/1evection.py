'''
TODO: try for a few different inclinations/eccentricities?
'''
from evection_solver import *
from multiprocessing import Pool
import os
import time
import lzma
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = int(1e4)
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
from scipy.integrate import solve_ivp

G = 39.4751488
c = 6.32397263*10**4
# length = AU
# c = 1AU / 499s
# unit of time = 499s * 6.32e4 = 1yr
# unit of mass = solar mass, solve for M using N1 + distance in correct units

def dydt(t, y, m1, m2, m3, fsec, fsl, fgwa, fgwe):
    m12 = m1 + m2
    mu = (m2 * m1) / m12
    m123 = m12 + m3
    muout = (m12 * m3) / m123

    Lin = y[0:3]
    ein = y[3:6]
    rout = y[6:9]
    vout = y[9:12]
    Linmag = np.sqrt(np.sum(Lin**2))
    Linhat = Lin / Linmag
    e = np.sqrt(np.sum(ein**2))
    einhat = ein / e
    routmag = np.sqrt(np.sum(rout**2))
    routhat = rout / routmag

    ain = Linmag**2 / (mu**2 * G * (m12) * (1 - e**2))
    phi0 = G * m123 / routmag
    n = np.sqrt(G * m12 / ain**3)
    WSA = fsec * 3/2 * (m3 / m12) * (ain / routmag)**3 * n
    phiQ = fsec * m3 / (4 * routmag) * (mu / muout) * (ain / routmag)**2
    WSL = fsl * 3 * G * n * m12 / (c**2 * ain * (1 - e**2))
    WGWL = fgwa * 32 / 5 * G**(7/2) / c**5 * (
        mu**2 * m12**(5/2) / ain**(7/2)
        * (1 + 7 * e**2 / 8) / (1 - e**2)**2
    )
    WGWe = fgwe * 304 / 15 * G**3 / (c**5) * (
        mu * m12**2 / ain**4
        * (1 + 121 * e**2 / 304) / (1 - e**2)**(5/2)
    )

    dLindt = Linmag / np.sqrt(1 - e**2) * WSA * (
        -(1 - e**2) * np.dot(Linhat, routhat) * np.cross(Linhat, routhat)
        + 5 * np.dot(ein, routhat) * np.cross(ein, routhat)
    ) - WGWL * Linhat
    deindt = WSA * np.sqrt(1 - e**2) * (
        -np.dot(Linhat, routhat) * np.cross(ein, routhat)
        -2 * np.cross(Linhat, ein)
        + 5 * np.dot(ein, routhat) * np.cross(Linhat, routhat)
    ) + WSL * np.cross(Linhat, ein) - WGWe * einhat

    droutdt = vout
    dvoutdt = -phi0 * routhat / routmag - phiQ * (
        -3 * (routhat / routmag) * (
            -1 + 6*e**2 + 3*(1 - e**2) * np.dot(Linhat, routhat)
            - 15 * np.dot(ein, routhat)
        ) + 6 * (1 - e**2) / routmag * np.dot(Linhat, routhat) * (
            Linhat - np.dot(Linhat, routhat) * routhat
        ) - 30 * np.dot(ein, routhat) / routmag * (
            ein - np.dot(ein, routhat) * routhat
        )
    )
    return np.concatenate((dLindt, deindt, droutdt, dvoutdt))
def solve_sa(
    tf, a0, e0, aout, eout=0, fout=0,
    m1=1, m2=1, m3=1,
    I0d=0, w0=np.random.uniform(2 * np.pi),
    W0=np.random.uniform(2 * np.pi),
    method='Radau', tol=1e-9,
    fsec=1, fsl=1, fgwa=1, fgwe=1,
    use_cython=False,
):
    m12 = m1 + m2
    mu = (m2 * m1) / m12
    m123 = m12 + m3
    muout = (m12 * m3) / m123
    Lmag = mu * np.sqrt(G * m12 * a0 * (1 - e0**2))
    I0 = np.radians(I0d)
    L0 = [
        Lmag * np.sin(I0) * np.sin(W0),
        -Lmag * np.sin(I0) * np.cos(W0),
        Lmag * np.cos(I0),
    ]
    e0 = [
        e0 * (np.cos(w0) * np.cos(W0) - np.sin(w0) * np.cos(I0) * np.sin(W0)),
        e0 * (np.cos(w0) * np.sin(W0) + np.sin(w0) * np.cos(I0) * np.cos(W0)),
        e0 * np.sin(w0) * np.sin(I0),
    ]

    # convention: pericenter for outer orbit in +x direction, fout=0
    rout = aout * (1 - eout**2) / (1 + eout * np.cos(fout))
    r0 = [
        rout * np.cos(fout),
        rout * np.sin(fout),
        0,
    ]
    # specific ang mom = rv = sqrt(Gma(1-e^2)) is conserved
    vout = np.sqrt(G * m123 * aout * (1 - eout**2)) / rout
    v0 = [
        -vout * np.sin(fout),
        vout * np.cos(fout),
        0,
    ]
    y0 = np.concatenate((L0, e0, r0, v0))
    def ain_thresh(t, y, m1, m2, m3, f1, f2, f3, f4):
        Lin = y[0:3]
        Linmag = np.sqrt(np.sum(Lin**2))
        ein = y[3:6]
        e = np.sqrt(np.sum(ein**2))
        ain = Linmag**2 / (mu**2 * G * (m12) * (1 - e**2))
        return ain - 0.0015
    ain_thresh.terminal = True
    if use_cython:
        return solve_ivp(dydt_cython, (0, tf), y0, events=[ain_thresh_cython],
                         method=method, atol=tol, rtol=tol,
                         args=(m1, m2, m3, fsec, fsl, fgwa, fgwe))
    else:
        return solve_ivp(dydt, (0, tf), y0, events=[ain_thresh],
                         method=method, atol=tol, rtol=tol,
                         args=(m1, m2, m3, fsec, fsl, fgwa, fgwe))

def run(fn, a0=0.002, e0=1e-3, tf=1e4, tol=1e-9, method='BDF',
        folder='1evection/', plot=False, **kwargs):
    os.makedirs(folder, exist_ok=True)
    m1 = 1
    m2 = 1
    m12 = m1 + m2
    mu = (m2 * m1) / m12
    pkl_fn = folder + fn + '.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        ret = solve_sa(tf, a0, e0, 2.38, m1=m1, m2=m2, method=method,
                       tol=tol, fgwa=0, fgwe=0, **kwargs)
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump((ret), f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            if plot:
                print('Loading %s' % pkl_fn)
            ret = pickle.load(f)
    y = ret.y

    Lin = y[0:3]
    ein = y[3:6]
    rout = y[6:9]
    vout = y[9:12]
    Linmag = np.sqrt(np.sum(Lin**2, axis=0))
    Linhat = Lin / Linmag
    e = np.sqrt(np.sum(ein**2, axis=0))
    einhat = ein / e
    routmag = np.sqrt(np.sum(rout**2, axis=0))
    routhat = rout / routmag
    ain = Linmag**2 / (mu**2 * G * (m12) * (1 - e**2))
    if not plot:
        return e.max() - e.min()

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        sharex=True,
        figsize=(8, 8))
    # ax1.plot(ret.t, ain)
    # ax1.set_ylabel(r'$a$')
    # ax1.plot(rout[0], rout[1])
    # ax1.set_xlabel(r'$x_{\rm out}$')
    # ax1.set_ylabel(r'$y_{\rm out}$')
    ax1.plot(ret.t, routmag)
    ax1.set_ylabel(r'$|r_{\rm out}|$')

    ax2.plot(ret.t, e)
    ax2.set_ylabel(r'$e$')

    # Win = np.arctan2(Linhat[0], Linhat[1]) # I=0, varpi = w
    varpi_in = np.unwrap(np.arctan2(einhat[1], einhat[0]))
    fout = np.unwrap(np.arctan2(routhat[1], routhat[0]))

    ax3.plot(ret.t, np.degrees(varpi_in - fout))
    ax3.set_ylabel(r'$\varpi - f_{\rm out}$ [Deg]')
    ax3.set_xlabel(r'$t$')

    fig.subplots_adjust(hspace=0.02)
    plt.tight_layout()
    plt.savefig(folder + fn, dpi=300)
    plt.close()

def scan_circ(to_run=False):
    p = Pool(4)
    idxs = range(100)
    args = [
        ('sim%d' % i, 0.002 + 0.00005 * (i - len(idxs) / 2) / len(idxs))
        for i in idxs
    ]

    if not to_run:
        emaxes = []
        for arg in args:
            emaxes.append(run(arg[0], arg[1], plot=True))
    else:
        emaxes = p.starmap(run, args)

    plt.plot([1e3 * a[1] for a in args], np.array(emaxes))
    plt.xlabel(r'$a$ [$10^{-3}$ AU]')
    plt.ylabel(r'$\Delta e$')
    plt.tight_layout()
    plt.savefig('1evection/composite.png', dpi=300)
    plt.close()

def cython_tests():
    start = time.time()
    run(folder='1test/', fn='default', tf=1e2, plot=True, W0=0, w0=0)
    print('Regular took %.2f' % (time.time() - start))

    start = time.time()
    run(folder='1test/', fn='cython', tf=1e2, use_cython=True, plot=True, W0=0,
        w0=0)
    print('Cython took %.2f' % (time.time() - start))

    start = time.time()
    run(folder='1test/', fn='cython10', tf=1e2, use_cython=True, plot=True, W0=0,
        w0=0, tol=1e-10)
    print('Cython -10 took %.2f' % (time.time() - start))

    start = time.time()
    run(folder='1test/', fn='cythonR', tf=1e2, use_cython=True, plot=True, W0=0,
        w0=0, method='Radau')
    print('Cython Radau took %.2f' % (time.time() - start))

    start = time.time()
    run(folder='1test/', fn='cythonDOP', tf=1e2, use_cython=True, plot=True, W0=0,
        w0=0, method='DOP853')
    print('Cython DOP853 took %.2f' % (time.time() - start))

    start = time.time()
    run(folder='1test/', fn='cythonDOP10', tf=1e2, use_cython=True, plot=True, W0=0,
        w0=0, method='DOP853', tol=1e-10)
    print('Cython DOP853-10 took %.2f' % (time.time() - start))

if __name__ == '__main__':
    # cython_tests()

    # scan_circ()
    scan_circ(to_run=True)
