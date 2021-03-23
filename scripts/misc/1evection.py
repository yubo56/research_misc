'''
TODO: try for a few different inclinations/eccentricities?
'''
from evection_solver import *
from scipy.optimize import bisect
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

def H(nout_rat, eps, I0d, phi, e, e0=1e-3):
    I0 = np.radians(I0d)
    J3 = np.sqrt(1 - e0**2) * (1 - np.cos(I0))
    P = 2 * (1 - nout_rat - 3 * eps * (4 * J3 - 3) / 4)
    Q = 4 - 3 * eps / 2
    R = 15 * eps / 2
    Gamma = -e**2/4
    return Gamma * P - Gamma**2 * Q + Gamma * R * np.cos(phi)

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

def run(fn, a0=0.002, e0=1e-3, tf=1e4, tol=1e-9, method='DOP853',
        aout=2.38, folder='1evection/', plot=False, foutmult=1, **kwargs):
    os.makedirs(folder, exist_ok=True)
    m1 = 1
    m2 = 1
    m3 = 1
    I0d = 0
    m12 = m1 + m2
    mu = (m2 * m1) / m12
    m123 = m12 + m3
    pkl_fn = folder + fn + '.pkl'

    nout_rat = (
        (a0 / aout)**(3/2) * (m123 / m12)**(1/2)
        / (3 * G * m12 / (c**2 * a0))
    )
    eout = kwargs.get('eout', 0)
    nout_rat_ecc = nout_rat * np.sqrt(1 + eout) / (1 - eout)**(3/2)
    print('nout ratios (circ, ecc)', nout_rat, nout_rat_ecc)

    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        ret = solve_sa(tf, a0, e0, 2.38, m1=m1, m2=m2, m3=m3, I0d=I0d,
                       method=method, tol=tol, fgwa=0, fgwe=0, **kwargs)
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

    # Win = np.arctan2(Linhat[0], Linhat[1]) # I=0, varpi = w
    varpi_in = np.unwrap(np.arctan2(einhat[1], einhat[0]))
    fout = np.unwrap(np.arctan2(routhat[1], routhat[0]))

    dfoutdt = np.diff(fout) / np.diff(ret.t)
    dvarpidt = np.diff(varpi_in) / np.diff(ret.t)
    # print('dfdt', np.min(dfoutdt), np.max(dfoutdt))
    # print('dvarpidt', np.min(dvarpidt), np.max(dvarpidt))

    eps = (m3 * a0**4 * c**2) / (3 * G * m12**2 * aout**3)
    H_simp = H(dfoutdt / dvarpidt, eps, I0d,
              2 * (varpi_in - fout)[ :-1], e[: -1])

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        sharex=True,
        figsize=(8, 8))
    ax1.plot(ret.t, routmag)
    ax1.set_ylabel(r'$|r_{\rm out}|$')
    ax1twin = ax1.twinx()
    # ax1twin.plot(ret.t, ain, ls='--')
    # ax1twin.set_ylabel(r'$a$')
    ax1twin.plot(ret.t[ :-1], H_simp, 'm:')
    ax1twin.set_ylabel(r'$H$')

    ax2.plot(ret.t, e)
    ax2.set_ylabel(r'$e$')

    ax3.plot(ret.t, 2 * np.degrees(varpi_in - foutmult * fout))
    if foutmult == 1:
        ax3.set_ylabel(r'$2(\varpi - f_{\rm out})$ [Deg]')
    else:
        ax3.set_ylabel(r'$2(\varpi - %df_{\rm out})$ [Deg]' % foutmult)
    ax3.set_xlabel(r'$t$')

    fig.subplots_adjust(hspace=0.02)
    plt.tight_layout()
    plt.savefig(folder + fn, dpi=300)
    plt.close()

def scan_circ(to_run=False, plot=True):
    p = Pool(4)
    idxs = range(100)
    args = [
        ('sim%d' % i, 0.002 + 0.00005 * (i - len(idxs) / 2) / len(idxs))
        for i in idxs
    ]

    if not to_run:
        emaxes = []
        for arg in args:
            emaxes.append(run(arg[0], arg[1], plot=plot))
            print(emaxes[-1])
    else:
        emaxes = p.starmap(run, args)

    plt.plot([1e3 * a[1] for a in args], np.array(emaxes))
    plt.xlabel(r'$a$ [$10^{-3}$ AU]')
    plt.ylabel(r'$\Delta e$')
    plt.tight_layout()
    plt.savefig('1evection/composite.png', dpi=300)
    plt.close()

def cython_tests():
    '''
    Results:

    Running 1test/default.pkl
    Regular took 8.10
    Running 1test/cython.pkl
    Cython took 1.79
    Running 1test/cython10.pkl
    Cython -10 took 2.42
    Running 1test/cythonR.pkl
    Cython Radau took 4.03
    Running 1test/cythonDOP.pkl
    Cython DOP853 took 0.53
    Running 1test/cythonDOP10.pkl
    Cython DOP853-10 took 0.57

    To date, DOP853-10 and Radau are the most accurate
    '''
    start = time.time()
    run(folder='1test/', fn='default', tf=1e2, use_cython=False, plot=True,
        W0=0, w0=0, tol=1e-9, method='BDF')
    print('Regular took %.2f' % (time.time() - start))

    start = time.time()
    run(folder='1test/', fn='cython', tf=1e2, use_cython=True, plot=True,
        W0=0, w0=0, tol=1e-9, method='BDF')
    print('Cython took %.2f' % (time.time() - start))

    start = time.time()
    run(folder='1test/', fn='cython10', tf=1e2, use_cython=True, plot=True, W0=0,
        w0=0, tol=1e-10, method='BDF')
    print('Cython -10 took %.2f' % (time.time() - start))

    start = time.time()
    run(folder='1test/', fn='cythonR', tf=1e2, use_cython=True, plot=True, W0=0,
        w0=0, tol=1e-9, method='Radau')
    print('Cython Radau took %.2f' % (time.time() - start))

    start = time.time()
    run(folder='1test/', fn='cythonDOP', tf=1e2, use_cython=True, plot=True, W0=0,
        w0=0, tol=1e-9, method='DOP853')
    print('Cython DOP853 took %.2f' % (time.time() - start))

    start = time.time()
    run(folder='1test/', fn='cythonDOP10', tf=1e2, use_cython=True, plot=True, W0=0,
        w0=0, method='DOP853', tol=1e-10)
    print('Cython DOP853-10 took %.2f' % (time.time() - start))

def plot_H(fn, m12=2, m3=1, a=0.001994, aout=2.38 - 1e-5, I0d=0, npts=400):
    m123 = m12 + m3

    nout_rat = (a / aout)**(3/2) * (m123 / m12)**(1/2) / (3 * G * m12 / (c**2 * a))
    eps = (m3 * a**4 * c**2) / (3 * G * m12**2 * aout**3)

    phi = np.linspace(0, 2 * np.pi, npts)
    e = np.linspace(0, 0.008, npts)
    egrid = np.outer(np.ones_like(phi), e)
    phigrid = np.outer(phi, np.ones_like(e))
    Hgrid = H(nout_rat, eps, I0d, phigrid, egrid)
    plt.contourf(
        phigrid,
        egrid,
        Hgrid,
        levels=30,
    )
    H_sep = Hgrid[0, :].max()
    plt.axhline(1e-3, c='k', ls='--')
    # plt.axhline((7.5e-3)**2, c='k')
    # plt.contour(
    #     phigrid,
    #     egrid,
    #     Hgrid,
    #     levels=[H_sep],
    #     linewidths=2,
    #     colors='k',
    # )
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\sqrt{-\Gamma} \approx e$')
    plt.tight_layout()
    plt.savefig(fn, dpi=300)
    plt.close()

def investigate_38():
    ''' too lazy to find another set of resonant params lol '''
    a0_38 = 0.001994
    run('sim38long', a0_38, tol=1e-12, plot=True, tf=1e5)
    # run('sim38long1', a0_38, tol=1e-12, plot=True, tf=1e5)
    # run('sim38long2', a0_38, tol=1e-12, plot=True, tf=1e5)
    # run('sim38long3', a0_38, tol=1e-12, plot=True, tf=1e5)
    # run('sim38long4', a0_38, tol=1e-12, plot=True, tf=1e5)

    eout = 0.6
    run('sim38ecc', a0_38, plot=True, tf=1e5, tol=1e-12, eout=eout)
    # run('sim38ecc1', a0_38, plot=True, tf=1e5, tol=1e-12, eout=eout)
    # run('sim38ecc2', a0_38, plot=True, tf=1e5, tol=1e-12, eout=eout)
    # run('sim38ecc3', a0_38, plot=True, tf=1e5, tol=1e-12, eout=eout)
    # run('sim38ecc4', a0_38, plot=True, tf=1e5, tol=1e-12, eout=eout)
    a0_38_ecc = a0_38 / (np.sqrt(1 + eout) / (1 - eout)**(3/2))**(2/5)
    run('sim38eccperi', a0_38_ecc, plot=True, tf=1e5, tol=1e-12, eout=eout,
        foutmult=5)
    # run('sim38eccperi1', a0_38_ecc, plot=True, tf=1e5, tol=1e-12, eout=eout,
    #     foutmult=5)
    # run('sim38eccperi2', a0_38_ecc, plot=True, tf=1e5, tol=1e-12, eout=eout,
    #     foutmult=5)
    # run('sim38eccperi3', a0_38_ecc, plot=True, tf=1e5, tol=1e-12, eout=eout,
    #     foutmult=5)
    # run('sim38eccperi4', a0_38_ecc, plot=True, tf=1e5, tol=1e-12, eout=eout,
    #     foutmult=5)

    # run('sim38longhighe', a0_38, plot=True, e0=0.1, tf=1e5)
    # plot_H('sim38_H', npts=100)

if __name__ == '__main__':
    # cython_tests()

    # scan_circ(to_run=True)
    # scan_circ(plot=False)
    investigate_38()
