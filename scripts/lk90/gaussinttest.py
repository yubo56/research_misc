'''
testing some gaussian integrals for delta q_eff calculations
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('lines', lw=3.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

def test1():
    ''' W_e const '''
    W = 2
    inttotal = 10
    Amax_arr = np.arange(0.5, 2, 0.1)
    tmid = 150
    t = np.linspace(-tmid, tmid, int(1e4))
    dt = np.full_like(t, t[1] - t[0])
    max_sums = []
    for Amax in Amax_arr:
        A_sigm2 = (inttotal / Amax)**2 / (2 * np.pi)
        A_vals = Amax * np.exp(-t**2 / (2 * A_sigm2))
        max_sum = -1
        for phi_offset in np.linspace(0, 2 * np.pi, endpoint=False):
            max_sum = max(max_sum, abs(np.sum(
                A_vals * dt * np.cos(W * t + phi_offset)
            )))
        max_sums.append(max_sum)
    plt.semilogy(Amax_arr, max_sums, 'bo')
    plt.plot(Amax_arr,
             inttotal * np.exp(-(W**2 * (inttotal / Amax_arr)**2 / (4 * np.pi))),
             'r')
    plt.ylim(bottom=1e-17)
    plt.savefig('gaussinttest.png', dpi=200)
    plt.close()

def test2():
    ''' W_e linear '''
    t = np.linspace(-1, 1, int(3e4))
    dt = np.full_like(t, t[1] - t[0])
    s = 0.3
    W = 100
    Wdots = np.linspace(10, 100, 10)
    integs = []
    for Wdot in Wdots:
        max_sum = -1
        for phi_offset in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            max_sum = max(max_sum, abs(np.sum(
                np.exp(-t**2 / (2 * s**2))
                * np.cos((Wdot * t + W) * t + phi_offset)
                * dt
            )))
        integs.append(max_sum)
    plt.semilogy(Wdots, integs)
    ylim = plt.ylim()
    guess = np.sqrt(np.pi / Wdots) * np.exp(-W**2 / (8 * s**2 * Wdots**2))
    plt.plot(Wdots, guess)
    plt.ylim(ylim)
    plt.savefig('gausstest2.png', dpi=200)
    plt.close()

def test_real():
    ''' semi-realistic W_e '''
    s = 0.05
    t = np.linspace(-1, 1, int(1e5))
    dt = np.full_like(t, t[1] - t[0])
    a = 1 / (t + 1.2)**4
    I = np.radians(65)
    Wdot = 3 / np.sqrt(a)
    Wsl = 0.1 * a**(-4.5)
    W_evec = np.array([Wsl * np.sin(I), Wdot + Wsl * np.cos(I)])
    W_e = np.sqrt(np.sum(W_evec**2, axis=0))
    Idot = 0.1 * np.exp(-t**2 / (2 * s**2))
    inttotal = 0.1 * s * np.sqrt(2 * np.pi)
    plt.semilogy(t, Wdot, 'b:')
    plt.semilogy(t, Wsl, 'r:')
    plt.semilogy(t, W_e, 'k')
    plt.semilogy(t, Idot, 'g:')
    plt.ylim(1e-4, 1e3)
    plt.savefig('gausstestreal_freqs.png', dpi=200)
    plt.clf()

    We_mults = np.geomspace(10, 100, 10)
    integs = []
    for We_mult in We_mults:
        max_sum = -1
        for phi_offset in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            max_sum = max(max_sum, abs(np.sum(
                Idot
                * np.cos(W_e * We_mult * t + phi_offset)
                * dt
            )))
        integs.append(max_sum)
    plt.loglog(We_mults, integs, 'ko')
    ylims = plt.ylim()
    plt.plot(We_mults,
             inttotal * np.exp(-((We_mults * W_e[len(t) // 2])**2
                                 * s**2 / (4 * np.pi))),
             'r')
    plt.ylim(ylims)
    plt.tight_layout()
    plt.savefig('gausstestreal.png', dpi=200)
    plt.close()

    smults = np.geomspace(1, 100, 10)
    integs = []
    for smult in smults:
        Idot = 0.1 / smult * np.exp(-t**2 / (2 * (s * smult)**2))
        max_sum = -1
        for phi_offset in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            max_sum = max(max_sum, abs(np.sum(
                Idot
                * np.cos(W_e * t + phi_offset)
                * dt
            )))
        integs.append(max_sum)
    plt.loglog(smults, integs, 'ko')
    Wdot = (W_e[int(0.6 * len(t))] - W_e[int(0.4 * len(t))]) / 0.2
    guess = 0.1 / smults * np.sqrt(np.pi / Wdot) * np.exp(-W_e[len(t) // 2]**2 /
                                            (8 * (s * smults)**2 * Wdot**2))
    plt.plot(smults, guess, 'r')
    plt.tight_layout()
    plt.savefig('gausstestreal_smult.png', dpi=200)
    plt.close()

if __name__ == '__main__':
    # test1()
    # test2()
    test_real()
