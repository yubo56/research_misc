from utils import *

def run(I=np.radians(90.3), e=0.001):
    m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
    getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)

    # note that a_f is in units of a0, not in AU!
    ret = solver(I, e,
                 atol=1e-10, rtol=1e-10,
                 getter_kwargs=getter_kwargs,
                 a_f=0.9,
                 tf=np.inf,
                 )
    plot_traj(ret, '1sim',
              m1, m2, m3, a0, a2, e2, I,
              use_start=False,
              use_stride=False,
              getter_kwargs=getter_kwargs,
              )

def plot_many(e=0.001):
    ''' run all the way to merger, measure q_sl at the end and plot '''
    m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
    getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)

    q_slfs = []
    tfs = []
    I_degs = np.linspace(90.05, 90.4, 8)
    for I_deg in I_degs:
        I = np.radians(I_deg)
        ret = solver(I, e,
                     atol=1e-10, rtol=1e-10,
                     getter_kwargs=getter_kwargs,
                     a_f=1e-6,
                     tf=np.inf,
                     )
        plot_traj(ret, '1sim_%s' % ('%.2f' % I_deg).replace('.', '_'),
                  m1, m2, m3, a0, a2, e2, I,
                  use_start=False,
                  use_stride=False,
                  getter_kwargs=getter_kwargs,
                  )
        L, _, s = to_vars(ret.y)
        Lhat = L / np.sqrt(np.sum(L**2, axis=0))
        q_slfs.append(np.degrees(np.arccos(np.dot(Lhat[:, -1], s[:, -1]))))
        tfs.append(ret.t[-1] * S_PER_UNIT / S_PER_YR)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax1.plot(I_degs, q_slfs)
    ax1.set_xlabel(r'$I$ (Deg)')
    ax1.set_ylabel(r'$\theta_{sl,f}$ (Deg)')

    ax2.plot(I_degs, tfs)
    ax2.set_xlabel(r'$I$ (Deg)')
    ax2.set_ylabel(r'$t_f$ (yr)')
    plt.savefig('1ensemble', dpi=400)
    plt.close()

if __name__ == '__main__':
    # run()
    plot_many()
