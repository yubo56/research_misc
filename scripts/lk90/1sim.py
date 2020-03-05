from utils import *
import os
import pickle

def run_one(I_deg=90.355, e=0.001, fn_template='1sim_%s',
            a_f=1e-4, wsl_mult=1,
            plot=True,
            plot_func=plot_traj, **kwargs):
    m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
    I = np.radians(I_deg)
    fn = fn_template % ('%.3f' % I_deg).replace('.', '_')
    getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    getter_kwargs['eps_sl'] *= wsl_mult

    if not os.path.exists('%s.pkl' % fn):
        print('Running %s' % fn)
        ret = solver(I, e,
                     atol=1e-6, rtol=1e-6,
                     getter_kwargs=getter_kwargs,
                     a_f=a_f,
                     tf=np.inf,
                     **kwargs,
                     )
        with open('%s.pkl' % fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        print('Loading %s' % fn)
        with open('%s.pkl' % fn, 'rb') as f:
            ret = pickle.load(f)
    if plot:
        plot_func(ret, fn,
                  m1, m2, m3, a0, a2, e2, I,
                  getter_kwargs=getter_kwargs,
                  plot_slice=np.where(ret.t > ret.t[-1] - 3)[0],
                  )
    L, _, s = to_vars(ret.y)
    Lhat = L / np.sqrt(np.sum(L**2, axis=0))
    q_slf = np.degrees(np.arccos(np.dot(Lhat[:, -1], s[:, -1])))
    return q_slf, ret

def plot_many(ens_fn='1ensemble', plot=True,
              I_degs=[], **kwargs):
    ''' run all the way to merger, measure q_sl at the end and plot '''
    q_slfs = []
    tfs = []
    for I_deg in I_degs:
        q_slf, ret = run_one(I_deg=I_deg, plot=True, **kwargs)
        q_slfs.append(q_slf)
        tfs.append(ret.t[-1] * S_PER_UNIT / S_PER_YR)
    # run_one(I_deg=I_degs[-1], plot=True, **kwargs)
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        ax1.plot(I_degs, q_slfs)
        ax1.set_xlabel(r'$I$ (Deg)')
        ax1.set_ylabel(r'$\theta_{sl,f}$ (Deg)')

        ax2.plot(I_degs, tfs)
        ax2.set_xlabel(r'$I$ (Deg)')
        ax2.set_ylabel(r'$t_f$ (yr)')
        plt.tight_layout()
        plt.savefig(ens_fn, dpi=400)
        plt.close()
    return q_slfs

def mkdirp(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    # run_one(I_deg=90.355, q_sl0=np.radians(20),
    #         fn_template='1sim_qsl020_%s',
    #         # plot_func=plot_traj_vecs,
    #         )
    # run_one(I_deg=90.30,
    #         a_f=0.3, fn_template='wsl_0/1sim_%s',
    #         # plot_func=plot_traj_vecs,
    #         )

    colors=['y', 'r', 'k', 'g', 'b']
    I_degs = np.arange(90.05, 90.3, 0.002)
    for idx, wsl_mult in enumerate([0.1, 1/3, 1, 3, 10]):
        mkdirp('wsl_%d' % idx)
        a_f = 0.2 * min(wsl_mult, 1)
        q_slfs = plot_many(a_f=a_f, fn_template='wsl_' + str(idx) + '/1sim_%s',
                           I_degs=I_degs, wsl_mult=wsl_mult, plot=False)
        plt.plot(I_degs, q_slfs, colors[idx],
                 label=r'$%.1f \times \Omega_{rm SL}$' % wsl_mult)
    plt.xlabel(r'$I$ (Deg)')
    plt.ylabel(r'$\theta_{sl,f}$ (Deg)')
    plt.legend(loc='best', fontsize=14)
    plt.savefig('1ensembles', dpi=400)
