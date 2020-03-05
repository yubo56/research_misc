from utils import *
import os
import pickle

def run_one(I_deg=90.355, e=0.001, fn_template='1sim_%s',
            plot_func=plot_traj, **kwargs):
    m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
    getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)

    I = np.radians(I_deg)
    fn = fn_template % ('%.2f' % I_deg).replace('.', '_')

    if not os.path.exists('%s.pkl' % fn):
        print('Running %s' % fn)
        ret = solver(I, e,
                     atol=1e-6, rtol=1e-6,
                     getter_kwargs=getter_kwargs,
                     a_f=1e-4,
                     tf=np.inf,
                     **kwargs,
                     )
        with open('%s.pkl' % fn, 'wb') as f:
            pickle.dump(ret, f)
    else:
        print('Loading %s' % fn)
        with open('%s.pkl' % fn, 'rb') as f:
            ret = pickle.load(f)
    plot_func(ret, fn,
              m1, m2, m3, a0, a2, e2, I,
              getter_kwargs=getter_kwargs,
              plot_slice=np.where(np.logical_and(ret.t > 25, ret.t < 27))[0],
              )
    L, _, s = to_vars(ret.y)
    Lhat = L / np.sqrt(np.sum(L**2, axis=0))
    q_slf = np.degrees(np.arccos(np.dot(Lhat[:, -1], s[:, -1])))
    return q_slf, ret

def plot_many(e=0.001):
    ''' run all the way to merger, measure q_sl at the end and plot '''
    q_slfs = []
    tfs = []
    I_degs = np.linspace(90.05, 90.355, 6)
    for I_deg in I_degs:
        q_slf, ret = run_one(I_deg=I_deg, e=e)
        q_slfs.append(q_slf)
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
    # plot_many()
    # run_one(I_deg=90.355, q_sl0=np.radians(20),
    #         fn_template='1sim_qsl020_%s',
    #         # plot_func=plot_traj_vecs,
    #         )
    run_one(I_deg=90.355,
            # plot_func=plot_traj_vecs,
            )
