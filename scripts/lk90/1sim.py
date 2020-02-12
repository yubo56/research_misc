from utils import *

def run():
    I = np.radians(85)
    e = 0.01

    # getter_kwargs_nogr = {'eps_sl': 1e3}
    # ret = solver(I, e, atol=1e-8, rtol=1e-8, tf=20,
    #              getter_kwargs=getter_kwargs_nogr)
    # plot_traj(ret, '1sim_nogr', getter_kwargs=getter_kwargs_nogr)

    getter_kwargs = {
        'eps_gr': 1e-5,
        'eps_sl': 1e-2,
        # 'kozai': 0,
    }
    ret = solver(I, e, getter=get_dydt_gr, atol=1e-8, rtol=1e-8,
                 getter_kwargs=getter_kwargs,
                 a_f=1e-4,
                 # method='Radau',
                 tf=np.inf)
    plot_traj(ret, '1sim_gr_late',
              # use_start=False,
              use_stride=False,
              getter_kwargs=getter_kwargs,
              )

if __name__ == '__main__':
    run()
