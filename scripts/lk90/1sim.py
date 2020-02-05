from utils import *

def run():
    I = np.radians(85)
    e = 0.01

    # ret = solver(I, e, atol=1e-9, rtol=1e-9, tf=50, method='BDF')
    # plot_traj(ret, '1sim_nogr')

    ret = solver(I, e, getter=get_dydt_gr, atol=1e-10, rtol=1e-10,
                 getter_kwargs={
                     'eps_gr': 1e-6,
                     'eps_sl': 3e-10,
                     # 'kozai': 0,
                 },
                 a_f=1e-9,
                 method='Radau',
                 tf=np.inf)
    plot_traj(ret, '1sim_gr_late',
              use_start=False,
              use_stride=False,
              )

if __name__ == '__main__':
    run()
