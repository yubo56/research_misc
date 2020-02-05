from utils import *

def run():
    I = np.radians(85)
    e = 0.01

    # ret = solver(I, e, atol=1e-9, rtol=1e-9, tf=50, method='BDF')
    # plot_traj(ret, '1sim_nogr')

    # debug: eps_gr = 0
    # ret = solver(I, e, getter=get_dydt_gr, atol=1e-9, rtol=1e-9,
    #              getter_kwargs={'eps_gr':0, 'eps_sl':0},
    #              tf=50)
    # plot_traj(ret, '1sim_gr')

    # debug: short
    # ret = solver(I, e, getter=get_dydt_gr, atol=1e-9, rtol=1e-9,
    #              getter_kwargs={'eps_gr':5e-10, 'eps_sl':5e-10},
    #              tf=50)
    # plot_traj(ret, '1sim_gr')

    # real? Radau is most accurate
    # 1e-10 tols, a_f = 0.15 takes 250s on YuboDesktop, 1e-9 is 167s
    # for idx, a_f in enumerate([1e-1, 5e-2, 2e-2, 1e-2]):
    #     ret = solver(I, e, getter=get_dydt_gr, atol=1e-9, rtol=1e-9,
    #                  getter_kwargs={'eps_gr':1e-9, 'eps_sl':1e-9},
    #                  a_f=a_f,
    #                  tf=np.inf)
    #     plot_traj(ret, '1sim_gr_%d' % idx)

    # later times
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
