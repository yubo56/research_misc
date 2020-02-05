from utils import *

def run():
    I = np.radians(85)
    e = 0.01

    # ret = solver(I, e, atol=1e-9, rtol=1e-9, tf=50, method='BDF')
    # plot_traj(ret, '1sim_nogr')

    # debug: eps = 0
    # ret = solver(I, e, getter=get_dydt_gr, atol=1e-9, rtol=1e-9,
    #              getter_kwargs={'eps':0, 'delta':0},
    #              tf=50)
    # plot_traj(ret, '1sim_gr')

    # debug: short
    # ret = solver(I, e, getter=get_dydt_gr, atol=1e-9, rtol=1e-9,
    #              getter_kwargs={'eps':5e-10, 'delta':5e-10},
    #              tf=50)
    # plot_traj(ret, '1sim_gr')

    # real? Radau is most accurate
    # 1e-10 tols, a_f = 0.15 takes 250s on YuboDesktop, 1e-9 is 167s
    for idx, a_f in enumerate([1e-1, 5e-2, 2e-2, 1e-2]):
        ret = solver(I, e, getter=get_dydt_gr, atol=1e-9, rtol=1e-9,
                     getter_kwargs={'eps':1e-9, 'delta':1e-9},
                     a_f=a_f,
                     tf=np.inf)
        plot_traj(ret, '1sim_gr_%d' % idx)

if __name__ == '__main__':
    run()
