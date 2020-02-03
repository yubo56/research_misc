from utils import *

def run():
    I = np.radians(85)
    e = 0.01

    # ret = solver(I, e, atol=1e-9, rtol=1e-9)
    # plot_traj(ret, '1sim_nogr')

    ret = solver(I, e, getter=get_dydt_gr,
                 getter_kwargs={'eps':5e-10, 'delta':5e-10},
                 tf=1500)
    plot_traj(ret, '1sim_gr')

if __name__ == '__main__':
    run()
