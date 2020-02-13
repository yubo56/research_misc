from utils import *

def run():
    I = np.radians(50)
    e = 0.01

    # getter_kwargs_nogr = {'eps_sl': 1e3}
    # ret = solver(I, e, atol=1e-8, rtol=1e-8, tf=20,
    #              getter_kwargs=getter_kwargs_nogr)
    # plot_traj(ret, '1sim_nogr', getter_kwargs=getter_kwargs_nogr)

    m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
    t_lk, getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    print(t_lk, getter_kwargs)
    getter_kwargs = {"eps_gw": 0, "eps_gr": 0, "eps_sl": 0}
    ret = solver(I, e, #atol=1e-8, rtol=1e-8,
                 getter_kwargs=getter_kwargs,
                 a_f=1e-4,
                 tf=10)
    plot_traj(ret, '1sim',
              use_start=False,
              use_stride=False,
              getter_kwargs=getter_kwargs,
              )

if __name__ == '__main__':
    run()
