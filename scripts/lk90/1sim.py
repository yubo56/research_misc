from utils import *

def run():
    I = np.radians(92)
    e = 0.001

    # param check w/ wolframalpha, right eps_gw
    # m1, m2, m3, a0, a2, e2 = 20, 20, 30, 0.2, 4500, 0
    # param check w/ wolframalpha, right eps_gr
    # m1, m2, m3, a0, a2, e2 = 20, 20, 30, 30, 4500, 0

    m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
    t_lk, getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    print(t_lk, getter_kwargs)
    getter_kwargs['eps_gr'] /= 1e6
    # getter_kwargs = {"eps_gw": 0, "eps_gr": 0, "eps_sl": 0}

    # note that a_f is in units of a0, not in AU!
    ret = solver(I, e,
                 # atol=1e-8, rtol=1e-8,
                 atol=1e-12, rtol=1e-12,
                 getter_kwargs=getter_kwargs,
                 a_f=0.1,
                 tf=3000)
    plot_traj(ret, '1sim',
              use_start=False,
              use_stride=False,
              getter_kwargs=getter_kwargs,
              t_lk=t_lk,
              )

if __name__ == '__main__':
    run()
