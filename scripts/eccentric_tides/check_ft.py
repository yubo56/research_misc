'''
debugging the FFT and checking Parseval's theorem
'''
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.optimize import bisect
from scipy.integrate import quad
from scipy.special import gamma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

from total_torque import get_coeffs_fft
import hansens

ecc = 0.9
N_modes = 998
def f(E, e=ecc):
     return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
def E(M, e=ecc):
     return bisect(lambda E_v: E_v - e * np.sin(E_v) - M, 0, 2 * np.pi)

def get_mv_coeffs():
    # Michelle's authoritative source (agrees w/ my manual integration)
    lines = open('ecc09.txt').readlines()
    vals = np.array([[float(word.strip())
                      for word in l.split(',')]
                     for l in lines])
    N_arr = vals[:, 0]
    FN2_arr = vals[:, 1]
    return N_arr[ :N_modes], FN2_arr[ :N_modes]

def eval_func_summed(m_vals):
    N_arr, FN2_arr = get_mv_coeffs()
    func = np.zeros_like(m_vals)
    for N, FN2 in zip(N_arr, FN2_arr):
        func += FN2 * np.cos(-N * m_vals)
    return func

def eval_func_fft(m_vals):
    N_arr, FN2_arr = get_mv_coeffs()
    func = np.real(fft(FN2_arr))
    return func

def eval_func_explicit(m_vals):
    f_vals = f(np.array([E(M) for M in m_vals]))
    func = ((1 + ecc * np.cos(f_vals)) / (1 - ecc**2))**3 * np.cos(-2 * f_vals)
    return func

def eval_func_complex(m_vals):
    f_vals = f(np.array([E(M) for M in m_vals]))
    func = ((1 + ecc * np.cos(f_vals)) / (1 - ecc**2))**3\
        * np.exp(-1j * 2 * f_vals)
    return func

def check_func_eval(m_vals):
    func = eval_func_summed(m_vals)
    funcft = eval_func_fft(m_vals)
    func2 = eval_func_explicit(m_vals)
    plt.plot(m_vals, func2 - func, 'b+', label='Exp-Sum')
    plt.plot(m_vals, funcft - func, 'g+', label='FT-Sum')
    plt.plot(m_vals, func2 - funcft, 'r+', label='Exp-FT')
    # plt.plot(m_vals, func2, label='Explicit')
    # print('mean/stdev of diff:', np.mean(func - func2), np.std(func - func2))
    # print('mean/stdev of fft:', np.mean(func - funcft), np.std(func - funcft))
    plt.legend()
    plt.tight_layout()
    plt.xlim([0, 0.1])
    plt.ylim([-15, 15])
    plt.savefig('check_ft_diffs', dpi=400)
    plt.clf()

    plt.plot(m_vals, funcft, label='FT')
    plt.plot(m_vals, func2, label='Explicit')
    plt.legend()
    plt.tight_layout()
    plt.xlim([0, 0.1])
    plt.savefig('check_ft', dpi=400)
    plt.clf()

def check_coeff_eval(m_vals):
    func2 = eval_func_complex(m_vals)
    N_arr, FN2_arr = get_mv_coeffs()
    func = fft(FN2_arr)
    FN2_ifft = np.real(ifft(func2))
    print(FN2_ifft[ :10])
    # I computed coefficients starting at N = 1, fft gives N = 0
    plt.plot(N_arr, FN2_arr, label='MV')
    plt.plot(N_arr - 1, FN2_ifft, label='ifft')
    plt.legend()
    plt.tight_layout()
    plt.xlim([0, 100])
    plt.savefig('check_fft', dpi=400)
    plt.clf()

def check_parsevals_int(ecc_vals):
    '''
    check agreement of parseval's with time/coefficient sums, as well as
    integral approximation to coefficient sum
    '''
    time_ints = []
    coeff_sums = []
    integ_approxs = []
    for ecc in ecc_vals:
        def f_sq(M): # f_squared to integrate
            ta = f(E(M, e=ecc), e=ecc) # true anomaly
            return ((1 + ecc * np.cos(ta)) / (1 - ecc**2))**6
        integ = quad(f_sq, 0, 2 * np.pi, limit=100)[0] / (2 * np.pi)
        time_int = (1 + 3 * ecc**2 + 3 * ecc**4 / 8) / (1 - ecc**2)**(9/2)

        coeffs_fft = get_coeffs_fft(1000, 2, ecc)
        coeffs_sum = np.sum(coeffs_fft**2)

        n_vals, coeffs_half, _ = hansens.get_coeffs_fft(1000, 2, ecc)
        c, p, a = hansens.fit_powerlaw_hansens(n_vals, coeffs_half)
        integ_approx = c**2 * (a / 2)**(2 * p + 1) * gamma(2 * p + 1)

        time_ints.append(time_int)
        coeff_sums.append(coeffs_sum)
        integ_approxs.append(integ_approx)
        print('Ran for', ecc)
    plt.semilogy(ecc_vals, time_ints, label='T-int')
    plt.semilogy(ecc_vals, coeff_sums, label='W-sum')
    plt.semilogy(ecc_vals, integ_approxs, label='I-approx')

    plt.xlabel(r'$e$')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('check_ft_parsevals', dpi=400)

if __name__ == '__main__':
    # m_vals = 2 * np.pi * np.arange(N_modes) / N_modes
    # check_func_eval(m_vals)
    # check_coeff_eval(m_vals)

    ecc_vals = np.arange(0.1, 0.96, 0.05)
    check_parsevals_int(ecc_vals)
