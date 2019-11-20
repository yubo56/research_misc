'''
debugging the FFT
'''
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.optimize import bisect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

e = 0.9
N_modes = 998
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
    def f(E):
         return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    def E(M):
         return bisect(lambda E_v: E_v - e * np.sin(E_v) - M, 0, 2 * np.pi)
    f_vals = f(np.array([E(M) for M in m_vals]))
    func = ((1 + e * np.cos(f_vals)) / (1 - e**2))**3 * np.cos(-2 * f_vals)
    return func

def eval_func_complex(m_vals):
    def f(E):
         return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    def E(M):
         return bisect(lambda E_v: E_v - e * np.sin(E_v) - M, 0, 2 * np.pi)
    f_vals = f(np.array([E(M) for M in m_vals]))
    func = ((1 + e * np.cos(f_vals)) / (1 - e**2))**3 * np.exp(-1j * 2 * f_vals)
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

if __name__ == '__main__':
    m_vals = 2 * np.pi * np.arange(N_modes) / N_modes
    check_func_eval(m_vals)
    check_coeff_eval(m_vals)
