import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('lines', lw=3.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

import pandas as pd
if __name__ == '__main__':
    # params of J0045-7319
    teff = 2.4 # 10^4 K
    l = 1.2e4 # solar luminosity
    terr = 0.1
    tsun = 0.5777 # 10^4 K
    rsun = 0.696 # 1e11 cm
    rcent = np.sqrt(l / (teff / tsun)**4) * rsun # 1e11 cm
    rerr = np.sqrt(l) / (teff / tsun)**3 * (terr / teff) # 1e11 cm

    dat = pd.read_csv('historyLasts.txt')
    lowx = dat[dat['X'] == 0]
    solars = dat[(dat['X'] == 1) & (dat['star_mass'] > 8)]

    fig = plt.figure()
    plt.semilogy(lowx['effective_T'] / 1e4, 10**(lowx['log_L']), 'bo',
                 label='Low-M')
    plt.semilogy(solars['effective_T'] / 1e4, 10**(solars['log_L']), 'g+',
                 label='Solar')
    plt.errorbar(teff, l, xerr=terr, fmt='rx', lw=1)
    plt.legend()
    plt.xlabel(r'$T_{\rm eff}$ ($10^4$ K)')
    plt.ylabel(r'$L / L_\odot$')
    plt.tight_layout()
    plt.savefig('LT', dpi=400)
    plt.clf()

    plt.plot(lowx['star_mass'], lowx['radius_cm'] / 1e11, 'bo',
             label='Low-M')
    plt.plot(solars['star_mass'], solars['radius_cm'] / 1e11, 'g+',
             label='Solar')
    # overplot theoretical curve R \propto M^{0.78}
    masses = np.array(solars['star_mass'])
    radii = np.array(solars['radius_cm'])
    idx = np.argsort(masses)
    # calibrate to Wikipedia Main_sequence article, Phi Orionis
    plt.plot(masses[idx],
             7.4 * rsun * (masses[idx] / 18)**(0.78),
             'r:',
             lw=1,
             alpha=0.5)
    plt.legend()
    plt.xlabel(r'$M / M_\odot$')
    plt.ylabel(r'$R$ ($10^{11}$ cm)')
    plt.tight_layout()
    # after tight_layout to get right xlims
    xlims = plt.xlim()
    plt.fill_between(xlims,
                     [rcent - rerr] * 2,
                     y2=[rcent + rerr] * 2,
                     color='r',
                     alpha=0.5)
    plt.xlim(*xlims)
    plt.savefig('MR', dpi=400)
    plt.clf()
