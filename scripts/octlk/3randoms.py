'''
misc plots
'''
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.0)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

from scipy.interpolate import interp1d

def plot_LIGOO3a_qhist():
    ''' convention: rm'd 2 NS-NS, kept 190814 '''
    dat = [
        [35.7, 30.6],
        [23.3, 13.6],
        [13.7, 7.7],
        [31.0, 20.0],
        [11.0, 7.6],
        [50.7, 34.0],
        [35.1, 23.8],
        [30.6, 25.3],
        [35.4, 26.7],
        [39.7, 29.0],
        [24.5, 18.3],
        [30.0, 8.3],
        [33.4, 23.4],
        [45.4, 30.9],
        [40.6, 31.4],
        [39.5, 31.0],
        [42.9, 28.5],
        [23.0, 12.5],
        [35.3, 18.1],
        [36.9, 27.5],
        [36.4, 24.8],
        [64.5, 39.9],
        [91.4, 66.8],
        [42.1, 32.7],
        [36.2, 22.8],
        [67.2, 47.4],
        [55.4, 35.0],
        [35.0, 23.6],
        [53.6, 40.8],
        [64.0, 38.5],
        [11.5, 8.4],
        [17.5, 13.1],
        [13.3, 7.8],
        [37.2, 28.8],
        [12.2, 8.1],
        [39.3, 28.0],
        [36.1, 26.7],
        [23.2, 2.59],
        [23.8, 10.2],
        [43.5, 35.1],
        [34.9, 24.4],
        [8.8, 5.0],
        [64.7, 25.7]
    ]
    qs = [d[1] / d[0] for d in dat]
    fig = plt.figure(figsize=(6, 4.5))
    plt.hist(qs, bins=np.linspace(0, 1, 31), color='k')
    plt.xlabel(r'$q$')
    plt.yticks([1, 3, 5, 7], labels=['%d' % d for d in [1, 3, 5, 7]])
    plt.xlim(0, 1)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('3qhist', dpi=300)
    plt.close()

def plot_qdists(p=2.35, num_masses=50000):
    fldr = '3plots/'
    os.makedirs(fldr, exist_ok=True)
    lowz_dat = np.array([
        [19.5, 4.0],
        [25.2, 10.7],
        [31.0, 21.3],
        [36.8, 33.0],
        [42.5, 38.1],
        [48.2, 43.2],
        [54.0, 48.3],
        [59.8, 53.4],
        [65.5, 48.5],
        [71.2, 28.8],
        [77.0, 31.1],
        [82.8, 33.8],
        [88.5, 36.6],
        [94.2, 39.2],
        [100.0, 41.9],
        [105.8, 44.8],
        [111.5, 12.5],
        [117.2, 0.1],
    ])
    lowz_premass = lowz_dat[:, 0]
    lowz_postmass = lowz_dat[:, 1]

    highz_dat = np.array([
        [19.5, 3.4],
        [25.2, 8.1],
        [31.0, 10.7],
        [36.8, 9.8],
        [42.5, 10.2],
        [48.2, 12.6],
        [54.0, 14.6],
        [59.8, 16.3],
        [65.5, 18.2],
        [71.2, 20.3],
        [77.0, 21.5],
        [82.8, 22.7],
        [88.5, 23.9],
        [94.2, 24.7],
        [100.0, 25.0],
        [105.8, 23.5],
        [111.5, 22.1],
        [117.2, 20.7],
    ])
    highz_premass = highz_dat[:, 0]
    highz_postmass = highz_dat[:, 1]

    fig = plt.figure(6)
    plt.plot(lowz_premass, lowz_postmass, 'b', label='Low Z')
    plt.plot(highz_premass, highz_postmass, 'k', label='High Z')
    plt.xlabel(r'$M_{\rm ZAMS}$')
    plt.ylabel(r'$M_{\rm rem}$')
    plt.legend(loc='upper left', fontsize=14)
    plt.tight_layout()
    plt.savefig(fldr + 'sne', dpi=300)
    plt.clf()

    mass_min = lowz_premass.min()
    mass_max = lowz_premass.max()
    masses = np.linspace(mass_min, mass_max, num_masses)
    lowz_mass_f = interp1d(lowz_premass, lowz_postmass)
    highz_mass_f = interp1d(highz_premass, highz_postmass)
    def get_mass_inverse_cdf(power=p):
        # generate normalized Salpeter PDF + CDF
        s_pdf = masses**(-power)
        s_pdf[0] = 0 # edge case, cdf needs to be in range [0, 1]
        s_pdf /= np.sum(s_pdf)
        s_cdf = np.cumsum(s_pdf)
        mass_prob_f = interp1d(s_cdf, masses)
        return mass_prob_f
    mass_prob_f = get_mass_inverse_cdf(p)

    m1 = mass_prob_f(np.random.uniform(size=num_masses))
    m2 = mass_prob_f(np.random.uniform(size=num_masses))
    n, bins, _ = plt.hist(m1, bins=30, alpha=0.7, label='M1')
    plt.hist(m2, bins=30, alpha=0.7, label='M2')
    plt.legend(fontsize=14)
    plt.plot(masses, n[0] * (masses / masses[0])**(-p), 'k--')
    plt.savefig(fldr + 'masses', dpi=300)
    plt.clf()

    q = np.maximum(m1 / m2, m2 / m1)
    n, bins, _ = plt.hist(q, bins=30)
    q_arr = np.linspace(q.max(), 1)
    pdist_sp2 = -q_arr**(-p) * ((q_arr / (mass_max / mass_min))**(2 * p -2) - 1)
    plt.plot(q_arr, n[0] * pdist_sp2, 'k--')
    plt.xlabel(r'$q$')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(fldr + 'qdist_salpeter2', dpi=300)
    plt.clf()

    q = np.minimum(m1 / m2, m2 / m1)
    n, bins, _ = plt.hist(q, bins=30)
    q_arr = np.linspace(q.min(), 1)
    pdist_sp = -q_arr**(p - 2) * (((mass_min / mass_max) / q_arr)**(2 * p -2) - 1)
    plt.plot(q_arr, np.mean(n[-3: ]) * pdist_sp, 'k--')
    plt.xlabel(r'$q$')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(fldr + 'qdist_salpeter', dpi=300)
    plt.clf()

    m1_lowz = lowz_mass_f(m1)
    m2_lowz = lowz_mass_f(m2)
    m1_highz = highz_mass_f(m1)
    m2_highz = highz_mass_f(m2)
    q_lowz = np.minimum(m1_lowz / m2_lowz, m2_lowz / m1_lowz)
    q_highz = np.minimum(m1_highz / m2_highz, m2_highz / m1_highz)
    plt.hist(q_lowz, bins=30, alpha=0.7, label='Low Z')
    plt.hist(q_highz, bins=30, alpha=0.7, label='High Z')
    plt.legend(fontsize=14)
    plt.xlabel(r'$q$')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(fldr + 'qdist_salpeter_sne', dpi=300)
    plt.clf()

    # this part is complicated, idk how to improve it
    qs_uniform = np.linspace(0.2, 1, 80, endpoint=False)
    q_lowz = []
    q_highz = []
    weights = []
    m_pdf = masses**(-p)
    m_pdf /= np.sum(m_pdf)
    m_pdf_fun = interp1d(masses, m_pdf)
    for q in qs_uniform:
        m2s = np.linspace(lowz_premass.min(),
                          (1 - 1e-10) * q * lowz_premass.max())
        m1s = m2s / q

        weights.extend(m_pdf_fun(m1s) * m_pdf_fun(m2s))
        q_lowzs = lowz_mass_f(m2s) / lowz_mass_f(m1s)
        q_lowz.extend(np.minimum(q_lowzs, 1 / q_lowzs))
        q_highzs = highz_mass_f(m2s) / highz_mass_f(m1s)
        q_highz.extend(np.minimum(q_highzs, 1 / q_highzs))
    weights = np.array(weights)
    weights /= np.sum(weights)
    plt.hist(q_lowz, bins=30, alpha=0.7, label='Low Z', weights=weights)
    plt.hist(q_highz, bins=30, alpha=0.7, label='High Z', weights=weights)
    plt.legend(fontsize=14)
    plt.xlabel(r'$q$')
    plt.ylabel('Probability')
    plt.tight_layout()
    plt.savefig(fldr + 'qdist_uniform', dpi=300)
    plt.clf()

    # log uniform = uniform grid in log q
    qs_logu = np.exp(np.linspace(np.log(0.2), np.log(1), 80, endpoint=False))
    q_lowz = []
    q_highz = []
    weights = []
    for q in qs_logu:
        m2s = np.linspace(lowz_premass.min(),
                          (1 - 1e-10) * q * lowz_premass.max())
        m1s = m2s / q

        weights.extend(m_pdf_fun(m1s) * m_pdf_fun(m2s))
        q_lowzs = lowz_mass_f(m2s) / lowz_mass_f(m1s)
        q_lowz.extend(np.minimum(q_lowzs, 1 / q_lowzs))
        q_highzs = highz_mass_f(m2s) / highz_mass_f(m1s)
        q_highz.extend(np.minimum(q_highzs, 1 / q_highzs))
    weights = np.array(weights)
    weights /= np.sum(weights)
    plt.hist(q_lowz, bins=30, alpha=0.7, label='Low Z', weights=weights)
    plt.hist(q_highz, bins=30, alpha=0.7, label='High Z', weights=weights)
    plt.legend(fontsize=14)
    plt.xlabel(r'$q$')
    plt.ylabel('Probability')
    plt.tight_layout()
    plt.savefig(fldr + 'qdist_logu', dpi=300)
    plt.clf()

if __name__ == '__main__':
    # plot_LIGOO3a_qhist()
    plot_qdists()
