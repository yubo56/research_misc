'''
misc plots
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('lines', lw=3.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
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
    plt.hist(qs, bins=np.linspace(0, 1, 31), color='k')
    plt.xlabel(r'$q$')
    plt.ylabel('Counts')
    plt.savefig('3qhist', dpi=300)
    plt.close()

if __name__ == '__main__':
    plot_LIGOO3a_qhist()
