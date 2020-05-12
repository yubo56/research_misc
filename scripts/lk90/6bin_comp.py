import os
import pickle
import numpy as np
from multiprocessing import Pool
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    plt.rc('lines', lw=3.5)
    plt.rc('xtick', direction='in', top=True, bottom=True)
    plt.rc('ytick', direction='in', left=True, right=True)
except ModuleNotFoundError:
    # on exo4, don't plot
    pass

from utils import *
from scipy import optimize as opt
from scipy.fft import dct, idct, fft
from scipy.interpolate import interp1d

N_THREADS = 60

# values from LL17
I_degs, qslfs = np.array([
    [2.56256, 1.7252483882292753],
    [6.27696, 4.219346735062742],
    [8.50227, 5.714295600421515],
    [10.2583, 6.889823238328885],
    [11.7572, 7.882902860872241],
    [13.0876, 8.768541748337839],
    [14.2967, 9.561690155797915],
    [15.413, 10.294153608904013],
    [16.4555, 10.974424960731971],
    [17.4374, 11.618653204432887],
    [18.3684, 12.233997148224166],
    [19.2559, 12.806922324419485],
    [20.1057, 13.345866587796168],
    [20.9224, 13.886113226171306],
    [21.7097, 14.393185551471028],
    [22.4707, 14.863930989311877],
    [23.208, 15.329518004783656],
    [23.9239, 15.784049278144359],
    [24.6201, 16.238374048642388],
    [25.2983, 16.645468320101816],
    [25.96, 17.06291601898261],
    [26.6063, 17.47881954439917],
    [27.2384, 17.858193978840973],
    [27.8572, 18.248788378838412],
    [28.4636, 18.60412311698293],
    [29.0584, 18.966264862282276],
    [29.6423, 19.31947911065476],
    [30.2159, 19.666508121348823],
    [30.7798, 20.00196831978425],
    [31.3346, 20.332855388775332],
    [31.8807, 20.659307418597844],
    [32.4185, 20.97444260744041],
    [32.9485, 21.28573971274869],
    [33.471, 21.607883434182124],
    [33.9864, 21.90125625559918],
    [34.4951, 22.198796063601794],
    [34.9972, 22.465712617203586],
    [35.4931, 22.74704381726423],
    [35.9831, 23.037281875613328],
    [36.4674, 23.31226905287581],
    [36.9462, 23.579764470549204],
    [37.4198, 23.836418437258228],
    [37.8883, 24.084015402991593],
    [38.3519, 24.34990405510149],
    [38.8108, 24.577806017772314],
    [39.2652, 24.838804726246455],
    [39.7152, 25.079539291507704],
    [40.161, 25.301853666047922],
    [40.6028, 25.54461194897818],
    [41.0406, 25.770600884563468],
    [41.4745, 25.99169281014202],
    [41.9049, 26.191831051413015],
    [42.3316, 26.41233859139507],
    [42.7549, 26.617302273976662],
    [43.1748, 26.83851455037676],
    [43.5914, 27.046153417059283],
    [44.8228, 27.62317940325634],
    [44.4154, 27.43500316100708],
    [44.0049, 27.24005739420256],
    [45.6291, 28.00710780193301],
    [45.2274, 27.807445764722118],
    [46.8182, 28.52238031320687],
    [46.4245, 28.36620270549223],
    [46.0281, 28.186761892555182],
    [47.9846, 29.025604609713902],
    [47.5982, 28.877447869828615],
    [47.2095, 28.7020525501466],
    [48.7504, 29.360774734948215],
    [48.3687, 29.20415642077577],
    [49.8827, 29.799563658558288],
    [49.5074, 29.6546479757655],
    [49.13, 29.49894759759345],
    [50.9963, 30.23130534893325],
    [50.6271, 30.07962678703457],
    [50.2559, 29.94071937157127],
    [51.3637, 30.348475891120582],
    [51.7291, 30.49471280233404],
    [52.0928, 30.606737641001907],
    [52.4546, 30.74557794456809],
    [52.8147, 30.860565060722728],
    [53.1731, 30.96659671578986],
    [53.5298, 31.079717197325635],
    [53.8848, 31.195092858120418],
    [54.2383, 29.637397006941473],
    [54.5902, 30.44307677248422],
    [54.9406, 25.023109810488585],
    [55.2895, 40.87807924384774],
    [55.6369, 83.52570783299788],
    [55.9829, 85.02647348832225],
    [56.6707, 145.19127222069557],
    [56.3275, 90.31947559249976],
    [57.6924, 137.66042112068283],
    [57.3531, 139.29976846705426],
    [57.0126, 126.49773854177914],
    [58.7026, 134.7529292928004],
    [58.3671, 97.39666938235739],
    [58.0303, 128.51649899685725],
    [59.7022, 144.93953459485954],
    [59.3701, 127.8524730046811],
    [59.037, 80.81263210410096],
    [60.6916, 146.5783829604984],
    [60.3629, 67.91426646250957],
    [60.0331, 95.055832717768],
    [61.9962, 84.07011700418825],
    [61.6716, 59.34798137995333],
    [61.3459, 91.84338928334161],
    [61.0193, 60.81520573217315],
    [62.9643, 26.135982505494262],
    [62.6426, 60.88243957961385],
    [62.3199, 143.46346296856933],
    [63.9242, 41.1192715733193],
    [63.6051, 41.61798097400314],
    [63.2852, 47.62168093664826],
    [64.8762, 31.618740073541627],
    [64.5597, 124.43075690122289],
    [64.2424, 118.77249213235746],
    [65.8209, 14.876466254663422],
    [65.5068, 98.24414685021905],
    [65.1919, 18.108569375804496],
    [66.7587, 63.41583008110677],
    [66.4468, 77.89171982116181],
    [66.1342, 73.28475693173515],
    [67.9989, 78.65663359939501],
    [67.6899, 87.54623865558887],
    [67.3802, 39.66501207769029],
    [67.0698, 47.887128925081754],
    [68.9219, 38.802480572364175],
    [68.6149, 65.71763308341303],
    [68.3072, 36.66466966881367],
    [69.8393, 31.85147368683731],
    [69.5341, 79.13627856488662],
    [69.2283, 34.117506524236916],
    [70.7513, 59.68329227220516],
    [70.4478, 27.051841283480332],
    [70.1438, 59.45064983569847],
    [71.0541, 78.89022136244995],
    [71.3564, 96.90202942622125],
    [71.6582, 32.148136154272045],
    [71.9595, 38.11711864145912],
    [72.2602, 75.01262165127477],
    [72.5604, 44.78890804207723],
    [72.8602, 85.5401108487261],
    [73.1594, 44.65409920822142],
    [73.4582, 22.747525163104854],
    [73.7565, 89.06322248629785],
    [74.0544, 60.20586407455485],
    [74.3518, 46.815979817741685],
    [74.6488, 46.454420525950994],
    [74.9454, 40.73972133896699],
    [75.2415, 30.059207508313524],
    [75.5373, 43.979119194771705],
    [75.8326, 48.66783269712936],
    [76.1276, 43.53508566110924],
    [76.4222, 31.408466199538754],
    [76.7165, 40.22029164715755],
    [77.0104, 38.38531013738017],
    [77.3039, 38.82590772618537],
    [77.5971, 36.34016189052759],
    [77.89, 49.46301755718182],
    [78.1825, 40.62900482510828],
    [78.4747, 41.38372048774677],
    [78.7667, 38.5336906602654],
    [79.0583, 34.73870971771819],
    [79.3497, 42.62596032537836],
    [79.6407, 27.56802595461914],
    [79.9315, 37.644472320508285],
    [80.2221, 139.08471666851267],
    [80.5124, 25.144235546279905],
    [80.8024, 36.306271005369496],
    [81.0922, 25.0980798853134],
    [81.3818, 40.63730520947132],
    [81.6711, 27.44881416901452],
    [81.9603, 59.577812076414524],
    [82.2492, 92.25895385401746],
    [82.5379, 132.06925390242964],
    [82.8265, 24.711660833684647],
    [83.1148, 25.53662202266227],
    [83.403, 27.10360221958225],
    [83.691, 29.652423137861422],
    [83.9789, 23.53566486625087],
    [84.2666, 30.05786470677312],
    [84.5541, 20.57773362801833],
    [84.8416, 27.927430440510843],
    [85.1289, 25.43719055715833],
    [85.416, 20.141136159411474],
    [85.7031, 23.31069802380958],
    [85.99, 22.532839645166955],
    [86.2769, 20.6003875327905],
    [86.5636, 17.84622540743524],
    [86.8503, 15.801815373692579],
    [87.1369, 16.422141762489897],
    [87.4234, 15.875252222183692],
    [87.7099, 14.929237753594805],
    [87.9962, 13.782081059664321],
    [88.2826, 12.544653940461295],
    [88.5689, 11.363385558907716],
    [88.8552, 10.472269076541693],
    [89.1414, 8.98834768397717],
    [89.4276, 7.825578533517066],
    [89.7138, 6.627856862731721],
    [90., 5.380327703228222],
    [90.2862, 4.0937047055778635],
    [90.5724, 2.7776260060723876],
    [90.8586, 1.4415757180112996],
    [91.1448, 0.21376805897279916],
    [91.4311, 1.327228990512798],
    [91.7174, 2.7042793900851163],
    [92.0038, 4.0918215332910535],
    [92.2901, 5.480373169175907],
    [92.5766, 6.86699362762936],
    [92.8631, 8.246299951705598],
    [93.1497, 9.612958584292759],
    [93.4364, 10.96299965043906],
    [93.7231, 12.288873200067956],
    [94.01, 13.586974902912514],
    [94.8711, 17.340207022920982],
    [94.584, 16.063077419783223],
    [94.2969, 14.845933125263006],
    [95.7334, 20.858925496025037],
    [95.4459, 19.678334204019098],
    [95.1584, 18.506323958177212],
    [96.8852, 23.311575725939264],
    [96.597, 23.406988065350486],
    [96.309, 22.857081740047057],
    [96.0211, 21.943578447741206],
    [97.7508, 28.04976797202689],
    [97.4621, 27.093194112673473],
    [97.1735, 25.46255022788396],
    [98.9078, 29.942644892740713],
    [98.6182, 30.15846668175505],
    [98.3289, 27.711260287072985],
    [98.0397, 27.844465476290946],
    [99.7779, 30.13566945945331],
    [99.4876, 30.469938257601626],
    [99.1976, 29.41628271806084],
    [100.942, 47.40403681330354],
    [100.65, 56.840496952746044],
    [100.359, 39.09942105901899],
    [100.068, 31.237544141067033],
    [101.817, 46.98352381621891],
    [101.525, 27.439044244213637],
    [101.233, 142.39693728692862],
    [102.99, 62.80552057956492],
    [102.696, 30.250464419628134],
    [102.403, 64.12783941038563],
    [102.11, 28.203882941658176],
    [103.872, 24.45009170987815],
    [103.578, 28.745764334030287],
    [103.284, 56.14815555612683],
    [104.758, 47.487214341465226],
    [104.463, 37.44738893053227],
    [104.167, 27.288476782067775],
    [105.946, 42.81683931867057],
    [105.648, 73.97389872223921],
    [105.351, 30.721941070582325],
    [105.055, 44.512570551897994],
    [106.841, 47.44326085048602],
    [106.542, 55.749019372005364],
    [106.243, 34.298156304396414],
    [107.74, 70.28190923033993],
    [107.44, 38.664868000433515],
    [107.14, 55.216962062912685],
    [108.946, 41.41327943127756],
    [108.644, 49.70371882682854],
    [108.342, 44.69812217314874],
    [108.041, 30.204895032001783],
    [109.856, 90.36960676306364],
    [109.552, 31.348443828193727],
    [109.249, 98.4289676236735],
    [110.772, 38.578108784211935],
    [110.466, 52.35793381143262],
    [110.161, 41.46519888981029],
    [111.693, 25.59754938759885],
    [111.385, 64.3025657358637],
    [111.078, 80.90405606705683],
    [112.93, 17.678711128943807],
    [112.62, 42.393423850685366],
    [112.31, 81.41499614883229],
    [112.001, 71.65509509910824],
    [113.866, 48.51600214211444],
    [113.553, 40.19070033495824],
    [113.241, 116.00123972410242],
    [114.808, 97.71714922600678],
    [114.493, 39.97441723357446],
    [114.179, 54.97016849726552],
    [115.758, 120.43835646282125],
    [115.44, 23.402764644373438],
    [115.124, 89.20155516844346],
    [116.076, 27.257716650468527],
    [116.395, 51.44517836544229],
    [116.715, 115.40187137798698],
    [117.036, 153.6978080151301],
    [117.357, 28.80382104352124],
    [117.68, 97.0906695618789],
    [118.004, 51.53728954875977],
    [118.328, 3.7756760070066084],
    [118.654, 67.14061715995115],
    [118.981, 13.71252898260946],
    [119.308, 164.5083555853703],
    [119.637, 16.6745866856939],
    [119.967, 51.02775875851736],
    [120.298, 153.74984265154114],
    [120.63, 52.674463354720515],
    [120.963, 77.55906547192663],
    [121.297, 66.60770785445337],
    [121.633, 118.960343007177],
    [121.97, 126.06894596514381],
    [122.308, 97.27348886334396],
    [122.647, 55.18278973047205],
    [122.987, 61.159030808074995],
    [123.329, 104.16388702432326],
    [123.672, 147.65494836934033],
    [124.017, 146.39035285096816],
    [124.363, 138.06535903872472],
    [124.71, 134.3555836993681],
    [125.059, 139.83979913703436],
    [125.41, 144.4137335196038],
    [125.762, 144.7240434936241],
    [126.115, 78.39712399828363],
    [126.47, 16.77147967274003],
    [126.827, 32.67281024055087],
    [127.185, 33.4872388372461],
    [127.545, 35.55823378088561],
    [127.907, 35.40660952546048],
    [128.271, 35.280116498664334],
    [128.636, 35.131516519595785],
    [129.004, 34.99439532199924],
    [129.373, 34.829637153242444],
    [129.744, 34.67764069769599],
    [130.117, 34.53955870778948],
    [130.493, 34.3642116840548],
    [130.87, 34.20066981438677],
    [131.25, 34.02294594750419],
    [131.631, 33.84941220575299],
    [132.015, 33.671028268942216],
    [132.402, 33.50180163832419],
    [132.791, 33.304246551567424],
    [133.182, 33.12038960854618],
    [133.576, 32.918682072954496],
    [133.972, 32.73283395879123],
    [134.773, 32.324080284552664],
    [134.371, 32.524212162999255],
    [135.995, 31.661809960900328],
    [135.585, 31.881322457074372],
    [135.177, 32.11731838365198],
    [136.825, 31.214317804661036],
    [136.409, 31.449221123385243],
    [137.668, 30.75048333496819],
    [137.245, 30.98388997558581],
    [138.959, 29.996549161729842],
    [138.525, 30.253207213551057],
    [138.095, 30.5079281153938],
    [139.839, 29.497792825129864],
    [139.397, 29.738565937661846],
    [140.735, 28.956274248364778],
    [140.285, 29.21922944455418],
    [141.648, 28.394867500467218],
    [141.189, 28.665326631153526],
    [142.58, 27.799104290321445],
    [142.112, 28.113463720943397],
    [143.533, 27.19649658892829],
    [143.054, 27.50116143289269],
    [144.507, 26.570091501131643],
    [144.017, 26.90616484305414],
    [145.505, 25.926589375106687],
    [145.003, 26.25509190137568],
    [146.529, 25.254427234220163],
    [146.014, 25.58053295614838],
    [147.582, 24.55066577916081],
    [147.052, 24.888531044973536],
    [148.665, 23.812168468388705],
    [148.119, 24.166581309588572],
    [149.784, 23.026497095554546],
    [149.22, 23.42272691180025],
    [150.942, 22.20977764707484],
    [150.358, 22.63340462165309],
    [151.536, 21.806040339641253],
    [152.762, 20.93374165705295],
    [152.143, 21.380311819607716],
    [153.394, 20.47671444084731],
    [154.702, 19.520251032023204],
    [154.04, 20.00508996581508],
    [155.38, 19.03513530516789],
    [156.792, 17.990017094296807],
    [156.076, 18.519078261712203],
    [157.529, 17.44858488677947],
    [158.29, 16.88453743176593],
    [159.894, 15.689040961052076],
    [159.078, 16.29231458316241],
    [160.744, 15.032179204539307],
    [161.632, 14.363109693055508],
    [162.563, 13.665377643611935],
    [163.544, 12.91587552491879],
    [164.587, 12.105736938205606],
    [165.703, 11.250682905410434],
    [166.912, 10.309768253659646],
    [168.243, 9.268761559795683],
    [169.742, 8.095418940391253],
    [171.498, 6.719842354614838],
    [173.723, 4.9716864134596515],
    [177.437, 2.0331217984109426],
]).T

def plot_anal():
    ''' plot Fig 4 of LL 17 w/ updated trend line '''
    plt.plot(I_degs, qslfs, 'bo', ms=1.0, label='Data')
    # analytic calculation
    I_rads = np.radians(I_degs)
    A = 0.37 # = WdS / WL (WdS = WSL)
    eta = np.sqrt(2 * 0.1) / np.sqrt(3 * 3)

    # LL17 estimate: units of m / 30M, a / AU, G = 1
    Lout_mag = 2/3 * np.sqrt(3 * 3)
    L_mag = 1/2 * np.sqrt(2 * 0.1)
    Lout_hat = np.array([
        np.zeros_like(I_rads),
        np.zeros_like(I_rads),
        np.ones_like(I_rads),
    ])
    L_hat = np.array([
        np.sin(I_rads),
        np.zeros_like(I_rads),
        np.cos(I_rads),
    ])
    Jtot = Lout_mag * Lout_hat + L_mag * L_hat
    # Jtot = Lout_mag * Lout_hat
    J_mag = np.sqrt(np.sum(Jtot**2, axis=0))
    J_hat = Jtot / J_mag
    WLp_mult = J_mag / Lout_mag
    Weff = A * L_hat / np.cos(I_rads) + WLp_mult * J_hat
    Weff_hat = Weff / np.sqrt(np.sum(Weff**2, axis=0))
    qeff_S1 = np.arccos(np.abs(ts_dot(L_hat, Weff_hat))) # L_hat = initial S
    plt.plot(I_degs, np.degrees(qeff_S1), 'r', lw=1.3, alpha=0.5, label='LL17')

    # my estimate (TODO jhat should be averaged too...)
    m1, m2, m3, a0, a2, e2 = 30, 30, 30, 0.1, 3, 0
    getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    qeff_my = []
    for I, lhat, jhat, qeffs1, wlpmult in\
            zip(I_rads, L_hat.T, J_hat.T, qeff_S1, WLp_mult):
        (dWz, dWx), (dWSLz, dWSLx) = get_dWjhat(
            0.001, I, Lout_mag, L_mag, **getter_kwargs)
        Weff_my = np.array([dWSLx - dWx, 0, dWSLz - dWz])
        Weff_my_hat = Weff_my / np.sqrt(np.sum(Weff_my**2, axis=0))
        qeff_I = np.arccos(np.abs(np.dot(lhat, Weff_my_hat)))
        qeff_my.append(qeff_I)
        print(np.degrees(I), np.degrees(qeff_I), np.degrees(qeffs1))
    qeff_my = np.array(qeff_my)
    plt.plot(I_degs, np.degrees(qeff_my), 'g', lw=1.3, alpha=0.5, label='YS')

    plt.xticks([0, 45, 90, 135, 180],
               labels=[r'$0$', r'$45$', r'$90$', r'$135$', r'$180$'])
    plt.xlabel(r'$I_0$ (Deg)')
    plt.ylabel(r'$\theta_{\rm SL,f}$')
    plt.axvline(126.78, c='k', ls=':', lw=1)
    plt.axvline(180 - 126.78, c='k', ls=':', lw=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('6bin_comp', dpi=200)
    plt.clf()

    # plot zoomed in plot w/ manual offset for intuition
    dIlim = 1.1448
    mid_idx = np.where(np.abs(I_degs - 90) < 20)[0]
    plt.plot(I_degs[mid_idx] - dIlim, qslfs[mid_idx], 'bo',
             ms=1.0, label='Data (Offset)')
    plt.plot(I_degs[mid_idx], np.degrees(qeff_S1[mid_idx]), 'r',
             lw=1.3, alpha=0.5, label='LL17')
    plt.plot(I_degs[mid_idx], np.degrees(qeff_my[mid_idx]), 'g',
             lw=1.3, alpha=0.5, label='YS')
    plt.xlabel(r'$I_0$ (Deg)')
    plt.ylabel(r'$\theta_{\rm SL,f}$')
    plt.axvline(102.6, c='k', ls=':', lw=1)
    plt.axvline(180 - 102.6, c='k', ls=':', lw=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('6bin_comp_zoom', dpi=200)
    plt.clf()

def Icrit_test():
    # m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
    m1, m2, m3, a0, a2, e2 = 30, 30, 30, 0.1, 3, 0
    m12 = m1 + m2
    m123 = m12 + m3
    mu = m1 * m2 / m12
    mu123 = m12 * m3 / m123
    eta = mu / mu123 * np.sqrt(m12 * a0 / (m123 * a2 * (1 - e2**2)))

    eps_gr = 3 * G * m12**2 * a2**3 * (1 - e2**2)**(3/2) / (a0**4 * m3)
    def get_Ilim():
        def f(j):
            return (
                3/8 * (j**2 - 1) * (
                    - 3 + eta**2 / 4 * (4 / 5 * j**2 - 1))
                + eps_gr * (1 - 1 / j))
        jlim = opt.brenth(f, 1e-10, 1 - 1e-10)
        return np.arccos(eta / 2 * (4 * jlim**2 / 5 - 1))
    print(np.degrees(get_Ilim())) # should be 92.16

TOY_FOLDER = '6toy/'
mkdirp(TOY_FOLDER)
def single_cycle_toy(getter_kwargs, e0=1e-3, I0=np.radians(95), intg_pts=int(3e5)):
    '''
    for far-out system params, solve toy problem in corotating frame
    '''
    eps_sl = getter_kwargs['eps_sl']
    eps_gr = getter_kwargs['eps_gr']
    a = 1
    W0 = 0
    w0 = 0

    def dydt(t, y):
        e, W, I, w = y
        x = 1 - e**2
        dedt = (
            15 * a**(3/2) * e * np.sqrt(x) * np.sin(2 * w)
                    * np.sin(I)**2 / 8
        )
        dWdt = (
            3 * a**(3/2) * np.cos(I) *
                    (5 * e**2 * np.cos(w)**2 - 4 * e**2 - 1)
                / (4 * np.sqrt(x))
        )
        dIdt = (
            -15 * a**(3/2) * e**2 * np.sin(2 * w)
                * np.sin(2 * I) / (16 * np.sqrt(x))
        )
        dwdt = (
            3 * a**(3/2)
                * (2 * x + 5 * np.sin(w)**2 * (e**2 - np.sin(I)**2))
                / (4 * np.sqrt(x))
            + eps_gr / (a**(5/2) * x)
        )
        return (dedt, dWdt, dIdt, dwdt)
    def start_event(t, y):
        return y[3] - np.pi / 2
    def term_event(t, y): # one Kozai cycle, starting from emax
        return y[3] - 3 * np.pi / 2
    term_event.terminal = True
    # use implicit method to ensure symmetry about half-period
    ret = solve_ivp(dydt, (0, np.inf), [e0, W0, I0, w0],
                    events=[start_event, term_event], atol=1e-10, rtol=1e-10,
                    method='Radau', dense_output=True)
    times = np.linspace(ret.t_events[0][0], ret.t[-1], intg_pts)
    # symmetrize for exact symmetry, time-shift to drop dominant phase shift
    def symmetrize(x):
        return (x + np.flip(x, axis=-1)) / 2
    e_arr, W_arr, I_arr, w_arr = ret.sol(times)
    e_arr = symmetrize(e_arr)
    I_arr = symmetrize(I_arr)
    x_arr = 1 - e_arr**2

    # compute W0_vec, WSL_vec, Weff_vec
    W0_vec = np.outer(
        np.array([0, 0, 1]),
        (
            3 * a**(3/2) * np.cos(I_arr) *
                    (5 * e_arr**2 * np.cos(w_arr)**2 - 4 * e_arr**2 - 1)
                / (4 * np.sqrt(x_arr))
        ))
    WSL_vec = np.array([
        eps_sl / (a**(5/2) * x_arr) * np.sin(I_arr),
        np.zeros(intg_pts),
        eps_sl / (a**(5/2) * x_arr) * np.cos(I_arr)])
    Weff_vec = WSL_vec - W0_vec
    return Weff_vec, times

def plot_weff_fft(Weff_vec, times, fn='6_vecfft', plot=True):
    # cosine transform
    x_coeffs = dct(Weff_vec[0], type=1)[::2] / (2 * len(times))
    z_coeffs = dct(Weff_vec[2], type=1)[::2] / (2 * len(times))
    # print(np.mean(Weff_vec[2]), z_coeffs[0]) # are equal
    # print(np.mean(Weff_vec[0]), x_coeffs[0]) # are equal

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
        # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 7))
        # ax3.semilogy(times, -Weff_vec[2], 'k', lw=1, label=r'$-z$')
        # ax3.semilogy(times, Weff_vec[0], 'b', lw=1, label=r'$x$')
        # ax3.legend(fontsize=12)
        # ax3.set_ylabel(r'$\Omega_{\rm eff}$')

        N = np.arange(len(z_coeffs))
        ax1.semilogy(N, x_coeffs, 'bo', ms=1)
        ax1.semilogy(N, -x_coeffs, 'ro', ms=1)
        # ax2.semilogy(N, z_coeffs, 'bo', ms=1)
        ax2.semilogy(N, -z_coeffs, 'ro', ms=1)
        ax1.set_ylabel(r'$\tilde{\Omega}_{\rm eff, x, N}$ (Rad / $t_{\rm LK,0}$)')
        ax2.set_ylabel(r'$\tilde{\Omega}_{\rm eff, z, N}$ (Rad / $t_{\rm LK,0}$)')
        ax2.set_xlabel(r'$N$')

        min_fact = 1e-6
        Nmax = np.where(abs(z_coeffs) / np.max(abs(z_coeffs)) < min_fact)[0][0]
        ax1.set_ylim(bottom=np.abs(x_coeffs).max() * min_fact / 3)
        ax2.set_ylim(bottom=np.abs(z_coeffs).max() * min_fact / 3)
        ax1.set_xlim((0, Nmax))
        ax2.set_xlim((0, Nmax))

        plt.tight_layout()
        plt.savefig(TOY_FOLDER + fn, dpi=200)
        plt.clf()

    return x_coeffs, z_coeffs

def get_amp(Weff_vec, t_vals, fn='6_devs', plot=False, q0=np.pi / 2, phi0=0,
            num_periods=100, tol=1e-6):
    '''
    run for num_periods, and at each period, compute theta_eff (s dot Weff)
    '''
    t0 = t_vals[0]
    tf = t_vals[-1]
    period = tf - t0
    Weff_x_interp = interp1d(t_vals, Weff_vec[0])
    Weff_z_interp = interp1d(t_vals, Weff_vec[2])
    Weff_x_mean = np.mean(Weff_vec[0])
    Weff_z_mean = np.mean(Weff_vec[2])
    def dydt(t, s):
        t_curr = (t - t0) % period + t0
        return np.cross(
            [Weff_x_interp(t_curr), 0, Weff_z_interp(t_curr)],
            s)

    def period_event(t, y):
        # want to transition continuously across (t - t0) % period = 0
        return (t - t0 + period / 2) % period - period / 2
    period_event.direction = +1
    s0 = [np.sin(q0) * np.cos(phi0), np.sin(q0) * np.sin(phi0), np.cos(q0)]
    ret = solve_ivp(dydt, (t0, t0 + num_periods * period), s0,
                    events=[period_event], dense_output=True,
                    atol=tol, rtol=tol, method='Radau')
    period_times = ret.t_events[0]
    period_spins = ret.sol(period_times)
    q_effs = ts_dot(period_spins,
                    [np.full_like(period_times, Weff_x_mean),
                     np.zeros_like(period_times),
                     np.full_like(period_times, Weff_z_mean)])
    if plot:
        plt.plot(period_times, np.degrees(q_effs), 'bo')

        plt.tight_layout()
        plt.savefig(TOY_FOLDER + fn, dpi=200)
        plt.close()

    return np.degrees(np.max(q_effs) - np.min(q_effs))

def get_devs(getter_kwargs, intg_pts=int(1e5), configs=[], outfile='',
             **kwargs):
    '''
    For various e0, I0, run for isotropic distribution, get difference between
    averaged + non-averaged (RMS diff), plot

    Also, calulate "epsilon" (N = 1) / (N = 0) magnitudes for each
    '''

    n_pts = 30
    mus_edges = np.linspace(-1, 1, n_pts + 1)
    phis_edges = np.linspace(0, 2 * np.pi, n_pts + 1)
    mus = (mus_edges[ :-1] + mus_edges[1: ]) / 2
    phis = (phis_edges[ :-1] + phis_edges[1: ]) / 2
    def inner_plot(cs):
        cosq_eff = cs[-2]
        res_nparr = np.array(cs[-1])
        phis_grid = np.outer(phis_edges, np.ones_like(mus_edges))
        mus_grid = np.outer(np.ones_like(phis_edges), mus_edges)
        plt.pcolormesh(phis_grid, mus_grid, res_nparr, cmap='viridis')
        plt.plot(0, cosq_eff, 'ro', ms=10)
        plt.colorbar()
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'$\cos \theta$')
        plt.tight_layout()
        plt.savefig(TOY_FOLDER + fn, dpi=200)
        plt.close()

    pkl_fn = TOY_FOLDER + outfile + '.pkl'
    config_stats = [] # store mean, stdev, lk_period, N=0, N=1 coeffs
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        for e0, I0, fn in configs:
            Weff_vec, t_vals = single_cycle_toy(
                getter_kwargs, e0=e0, I0=I0, intg_pts=intg_pts)
            res_grid = []
            for q in np.arccos(mus):
                res = []
                for phi in phis:
                    # in degrees
                    amp = get_amp(Weff_vec, t_vals, q0=q, phi0=phi, **kwargs)
                    print(e0, I0, q, phi, amp)
                    res.append(amp)
                res_grid.append(res)
            res_nparr = np.array(res_grid)

            x_coeffs, z_coeffs = plot_weff_fft(Weff_vec, t_vals, plot=False)
            cosq_eff = z_coeffs[0] / np.sqrt(x_coeffs[0]**2 + z_coeffs[0]**2)
            cs = (
                np.max(res_nparr),
                np.median(res_nparr),
                x_coeffs[ :2],
                z_coeffs[ :2],
                t_vals[-1] - t_vals[0],
                cosq_eff,
                res_nparr)
            config_stats.append(cs)
            if fn is not None:
                inner_plot(cs) # it's okay to plot twice
        with open(pkl_fn, 'wb') as f:
            pickle.dump(config_stats, f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            config_stats = pickle.load(f)
    with open(TOY_FOLDER + outfile, 'w') as f:
        f.write('e, I, max, median, period, (N = 0), (N = 1)\n')
        for c, cs in zip(configs, config_stats):
            outs = ('%s, %s, %s, %s, %s, %s, %s\n' % (
                c[0],
                np.degrees(c[1]),
                cs[0],
                cs[1],
                cs[4],
                (cs[2][0], cs[3][0]),
                (cs[2][1], cs[3][1])))
            print(outs)
            f.write(outs)

            fn = c[2]
            if fn is not None:
                inner_plot(cs)

def Iscan(getter_kwargs, intg_pts=int(1e5), fn='6_Iscan',
          e0=0.003, I_vals=np.radians(np.linspace(95, 135, 1000)), **kwargs):
    amps = []
    for I in I_vals:
        Weff_vec, t_vals = single_cycle_toy(
            getter_kwargs, e0=e0, I0=I, intg_pts=intg_pts)
        amp = get_amp(Weff_vec, t_vals, q0=I, **kwargs)
        print('Ran for I = %.3f, e0=%.3f, amp=%.3f' % (np.degrees(I), e0, amp))
        amps.append(amp)
    plt.plot(np.degrees(I_vals), amps, 'ko', ms=1)
    plt.xlabel(r'$I_0$')
    plt.ylabel(r'$\Delta \theta_{\rm eff}$')
    plt.savefig(TOY_FOLDER + fn, dpi=200)
    plt.close()

def Iscan_N1Component(getter_kwargs, intg_pts=int(1e5), e0=0.003, fn='6_IscanN1',
                      I_vals=np.radians(np.linspace(95, 135, 40)), **kwargs):
    ratios = []
    angles = []
    dWs = []
    for I in I_vals:
        Weff_vec, t_vals = single_cycle_toy(
            getter_kwargs, e0=e0, I0=I, intg_pts=intg_pts)
        x_coeffs, z_coeffs = plot_weff_fft(Weff_vec, t_vals, plot=False)
        print('Ran for I = %.3f, e0=%.3f' % (np.degrees(I), e0))
        get_mag = lambda idx: np.sqrt(x_coeffs[idx]**2 + z_coeffs[idx]**2)
        ratios.append(get_mag(1) / get_mag(0))
        angles.append(
            (x_coeffs[0] * x_coeffs[1] + z_coeffs[0] * z_coeffs[1]) /
            (get_mag(0) * get_mag(1)))
        dWs.append(get_mag(0) * (t_vals[-1] - t_vals[0]) / (2 * np.pi))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
    ax1.plot(np.degrees(I_vals), ratios, 'bo', ms=1)
    ax1.set_ylabel(r'$\Omega_{\rm eff, 1} / \Omega_{\rm eff}$')
    ax2.plot(np.degrees(I_vals), angles, 'bo', ms=1)
    ax2.set_ylabel(r'$\hat{\Omega_{\rm eff, 1}} \cdot \hat{\Omega_{\rm eff}}$')
    ax2.set_xlabel(r'$I_0$')
    ax3.plot(np.degrees(I_vals), dWs, 'ro', ms=1)
    ax3.set_ylabel(r'$\Omega_{\rm eff} P_{\rm LK} / (2 \pi)$')
    plt.tight_layout()
    plt.savefig(TOY_FOLDER + fn, dpi=200)
    plt.close()

def do_getamp(getter_kwargs, e0, I, intg_pts):
    Weff_vec, t_vals = single_cycle_toy(
        getter_kwargs, e0=e0, I0=I, intg_pts=intg_pts)
    amp = get_amp(Weff_vec, t_vals, q0=I)
    print('Ran for I = %.3f, e0=%.3f, amp=%.3f' % (np.degrees(I), e0, amp))
    return amp

def get_905():
    t_arr = []
    e0arr = []
    I0arr = []
    a0arr = []
    with open('4sims/4sim_lk_90_500.pkl', 'rb') as f:
        t, (a, e, W, I, w), t_events = pickle.load(f)

    for ti, tf in zip(t_events[0][ :-1], t_events[0][1: ]):
        where_idx = np.where(np.logical_and(
            t < tf, t > ti))[0]
        min_idx = np.argmin(e[where_idx])
        e0arr.append(e[where_idx][min_idx])
        I0arr.append(I[where_idx][min_idx])
        a0arr.append(a[where_idx][min_idx])
        t_arr.append(t[where_idx][min_idx])
    return t_arr, e0arr, I0arr, a0arr

def Iscan_grid(getter_kwargs,
               intg_pts=int(1e5),
               fn='6_Iscan',
               e_vals=np.geomspace(1e-3, 0.9, 120),
               I_vals=np.radians(np.linspace(90.5, 130, 200)),
               plot_905=False,
               **kwargs):
    pkl_fn = TOY_FOLDER + fn + '.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        args = []
        for e0 in e_vals:
            for I in I_vals:
                args.append((getter_kwargs, e0, I, intg_pts))
        p = Pool(N_THREADS)
        res = p.starmap(do_getamp, args)
        res = np.reshape(res, (len(e_vals), len(I_vals)))
        with open(pkl_fn, 'wb') as f:
            pickle.dump(res, f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            res = pickle.load(f)
    dI = np.mean(np.diff(I_vals))
    de = e_vals[-1] / e_vals[-2]
    I_edges = np.degrees(np.concatenate((I_vals - dI / 2,
                                         [I_vals[-1] + dI / 2])))
    e_edges = np.concatenate((e_vals / np.sqrt(de), [e_vals[-1] * np.sqrt(de)]))
    e_grid = np.outer(e_edges, np.ones_like(I_edges))
    I_grid = np.outer(np.ones_like(e_edges), I_edges)
    plt.pcolormesh(e_grid, I_grid, res)
    plt.xlim(left=1, right=1e-3)
    plt.xscale('log')

    # overplot lines of constant K for e0=1e-3
    e0 = 1e-3
    I_trunc = I_edges[:-1]
    stride = len(I_trunc) // 4
    for I0 in I_trunc[::stride]:
        e_match_sq = 1 - cosd(I0)**2 / cosd(I_trunc)**2 * (1 - e0**2)
        plt.plot(np.sqrt(e_match_sq), I_trunc, 'r:', lw=1)
    # overplot I0=90.5 from simulation
    if plot_905:
        _, e0arr, I0arr, _ = get_905()
        plt.plot(e0arr, np.degrees(I0arr), 'r', lw=2)

    plt.xlabel(r'$e_{\min}$')
    plt.ylabel(r'$I_{\min}$')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(TOY_FOLDER + fn, dpi=200)
    plt.close()

def plot_905(n_pts=50):
    m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
    t_arr, e0arr, I0arr, a0arr = get_905()
    # resample, e0arr and I0arr are very dense at late times
    t_vals = np.linspace(t_arr[0], t_arr[-1], n_pts)
    e0_new = interp1d(t_arr, e0arr)(t_vals)
    I0_new = interp1d(t_arr, I0arr)(t_vals)
    a0_new = interp1d(t_arr, a0arr)(t_vals)

    amps = []
    for e0, I0, a0_new in zip(e0_new, I0_new, a0_new):
        getter_kwargs = get_eps(m1, m2, m3, a0_new * a0, a2, e2)
        amps.append(do_getamp(getter_kwargs, e0, I0, int(1e5)))
        print('90.5sim, Did', e0, I0, a0_new, amps[-1])
    plt.plot(t_vals, amps)
    plt.xlabel(r'$t / t_{\rm LK, 0}$')
    plt.ylabel(r'$\Delta \theta_{\rm eff, \max}$ (Deg)')
    plt.savefig('6toy/6_905', dpi=200)
    plt.close()

if __name__ == '__main__':
    # plot_anal()
    # Icrit_test()

    # m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
    # m1, m2, m3, a0, a2, e2 = 30, 30, 30, 0.1, 3, 0
    # getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    # Weff_vec, t_vals = single_cycle_toy(getter_kwargs, I0=np.radians(90.5))
    # # plot_weff_fft(Weff_vec, t_vals, fn='6_vecfft_90_5')
    # get_amp(Weff_vec, t_vals, plot=True, fn='6_devs90_5')

    # Weff_vec, t_vals = single_cycle_toy(getter_kwargs, e0=0.9, I0=np.radians(90.5))
    # plot_weff_fft(Weff_vec, t_vals, fn='6_vecfft_90_5_highe')
    # get_amp(Weff_vec, t_vals, plot=True, fn='6_devs90_5_highe')

    # Weff_vec, t_vals = single_cycle_toy(getter_kwargs, I0=np.radians(95))
    # plot_weff_fft(Weff_vec, t_vals, fn='6_vecfft_95')
    # get_amp(Weff_vec, t_vals, plot=True, fn='6_devs95')

    # Weff_vec, t_vals = single_cycle_toy(getter_kwargs, I0=np.radians(125))
    # plot_weff_fft(Weff_vec, t_vals, fn='6_vecfft_125')
    # get_amp(Weff_vec, t_vals, plot=True, fn='6_devs125')

    # key takeaways:
    # configs1 = [
    #     (0.9, np.radians(90.5), '6_ampgrid_frac_90_5_p9'),
    #     (0.001, np.radians(90.5), '6_ampgrid_frac_90_5'),
    #     (0.001, np.radians(92.5), '6_ampgrid_frac_92_5'),
    #     (0.001, np.radians(94.5), '6_ampgrid_frac_94_5'),
    #     (0.001, np.radians(96.5), '6_ampgrid_frac_96_5'),
    #     (0.001, np.radians(98.5), '6_ampgrid_frac_98_5'),
    #     (0.001, np.radians(105), '6_ampgrid_frac_105'),
    #     (0.001, np.radians(125), '6_ampgrid_frac_125'),
    #     (0.01, np.radians(92.5), '6_ampgrid_frac_92_5_n2'),
    #     (0.1, np.radians(92.5), '6_ampgrid_frac_92_5_n1'),
    #     (0.3, np.radians(92.5), '6_ampgrid_frac_92_5_p3'),
    # ]
    # get_devs(getter_kwargs, configs=configs1, outfile='6ampgrid_frac.txt',
    #          num_periods=50)
    # configs3 = [
    #     (0.003, np.radians(125), '6_ampgrid_frac_large'),
    # ]
    # get_devs(getter_kwargs, configs=configs3, outfile='6ampgrid_outlarge.txt',
    #          num_periods=50)
    # configs2 = [
    #     (0.003, np.radians(117.543), '6_ampgrid_inner117'),
    #     (0.003, np.radians(125.15), '6_ampgrid_inner125'),
    # ]
    # get_devs(getter_kwargs, configs=configs2, outfile='6ampgridinner_frac.txt',
    #          num_periods=50)

    # Weff_vec, t_vals = single_cycle_toy(getter_kwargs, e0=0.003,
    #                                     I0=np.radians(125.15))
    # plot_weff_fft(Weff_vec, t_vals, fn='6_vecfft_inner125')
    # get_amp(Weff_vec, t_vals, plot=True, fn='6_devs_inner125',
    #         q0=np.radians(125.15),
    #         num_periods=500)
    # Weff_vec, t_vals = single_cycle_toy(getter_kwargs, e0=0.003,
    #                                     I0=np.radians(117.543))
    # plot_weff_fft(Weff_vec, t_vals, fn='6_vecfft_inner117')
    # get_amp(Weff_vec, t_vals, plot=True, fn='6_devs_inner117',
    #         q0=np.radians(117.543),
    #         num_periods=500)
    # Weff_vec, t_vals = single_cycle_toy(getter_kwargs, e0=0.003,
    #                                     I0=np.radians(120))
    # plot_weff_fft(Weff_vec, t_vals, fn='6_vecfft_inner120')
    # get_amp(Weff_vec, t_vals, plot=True, fn='6_devs_inner120',
    #         q0=np.radians(120),
    #         num_periods=500)

    # m1, m2, m3, a0, a2, e2 = 30, 20, 30, 100, 4500, 0
    # getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    # Iscan_N1Component(getter_kwargs, fn='6_IscanN1outer',
    #                   I_vals=np.radians(np.linspace(90.5, 135, 200)))
    # Iscan(getter_kwargs, fn='6_Iscan_out')
    # Iscan(getter_kwargs, fn='6_Iscan_outnarrow',
    #       I_vals=np.radians(np.linspace(90.5, 95, 200)))
    # Iscan_grid(getter_kwargs, fn='6_Iscangrid', plot_905=True)
    plot_905()

    # m1, m2, m3, a0, a2, e2 = 30, 30, 30, 0.1, 3, 0
    # getter_kwargs = get_eps(m1, m2, m3, a0, a2, e2)
    # Iscan_N1Component(getter_kwargs, fn='6_IscanN1inner',
    #                   I_vals=np.radians(np.linspace(90.5, 135, 200)))
    # Iscan(getter_kwargs, fn='6_Iscan')
    # Iscan_grid(getter_kwargs, fn='6_Iscangrid_inner')
