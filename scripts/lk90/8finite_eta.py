'''
self-contained file for the finite-eta simulations
'''
import os
import gc
import pickle
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    plt.rc('lines', lw=3.5)
    plt.rc('xtick', direction='in', top=True, bottom=True)
    plt.rc('ytick', direction='in', left=True, right=True)
    plt.rc('text.latex', preamble=r'\usepackage{newtxmath}')
except:
    pass

from utils import ts_dot, get_vals, cosd, mkdirp
from multiprocessing import Pool

from scipy.integrate import solve_ivp
from scipy.optimize import brenth
from scipy.interpolate import interp1d
from scipy.fft import dct

params = 30, 20, 30, 100, 4500, 0
params_in = 30, 30, 30, 0.1, 3, 0
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

def get_I1(I0d, eta):
    ''' given total inclination between Lout and L, returns I_tot '''
    I0 = np.radians(I0d)
    def I2_constr(_I2):
        return np.sin(_I2) - eta * np.sin(I0 - _I2)
    I2 = brenth(I2_constr, 0, np.pi, xtol=1e-12)
    return np.degrees(I0 - I2)

# by convention, use solar masses, AU, and set c = 1, in which case G = 9.87e-9
G = 9.87e-9
def get_eps(*params):
    m1, m2, m3, a0, a2, e2 = params
    m12 = m1 + m2
    mu = m1 * m2 / m12
    n = np.sqrt(G * m12 / a0**3)
    eps_gw = (1 / n) * (m12 / m3) * (a2**3 / a0**7) * G**3 * mu * m12**2
    eps_gr = (m12 / m3) * (a2**3 / a0**4) * (1 - e2**2)**(3/2) * 3 * G * m12
    eps_sl = (m12 / m3) * (a2**3 / a0**4) * (1 - e2**2)**(3/2) * (
        3 * G * (m2 + mu / 3) / 2)
    L1 = mu * np.sqrt(G * (m12) * a0)
    L2 = m3 * m12 / (m3 + m12) * np.sqrt(G * (m3 + m12) * a2)
    eta = L1 / L2
    return {'eps_gw': eps_gw, 'eps_gr': eps_gr, 'eps_sl': eps_sl,
            'eta': eta}

def get_Ilimd(eta=0, eps_gr=0, **kwargs):
    def jlim_criterion(j): # eq 44, satisfied when j = jlim
        return (
            3/8 * (j**2 - 1) * (
                - 3 + eta**2 / 4 * (4 * j**2 / 5 - 1))
            + eps_gr * (1 - 1 / j))
    jlim = brenth(jlim_criterion, 1e-15, 1 - 1e-15)
    Ilim = np.arccos(eta / 2 * (4 * jlim**2 / 5 - 1))
    Ilimd = np.degrees(Ilim)
    return Ilimd

def get_dydt(eps_gw=0, eps_gr=0, eps_sl=0, eta=0, e2=0):
    def dydt(t, y):
        '''
        dydt for all useful of 10 orbital elements + spin, eps_oct = 0 in LML15.
        eta = L / Lout
        '''
        a1, e1, W, I1, w1, I2, *svecs = y
        Itot = I1 + I2
        x1 = 1 - e1**2

        # orbital evolution
        da1dt =  (
            -eps_gw * (64 * (1 + 73 * e1**2 / 24 + 37 * e1**4 / 96)) / (
                5 * a1**3 * x1**(7/2))
        )
        de1dt = (
            15 * a1**(3/2) * e1 * np.sqrt(x1) * np.sin(2 * w1)
                    * np.sin(Itot)**2 / 8
                - eps_gw * 304 * e1 * (1 + 121 * e1**2 / 304)
                    / (15 * a1**4 * x1**(5/2))
        )
        dWdt = (
            -3 * a1**(3/2) / (np.sin(I1) * 32 * np.sqrt(x1)) * (
                2 * (2 + 3 * e1**2 - 5 * e1**2 * np.cos(2 * w1))
                * np.sin(2 * Itot))
        )
        dI1dt = (
            -15 * a1**(3/2) * e1**2 * np.sin(2 * w1)
                * np.sin(2 * Itot) / (16 * np.sqrt(x1))
        )
        dI2dt = eta * a1**(1/2) * (
            -15 * a1**(3/2) * e1**2 * np.sin(2 * w1) * np.sin(Itot) / 8
        )
        dw1dt = (
            3 * a1**(3/2) * (
                (4 * np.cos(Itot)**2 +
                 (5 * np.cos(2 * w1) - 1) * (1 - e1**2 - np.cos(Itot)**2))
                  / (8 * np.sqrt(x1))
                + eta * a1**(1/2) * np.cos(Itot) * (
                    2 + e1**2 * (3 - 5 * np.cos(2 * w1))) / 8
            )
            + eps_gr / (a1**(5/2) * x1)
        )

        # spin evolution
        Lhat = [np.sin(I1) * np.cos(W), np.sin(I1) * np.sin(W), np.cos(I1)]
        dsdt = np.zeros_like(svecs)
        for i in range(len(svecs) // 3):
            dsdt[3 * i: 3 * (i + 1)] = eps_sl *\
                np.cross(Lhat, svecs[3 * i: 3 * (i + 1)]) / (a1**(5/2) * x1)
        ret = [da1dt, de1dt, dWdt, dI1dt, dw1dt, dI2dt, *dsdt]
        return ret
    return dydt

def get_qslf_for_I0(I0, tf=np.inf, plot=False, tol=1e-9, params=params):
    print('Running for', np.degrees(I0), tf)
    af = 5e-3
    getter_kwargs = get_eps(*params)
    # getter_kwargs['eta'] = 0
    # getter_kwargs['eps_gw'] = 0
    # getter_kwargs['eps_gr'] = 0
    dydt = get_dydt(**getter_kwargs)

    # a1, e1, W, I1, w1, I2, sx, sy, sz = y
    I1 = np.radians(get_I1(np.degrees(I0), getter_kwargs['eta']))
    I2 = I0 - I1
    s0 = [np.sin(I1), 0, np.cos(I1)] # initial alignment
    y0 = [1, 1e-3, 0, I1, 0, I2, *s0]

    a_term_event = lambda t, y: y[0] - af
    a_term_event.terminal = True
    ret = solve_ivp(dydt, (0, tf), y0, events=[a_term_event],
                    method='BDF', atol=tol, rtol=tol)

    _, _, W_arr, I_arr, _, _, *s_arr = ret.y
    Lhat_arr = [np.sin(I_arr) * np.cos(W_arr),
                np.sin(I_arr) * np.sin(W_arr),
                np.cos(I_arr)]
    qslfd = np.degrees(np.arccos(ts_dot(Lhat_arr, s_arr)))
    print('Ran for', np.degrees(I0), qslfd[0], qslfd[-1])

    if plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
        ax1.semilogy(ret.t, ret.y[0], 'k', alpha=0.7, lw=0.7)
        ax2.semilogy(ret.t, 1 - ret.y[1], 'k', alpha=0.7, lw=0.7)
        print(ret.y[1,0], np.degrees(ret.y[3,0]), np.max(ret.y[1]), np.max(np.degrees(ret.y[3])))
        ax3.plot(ret.t, np.degrees(ret.y[3]), 'k', alpha=0.7, lw=0.7)
        ax4.plot(ret.t, qslfd, 'k', alpha=0.7, lw=0.7)
        plt.savefig('/tmp/8sim', dpi=400)
        plt.close()

    return qslfd[-1], ret.t[-1]

# n2 is fixed: n2 \propto eta * j1 * x1 + j2, e2=0
def get_dydt_vec(eps_gw=0, eps_gr=0, eps_sl=0, eta=0, **kwargs):
    def dydt(t, y):
        '''
        dydt for all useful of 10 orbital elements + spin, eps_oct = 0 in LML15.
        eta = L / Lout
        '''
        a, j1, e1, j2, s = y[0], y[1:4], y[4:7], y[7:10], y[10:13]
        e1s = np.sqrt(np.sum(e1**2)) # scalar
        x1 = 1 - e1s**2
        x2 = 1
        l1hat = j1 / np.sqrt(x1)
        n2 = j2 # e2 = 0

        # orbital evolution
        dadt = (
            -eps_gw * (64 * (1 + 73 * e1s**2 / 24 + 37 * e1s**4 / 96)) / (
                5 * a**3 * x1**(7/2))
        )
        dj1dt = 3 * a**(3/2) / 4 * (
            np.dot(j1, n2) * np.cross(j1, n2)
            - 5 * np.dot(e1, n2) * np.cross(e1, n2)
        )
        de1dt_lk = 3 * a**(3/2) / 4 * (
            np.dot(j1, n2) * np.cross(e1, n2)
            - 5 * np.dot(e1, n2) * np.cross(j1, n2)
            + 2 * np.cross(j1, e1)
        )
        de1dt_gw = -(
            eps_gw * (304 / 15) * (1 + 121 / 304 * e1s**2) /
            (a**4 * x1**(5/2))
        ) * e1
        de1dt_gr = eps_gr * np.cross(l1hat, e1) / (x1 * a**(5/2))
        de1dt = de1dt_lk + de1dt_gw + de1dt_gr
        dj2dt = eta * 3 * a**(3/2) / 4 * (
            np.dot(j1, n2) * np.cross(n2, j1)
            - 5 * np.dot(e1, n2) * np.cross(n2, e1)
        )
        dsdt = eps_sl * np.cross(l1hat, s) / (a**(5/2) * x1)
        ret = [dadt, *dj1dt, *de1dt, *dj2dt, *dsdt]
        return ret
    return dydt

def get_qslf_for_I0_vec(I0, tf=np.inf, plot=False, tol=1e-9, params=params):
    print('Running vec for', np.degrees(I0), tf)
    af = 5e-3
    W0 = 0
    w0 = 0 # I don't actually support anything more general than w0=0 lol
    e0s = 1e-3
    getter_kwargs = get_eps(*params)
    # getter_kwargs['eta'] = 0
    # getter_kwargs['eps_gw'] = 0
    # getter_kwargs['eps_gr'] = 0
    eta = getter_kwargs['eta']

    I1 = np.radians(get_I1(np.degrees(I0), getter_kwargs['eta']))
    I2 = I0 - I1
    s0 = np.array([np.sin(I1), 0, np.cos(I1)]) # initial alignment
    j1 = np.sqrt(1 - e0s**2) * s0
    e1 = np.array([0, e0s, 0])
    j2 = np.array([-np.sin(I2), 0, np.cos(I2)])
    y0 = [1, *j1, *e1, *j2, *s0]

    dydt = get_dydt_vec(**getter_kwargs)

    a_term_event = lambda t, y: y[0] - af
    a_term_event.terminal = True
    ret = solve_ivp(dydt, (0, tf), y0, events=[a_term_event],
                    method='BDF', atol=tol, rtol=tol)

    a, j1, e1, s = ret.y[0], ret.y[1:4], ret.y[4:7], ret.y[10:13]
    l1hat = j1 / np.sqrt(np.sum(j1**2, axis=0))
    qslfd = np.degrees(np.arccos(ts_dot(l1hat, s)))
    e1s = np.sqrt(np.sum(e1**2, axis=0))
    I1 = np.arccos(l1hat[2])
    print('Ran for', np.degrees(I0), qslfd[0], qslfd[-1])

    if plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
        print(e1s[0], np.degrees(I1[0]), np.max(e1s), np.max(np.degrees(I1)))
        ax1.semilogy(ret.t, a, 'k', alpha=0.7, lw=0.7)
        ax2.semilogy(ret.t, 1 - e1s, 'k', alpha=0.7, lw=0.7)
        ax3.plot(ret.t, np.degrees(I1), 'k', alpha=0.7, lw=0.7)
        ax4.plot(ret.t, qslfd, 'k', alpha=0.7, lw=0.7)
        plt.savefig('/tmp/8vecsim', dpi=400)
        plt.close()

    return qslfd[-1], ret.t[-1]

def get_qeff0(I0, params, tol=1e-10, e0=1e-3):
    print('Running for', np.degrees(I0))
    tf = 20
    getter_kwargs = get_eps(*params)
    # getter_kwargs['eta'] = 0
    getter_kwargs['eps_gw'] = 0
    dydt = get_dydt(**getter_kwargs)
    Ilimd = get_Ilimd(**getter_kwargs)

    other_svecs = [1, 0, 0, 0, 1, 0, 0, 0, 1] # for monodromy
    # s0 = [np.sin(I0), 0, np.cos(I0)] # initial alignment
    # y0 = [1, e0, 0, I0, 0, 0, *s0, *other_svecs]
    I1 = np.radians(get_I1(np.degrees(I0), getter_kwargs['eta']))
    I2 = I0 - I1
    s0 = [np.sin(I1), 0, np.cos(I1)] # initial alignment
    y0 = [1, e0, 0, I1, 0, I2, *s0, *other_svecs]

    period_event = lambda t, y: y[4] - np.pi
    period_event.terminal = True
    ret = solve_ivp(dydt, (0, tf), y0, events=[period_event],
                    method='BDF', atol=tol, rtol=tol, dense_output=True)
    t = ret.t
    a1, e1, W, I1, w1, I2, *svecs = ret.y
    svec = np.array(svecs[ :3])

    # calculate monodromy matrix in corotating frame (rotate by -W[-1] * zhat)
    Wf = W[-1]
    rot_mat = np.array([
        [np.cos(Wf), -np.sin(Wf), 0],
        [np.sin(Wf), np.cos(Wf), 0],
        [0, 0, 1]])
    svecs_f = np.array(svecs)[3: , -1]
    mono_mat = np.matmul(np.reshape(svecs_f, (3, 3)), rot_mat)
    # NB: in corotating frame, svec[:,-1] = np.dot(mono_mat.T, svec[:,0])
    eigs, eigv = np.linalg.eig(mono_mat.T)
    one_eig_idx = np.where(abs(np.imag(eigs)) < 1e-8)[0][0]
    mono_eig = np.real(eigv[:, one_eig_idx])
    Im = np.arccos(mono_eig[2])

    Iall = I1 + I2
    Itot = [get_I1(_I1, getter_kwargs['eta']) for _I1 in I1]
    x1 = 1 - e1**2
    dWdt = (
        3 * a1**(3/2) * np.sin(2 * Iall) / np.sin(I1) *
                (5 * e1**2 * np.cos(w1)**2 - 4 * e1**2 - 1)
            / (8 * np.sqrt(x1))
    )
    Jvec = np.array([0, 1])
    dWdt_mean = np.array([
        np.sum(dWdt * Jvec[0] * np.gradient(t)) / t[-1],
        0,
        np.sum(dWdt * Jvec[1] * np.gradient(t)) / t[-1]])

    dWslx = getter_kwargs['eps_sl'] * np.sin(I1) / (a1**(5/2) * x1)
    dWslz = getter_kwargs['eps_sl'] * np.cos(I1) / (a1**(5/2) * x1)
    dWslx_mean = np.sum(dWslx * np.gradient(t)) / t[-1]
    dWslz_mean = np.sum(dWslz * np.gradient(t)) / t[-1]

    dWsl_vec = np.array([
        dWslx,
        np.zeros_like(dWslx),
        dWslz,
    ])
    dWdt_vec = np.outer(np.array([0, 0, 1]), dWdt)
    Weff_vec_t = dWsl_vec - dWdt_vec
    dWsl_mean = np.sum(dWsl_vec * np.gradient(t) / t[-1], axis=1)
    dWdt_mean = np.sum(dWdt_vec * np.gradient(t) / t[-1], axis=1)
    _Weff_vec = np.sum(Weff_vec_t * np.gradient(t) / t[-1], axis=1)

    Weffmag = np.sqrt(np.sum(_Weff_vec**2))
    Weff_vec = _Weff_vec / Weffmag
    # |A|
    A = np.sqrt(np.sum(dWsl_mean**2)) / np.sqrt(np.sum(dWdt_mean**2))

    # analytic check
    # sign = 1 if np.degrees(I0) < Ilimd else -1
    # I2_anal = np.arctan2(Jvec[0], Jvec[1])
    # I1_anal = I0 - I2_anal # indeed equals I1_anal from my func
    # I_e_tot = np.arctan2(sign * A * np.sin(I1_anal),
    #                      (1 + sign * A * np.cos(I1_anal)))
    # I_e_out = np.arctan2(Weff_vec[0], Weff_vec[2]) + (
    #     0 if sign == 1 else np.pi)
    # print(np.degrees(I_e_out), np.degrees(I_e_tot + I2_anal))

    qeff0 = np.arccos(np.dot(Weff_vec, s0))

    # my best qeff estimator: <We> dot <svec>
    # seems like qeff0 is sufficient
    # dWdt_inertial = np.outer([Jvec[0], 0, Jvec[1]], dWdt)
    # Wsl_inertial = np.outer(
    #     np.ones(3),
    #     getter_kwargs['eps_sl'] / (a1**(5/2) * x1)
    # ) * np.array([
    #     np.sin(I1) * np.cos(W),
    #     np.sin(I1) * np.sin(W),
    #     np.cos(I1),
    # ])
    # Weffvec_inertial = np.sum(
    #     (Wsl_inertial - dWdt_inertial) * np.gradient(t) / t[-1], axis=1)
    # Weffvec_inertial /= np.sqrt(np.sum(Weffvec_inertial**2))
    # qeff_est = np.arccos(np.dot(Weffvec_inertial, np.mean(svec, axis=1)))
    # print(np.degrees(qeff0), np.degrees(qeff_est))

    Ie = np.arctan2(Weff_vec[0], Weff_vec[2])
    ratio = Weffmag * t[-1] / (2 * np.pi)

    # Weff harmonics
    t_vals = np.linspace(t[0], t[-1], 4 * len(t))
    Weffx_interp = interp1d(t, Weff_vec_t[0])(t_vals)
    Weffz_interp = interp1d(t, Weff_vec_t[2])(t_vals)
    x_coeffs = dct(Weffx_interp, type=1)[::2] / (2 * len(t_vals))
    z_coeffs = dct(Weffz_interp, type=1)[::2] / (2 * len(t_vals))
    Ie1 = np.arctan2(x_coeffs[1], z_coeffs[1])
    ratio1 = np.sqrt(x_coeffs[1]**2 + z_coeffs[1]**2) * t[-1] / (2 * np.pi)

    print(A, np.degrees(Ie), np.degrees(qeff0), np.degrees(Im), ratio,
          np.degrees(Ie1), ratio1)
    return A, Ie, qeff0, Im, ratio, Ie1, ratio1

def qslfs_run(npts=200):
    getter_kwargs = get_eps(*params)
    # getter_kwargs['eta'] = 0
    Ilimd = get_Ilimd(**getter_kwargs)

    pkl_fn = '8finite_qslfs.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)

        incs1 = np.radians(np.linspace(Ilimd + 0.5, Ilimd, npts))
        incs2 = np.radians(
            np.linspace(Ilimd - 0.5, Ilimd, npts - 1, endpoint=False)
        )
        incs = np.array(list(zip(incs1, incs2))).flatten()
        with Pool(64) as p:
            dat = p.map(get_qslf_for_I0, incs)
        qslfds, t_merges = np.array(dat).T
        with open(pkl_fn, 'wb') as f:
            pickle.dump((incs, qslfds, t_merges), f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            incs, qslfds, t_merges = pickle.load(f)
    sort_idx = np.argsort(incs)
    I_degs = np.degrees(incs)[sort_idx]
    tf = t_merges[sort_idx]
    qslfd_arr = qslfds[sort_idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 2]})
    t_lk, _, _, _ = get_vals(*params, np.radians(90.5))
    ax1.plot(I_degs, tf * t_lk, 'b', lw=1.5)
    ax1.set_yscale('log')
    ax1.set_ylabel('Merger Time (yr)')
    ax1.set_yticks([1e6, 1e8, 1e10])
    ax1.set_yticklabels([r'$10^{6}$', r'$10^{8}$', r'$10^{10}$'])

    ax2.plot(I_degs, qslfd_arr, 'b', lw=1.5)
    ax2.set_xlabel(r'$I_{\rm i}$ (Deg)')
    ax2.set_ylabel(r'$\theta_{\rm sl, f}$ (Deg)')
    ax2.set_yticks([0, 30, 60, 90])
    ax2.set_yticklabels([r'$0$', r'$30$', r'$60$', r'$90$'])
    ax1.axvline(Ilimd, c='k', lw=0.7)
    ax2.axvline(Ilimd, c='k', lw=0.7)

    I_left = I_degs[np.where(I_degs < Ilimd)[0]]
    I_right = I_degs[np.where(I_degs > Ilimd)[0]]
    I_leftlim = np.degrees(get_qeff0(np.radians(I_degs.min()), params)[2])
    I_rightlim = np.degrees(get_qeff0(np.radians(I_degs.max()), params)[2])
    ax2.plot(I_left,
             I_leftlim
                - (cosd(90.3)**2 / cosd(I_left - Ilimd + 90)**2)**(37/16),
             'k:', lw=1, alpha=0.7)
    ax2.plot(I_left,
             I_leftlim
                + (cosd(90.3)**2 / cosd(I_left - Ilimd + 90)**2)**(37/16),
             'k:', lw=1, alpha=0.7)
    ax2.plot(I_right,
             I_rightlim
                - (cosd(90.3)**2 / cosd(I_right - Ilimd + 90)**2)**(37/16),
             'k:', lw=1, alpha=0.7)
    ax2.plot(I_right,
             I_rightlim
                + (cosd(90.3)**2 / cosd(I_right - Ilimd + 90)**2)**(37/16),
             'k:', lw=1, alpha=0.7)
    ax2.set_ylim(bottom=0, top=120)

    ax2.plot(I_left, np.full_like(I_left, I_leftlim), 'k', lw=1, alpha=0.7)
    ax2.plot(I_right, np.full_like(I_right, I_rightlim), 'k', lw=1, alpha=0.7)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.03)
    plt.savefig('8finite_qslfs.png', dpi=400)
    plt.close()

def bin_comp(e0=1e-3, fn='8bin_comp'):
    ''' plot Fig 4 of LL 17 w/ updated trend line '''
    params = params_in
    getter_kwargs = get_eps(*params)
    Ilimd = get_Ilimd(**getter_kwargs)
    I_d = np.linspace(I_degs.min(), I_degs.max(), 2000)

    pkl_fn = fn + '.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        rets = []
        for I_val in I_d:
            ret = get_qeff0(np.radians(I_val), params, e0=e0)
            rets.append(ret)
        with open(pkl_fn, 'wb') as f:
            pickle.dump(np.array(rets).T, f)
        rets = np.array(rets).T
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            rets = pickle.load(f)
    As, Ies, qeis, Ims, ratios, Ie1s, ratio1s = rets # whoops, double .T
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(6, 12), sharex=True,
        gridspec_kw={'height_ratios': [1, 1, 1, 3]})
    # convention: Im is in [0, 90] or [90, 180] with Ies
    corrected_Ims = []
    for Im, Ie in zip(Ims, Ies):
        if Ie < np.pi / 2 and Ie > 0:
            if Im < np.pi / 2 and Im > 0:
                corrected_Ims.append(Im)
            elif Im > np.pi / 2 and Im < np.pi:
                corrected_Ims.append(np.pi - Im)
        elif Ie > np.pi / 2 and Ie < np.pi:
            if Im < np.pi / 2 and Im > 0:
                corrected_Ims.append(np.pi - Im)
            elif Im > np.pi / 2 and Im < np.pi:
                corrected_Ims.append(Im)
    assert len(corrected_Ims) == len(Ims)
    Ims = np.array(corrected_Ims)
    Ies = np.array(Ies)

    ax1.semilogy(I_d, As, 'k', lw=1)
    ax1.axhline(1, c='k', ls=':', lw=0.7, alpha=0.5)
    ax1.set_ylabel(r'$|\mathcal{A}|$')

    ax2.plot(I_d, ratios, 'k',
             label=r'$\bar{\Omega}_{\rm e} / \Omega_{\rm LK}$', lw=1, alpha=0.7)
    ax2.plot(I_d, ratio1s, 'g',
             label=r'$\Omega_{\rm e1} / \Omega_{\rm LK}$', lw=1, alpha=0.7)
    ax2.legend(fontsize=12, loc='upper right')
    ax2.set_ylabel('Freq. Ratios')

    # angle between averaged rotation axis and true monodromy eigenvector
    delta_Im = np.degrees(Ims - Ies)
    ax3.plot(I_d, np.abs(delta_Im), 'k', label='Num', lw=1)
    ylims = ax3.get_ylim()
    ax3.set_ylabel(r'$|\bar{I}_{\rm m, i} - I_{\rm e, i}|$')
    ax3.plot(
        I_d, np.degrees(np.abs(
            np.sin(Ies - Ie1s) * ratio1s / (np.abs(ratios) - 1))),
        'g', label=r'$N = 1$', lw=1)
    # TODO use Ie2, ratio2s for this
    ax3.plot(
        I_d, np.degrees(np.abs(
            np.sin(Ies - Ie1s) * ratio1s / (np.abs(ratios) - 2))),
        'b', label=r'$N = 2$', lw=1)
    ax3.legend(fontsize=12)
    ax3.set_ylim(ylims)

    # correct range?
    # qslfs_ranged = np.minimum(qslfs, 180 - qslfs)
    ax4.plot(I_degs, qslfs, 'bo', ms=1.0, label='Data')

    # same as dl_prediction
    # ax4.plot(I_d, np.degrees(qeis), 'g', lw=1.5, alpha=0.7)
    Itot = [get_I1(_I1, getter_kwargs['eta']) for _I1 in I_d]
    qslf_dl = np.degrees(np.abs(Ies - np.radians(Itot)))
    ax4.plot(I_d, qslf_dl, 'r', lw=2)
    ax4.fill_between(
        I_d,
        np.minimum(np.abs(delta_Im - qslf_dl), np.abs(delta_Im + qslf_dl)),
        np.maximum(np.abs(delta_Im - qslf_dl), np.abs(delta_Im + qslf_dl)),
        color='r',
        alpha=0.3)

    ax4.set_xticks([0, 45, 90, 135, 180])
    ax4.set_xticklabels([r'$0$', r'$45$', r'$90$', r'$135$', r'$180$'])
    ax4.set_xlabel(r'$I_{\rm i}$ (Deg)')
    ax4.set_ylabel(r'$\theta_{\rm sl, f}$')

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.03)
    plt.savefig(fn, dpi=600)
    plt.clf()

def run_long(I0d, tol=1e-8, tf=np.inf, t_inc=1000):
    folder = '8long'
    mkdirp(folder)
    I0 = np.radians(I0d)
    params = [30, 30, 30, 0.1, 3, 0]
    t_lk, _, _, _ = get_vals(*params, 80)
    getter_kwargs = get_eps(*params)
    af = 0.1

    fn = '%s/%d_run' % (folder, I0d)
    pkl_fn = fn + '.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        num_ran = 0
        t_emaxes = []
        y_emaxes = []

        dydt = get_dydt(**getter_kwargs)

        # a1, e1, W, I1, w1, I2, sx, sy, sz = y
        I1 = np.radians(get_I1(I0d,
                               getter_kwargs['eta'] * np.sqrt(1 - 1e-6)))
        I2 = I0 - I1
        s0 = [np.sin(I1), 0, np.cos(I1)] # initial alignment
        y0 = [1, 1e-3, 0, I1, 0, I2, *s0]

        peak_event = lambda t, y: (y[4] % np.pi) - (np.pi / 2)
        peak_event.direction = +1 # only when w is increasing
        a_term_event = lambda t, y: y[0] - af
        a_term_event.terminal = True
        events=[peak_event, a_term_event]

        print('Running for I0=%d, t_i=%d' % (I0d, num_ran * t_inc))
        ret = solve_ivp(dydt, (0, t_inc), y0, events=events,
                        method='BDF', atol=tol, rtol=tol,
                        dense_output=True)
        t_emax = ret.t_events[0]
        t_emaxes.extend(t_emax)
        y_emaxes.extend(ret.sol(t_emax).T)

        while ret.y[0, -1] > af and ret.t[-1] < tf:
            num_ran += 1
            print('Running for I0=%d, t_i=%d, a=%.5f, emax=%.3f' %
                  (I0d, num_ran * t_inc, ret.y[0, -1],
                   ret.sol(t_emax[-1])[1]))
            y0 = ret.y[:,-1]
            del ret
            gc.collect()
            ret = solve_ivp(dydt,
                            (num_ran * t_inc, (num_ran + 1) * t_inc),
                            y0,
                            events=[peak_event],
                            method='BDF', atol=tol, rtol=tol,
                            dense_output=True)
            t_emax = ret.t_events[0]
            t_emaxes.extend(t_emax)
            y_emaxes.extend(ret.sol(t_emax).T)
            # dump incrementally
            with open(pkl_fn, 'wb') as f:
                pickle.dump((t_emaxes, y_emaxes), f)
    else:
        with open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            t_emaxes, y_emaxes = pickle.load(f)
    t_evals = np.array(t_emaxes) * t_lk

    a_vals, e_vals, W_vals, I1_vals, _, I2_vals, *s_vals = np.array(y_emaxes).T
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
    l1hat = np.array([
        np.sin(I1_vals) * np.cos(W_vals),
        np.sin(I1_vals) * np.sin(W_vals),
        np.cos(I1_vals),
    ])
    qslfd = np.degrees(np.arccos(ts_dot(l1hat, s_vals)))

    ax1.plot(t_evals, a_vals * 0.1, 'k', alpha=0.7, lw=0.7)
    ax1.set_ylabel(r'$a$ (AU)')
    ax2.plot(t_evals, e_vals, 'k', alpha=0.7, lw=0.7)
    ax2.set_ylabel(r'$e$')
    ax3.plot(t_evals, np.degrees(I1_vals + I2_vals), 'k', alpha=0.7, lw=0.7)
    ax3.set_ylabel(r'$I$ (deg)')
    ax4.plot(t_evals, qslfd, 'k', alpha=0.7, lw=0.7)
    ax4.set_ylabel(r'$\theta_{\rm sl}$')
    ax3.set_xlabel(r'$t$ (yr)')
    ax4.set_xlabel(r'$t$ (yr)')
    plt.tight_layout()
    plt.savefig(fn, dpi=200)
    plt.close()

if __name__ == '__main__':
    # getter_kwargs = get_eps(*params)
    # Ilimd = get_Ilimd(**getter_kwargs)
    # get_qslf_for_I0_vec(np.radians(Ilimd + 0.35), tf=10, plot=True, tol=1e-10)
    # get_qslf_for_I0(np.radians(Ilimd + 0.35), tf=10, plot=True, tol=1e-10)
    # get_qslf_for_I0(np.radians(Ilimd - 0.35))
    # print(np.degrees(get_qeff0(np.radians(Ilimd + 0.35), params)[2]))
    # print(np.degrees(get_qeff0(np.radians(Ilimd - 0.35), params)[2]))
    # these two finally agree

    # qslfs_run()

    # get_qeff0(np.radians(85), params_in)
    bin_comp()
    bin_comp(e0=1e-2, fn='8bin_comp_en2')
    bin_comp(e0=3e-3, fn='8bin_comp_en1')

    # run_long(80, tol=1e-8)
    # run_long(70, tol=1e-8)
    # run_long(60, tol=1e-8)
    pass
