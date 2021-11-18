import numpy as np
import pandas as pd
import xarray as xr
import Grid
import pf_dynamic_sph
from scipy.io import savemat, loadmat
import os
from timeit import default_timer as timer
import sys
from copy import deepcopy
# import matplotlib
# import matplotlib.pyplot as plt
# from mpmath import polylog

if __name__ == "__main__":

    start = timer()

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (20, 20, 20)
    (dx, dy, dz) = (0.2, 0.2, 0.2)

    xgrid = Grid.Grid('CARTESIAN_3D')
    xgrid.initArray('x', -Lx, Lx, dx); xgrid.initArray('y', -Ly, Ly, dy); xgrid.initArray('z', -Lz, Lz, dz)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    NGridPoints_desired = (1 + 2 * Lx / dx) * (1 + 2 * Lz / dz)
    Ntheta = 50
    Nk = np.ceil(NGridPoints_desired / Ntheta).astype(int)

    theta_max = np.pi
    thetaArray, dtheta = np.linspace(0, theta_max, Ntheta, retstep=True)

    k_max = ((2 * np.pi / dx)**3 / (4 * np.pi / 3))**(1 / 3)

    k_min = 1e-5
    kArray, dk = np.linspace(k_min, k_max, Nk, retstep=True)
    if dk < k_min:
        print('k ARRAY GENERATION ERROR')

    kgrid = Grid.Grid("SPHERICAL_2D")
    kgrid.initArray_premade('k', kArray)
    kgrid.initArray_premade('th', thetaArray)

    tMax = 6000; dt = 0.5
    # tMax = 500; dt = 0.5
    # tMax = 900; dt = 0.5
    # tMax = 0.5; dt = 0.5
    tgrid = np.arange(0, tMax + dt, dt)

    gParams = [xgrid, kgrid, tgrid]
    NGridPoints = kgrid.size()

    print('Total time steps: {0}'.format(tgrid.size))
    print('UV cutoff: {0}'.format(k_max))
    print('dk: {0}'.format(dk))
    print('NGridPoints: {0}'.format(NGridPoints))

    # Experimental params

    expParams = pf_dynamic_sph.Zw_expParams_2021()
    L_exp2th, M_exp2th, T_exp2th = pf_dynamic_sph.unitConv_exp2th(expParams['n0_BEC_scale'], expParams['mB'])

    kB = 1.38064852e-23  # Boltzmann constant in J/K
    hbar = 1.0555e-34  # reduced Planck's constant (J*s/rad)

    aIBexp_Vals = np.array([-1000, -750, -500, -375, -250, -125, -60, -20, 0, 20, 50, 125, 175, 250, 375, 500, 750, 1000])

    n0_BEC = np.array([5.51533197e+19, 5.04612835e+19, 6.04947525e+19, 5.62709096e+19, 6.20802175e+19, 7.12364194e+19, 6.74430590e+19, 6.52854564e+19, 5.74487521e+19, 6.39240612e+19, 5.99344093e+19, 6.12326489e+19, 6.17370181e+19, 5.95291621e+19, 6.09224617e+19, 6.35951755e+19, 5.52594316e+19, 5.94489028e+19])  # peak BEC density (given in in m^(-3))
    RTF_BEC_Y = np.array([11.4543973014280, 11.4485027292274, 12.0994087866866, 11.1987472415996, 12.6147755284164, 13.0408759297917, 12.8251948079726, 12.4963915490121, 11.6984708883771, 12.1884624646191, 11.7981246004719, 11.8796464214276, 12.4136593404667, 12.3220325703494, 12.0104329130883, 12.1756670927480, 10.9661042681457, 12.1803009563806])  # Thomas-Fermi radius of BEC in direction of oscillation (given in um)

    Na_displacement = np.array([26.2969729628679, 22.6668334850173, 18.0950989598699, 20.1069898676222, 14.3011351453467, 18.8126473489499, 17.0373115356076, 18.6684373282353, 18.8357213162278, 19.5036039713438, 21.2438389441807, 18.2089748680659, 18.0433963046778, 8.62940156299093, 16.2007030552903, 23.2646987822343, 24.1115616621798, 28.4351972435186])  # initial position of the BEC (in um)
    K_displacement_raw = np.array([0.473502276902047, 0.395634326123081, 8.66936929134637, 11.1470221226478, 9.34778274195669, 16.4370036199872, 19.0938486958001, 18.2135041439547, 21.9211790347041, 20.6591098913628, 19.7281375591975, 17.5425503131171, 17.2460344933717, 11.7179407507981, 12.9845862662090, 9.18113956217101, 11.9396846941782, 4.72461841775226])  # initial position of the impurity (in um)
    K_displacement_scale = np.mean(K_displacement_raw[6:11] / Na_displacement[6:11])
    K_displacement = deepcopy(K_displacement_raw); K_displacement[0:6] = K_displacement_scale * Na_displacement[0:6]; K_displacement[11::] = K_displacement_scale * Na_displacement[11::]   # in um
    K_relPos = K_displacement - Na_displacement   # in um
    # print('init ', K_relPos[0])
    # K_relPos[0] = 0

    # K_relPos[0] = -15
    # K_relPos[1] = -15
    # K_relPos[2] = -7

    omega_Na = np.array([465.418650581347, 445.155256942448, 461.691943131414, 480.899902898451, 448.655522184374, 465.195338759998, 460.143258369460, 464.565377197007, 465.206177963899, 471.262139163205, 471.260672147216, 473.122081065092, 454.649394420577, 449.679107889662, 466.770887179217, 470.530355145510, 486.615655444221, 454.601540658640])   # in rad*Hz
    omega_K_raw = np.array([764.649207995890, 829.646158322623, 799.388442120805, 820.831266284088, 796.794204312379, 810.331402280747, 803.823888714144, 811.210511844489, 817.734286423120, 809.089608774626, 807.885837386121, 808.334196591376, 782.788534907910, 756.720677755942, 788.446619623011, 791.774719564856, 783.194731826180, 754.641677886382])   # in rad*Hz
    omega_K_scale = np.mean(omega_K_raw[6:11] / omega_Na[6:11])
    omega_K = deepcopy(omega_K_raw); omega_K[0:6] = omega_K_scale * omega_Na[0:6]; omega_K[11::] = omega_K_scale * omega_Na[11::]  # in rad*Hz
    omega_x_K = 2 * np.pi * 141; omega_y_K = 2 * np.pi * 130; omega_z_K = 2 * np.pi * 15  # should get more accurate estimate for omega_x_K

    K_relVel = np.array([1.56564660488838, 1.31601642026105, 0.0733613860991014, 1.07036861258786, 1.22929932184982, -13.6137940945403, 0.0369377794311800, 1.61258456681232, -1.50457700049200, -1.72583008593939, 4.11884512615162, 1.04853747806043, -0.352830359266360, -4.00683426531578, 0.846101589896479, -0.233660196108278, 4.82122627459411, -1.04341939663180])  # in um/ms

    phi_Na = np.array([-0.2888761, -0.50232022, -0.43763589, -0.43656233, -0.67963017, -0.41053479, -0.3692152, -0.40826816, -0.46117853, -0.41393032, -0.53483635, -0.42800711, -0.3795508, -0.42279337, -0.53760432, -0.4939509, -0.47920687, -0.51809527])  # phase of the BEC oscillation in rad
    gamma_Na = np.array([4.97524294, 14.88208436, 4.66212187, 6.10297397, 7.77264927, 4.5456649, 4.31293083, 7.28569606, 8.59578888, 3.30558254, 8.289436, 4.14485229, 7.08158476, 4.84228082, 9.67577823, 11.5791718, 3.91855863, 10.78070655])  # decay rate of the BEC oscillation in Hz

    N_K = np.array([2114.31716217314, 3040.54086059863, 3788.54290366850, 2687.53370686094, 2846.49206660163, 1692.49722769915, 1813.12703968803, 2386.60764443984, 2532.45824159990, 2361.26046445201, 2466.63648224567, 2206.34584323146, 2113.15620874362, 3755.19098529495, 2163.29615872937, 2042.58962172497, 4836.09854876457, 3044.93792941312])  # total number of fermions in K gas
    TFermi = np.array([6.83976585132807e-08, 7.93313829893224e-08, 8.43154444077350e-08, 7.58635297351284e-08, 7.65683267650816e-08, 6.47481434584840e-08, 6.60734255262424e-08, 7.26332216239745e-08, 7.42817184102838e-08, 7.23120402195269e-08, 7.33357082077064e-08, 7.06727442566945e-08, 6.89216704173642e-08, 8.25441536498287e-08, 6.96294877404586e-08, 6.84055531750863e-08, 9.08417325299114e-08, 7.69018614503965e-08])  # Fermi temperature of K gas (in K)
    T_K_ratio = np.array([1.16963068237879, 1.00842815271187, 0.948817865599258, 1.05452514903161, 1.04481844360328, 1.23555666196507, 1.21077421615179, 1.10142436492992, 1.07698100841087, 1.10631645514542, 1.09087376334348, 1.13197811746813, 1.16073797276748, 0.969178269600757, 1.14893851148521, 1.16949569569648, 0.880652512584549, 1.04028691232139])  # Ratio of temperature T to Fermi temperature T_Fermi of K gas
    T = 80e-9  # Temperature T of K gas (in K) --> equivalent to T_K_ratio * T_Fermi
    mu_div_hbar_K = np.array([21527.623521898644, 17656.025221467124, 15298.569367268587, 18973.981143581444, 18360.701066883277, 23888.301168354345, 23158.661546706127, 20239.341737009476, 19607.6059603436, 20352.99023696009, 19888.153905968644, 21074.805169679148, 21533.45904566066, 15393.579214021502, 21284.26382771103, 21894.22770364862, 12666.509194815215, 17640.573345313787])  # Chemical potential of the K gas (in rad*Hz) - computed using the code below
    # print(mu_div_hbar_K / (2 * np.pi))

    # Optical trap experimental parameters (for K gas)

    A_ODT1_uK = -4.25; A_ODT2_uK = -3.44
    # A_TiSa_uK = -6.9  # Amplitude of each gaussian beam in uK
    A_TiSa_uK = -6.97  # Amplitude of each gaussian beam in uK
    A_ODT1_Hz = kB * A_ODT1_uK * 1e-6 / hbar / (2 * np.pi); A_ODT2_Hz = kB * A_ODT2_uK * 1e-6 / hbar / (2 * np.pi); A_TiSa_Hz = kB * A_TiSa_uK * 1e-6 / hbar / (2 * np.pi)  # Amplitude of each gaussian beam in Hz
    wx_ODT1 = 82; wy_ODT1 = 82  # beam waists of ODT1 in um
    ep_ODT2 = 4  # ellipticity of ODT2 beam
    wx_ODT2 = 144; wz_ODT2 = wx_ODT2 / ep_ODT2  # beam waists of ODT2 in um
    wx_TiSa = 95; wy_TiSa = 95  # beam waists of TiSa in um

    # def omega_gaussian_approx(A, m):
    #     return np.sqrt((-4) * hbar * 2 * np.pi * A / (m * (wy_TiSa * 1e-6)**2))

    # A_TiSa_uK_Na = -1.39; A_TiSa_Hz_Na = kB * A_TiSa_uK_Na * 1e-6 / hbar / (2 * np.pi)
    # # scale_fac = np.array([0.9724906988829373, 0.8896535250404035, 0.9569791653444897, 1.0382627423131257, 0.903699258873818, 0.971557704172668, 0.9505698121511785, 0.9689281438661883, 0.9716029799261697, 0.9970638982528262, 0.9970576906401939, 1.00494970248614, 0.9280067282060056, 0.9078274542675744, 0.9781498957395225, 0.9939697862098377, 1.0630900294469736, 0.9278113852604492])
    # # A_TiSa_Hz = A_TiSa_Hz * scale_fac
    # # A_TiSa_Hz_Na = A_TiSa_Hz_Na * scale_fac
    # print(omega_K / (2 * np.pi))
    # print(omega_gaussian_approx(A_TiSa_Hz, expParams['mI']) / (2 * np.pi))
    # print(omega_Na / (2 * np.pi))
    # print(omega_gaussian_approx(A_TiSa_Hz_Na, expParams['mB']) / (2 * np.pi))

    # # from scipy.optimize import root_scalar
    # # sf_List = []
    # # for indw, wy in enumerate(omega_Na):
    # #     muScale = 2 * np.pi * 1e3
    # #     # n = N3D(muScale, omega_x_K, omega_y_K, omega_z_K, T)
    # #     def f(sf): return (omega_gaussian_approx(sf * A_TiSa_Hz_Na, expParams['mB']) - wy)
    # #     sol = root_scalar(f, bracket=[0.5, 1.5], method='brentq')
    # #     sf_List.append(sol.root)
    # # print(sf_List)

    # # Compute chemical potential of the K gas

    # # mu_K = -1 * kB * T * np.log(np.exp(N_K) - 1) / hbar  # chemical potential of K gas (in rad*Hz)
    # # mu_K = -1 * kB * T * N_K / hbar  # chemical potential of K gas (in rad*Hz) -- approximation of above line since np.exp(N_K) >> 1. Get chemical potential ~5 MHz compared to ~1 kHz for Na gas
    # # print(mu_K / (2 * np.pi) * 1e-6)

    # from scipy.optimize import root_scalar

    # def N3D(mu_div_hbar, omega_x, omega_y, omega_z, T):
    #     mu = hbar * mu_div_hbar
    #     beta = 1 / (kB * T)
    #     prefac = 1 / ((hbar**3) * (beta**3) * np.sqrt((omega_x**2) * (omega_y**2) * (omega_z**2)))
    #     return -1 * prefac * polylog(3, -1 * np.exp(-1 * beta * mu))

    # mu_div_hbar_List = []
    # for indw, wy in enumerate(omega_K):
    #     muScale = 2 * np.pi * 1e3
    #     # n = N3D(muScale, omega_x_K, omega_y_K, omega_z_K, T)
    #     def f(mu): return (N3D(mu, omega_x_K, omega_y_K, omega_z_K, T) - N_K[indw])
    #     sol = root_scalar(f, bracket=[0.1 * muScale, 10 * muScale], method='brentq')
    #     # print(sol.root / (2 * np.pi)); n = N3D(sol.root, omega_x_K, omega_y_K, omega_z_K, T); print(N_K[indw], n)
    #     mu_div_hbar_List.append(sol.root)
    # # print(mu_div_hbar_List)

    # Basic parameters

    n0 = n0_BEC / (L_exp2th**3)  # converts peak BEC density (for each interaction strength) to theory units
    mB = expParams['mB'] * M_exp2th  # should = 1
    mI = expParams['mI'] * M_exp2th
    aBB = expParams['aBB'] * L_exp2th
    gBB = (4 * np.pi / mB) * aBB

    # # Derived quantities

    nu = pf_dynamic_sph.nu(mB, n0, gBB)
    xi = (8 * np.pi * n0 * aBB)**(-1 / 2)
    c_BEC_um_Per_ms = (nu * T_exp2th / L_exp2th) * (1e6 / 1e3)  # speed of sound in um/ms
    print(1e3 * tgrid[-1] / T_exp2th)
    # print(c_BEC_um_Per_ms)

    # # # Create density splines

    # from scipy import interpolate
    # yMat = loadmat('zwData/yMat_InMuM.mat')['yMat']  # grid of positions in the BEC oscillation direction ranging from -4*R_TF to +4*R_TF for each interaction strength (given in in um)
    # densityMat = loadmat('zwData/densityMat_InM-3.mat')['densityMat']  # BEC density for each experimentally measured interaction strength evaluated at each position in yMat (given in in m^(-3))
    # yMat_th = yMat * 1e-6 * L_exp2th  # converts grid positions from um to m and then converts to theory units
    # densityMat_th = densityMat / (L_exp2th**3)  # converts BEC density arrays (for each interaction strength) to theory units

    # # for ind, aIB_exp in enumerate(aIBexp_Vals):
    # #     den_tck = interpolate.splrep(yMat_th[ind, :], densityMat_th[ind, :], s=0)
    # #     np.save('zwData/densitySplines/nBEC_aIB_{0}a0.npy'.format(aIB_exp), den_tck)

    # import matplotlib
    # import matplotlib.pyplot as plt
    # yMat_th_interp = np.zeros((18, 1000))
    # densityMat_th_interp = np.zeros((18, 1000))
    # fig, ax = plt.subplots()
    # for ind, aIB_exp in enumerate(aIBexp_Vals):
    #     den_tck_load = np.load('zwData/densitySplines/nBEC_aIB_{0}a0.npy'.format(aIB_exp), allow_pickle=True)
    #     yMat_th_interp[ind, :] = np.linspace(yMat_th[ind, 0], yMat_th[ind, -1], 1000)
    #     densityMat_th_interp[ind, :] = interpolate.splev(yMat_th_interp[ind, :], den_tck_load)
    #     ax.plot(1e6 * yMat_th_interp[ind, :] / L_exp2th, densityMat_th_interp[ind, :] * (L_exp2th**3), label='{0}'.format(aIB_exp))
    # ax.legend()
    # plt.show()

    # # # Create thermal distribution of initial conditions

    # from arspy.ars import adaptive_rejection_sampling as ars
    # import matplotlib.pyplot as plt

    # # gaussian_logpdf = lambda x, sigma=1: np.log(np.exp(-x ** 2 / sigma))
    # # a, b = -2, 2  # a < b must hold
    # # domain = (float("-inf"), float("inf"))
    # # n_samples = 10000
    # # samples = ars(logpdf=gaussian_logpdf, a=a, b=b, domain=domain, n_samples=n_samples)
    # # print(np.isclose(np.mean(samples), 0.0, atol=1e-02))
    # # print(np.min(samples), np.max(samples))

    # # plt.hist(samples, bins=1000)
    # # plt.show()

    # def Vho_3D(x, y, z, omega_x, omega_y, omega_z, mI):
    #     return (mI / 2) * ((omega_x * x)**2 + (omega_y * y)**2 + (omega_z * z)**2)

    # def log_f_3D(x, y, z, px, py, pz, mI, mu, Vpot, T):
    #     return -1 * np.log(1 + np.exp(((px**2 + py**2 + pz**2) / (2 * mI) + Vpot(x, y, z) - mu) / (kB * T)))

    # def log_fxp_1D_slice(y, py, omega_x, omega_y, omega_z, mI, mu_div_hbar, T):
    #     return

    # def log_fx_1D_slice(y, omega_x, omega_y, omega_z, mI, mu_div_hbar, T):
    #     return

    # def log_fxp_1D_int(y, py, omega_x, omega_y, omega_z, mI, mu_div_hbar, T):
    #     return

    # def log_fx_1D_int(y, omega_x, omega_y, omega_z, mI, mu_div_hbar, T):
    #     return

    # Convert experimental parameters to theory parameters

    x0_imp = K_relPos * 1e-6 * L_exp2th  # initial positions of impurity in BEC frame (relative to the BEC)
    v0_imp = K_relVel * (1e-6 / 1e-3) * (L_exp2th / T_exp2th)  # initial velocities of impurity in BEC frame (relative to BEC)
    RTF_BEC = RTF_BEC_Y * 1e-6 * L_exp2th  # BEC oscillation amplitude (carries units of position)

    omega_BEC_osc = omega_Na / T_exp2th
    gamma_BEC_osc = gamma_Na / T_exp2th
    phi_BEC_osc = phi_Na
    amp_BEC_osc = (Na_displacement * 1e-6 * L_exp2th) / np.cos(phi_Na)  # BEC oscillation amplitude (carries units of position)
    # print(amp_BEC_osc[0])
    # amp_BEC_osc[0] = 0.15 * amp_BEC_osc[0]
    # print(amp_BEC_osc[0])

    omega_Imp_x = omega_K / T_exp2th

    gaussian_amp = 2 * np.pi * A_TiSa_Hz / T_exp2th  # converts optical potential amplitude to radHz, then converts this frequency to theory units. As we choose hbar = 1 in theory units, this is also units of energy.
    gaussian_width = wy_TiSa * 1e-6 * L_exp2th

    a0_exp = 5.29e-11  # Bohr radius (m)
    aIBi_Vals = 1 / (aIBexp_Vals * a0_exp * L_exp2th)

    P0_scaleFactor = np.array([1.2072257743801846, 1.1304777274446096, 1.53, 1.0693683091221053, 1.0349159867400886, 0.95, 2.0, 1.021253131703251, 0.9713134192266438, 0.9781007832739641, 1.0103135855263197, 1.0403095335853234, 0.910369833990368, 0.9794720983829749, 1.0747443076336567, 0.79, 1.0, 0.8127830658214898])  # scales the momentum of the initial polaron state (scaleFac * mI * v0) so that the initial impurity velocity matches the experiment. aIB= -125a0 and +750a0 have ~8% and ~20% error respectively

    # Create dicts

    toggleDict = {'Location': 'cluster', 'Dynamics': 'real', 'Interaction': 'on', 'InitCS': 'steadystate', 'InitCS_datapath': '', 'Coupling': 'twophonon', 'Grid': 'spherical',
                  'F_ext': 'off', 'PosScat': 'off', 'BEC_density': 'on', 'BEC_density_osc': 'on', 'Imp_trap': 'on', 'ImpTrap_Type': 'gaussian', 'CS_Dyn': 'on', 'Polaron_Potential': 'on', 'PP_Type': 'smarter'}

    # ---- SET OUTPUT DATA FOLDER ----

    if toggleDict['Location'] == 'personal':
        datapath = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/2021'
    elif toggleDict['Location'] == 'cluster':
        datapath = '/n/holyscratch01/demler_lab/kis/data/ZwierleinExp_data/2021'

    # if toggleDict['BEC_density'] == 'off':
    #     innerdatapath = innerdatapath + '/HomogBEC'
    #     toggleDict['Polaron_Potential'] = 'off'

    if toggleDict['ImpTrap_Type'] == 'harmonic':
        innerdatapath = datapath + '/harmonicTrap'
    else:
        innerdatapath = datapath + '/gaussianTrap'

    if toggleDict['Polaron_Potential'] == 'off':
        toggleDict['Polaron_Potential'] = 'off'
        innerdatapath += '/NoPolPot'
    else:
        innerdatapath += '/PolPot'
        if toggleDict['PP_Type'] == 'naive':
            innerdatapath += '/naivePP'
        else:
            innerdatapath += '/smarterPP'

    jobList = []
    for ind, aIBi in enumerate(aIBi_Vals):
        sParams = [mI, mB, n0[ind], gBB]
        den_tck = np.load('zwData/densitySplines/nBEC_aIB_{0}a0.npy'.format(aIBexp_Vals[ind]), allow_pickle=True)
        trapParams = {'nBEC_tck': den_tck, 'RTF_BEC': RTF_BEC[ind], 'omega_BEC_osc': omega_BEC_osc[ind], 'gamma_BEC_osc': gamma_BEC_osc[ind], 'phi_BEC_osc': phi_BEC_osc[ind], 'amp_BEC_osc': amp_BEC_osc[ind], 'omega_Imp_x': omega_Imp_x[ind], 'gaussian_amp': gaussian_amp, 'gaussian_width': gaussian_width, 'X0': x0_imp[ind], 'P0': P0_scaleFactor[ind] * mI * v0_imp[ind]}
        filepath = innerdatapath + '/aIB_{0}a0.nc'.format(aIBexp_Vals[ind])
        jobList.append((aIBi, sParams, trapParams, filepath))

    # print(len(jobList))

    # # ---- COMPUTE DATA ON COMPUTER ----

    # runstart = timer()
    # for ind, tup in enumerate(jobList):
    #     if ind != 1:
    #         continue
    #     print('aIB: {0}a0'.format(aIBexp_Vals[ind]))
    #     loopstart = timer()
    #     (aIBi, sParams, trapParams, filepath) = tup
    #     cParams = {'aIBi': aIBi}
    #     ds = pf_dynamic_sph.zw2021_quenchDynamics(cParams, gParams, sParams, trapParams, toggleDict)
    #     ds.to_netcdf(filepath)

    #     # v0 = ds['V'].values[0] * (T_exp2th / L_exp2th) * (1e6 / 1e3)
    #     # print(v0_imp[ind], K_relVel[ind], v0, K_relVel[ind] / v0)

    #     x0 = ds['X'].values[0] * 1e6 / L_exp2th
    #     print(K_relPos[ind], K_displacement[ind], x0)

    #     loopend = timer()
    #     print('aIBi: {:.2f}, Time: {:.2f}'.format(aIBi, loopend - loopstart))
    # end = timer()
    # print('Total Time: {:.2f}'.format(end - runstart))

    # ---- COMPUTE DATA ON CLUSTER ----

    runstart = timer()

    taskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    taskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    # taskCount = len(jobList); taskID = 4

    if(taskCount > len(jobList)):
        print('ERROR: TASK COUNT MISMATCH')
        sys.exit()
    else:
        tup = jobList[taskID]
        (aIBi, sParams, trapParams, filepath) = tup

    cParams = {'aIBi': aIBi}
    ds = pf_dynamic_sph.zw2021_quenchDynamics(cParams, gParams, sParams, trapParams, toggleDict)
    ds.to_netcdf(filepath)

    end = timer()
    print('Task ID: {:d}, aIBi: {:.2f}, Time: {:.2f}'.format(taskID, aIBi, end - runstart))
