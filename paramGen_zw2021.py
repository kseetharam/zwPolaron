import numpy as np
import pandas as pd
import xarray as xr
import Grid
import pf_dynamic_sph
import pf_static_sph
from scipy.io import savemat, loadmat
from scipy.optimize import root_scalar, minimize_scalar, minimize
from scipy.integrate import simpson
from scipy import interpolate
import os
from timeit import default_timer as timer
import sys
from copy import deepcopy
import mpmath as mpm
import matplotlib
import matplotlib.pyplot as plt

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

    # n0_BEC = np.array([5.51533197e+19, 5.04612835e+19, 6.04947525e+19, 5.62709096e+19, 6.20802175e+19, 7.12364194e+19, 6.74430590e+19, 6.52854564e+19, 5.74487521e+19, 6.39240612e+19, 5.99344093e+19, 6.12326489e+19, 6.17370181e+19, 5.95291621e+19, 6.09224617e+19, 6.35951755e+19, 5.52594316e+19, 5.94489028e+19])  # peak BEC density (given in in m^(-3))
    # n0_BEC = np.array([5.50743315e+19, 5.03889459e+19, 6.04081899e+19, 5.61903369e+19, 6.19914061e+19, 7.11346218e+19, 6.73466436e+19, 6.51920977e+19, 5.73665093e+19, 6.38326341e+19, 5.98486416e+19, 6.11450398e+19, 6.16486935e+19, 5.94439691e+19, 6.08352926e+19, 6.35042149e+19, 5.51802931e+19, 5.93638236e+19])  # peak density including both condensate and thermal cloud
    n0_BEC = np.array([5.46880416e+19, 4.99782123e+19, 6.00472624e+19, 5.58095582e+19, 6.16375560e+19, 7.08181465e+19, 6.70153652e+19, 6.48519814e+19, 5.69914183e+19, 6.34867755e+19, 5.94851655e+19, 6.07874304e+19, 6.12933278e+19, 5.90786343e+19, 6.04762935e+19, 6.31569492e+19, 5.47945313e+19, 5.89981190e+19])  # peak density of condensate
    RTF_BEC_X = np.array([8.48469347093994, 8.11111072629368, 8.89071272031954, 8.57125199684266, 9.00767433275159, 9.65522167387697, 9.39241266912852, 9.23956650925869, 8.66153179309422, 9.14179769236378, 8.84900230929328, 8.94534024135962, 8.98248647105392, 8.81871271135454, 8.92241777405925, 9.11802005065468, 8.49295023977057, 8.81270137636933])  # Thomas-Fermi radius of BEC in x-direction (given in um)
    RTF_BEC_Y = np.array([11.4543973014280, 11.4485027292274, 12.0994087866866, 11.1987472415996, 12.6147755284164, 13.0408759297917, 12.8251948079726, 12.4963915490121, 11.6984708883771, 12.1884624646191, 11.7981246004719, 11.8796464214276, 12.4136593404667, 12.3220325703494, 12.0104329130883, 12.1756670927480, 10.9661042681457, 12.1803009563806])  # Thomas-Fermi radius of BEC in direction of oscillation (given in um)
    RTF_BEC_Z = np.array([70.7057789244995, 67.5925893857806, 74.0892726693295, 71.4270999736888, 75.0639527729299, 80.4601806156414, 78.2701055760710, 76.9963875771558, 72.1794316091185, 76.1816474363648, 73.7416859107773, 74.5445020113302, 74.8540539254493, 73.4892725946212, 74.3534814504937, 75.9835004221224, 70.7745853314214, 73.4391781364111])  # Thomas-Fermi radius of BEC in z-direction (given in um)

    Na_displacement = np.array([26.2969729628679, 22.6668334850173, 18.0950989598699, 20.1069898676222, 14.3011351453467, 18.8126473489499, 17.0373115356076, 18.6684373282353, 18.8357213162278, 19.5036039713438, 21.2438389441807, 18.2089748680659, 18.0433963046778, 8.62940156299093, 16.2007030552903, 23.2646987822343, 24.1115616621798, 28.4351972435186])  # initial position of the BEC (in um) -> assumes that lab frame origin is the center of the TiSa beam (to the left of BEC)
    K_displacement_raw = np.array([0.473502276902047, 0.395634326123081, 8.66936929134637, 11.1470221226478, 9.34778274195669, 16.4370036199872, 19.0938486958001, 18.2135041439547, 21.9211790347041, 20.6591098913628, 19.7281375591975, 17.5425503131171, 17.2460344933717, 11.7179407507981, 12.9845862662090, 9.18113956217101, 11.9396846941782, 4.72461841775226])  # initial position of the impurity (in um)
    K_displacement_scale = np.mean(K_displacement_raw[6:11] / Na_displacement[6:11])
    K_displacement = deepcopy(K_displacement_raw); K_displacement[0:6] = K_displacement_scale * Na_displacement[0:6]; K_displacement[11::] = K_displacement_scale * Na_displacement[11::]   # in um
    K_relPos = K_displacement - Na_displacement   # in um

    omega_Na = np.array([465.418650581347, 445.155256942448, 461.691943131414, 480.899902898451, 448.655522184374, 465.195338759998, 460.143258369460, 464.565377197007, 465.206177963899, 471.262139163205, 471.260672147216, 473.122081065092, 454.649394420577, 449.679107889662, 466.770887179217, 470.530355145510, 486.615655444221, 454.601540658640])   # in rad*Hz
    omega_x_Na = 2 * np.pi * 100; omega_z_Na = 2 * np.pi * 12   # trap frequencies in rad*Hz
    omega_K_raw = np.array([764.649207995890, 829.646158322623, 799.388442120805, 820.831266284088, 796.794204312379, 810.331402280747, 803.823888714144, 811.210511844489, 817.734286423120, 809.089608774626, 807.885837386121, 808.334196591376, 782.788534907910, 756.720677755942, 788.446619623011, 791.774719564856, 783.194731826180, 754.641677886382])   # in rad*Hz
    omega_K_scale = np.mean(omega_K_raw[6:11] / omega_Na[6:11])
    # omega_K = deepcopy(omega_K_raw); omega_K[0:6] = omega_K_scale * omega_Na[0:6]; omega_K[11::] = omega_K_scale * omega_Na[11::]  # in rad*Hz
    omega_K = omega_K_raw
    omega_x_K = 2 * np.pi * 141; omega_y_K = 2 * np.pi * 130; omega_z_K = 2 * np.pi * 15  # should get more accurate estimate for omega_x_K

    K_relVel = np.array([1.56564660488838, 1.31601642026105, 0.0733613860991014, 1.07036861258786, 1.22929932184982, -13.6137940945403, 0.0369377794311800, 1.61258456681232, -1.50457700049200, -1.72583008593939, 4.11884512615162, 1.04853747806043, -0.352830359266360, -4.00683426531578, 0.846101589896479, -0.233660196108278, 4.82122627459411, -1.04341939663180])  # in um/ms

    phi_Na = np.array([-0.2888761, -0.50232022, -0.43763589, -0.43656233, -0.67963017, -0.41053479, -0.3692152, -0.40826816, -0.46117853, -0.41393032, -0.53483635, -0.42800711, -0.3795508, -0.42279337, -0.53760432, -0.4939509, -0.47920687, -0.51809527])  # phase of the BEC oscillation in rad
    gamma_Na = np.array([4.97524294, 14.88208436, 4.66212187, 6.10297397, 7.77264927, 4.5456649, 4.31293083, 7.28569606, 8.59578888, 3.30558254, 8.289436, 4.14485229, 7.08158476, 4.84228082, 9.67577823, 11.5791718, 3.91855863, 10.78070655])  # decay rate of the BEC oscillation in Hz

    N_K = np.array([2114.31716217314, 3040.54086059863, 3788.54290366850, 2687.53370686094, 2846.49206660163, 1692.49722769915, 1813.12703968803, 2386.60764443984, 2532.45824159990, 2361.26046445201, 2466.63648224567, 2206.34584323146, 2113.15620874362, 3755.19098529495, 2163.29615872937, 2042.58962172497, 4836.09854876457, 3044.93792941312])  # total number of fermions in K gas
    TFermi = np.array([6.83976585132807e-08, 7.93313829893224e-08, 8.43154444077350e-08, 7.58635297351284e-08, 7.65683267650816e-08, 6.47481434584840e-08, 6.60734255262424e-08, 7.26332216239745e-08, 7.42817184102838e-08, 7.23120402195269e-08, 7.33357082077064e-08, 7.06727442566945e-08, 6.89216704173642e-08, 8.25441536498287e-08, 6.96294877404586e-08, 6.84055531750863e-08, 9.08417325299114e-08, 7.69018614503965e-08])  # Fermi temperature of K gas (in K)
    T_K_ratio = np.array([1.16963068237879, 1.00842815271187, 0.948817865599258, 1.05452514903161, 1.04481844360328, 1.23555666196507, 1.21077421615179, 1.10142436492992, 1.07698100841087, 1.10631645514542, 1.09087376334348, 1.13197811746813, 1.16073797276748, 0.969178269600757, 1.14893851148521, 1.16949569569648, 0.880652512584549, 1.04028691232139])  # Ratio of temperature T to Fermi temperature T_Fermi of K gas
    T = 80e-9  # Temperature T of K gas (in K) --> equivalent to T_K_ratio * T_Fermi
    mu_div_hbar_K = -1 * np.array([21527.623521898644, 17656.025221467124, 15298.569367268587, 18973.981143581444, 18360.701066883277, 23888.301168354345, 23158.661546706127, 20239.341737009476, 19607.6059603436, 20352.99023696009, 19888.153905968644, 21074.805169679148, 21533.45904566066, 15393.579214021502, 21284.26382771103, 21894.22770364862, 12666.509194815215, 17640.573345313787])  # Chemical potential of the K gas (in rad*Hz) - computed using the code below
    # print(mu_div_hbar_K / (2 * np.pi)/1e3)
    print(T)

    # n0_BEC = np.zeros(aIBexp_Vals.size)
    # for inda, a in enumerate(aIBexp_Vals):
    #     n0_BEC[inda] = pf_dynamic_sph.becdensity_zw2021(0, 0, 0, omega_x_Na, omega_Na[inda], omega_z_Na, T, RTF_BEC_Z[inda])  # computes BEC density in experimental units
    # print(n0_BEC)

    # Optical trap experimental parameters (for K gas)

    A_ODT1_uK = -4.25; A_ODT2_uK = -3.44; A_TiSa_uK = -6.9  # Amplitude of each gaussian beam in uK
    A_ODT1_Hz = kB * A_ODT1_uK * 1e-6 / hbar / (2 * np.pi); A_ODT2_Hz = kB * A_ODT2_uK * 1e-6 / hbar / (2 * np.pi); A_TiSa_Hz = kB * A_TiSa_uK * 1e-6 / hbar / (2 * np.pi)  # Amplitude of each gaussian beam in Hz
    wx_ODT1 = 82; wy_ODT1 = 82  # beam waists of ODT1 in um
    ep_ODT2 = 4  # ellipticity of ODT2 beam
    wx_ODT2 = 144; wz_ODT2 = wx_ODT2 / ep_ODT2  # beam waists of ODT2 in um
    wx_TiSa = 95; wy_TiSa = 95  # beam waists of TiSa in um

    A_ODT1_Na_uK = -1.66; A_ODT2_Na_uK = -1.35; A_TiSa_Na_uK = -1.39  # Amplitude of each gaussian beam in uK (for Na)
    A_ODT1_Na_Hz = kB * A_ODT1_Na_uK * 1e-6 / hbar / (2 * np.pi); A_ODT2_Na_Hz = kB * A_ODT2_Na_uK * 1e-6 / hbar / (2 * np.pi); A_TiSa_Na_Hz = kB * A_TiSa_Na_uK * 1e-6 / hbar / (2 * np.pi)  # Amplitude of each gaussian beam in Hz

    def omega_gaussian_approx(A, m):
        return np.sqrt((-4) * hbar * 2 * np.pi * A / (m * (wy_TiSa * 1e-6)**2))

    scale_fac = np.array([0.9191250336816127, 1.0820215784890415, 1.0045367702210797, 1.059151023960805, 0.998027347459867, 1.0322275524975513, 1.015715140919877, 1.0344684463876583, 1.0511737903469414, 1.0290662994361741, 1.0260064676580842, 1.027145606565003, 0.9632500076336702, 0.9001633188311512, 0.9772252948388005, 0.9854926080339469, 0.9642499474231473, 0.8952239304967515])  # fitting the optical trap depth to the K oscillation frequencies
    # scale_fac = np.array([0.9724906988829373, 0.8896535250404035, 0.9569791653444897, 1.0382627423131257, 0.903699258873818, 0.971557704172668, 0.9505698121511785, 0.9689281438661883, 0.9716029799261697, 0.9970638982528262, 0.9970576906401939, 1.00494970248614, 0.9280067282060056, 0.9078274542675744, 0.9781498957395225, 0.9939697862098377, 1.0630900294469736, 0.9278113852604492])  # fitting the optical trap depth to the Na oscillation frequencies
    A_TiSa_Hz_scaled = A_TiSa_Hz * scale_fac
    A_TiSa_Na_Hz_scaled = A_TiSa_Na_Hz * scale_fac

    ODT1_displacement = np.array([39.9734769508128, 37.20726134699691, 29.022743967492712, 32.85605962371015, 23.00479821032066, 30.475997313212293, 27.49539761274011, 30.277006179531572, 30.746034106569127, 31.517392916389632, 34.17496197024173, 29.467112794532262, 28.46260872772458, 13.428923709748158, 25.777101525763207, 36.645281366522546, 37.56837023644184, 42.51753230100077])  # initial position of the ODT1 beam (in um) before it is turned off -> assumes that lab frame origin is the center of the TiSa beam (to the left of the ODT1 beam)

    # print(omega_K / (2 * np.pi))
    # print(omega_gaussian_approx(A_TiSa_Hz, expParams['mI']) / (2 * np.pi))
    # print(omega_gaussian_approx(A_TiSa_Hz_scaled, expParams['mI']) / (2 * np.pi))
    # print(omega_Na / (2 * np.pi))
    # print(omega_gaussian_approx(A_TiSa_Hz_Na, expParams['mB']) / (2 * np.pi))
    # print(omega_gaussian_approx(A_TiSa_Hz_Na_scaled, expParams['mB']) / (2 * np.pi))
    # print(np.sqrt((-4) * hbar * 2 * np.pi * A_TiSa_Hz / (expParams['mI'] * (wx_TiSa * 1e-6)**2)) / (2 * np.pi))
    # print(np.sqrt((-4) * hbar * 2 * np.pi * A_ODT2_Hz / (expParams['mI'] * (wx_ODT2 * 1e-6)**2)) / (2 * np.pi))

    # sf_List = []
    # omega_fit = omega_K
    # # omega_fit = omega_Na
    # for indw, wy in enumerate(omega_fit):
    #     def f(sf): return (omega_gaussian_approx(sf * A_TiSa_Hz, expParams['mI']) - wy)
    #     # def f(sf): return (omega_gaussian_approx(sf * A_TiSa_Na_Hz, expParams['mB']) - wy)
    #     sol = root_scalar(f, bracket=[0.5, 1.5], method='brentq')
    #     sf_List.append(sol.root)
    # print(sf_List)

    # def UNa_deriv(yODT1, yNa, A_ODT1, w_ODT1, A_TiSa, w_TiSa):
    #     A_ODT1_en = hbar * 2 * np.pi * A_ODT1
    #     A_TiSa_en = hbar * 2 * np.pi * A_TiSa
    #     return (-4 * A_TiSa_en / (w_TiSa**2)) * yNa * np.exp(-2 * (yNa / w_TiSa)**2) + (-4 * A_ODT1_en / (w_ODT1**2)) * (yNa - yODT1) * np.exp(-2 * ((yNa - yODT1) / w_ODT1)**2)

    # yODT1_List = []
    # for indy, yNa in enumerate(Na_displacement*np.cos(phi_Na)):
    #     def f(yODT1): return UNa_deriv(yODT1 * 1e-6, yNa * 1e-6, A_ODT1_Na_Hz, wy_ODT1 * 1e-6, A_TiSa_Na_Hz_scaled[indy], wy_TiSa * 1e-6)
    #     # print(indy, f(0 * yNa), f(2 * yNa))
    #     sol = root_scalar(f, bracket=[0 * yNa, 2 * yNa], method='brentq')
    #     yODT1_List.append(sol.root)
    # print(yODT1_List)
    # print(A_ODT1_Hz/A_ODT1_Na_Hz)
    # print(A_TiSa_Hz/A_TiSa_Na_Hz_scaled)

    # # Compute chemical potential of the K gas

    # # mu_K = -1 * kB * T * np.log(np.exp(N_K) - 1) / hbar  # chemical potential of K gas (in rad*Hz)
    # # mu_K = -1 * kB * T * N_K / hbar  # chemical potential of K gas (in rad*Hz) -- approximation of above line since np.exp(N_K) >> 1. Get chemical potential ~5 MHz compared to ~1 kHz for Na gas
    # # print(mu_K / (2 * np.pi) * 1e-6)

    # def N3D(mu_div_hbar, omega_x, omega_y, omega_z, T):
    #     mu = hbar * mu_div_hbar
    #     beta = 1 / (kB * T)
    #     prefac = 1 / ((hbar**3) * (beta**3) * np.sqrt((omega_x**2) * (omega_y**2) * (omega_z**2)))
    #     return -1 * prefac * polylog(3, -1 * np.exp(beta * mu))

    # mu_div_hbar_List = []
    # for indw, wy in enumerate(omega_K):
    #     muScale = 2 * np.pi * 1e3
    #     # n = N3D(muScale, omega_x_K, omega_y_K, omega_z_K, T)
    #     def f(mu): return (N3D(mu, omega_x_K, omega_y_K, omega_z_K, T) - N_K[indw])
    #     sol = root_scalar(f, bracket=[-10 * muScale, 10 * muScale], method='brentq')
    #     # print(sol.root / (2 * np.pi)); n = N3D(sol.root, omega_x_K, omega_y_K, omega_z_K, T); print(N_K[indw], n)
    #     mu_div_hbar_List.append(sol.root)
    # print(mu_div_hbar_List)

    # # Density test - 10x faster to evaluate spline than compute density directly using polylog

    # xMat = loadmat('zwData/xMat_xy_InMuM_N_1000.mat')['zMat']  # grid of positions in the BEC oscillation direction ranging from -4*R_TF_X to +4*R_TF_X for each interaction strength (given in in um)
    # yMat = loadmat('zwData/yMat_xy_InMuM_N_1000.mat')['yMat']  # grid of positions in the BEC oscillation direction ranging from -4*R_TF_Y to +4*R_TF_Y for each interaction strength (given in in um)
    # densityMat = loadmat('zwData/densityMat_xy_InM-3_N_1000.mat')['densityMat']  # 2D slices (z=0) of BEC density for each experimentally measured interaction strength evaluated at each position in xMat and yMat (given in in m^(-3))

    # inda = 3
    # RTF_X = RTF_BEC_X[inda]; RTF_Y = RTF_BEC_Y[inda]; RTF_Z = RTF_BEC_Z[inda]
    # xVals = np.linspace(-4 * RTF_X, 4 * RTF_X, 10000)
    # yVals = np.linspace(-4 * RTF_Y, 4 * RTF_Y, 10000)

    # xInd = 10000 // 2
    # denVals = np.zeros((xVals.size, yVals.size))
    # start = timer()
    # for yInd, y in enumerate(yVals):
    #     denVals[xInd, yInd] = pf_dynamic_sph.becdensity_zw2021(xVals[xInd], y, 0, omega_x_Na, omega_Na[inda], omega_z_Na, T, RTF_Z)
    # print(timer() - start)

    # import matplotlib
    # import matplotlib.pyplot as plt
    # from scipy import interpolate

    # den_tck = interpolate.splrep(yVals, denVals[xInd, :], s=0)
    # yVals_interp = np.linspace(-4 * RTF_Y, 4 * RTF_Y, 1000)
    # den_interp = interpolate.splev(yVals_interp, den_tck)

    # s1 = timer()
    # d1 = pf_dynamic_sph.becdensity_zw2021(xVals[xInd], yVals[1000], 0, omega_x_Na, omega_Na[inda], omega_z_Na, T, RTF_Z)
    # f1 = timer()
    # print(f1 - s1)
    # s2 = timer()
    # d2 = interpolate.splev(yVals[1000], den_tck)
    # f2 = timer()
    # print(f2 - s2)
    # print((f1 - s1) / (f2 - s2))
    # print(d1, d2)

    # fig, ax = plt.subplots()
    # ax.plot(yVals, denVals[xInd, :])
    # ax.plot(yVals_interp, den_interp)
    # plt.show()

    # # # Create density splines

    # from scipy import interpolate
    # xMat = loadmat('zwData/xMat_xy_InMuM_N_5000.mat')['zMat']  # grid of positions in the BEC oscillation direction ranging from -4*R_TF_X to +4*R_TF_X for each interaction strength (given in in um)
    # yMat = loadmat('zwData/yMat_xy_InMuM_N_1000.mat')['yMat']  # grid of positions in the BEC oscillation direction ranging from -4*R_TF_Y to +4*R_TF_Y for each interaction strength (given in in um)
    # densityMat = loadmat('zwData/densityMat_xy_InM-3_N_1000.mat')['densityMat']  # 2D slices (z=0) of BEC density for each experimentally measured interaction strength evaluated at each position in xMat and yMat (given in in m^(-3))
    # N = yMat[0,:].size

    # xMat_th = xMat * 1e-6 * L_exp2th  # converts grid positions from um to m and then converts to theory units
    # yMat_th = yMat * 1e-6 * L_exp2th  # converts grid positions from um to m and then converts to theory units
    # densityMat_th = densityMat / (L_exp2th**3)  # converts BEC density arrays (for each interaction strength) to theory units

    # # for ind, aIB_exp in enumerate(aIBexp_Vals):
    # #     start=timer()

    # #     xVec = xMat_th[ind,:]; yVec = yMat_th[ind,:]
    # #     RTF_X = np.max(xVec)/4; RTF_Y = np.max(yVec)/4
    # #     xMask_inner = (xVec >= -RTF_X) * (xVec <= RTF_X)
    # #     yMask_inner = (yVec >= -RTF_Y) * (yVec <= RTF_Y)
    # #     xVec_inner = xVec[xMask_inner]; yVec_inner = yVec[yMask_inner]
    # #     xVec_outer = xVec[np.logical_not(xMask_inner)]; yVec_outer = yVec[np.logical_not(yMask_inner)]
    # #     xVec_downsample = np.sort(np.concatenate((xVec_inner[::10], xVec_outer[::30]))); yVec_downsample = np.sort(np.concatenate((yVec_inner[::10], yVec_outer[::30])))
    # #     xInds = np.where(np.in1d(xVec, xVec_downsample))[0]; yInds = np.where(np.in1d(yVec, yVec_downsample))[0]

    # #     xg, yg = np.meshgrid(xVec, yVec, indexing='ij')
    # #     xg_downsample = xg[xInds,:][:,yInds]; yg_downsample = yg[xInds,:][:,yInds]
    # #     den = densityMat_th[ind,:,:]
    # #     print(xg_downsample.shape)
    # #     den_tck = interpolate.bisplrep(xg_downsample, yg_downsample, den[xInds,:][:,yInds],s=0)
    # #     np.save('zwData/densitySplines/nBEC_aIB_{0}a0.npy'.format(aIB_exp), den_tck)

    # #     print(timer()-start)

    # import matplotlib
    # import matplotlib.pyplot as plt
    # for ind, aIB_exp in enumerate(aIBexp_Vals):
    #     if ind > 0:
    #         continue
    #     den_tck_load = np.load('zwData/densitySplines/div5/nBEC_aIB_{0}a0.npy'.format(aIB_exp), allow_pickle=True)
    #     xVec = np.linspace(xMat_th[ind, 0], xMat_th[ind, -1], 1000)
    #     yVec = np.linspace(yMat_th[ind, 0], yMat_th[ind, -1], 1000)
    #     xg, yg = np.meshgrid(xVec, yVec, indexing='ij')
    #     den = interpolate.bisplev(xVec, yVec, den_tck_load)
    #     fig, ax = plt.subplots()
    #     c = ax.pcolormesh(1e6 * xg / L_exp2th, 1e6 * yg / L_exp2th, den * (L_exp2th**3),cmap='RdBu')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     fig.colorbar(c, ax=ax)

    #     fig2, ax2 = plt.subplots()
    #     yVec_dat = yMat_th[ind,:]
    #     den_dat = densityMat[ind,N//2,:]
    #     den_slice_interp = interpolate.bisplev(0, yVec, den_tck_load)
    #     # ax2.plot(1e6 * yVec / L_exp2th, den_slice_interp * (L_exp2th**3))
    #     ax2.plot(1e6 * yVec_dat / L_exp2th, den_dat * (L_exp2th**3))
    #     ax2.set_ylabel('y')

    #     plt.show()

    # # Test density gradient evaluation

    # import numdifftools as nd

    # inda = 3

    # RTF_BEC_X_th = RTF_BEC_X[inda] * 1e-6 * L_exp2th
    # RTF_BEC_Y_th = RTF_BEC_Y[inda] * 1e-6 * L_exp2th 

    # omegaX = omega_x_Na
    # omegaY = omega_Na[inda]
    # omegaZ = omega_z_Na
    # zTF = RTF_BEC_Z[inda]
    # xTF = zTF * omegaZ / omegaX
    # yTF = zTF * omegaZ / omegaY

    # # fun = lambda coords: pf_dynamic_sph.becdensity_zw2021(coords[0] * (1e6) / L_exp2th, coords[1] * (1e6) / L_exp2th, 0, omega_x_Na, omega_Na[inda], omega_z_Na, T, RTF_BEC_Z[inda]) / (L_exp2th**3)
    # # dfun = nd.Gradient(fun, method='central')

    # fun = lambda coords: pf_dynamic_sph.becdensity_zw2021(coords[0] * (1e6) / L_exp2th, coords[1] * (1e6) / L_exp2th, 0, omega_x_Na, omega_Na[inda], omega_z_Na, T, RTF_BEC_Z[inda]) / (L_exp2th**3)
    # dfun = lambda coords: pf_dynamic_sph.becdensityGrad_zw2021(coords[0] * (1e6) / L_exp2th, coords[1] * (1e6) / L_exp2th, 0, omega_x_Na, omega_Na[inda], omega_z_Na, T, RTF_BEC_Z[inda]) / (L_exp2th**4)

    # # # Check slices along x

    # # xVals = np.linspace(-0.9 * RTF_BEC_X_th, 0.9 * RTF_BEC_X_th, 3)
    # # yVals = np.linspace(-2 * RTF_BEC_Y_th, 2 * RTF_BEC_Y_th, 1000)

    # # for indx, x in enumerate(xVals):
    # #     denVals = np.zeros(yVals.size)
    # #     ndVals = np.zeros(yVals.size)
    # #     for indy, y in enumerate(yVals):
    # #         x_MuM = x * (1e6) / L_exp2th; y_MuM = y * (1e6) / L_exp2th; 
    # #         n_exp = pf_dynamic_sph.becdensity_zw2021(x_MuM, y_MuM, 0, omega_x_Na, omega_Na[inda], omega_z_Na, T, RTF_BEC_Z[inda])  # computes BEC density in experimental units
    # #         # denVals[indy] = n_exp / (L_exp2th**3)  # converts density in SI units to theory units
    # #         denVals[indy] = fun([x, y])
    # #         ndVals[indy] = dfun([x,y])[1]

    # #     nBEC_tck = interpolate.splrep(yVals, denVals, s=0)

    # #     print(x_MuM / RTF_BEC_X[inda])
    # #     n = interpolate.splev(yVals, nBEC_tck)
    # #     dndy = interpolate.splev(yVals, nBEC_tck, der=1)
    # #     dgrad = np.gradient(denVals, yVals)
    # #     print(np.max(ndVals)/np.max(dndy), np.max(ndVals)/np.max(dgrad), np.max(dndy)/np.max(dgrad))
    # #     fig, ax = plt.subplots(ncols=2)
    # #     ax[0].plot(yVals/RTF_BEC_Y_th, n)
    # #     ax[0].plot(yVals/RTF_BEC_Y_th, denVals,'--')
    # #     ax[1].plot(yVals/RTF_BEC_Y_th,dndy)
    # #     ax[1].plot(yVals/RTF_BEC_Y_th,ndVals,'--')
    # #     ax[1].plot(yVals/RTF_BEC_Y_th,dgrad,'--')

    # #     plt.show()

    # # # Check slices along y

    # # xVals = np.linspace(-2 * RTF_BEC_X_th, 2 * RTF_BEC_X_th, 1000)
    # # yVals = np.linspace(-0.9 * RTF_BEC_Y_th, 0.9 * RTF_BEC_Y_th, 5)

    # # for indy, y in enumerate(yVals):
    # #     denVals = np.zeros(xVals.size)
    # #     ndVals = np.zeros(xVals.size)
    # #     for indx, x in enumerate(xVals):
    # #         x_MuM = x * (1e6) / L_exp2th; y_MuM = y * (1e6) / L_exp2th; 
    # #         n_exp = pf_dynamic_sph.becdensity_zw2021(x_MuM, y_MuM, 0, omega_x_Na, omega_Na[inda], omega_z_Na, T, RTF_BEC_Z[inda])  # computes BEC density in experimental units
    # #         # denVals[indx] = n_exp / (L_exp2th**3)  # converts density in SI units to theory units
    # #         denVals[indx] = fun([x, y])
    # #         ndVals[indx] = dfun([x,y])[0]

    # #     nBEC_tck = interpolate.splrep(xVals, denVals, s=0)

    # #     print(y / RTF_BEC_Y_th)
    # #     n = interpolate.splev(xVals, nBEC_tck)
    # #     dndx = interpolate.splev(xVals, nBEC_tck, der=1)
    # #     dgrad = np.gradient(denVals, xVals)
    # #     print(np.max(ndVals)/np.max(dndx), np.max(ndVals)/np.max(dgrad), np.max(dndx)/np.max(dgrad))
    # #     fig, ax = plt.subplots(ncols=2)
    # #     ax[0].plot(xVals/RTF_BEC_X_th, n)
    # #     ax[0].plot(xVals/RTF_BEC_X_th, denVals,'--')
    # #     ax[1].plot(xVals/RTF_BEC_X_th,dndx)
    # #     ax[1].plot(xVals/RTF_BEC_X_th,ndVals,'--')
    # #     ax[1].plot(xVals/RTF_BEC_X_th,dgrad,'--')

    # #     plt.show()


    # Convert experimental parameters to theory parameters

    n0 = n0_BEC / (L_exp2th**3)  # converts peak BEC density (for each interaction strength) to theory units
    mB = expParams['mB'] * M_exp2th  # should = 1
    mI = expParams['mI'] * M_exp2th
    aBB = expParams['aBB'] * L_exp2th
    gBB = (4 * np.pi / mB) * aBB
    nu = pf_dynamic_sph.nu(mB, n0, gBB)

    y0_imp = K_relPos * 1e-6 * L_exp2th  # initial positions of impurity in BEC frame (relative to the BEC)
    v0_imp = K_relVel * (1e-6 / 1e-3) * (L_exp2th / T_exp2th)  # initial velocities of impurity in BEC frame (relative to BEC)
    RTF_BEC_Y_th = RTF_BEC_Y * 1e-6 * L_exp2th  # BEC oscillation amplitude (carries units of position)
    RTF_BEC_X_th = RTF_BEC_X * 1e-6 * L_exp2th  # BEC oscillation amplitude (carries units of position)

    omega_BEC_osc = omega_Na / T_exp2th
    gamma_BEC_osc = gamma_Na / T_exp2th
    phi_BEC_osc = phi_Na
    amp_BEC_osc = (Na_displacement * 1e-6 * L_exp2th) / np.cos(phi_Na)  # BEC oscillation amplitude (carries units of position)
    y0_BEC_lab = Na_displacement * 1e-6 * L_exp2th  # starting position of the BEC in the lab frame (lab frame origin = center of the TiSa beam)
    y0_imp_lab = y0_imp + y0_BEC_lab  # starting position of the impurity in the lab frame (lab frame origin = center of the TiSa beam)
    y0_ODT1_lab = ODT1_displacement * 1e-6 * L_exp2th  # starting position of the ODT1 beam in the lab frame (lab frame origin = center of the TiSa beam)

    omega_Imp_y = omega_K / T_exp2th

    # gaussian_amp = 2 * np.pi * A_TiSa_Hz * np.ones(aIBexp_Vals) / T_exp2th  # converts optical potential amplitude to radHz, then converts this frequency to theory units. As we choose hbar = 1 in theory units, this is also units of energy.
    gaussian_amp = 2 * np.pi * A_TiSa_Hz_scaled / T_exp2th  # converts optical potential amplitude to radHz, then converts this frequency to theory units. As we choose hbar = 1 in theory units, this is also units of energy.
    gaussian_width = wy_TiSa * 1e-6 * L_exp2th

    a0_exp = 5.29e-11  # Bohr radius (m)
    aIBi_Vals = 1 / (aIBexp_Vals * a0_exp * L_exp2th)

    P0_scaleFactor = np.array([1.2072257743801846, 1.1304777274446096, 1.53, 1.0693683091221053, 1.0349159867400886, 0.95, 2.0, 1.021253131703251, 0.9713134192266438, 0.9781007832739641, 1.0103135855263197, 1.0403095335853234, 0.910369833990368, 0.9794720983829749, 1.0747443076336567, 0.79, 1.0, 0.8127830658214898])  # scales the momentum of the initial polaron state (scaleFac * mI * v0) so that the initial impurity velocity matches the experiment. aIB= -125a0 and +750a0 have ~8% and ~20% error respectively
    P0_imp = P0_scaleFactor * mI * v0_imp

    # print(P0_imp)
    # print(mI * nu)

    # print(K_relPos)
    # print(y0_imp)

    # # Create density splines (ground state PB and aSi vs DP) for peak BEC density (n0). Also compute the E_pol_gs offset (min(E_pol_gs) occurs at momentum P=0 and BEC density n=max(n)=n0)

    # E_pol_offset = np.zeros(aIBexp_Vals.size)
    # Nsteps = 1e2
    # for ind, aIB_exp in enumerate(aIBexp_Vals):
    #     if aIB_exp >= 0:
    #         E_pol_offset[ind] = 0
    #         continue
    #     # aSi0_tck, PBint0_tck = pf_static_sph.createSpline_grid(Nsteps, kgrid, mI, mB, n0[ind], gBB)
    #     # np.save('zwData/densitySplines/n0/aSi0_aIB_{0}a0.npy'.format(aIB_exp), aSi0_tck)
    #     # np.save('zwData/densitySplines/n0/PBint0_aIB_{0}a0.npy'.format(aIB_exp), PBint0_tck)
    #     aSi0 = pf_static_sph.aSi_grid(kgrid, 0, mI, mB, n0[ind], gBB)
    #     PB0 = pf_static_sph.PB_integral_grid(kgrid, 0, mI, mB, n0[ind], gBB)
    #     E_pol_offset[ind] = pf_static_sph.Energy(0, PB0, aIBi_Vals[ind], aSi0, mI, mB, n0[ind])
    # print(E_pol_offset)

    # Sample positions and momenta

    A_TiSa_th = 2 * np.pi * A_TiSa_Hz_scaled / T_exp2th  # equivalent to gaussian_amp above
    A_ODT1_th = 2 * np.pi * A_ODT1_Hz / T_exp2th
    # A_ODT1_th = 0
    A_ODT2_th = 2 * np.pi * A_ODT2_Hz / T_exp2th
    wx_ODT1_th = wx_ODT1 * 1e-6 * L_exp2th; wy_ODT1_th = wy_ODT1 * 1e-6 * L_exp2th; wx_ODT2_th = wx_ODT2 * 1e-6 * L_exp2th; wz_ODT2_th = wz_ODT2 * 1e-6 * L_exp2th; wx_TiSa_th = wx_TiSa * 1e-6 * L_exp2th; wy_TiSa_th = wy_TiSa * 1e-6 * L_exp2th
    beta_exp = 1 / (kB * T / hbar)  # inverse temperature in units of 1/(rad*Hz) = s/rad
    beta_th = beta_exp * T_exp2th

    # U_opt_offset = np.array([-28.60814655, -31.21901016, -30.87067425, -31.3245341, -31.22762206, -31.15045883, -31.15850962, -31.20071966, -31.40204218, -31.0106113, -30.71828868, -31.16363115, -30.31259895, -30.27470958, -30.72544145, -29.89234401, -29.49573856, -28.0016819])  # (density is condensate and thermal gas) constant energy offset (theory units) of U_tot_opt to make sure the minimum value U_tot_opt(0,ymin,0) = 0. This offset is determined by numerically determining min(U_tot_opt(0,y,0))
    U_opt_offset = np.array([-28.60825064869891, -31.21994067822379, -30.8707107193229, -31.325119904209522, -31.228746220118186, -31.150586511632262, -31.15897717819624, -31.200727683513215, -31.403029121814292, -31.011718680361117, -30.718864141323024, -31.163711698459377, -30.313300309127857, -30.275118254553636, -30.72567395606398, -29.892556211531456, -29.495819535657738, -28.00228060325752])  # (density is just condensate) constant energy offset (theory units) of U_tot_opt to make sure the minimum value U_tot_opt(0,ymin,0) = 0. This offset is determined by numerically determining min(U_tot_opt(0,y,0))
    U0_opt_offset = A_ODT1_th + A_ODT2_th + A_TiSa_th  # constant energy offset (theory units) of U_tot_opt to make sure the minimum value U_tot_opt(0,ymin,0) = 0 when AODT1 = 0 (the ODT1 beam is turned off)
    # E_pol_offset = np.array([-1.18662877e+00, -8.41179486e-01, -6.87998107e-01, -4.88436591e-01, -3.64301316e-01, -2.12254441e-01, -9.73809669e-02, -3.16003180e-02, -8.62356415e-36, 3.11130148e-02, 7.32152511e-02, 1.88938399e-01, 2.68558712e-01, 3.73587857e-01, 5.83872838e-01, 8.28556730e-01, 1.11273234e+00, 1.66368733e+00])  # when density includes condensate and thermal gas
    # E_pol_offset = np.array([-1.17876723e+00, -8.34604734e-01, -6.84013469e-01, -4.85201407e-01, -3.62255174e-01, -2.11318340e-01, -9.69039151e-02, -3.14356780e-02, -5.28352867e-37,  3.09442108e-02,  7.27691389e-02,  1.87824142e-01, 2.66992312e-01,  3.71253309e-01,  5.80338018e-01,  8.23862869e-01, 1.10454958e+00,  1.65267226e+00])  # when density is just of the condensate
    E_pol_offset = np.array([-1.17876723e+00, -8.34604734e-01, -6.84013469e-01, -4.85201407e-01, -3.62255174e-01, -2.11318340e-01, -9.69039151e-02, -3.14356780e-02, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0])  # when density is just of the condensate

    # f_thermal_maxVals = np.array([0.7367184970676869, 0.7023880636086897, 0.7459548034386865, 0.79242602587235, 0.8340965891576568, 0.8912926562637322, 0.8949215118953942, 0.8698744612202944, 0.8675373653753994, 0.8801537515385071, 0.885519273327338, 0.9098398084573248, 0.9212560518600181, 0.8316673727452937, 0.9746952226616081, 0.997698493028504, 0.9991709720238214, 0.9999781206212883])  # Numerically determined maximum value of f_thermal for each interaction (occurs at x=0, y=yMax, z=0, p=0 where yMax is given below)
    # y_thermal_maxVals = np.array([-12.358738440529192, -15.142125013161545, -13.144260996614438, -16.589728903343367, -13.283761232947477, -21.15224812723023, -20.742902013227198, -24.997818462639284, -26.498895342912757, -28.469193967012014, -33.970099935372986, -33.66816313625496, -36.2296199300486, -18.44950460158087, -55.65474817181998, -56.536254230789034, -50.99473895410588, -56.758005648192395])  # the y values y=yMax at which f_thermal has a maximum (positive Mu)

    # f_thermal_maxVals = np.array([0.04371091507791199, 0.07475837588152232, 0.13625255887903176, 0.09222453506832799, 0.13076334263340406, 0.07859344842990887, 0.09244448365158987, 0.1225636725220138, 0.1337588086527938, 0.13055540035829122, 0.14737558497429884, 0.15235876040489293, 0.16029655986623156, 0.20675607487658249, 0.3972825025010302, 0.8684560204142703, 0.9907473907238571, 0.9993631937004797])  # Numerically determined maximum value of f_thermal for each interaction (occurs at x=0, y=yMax, z=0, p=0 where yMax is given below) [T = 80 nK]
    # y_thermal_maxVals = np.array([-12.358738846077586, -15.14212490509725, -13.144260921615512, -16.58972888463607, -13.283761360003684, -21.15224703048301, -20.742905297539455, -24.997818472525328, -26.498896331698784, -28.469197495859653, -33.97009782737399, -33.66816330606461, -36.2296200495234, -18.449504042252958, -55.6547478236438, -56.53625335500957, -50.994739542476275, -56.75799513288452])  # the y values y=yMax at which f_thermal has a maximum [T = 80 nK] (density is condensate + thermal)

    # f_thermal_maxVals = np.array([0.04360501531188545, 0.07423627004068983, 0.13610859139446085, 0.09182274260636307, 0.12998302782024626, 0.07849750955959393, 0.09219732158187448, 0.12254222660907568, 0.13310938618629403, 0.12986652280788644, 0.1470784775705773, 0.15258326647475098, 0.16020672201743388, 0.20654702353157303, 0.40463678278705134, 0.8685179744643099, 0.9906004824193367, 0.9993326857041089])  # Numerically determined maximum value of f_thermal for each interaction (occurs at x=0, y=yMax, z=0, p=0 where yMax is given below) [T = 80 nK]
    # y_thermal_maxVals = np.array([-12.310607736190837, -15.088006758036995, -13.113276449372993, -16.549741670703483, -13.265427968228, -21.13428630190987, -20.733336045364, -24.993011705445387, -26.498896082566567, -28.476128183706134, -33.999400717547196, -33.7479660563224, -36.36171863701398, -18.5225207508263, -55.19001187934648, -55.949291403532044, -50.39113955063414, -55.97058158931767])  # the y values y=yMax at which f_thermal has a maximum [T = 80 nK] (density is just condensate)

    f_thermal_maxVals = np.array([0.043605015311886194, 0.07423627004069085, 0.1361085913944598, 0.09182274260636307, 0.12998302782024512, 0.07849750955959416, 0.09219732158187481, 0.12254222660907683, 0.1331093861862952, 0.11123036742563963, 0.10231059870341767, 0.05823118071011247, 0.040076160211829914, 0.03051433752079719, 0.024649193052760223, 0.06043817659525932, 0.17758415129620675, 0.1249981760849305])
    y_thermal_maxVals = np.array([-12.310607817543131, -15.088006673500407, -13.113276319803973, -16.549741644108156, -13.265428231999312, -21.134286502941773, -20.733336172212375, -24.993012099767803, -26.498896338575584, -28.476128846728432, -33.999401509672516, -33.747966631241525, -36.361726062306474, -18.52251790084638, -55.7474894286806, -56.514441692738274, -50.90014689465987, -56.535953938856046])

    # f_thermal_maxVals = np.array([0.04360501531169881, 0.07423627004053718, 0.13610859139410567, 0.09182274260621211, 0.12998302781971663, 0.0784975095594019, 0.0921973215816414, 0.12254222660879804, 0.13310938618614587, 0.11123036742547981, 0.10231059870333899, 0.05823118070999939, 0.040076160211768096, 0.030514337520763858, 0.02464919488645606, 0.06043818202879863, 0.17758415265276545, 0.12499819573605757])
    # y_thermal_maxVals = np.array([-12.31058850810297, -15.088025052975922, -13.113273308950165, -16.549767540237884, -13.265409968410175, -21.134262235115173, -20.733358828979426, -24.993021554605832, -26.498905219081237, -28.476104294810256, -33.999377291733026, -33.74799171913338, -36.36168024321722, -18.522515249904494, -55.74749070725707, -56.51443997036429, -50.90014665499372, -56.535948430555266])
    # x_thermal_maxVals = np.array([-2.3736941359589154e-05, 1.588876936985104e-05, 2.7488943858647048e-05, 7.913003247694235e-06, -3.708408207065498e-05, 2.6413803527583937e-05, 3.108020929578367e-05, -3.6415173040755955e-05, -2.5564260198023278e-05, -2.275845816817349e-05, -1.2074836714797555e-05, 3.53577807465093e-05, 6.377605683146194e-06, -5.118896740736545e-05, -0.002810005050657696, 0.00039005153833080976, -6.466323088919726e-05, 5.917094406243484e-05])

    # f_thermal_maxVals = np.array([3.6378912844041568e-06, 3.652290496943756e-05, 0.0005640245814312913, 9.857039995630772e-05, 0.0004881293664369387, 5.171451969872132e-05, 0.00010638286166695999, 0.0003789391523830838, 0.0005681819252681373, 0.0005104871818073756, 0.0009037638619273189, 0.001077650557330782, 0.0013902800196785602, 0.004857527551207323, 0.19032061782479667, 0.9996093970094144, 0.9999999951103646, 0.9999999999999079])  # Numerically determined maximum value of f_thermal for each interaction (occurs at x=0, y=yMax, z=0, p=0 where yMax is given below) [T = 20 nK]
    # y_thermal_maxVals = np.array([-12.313756173386173, -15.092081246922893, -13.115154653923879, -16.55256620596325, -13.266519101775238, -21.135281524323595, -20.733909803778854, -24.993339840222134, -26.498895611802926, -28.475590509794532, -33.99651731075096, -33.74027937637036, -36.34859113208791, -18.51735283512218, -55.740980406094394, -56.51580566054858, -50.90619908248219, -56.54844009927777])  # the y values y=yMax at which f_thermal has a maximum [T = 20 nK]


    inda = 6
    true2D = True

    sampleParams = {'omegaX_radHz': omega_x_Na, 'omegaY_radHz': omega_Na[inda], 'omegaZ_radHz': omega_z_Na, 'temperature_K': T, 'zTF_MuM': RTF_BEC_Z[inda], 'y0_BEC': y0_BEC_lab[inda], 'y0_ODT1': y0_ODT1_lab[inda], 'omega_Imp_y': omega_Imp_y[inda], 'n0_BEC_m^-3': n0_BEC[inda], 'L_exp2th': L_exp2th,
                    'U_opt_offset': U_opt_offset[inda], 'U0_opt_offset': U0_opt_offset[inda], 'E_pol_offset': E_pol_offset[inda], 'A_ODT1': A_ODT1_th, 'wx_ODT1': wx_ODT1_th, 'wy_ODT1': wy_ODT1_th, 'A_ODT2': A_ODT2_th, 'wx_ODT2': wx_ODT2_th, 'wz_ODT2': wz_ODT2_th, 'A_TiSa': A_TiSa_th[inda], 'wx_TiSa': wx_TiSa_th, 'wy_TiSa': wy_TiSa_th}
    cParams = {'aIBi': aIBi_Vals[inda]}
    sParams = [mI, mB, n0[inda], gBB]
    mu_th = mu_div_hbar_K[inda] / T_exp2th  # converts chemical potential in rad*Hz to theory units

    y0 = y0_imp[inda]; p0 = P0_imp[inda]  # desired mean starting position and total momentum of the initial polaron for motion across the x=z=0 slice of the density (in theory units)

    # xMin = -1 * RTF_BEC_X_th[inda]; xMax = 1 * RTF_BEC_X_th[inda]
    # yMin = -1 * RTF_BEC_Y_th[inda]; yMax = 1 * RTF_BEC_Y_th[inda]
    xMin = -2 * RTF_BEC_X_th[inda]; xMax = 2 * RTF_BEC_X_th[inda]
    yMin = -2 * RTF_BEC_Y_th[inda]; yMax = 2 * RTF_BEC_Y_th[inda]


    pMin = -1 * mI * nu[inda]; pMax = 1 * mI * nu[inda]

    fMax = f_thermal_maxVals[inda]
    print(aIBexp_Vals[inda], y0, p0/(mI * nu[inda]))
    print(xMax, yMax, pMax, fMax)

    # Ns = 1000  # number of desired samples
    # evals = 0
    # counter = 0
    # if true2D:
    #     samples = np.zeros((Ns, 4))
    # else:
    #     samples = np.zeros((Ns, 3))

    # start = timer()
    # while counter < Ns:
    #     f = np.random.uniform(low=0, high=fMax)
    #     x = np.random.uniform(low=xMin, high=xMax)
    #     y = np.random.uniform(low=yMin, high=yMax)
    #     py = np.random.uniform(low=pMin, high=pMax)
    #     px = np.random.uniform(low=pMin, high=pMax)

    #     # px, py, pz = np.random.uniform(low=pMin,high=pMax, size=3)
    #     # p = np.sqrt(px**2 + py**2 + pz**2)
    #     # if p > pMax:
    #     #     continue

    #     # x = 0
    #     # y = y0
    #     # py = p0
    #     # px = 0

    #     if true2D:
    #         f_eval = pf_dynamic_sph.f_thermal_true2D(x, y, 0, px, py, beta_th, mu_th, kgrid, cParams, sParams, sampleParams)  # we assume we only sample initial particles with z=0, px=0, pz=0 (so p=sqrt(px^2+py^2+pz^2)=py)
    #         if f < f_eval:
    #             samples[counter, :] = [x, y, px, py]
    #             counter += 1

    #     else:
    #         f_eval = pf_dynamic_sph.f_thermal(x, y, 0, py, beta_th, mu_th, kgrid, cParams, sParams, sampleParams)  # we assume we only sample initial particles with z=0, px=0, pz=0 (so p=sqrt(px^2+py^2+pz^2)=py)
    #         if f < f_eval:
    #             samples[counter, :] = [x, y, py]
    #             counter += 1

    #     evals += 1
    #     print(counter, evals, f, f_eval)
    # print(timer() - start)

    # sample_datapath = 'zwData/samples/'
    # savemat(sample_datapath + 'aIB_{0}a0_true2D.mat'.format(aIBexp_Vals[inda]), {'samples': samples})
    # # savemat(sample_datapath + 'aIB_{0}a0.mat'.format(aIBexp_Vals[inda]), {'samples': samples})
    # # savemat(sample_datapath + 'aIB_{0}a0_P_P0.mat'.format(aIBexp_Vals[inda]), {'samples': samples})
    # # savemat(sample_datapath + 'aIB_{0}a0_P_P0_Y_Y0.mat'.format(aIBexp_Vals[inda]), {'samples': samples})
    # # savemat(sample_datapath + 'aIB_{0}a0_X_0.mat'.format(aIBexp_Vals[inda]), {'samples': samples})
    # # savemat(sample_datapath + 'aIB_{0}a0_T_20nk.mat'.format(aIBexp_Vals[inda]), {'samples': samples})

    # Visualize distribution of samples

    cmap = 'gist_heat_r'
    my_cmap = matplotlib.cm.get_cmap(cmap)
    my_cmap.set_under('w')

    # inda = 3
    print(aIBexp_Vals[inda])

    xMin = -2 * RTF_BEC_X_th[inda]; xMax = 2 * RTF_BEC_X_th[inda]
    yMin = -2 * RTF_BEC_Y_th[inda]; yMax = 2 * RTF_BEC_Y_th[inda]

    pMin = -1 * mI * nu[inda]; pMax = mI * nu[inda]
    # samples = loadmat('zwData/samples/posMu/aIB_{0}a0.mat'.format(aIBexp_Vals[inda]))['samples']  # loads matrix representing samples of initial conditions: each row is a different initial condition and the columns represent (x0, y0, p0) in theory units
    # samples = loadmat('zwData/samples/aIB_{0}a0.mat'.format(aIBexp_Vals[inda]))['samples']  # loads matrix representing samples of initial conditions: each row is a different initial condition and the columns represent (x0, y0, p0) in theory units
    # samples = loadmat('zwData/samples/aIB_{0}a0_P_P0.mat'.format(aIBexp_Vals[inda]))['samples']  # loads matrix representing samples of initial conditions: each row is a different initial condition and the columns represent (x0, y0, p0) in theory units
    # samples = loadmat('zwData/samples/aIB_{0}a0_P_P0_Y_Y0.mat'.format(aIBexp_Vals[inda]))['samples']  # loads matrix representing samples of initial conditions: each row is a different initial condition and the columns represent (x0, y0, p0) in theory units
    samples = loadmat('zwData/samples/aIB_{0}a0_true2D.mat'.format(aIBexp_Vals[inda]))['samples']  # loads matrix representing samples of initial conditions: each row is a different initial condition and the columns represent (x0, y0, p0) in theory units

    # samples = loadmat('zwData/samples/aIB_{0}a0_fMaxLarge.mat'.format(aIBexp_Vals[inda]))['samples']  # loads matrix representing samples of initial conditions: each row is a different initial condition and the columns represent (x0, y0, p0) in theory units

    # samples = loadmat('zwData/samples/aIB_{0}a0_X_0.mat'.format(aIBexp_Vals[inda]))['samples']  # loads matrix representing samples of initial conditions: each row is a different initial condition and the columns represent (x0, y0, p0) in theory units

    Ns = samples.shape[0]
    # xSamples = samples[:, 0]; ySamples = samples[:, 1]; pSamples = samples[:, 2]
    xSamples = samples[:, 0]; ySamples = samples[:, 1]; pxSamples = samples[:, 2]; pySamples = samples[:, 3]
    pSamples = pySamples

    # pSamples = np.abs(pSamples)
    xmean = np.mean(xSamples); ymean = np.mean(ySamples); pmean = np.mean(pSamples)
    print(xmean, 0, ymean, y0_imp[inda], pmean, P0_imp[inda])
    # print(xmean/RTF_BEC_X_th[inda], 0, ymean/RTF_BEC_Y_th[inda], y0_imp[inda]/RTF_BEC_Y_th[inda], pmean/(mI * nu[inda]), P0_imp[inda]/(mI * nu[inda]))
    # print(xmean, 0, ymean, y0_imp[inda], pmean, P0_imp[inda])

    H, xedges, yedges = np.histogram2d(xSamples, ySamples)
    H = H.T
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(xedges, yedges)
    quad = ax.pcolormesh(X, Y, H, cmap=my_cmap, rasterized=True)
    ax.plot(xmean, ymean, marker='x', markersize=10, zorder=11, color="blue")[0]
    ax.plot(0, y0_imp[inda], marker='x', markersize=10, zorder=11, color="green")[0]
    ax.set_xlim([xMin, xMax])
    ax.set_ylim([yMin, yMax])
    fig.colorbar(quad,extend='max')

    fig2, ax2 = plt.subplots()
    ax2.hist(pSamples/(mI * nu[inda]), bins=100, color='k')
    ax2.axvline(x=pmean/(mI * nu[inda]), ymin=0, ymax=Ns, linestyle=':', color="blue", lw=2)
    ax2.axvline(x=P0_imp[inda]/(mI * nu[inda]), ymin=0, ymax=Ns, linestyle=':', color="green", lw=2)
    ax2.set_xlim([pMin/(mI * nu[inda]), pMax/(mI * nu[inda])])
    # ax2.set_ylim([])

    my_cmap='RdBu'

    H, yedges, pedges = np.histogram2d(ySamples, pSamples, bins=15)
    H = H.T
    fig, ax = plt.subplots()
    Y, P = np.meshgrid(yedges, pedges)
    quad = ax.pcolormesh(Y * 1e6 / L_exp2th, P / (mI * nu[inda]), H / Ns, cmap=my_cmap, rasterized=True)
    ax.set_xlim([yMin * 1e6 / L_exp2th, yMax * 1e6 / L_exp2th])
    ax.set_ylim([pMin / (mI * nu[inda]), pMax / (mI * nu[inda])])
    fig.colorbar(quad,extend='max')

    plt.show()

    # # Polaron energy exploration

    # A_TiSa_th = 2 * np.pi * A_TiSa_Hz_scaled / T_exp2th  # equivalent to gaussian_amp above
    # A_ODT1_th = 2 * np.pi * A_ODT1_Hz / T_exp2th
    # # A_ODT1_th = 0
    # A_ODT2_th = 2 * np.pi * A_ODT2_Hz / T_exp2th
    # wx_ODT1_th = wx_ODT1 * 1e-6 * L_exp2th; wy_ODT1_th = wy_ODT1 * 1e-6 * L_exp2th; wx_ODT2_th = wx_ODT2 * 1e-6 * L_exp2th; wz_ODT2_th = wz_ODT2 * 1e-6 * L_exp2th; wx_TiSa_th = wx_TiSa * 1e-6 * L_exp2th; wy_TiSa_th = wy_TiSa * 1e-6 * L_exp2th
    # beta_exp = 1 / (kB * T / hbar)  # inverse temperature in units of 1/(rad*Hz) = s/rad
    # beta_th = beta_exp * T_exp2th

    # U_opt_offset = np.array([-28.60814655, -31.21901016, -30.87067425, -31.3245341, -31.22762206, -31.15045883, -31.15850962, -31.20071966, -31.40204218, -31.0106113, -30.71828868, -31.16363115, -30.31259895, -30.27470958, -30.72544145, -29.89234401, -29.49573856, -28.0016819])  # constant energy offset (theory units) of U_tot_opt to make sure the minimum value U_tot_opt(0,ymin,0) = 0. This offset is determined by numerically determining min(U_tot_opt(0,y,0))
    # U0_opt_offset = A_ODT1_th + A_ODT2_th + A_TiSa_th  # constant energy offset (theory units) of U_tot_opt to make sure the minimum value U_tot_opt(0,ymin,0) = 0 when AODT1 = 0 (the ODT1 beam is turned off)
    # E_pol_offset = np.array([-1.18662877e+00, -8.41179486e-01, -6.87998107e-01, -4.88436591e-01, -3.64301316e-01, -2.12254441e-01, -9.73809669e-02, -3.16003180e-02, -8.62356415e-36, 3.11130148e-02, 7.32152511e-02, 1.88938399e-01, 2.68558712e-01, 3.73587857e-01, 5.83872838e-01, 8.28556730e-01, 1.11273234e+00, 1.66368733e+00])

    # inda = 3

    # sampleParams = {'omegaX_radHz': omega_x_Na, 'omegaY_radHz': omega_Na[inda], 'omegaZ_radHz': omega_z_Na, 'temperature_K': T, 'zTF_MuM': RTF_BEC_Z[inda], 'y0_BEC': y0_BEC_lab[inda], 'y0_ODT1': y0_ODT1_lab[inda], 'omega_Imp_y': omega_Imp_y[inda], 'n0_BEC_m^-3': n0_BEC[inda], 'L_exp2th': L_exp2th,
    #                 'U_opt_offset': U_opt_offset[inda], 'U0_opt_offset': U0_opt_offset[inda], 'E_pol_offset': E_pol_offset[inda], 'A_ODT1': A_ODT1_th, 'wx_ODT1': wx_ODT1_th, 'wy_ODT1': wy_ODT1_th, 'A_ODT2': A_ODT2_th, 'wx_ODT2': wx_ODT2_th, 'wz_ODT2': wz_ODT2_th, 'A_TiSa': A_TiSa_th[inda], 'wx_TiSa': wx_TiSa_th, 'wy_TiSa': wy_TiSa_th}
    # cParams = {'aIBi': aIBi_Vals[inda]}
    # sParams = [mI, mB, n0[inda], gBB]
    # mu_th = mu_div_hbar_K[inda] / T_exp2th  # converts chemical potential in rad*Hz to theory units

    # y0 = y0_imp[inda]; p0 = P0_imp[inda]  # desired mean starting position and total momentum of the initial polaron for motion across the x=z=0 slice of the density (in theory units)

    # pMin = -1 * mI * nu[inda]; pMax = mI * nu[inda]
    # xMin = -2 * RTF_BEC_X_th[inda]; xMax = 2 * RTF_BEC_X_th[inda]
    # yMin = -2 * RTF_BEC_Y_th[inda]; yMax = 2 * RTF_BEC_Y_th[inda]
    # pVals = np.linspace(pMin,pMax,100)
    # EVals = np.zeros(pVals.size)

    # start = timer()

    # for indp, P in enumerate(pVals):
    #     EVals[indp] = pf_dynamic_sph.E_Pol_gs(xMax, yMin, 0, P, kgrid, cParams, sParams, sampleParams)   # we assume we only sample initial particles with z=0, px=0, pz=0 (so p=sqrt(px^2+py^2+pz^2)=py)

    # print(timer()-start)

    # fig, ax = plt.subplots()
    # ax.plot(pVals,EVals)
    # plt.show()

    # # X Potential exploration

    # from scipy.optimize import curve_fit

    # inda = 3
    # aIBi = aIBi_Vals[inda]
    # gnum = pf_dynamic_sph.g(kgrid, aIBi, mI, mB, n0, gBB)
    # # gnum = 0


    # A_TiSa_th = 2 * np.pi * A_TiSa_Hz_scaled[inda] / T_exp2th  # equivalent to gaussian_amp above
    # A_ODT1_th = 2 * np.pi * A_ODT1_Hz / T_exp2th
    # # A_ODT1_th = 0
    # A_ODT2_th = 2 * np.pi * A_ODT2_Hz / T_exp2th
    # wx_ODT1_th = wx_ODT1 * 1e-6 * L_exp2th; wy_ODT1_th = wy_ODT1 * 1e-6 * L_exp2th; wx_ODT2_th = wx_ODT2 * 1e-6 * L_exp2th; wz_ODT2_th = wz_ODT2 * 1e-6 * L_exp2th; wx_TiSa_th = wx_TiSa * 1e-6 * L_exp2th; wy_TiSa_th = wy_TiSa * 1e-6 * L_exp2th
    # RTF_BEC_X_th = RTF_BEC_X[inda] * 1e-6 * L_exp2th

    # U0_dens = gnum * pf_dynamic_sph.becdensity_zw2021(0, 0, 0,omega_x_Na, omega_Na[inda], omega_z_Na, T, RTF_BEC_Z[inda]) / (L_exp2th**3)  # function that gives the BEC density (expressed in theory units) given a coordinates (x,y) (expressed in theory units)
    # U0_trap = pf_dynamic_sph.U_TiSa(0, 0, A_TiSa_th, wx_TiSa_th, wy_TiSa_th) + pf_dynamic_sph.U_ODT2(0, 0, A_ODT2_th, wx_ODT2_th, wz_ODT2_th)


    # def U_trap_X(X, Y):
    #         U_trap = -1*U0_trap + pf_dynamic_sph.U_TiSa(X, 0, A_TiSa_th, wx_TiSa_th, wy_TiSa_th) + pf_dynamic_sph.U_ODT2(X, 0, A_ODT2_th, wx_ODT2_th, wz_ODT2_th)
    #         return U_trap

    # def U_dens_X(X, Y):
    #         U_dens = -1*U0_dens + gnum * pf_dynamic_sph.becdensity_zw2021(X * (1e6) / L_exp2th, Y * (1e6) / L_exp2th, 0,omega_x_Na, omega_Na[inda], omega_z_Na, T, RTF_BEC_Z[inda]) / (L_exp2th**3)  # function that gives the BEC density (expressed in theory units) given a coordinates (x,y) (expressed in theory units)
    #         return U_dens

    # def U_Imp_trap_X(X, Y):
    #         return U_trap_X(X,Y) + U_dens_X(X, Y)

    # def harmonic_approx(X, A, omega):
    #     return A + 0.5 * mI * omega**2 * X**2 


    # Y0_vals = [4.6, -27.8, -17.7, -24.4, 0]
    # Y0 = Y0_vals[-1]
    # xVals = np.linspace(-1*RTF_BEC_X_th, 1*RTF_BEC_X_th, 1000)
    # data = U_Imp_trap_X(xVals, Y0)

    # popt, cov = curve_fit(harmonic_approx, xVals, data)
    # omega_x_fit = popt[1]*T_exp2th / (2 * np.pi)
    # print(omega_x_fit)

    # fig, ax = plt.subplots()
    # ax.plot(xVals, data)
    # ax.plot(xVals, harmonic_approx(xVals, *popt))
    # # ax.plot(xVals, U_trap_X(xVals,Y0))
    # # ax.plot(xVals, U_dens_X(xVals,Y0))

    # plt.show()

    # # Finding minimum value of Uopt_tot for each interaction strength

    # yMin_vals = [] # location of minimum of Uopt_tot
    # Uopt_offset_vals = []  # minimum value of Uopt_tot

    # for inda, aIBi in enumerate(aIBi_Vals):
    #     sampleParams = {'omegaX_radHz': omega_x_Na, 'omegaY_radHz': omega_Na[inda], 'omegaZ_radHz': omega_z_Na, 'temperature_K': T, 'zTF_MuM': RTF_BEC_Z[inda], 'y0_BEC': y0_BEC_lab[inda], 'y0_ODT1': y0_ODT1_lab[inda], 'omega_Imp_y': omega_Imp_y[inda], 'n0_BEC_m^-3': n0_BEC[inda], 'L_exp2th': L_exp2th,
    #                     'U_opt_offset': U_opt_offset[inda], 'U0_opt_offset': U0_opt_offset[inda],'E_pol_offset': E_pol_offset[inda], 'A_ODT1': A_ODT1_th, 'wx_ODT1': wx_ODT1_th, 'wy_ODT1': wy_ODT1_th, 'A_ODT2': A_ODT2_th, 'wx_ODT2': wx_ODT2_th, 'wz_ODT2': wz_ODT2_th, 'A_TiSa': A_TiSa_th[inda], 'wx_TiSa': wx_TiSa_th, 'wy_TiSa': wy_TiSa_th}
    #     cParams = {'aIBi': aIBi_Vals[inda]}
    #     sParams = [mI, mB, n0[inda], gBB]
    #     mu_th = mu_div_hbar_K[inda] / T_exp2th  # converts chemical potential in rad*Hz to theory units

    #     def g(y): return (pf_dynamic_sph.U_ODT1(0, y - (sampleParams['y0_ODT1'] - sampleParams['y0_BEC']), sampleParams['A_ODT1'], sampleParams['wx_ODT1'], sampleParams['wy_ODT1']) + pf_dynamic_sph.U_ODT2(0, 0, sampleParams['A_ODT2'], sampleParams['wx_ODT2'], sampleParams['wz_ODT2']) + pf_dynamic_sph.U_TiSa(0, y + sampleParams['y0_BEC'], sampleParams['A_TiSa'], sampleParams['wx_TiSa'], sampleParams['wy_TiSa']))
    #     sol_U = minimize_scalar(g, bounds=[-4 * RTF_BEC_Y_th[inda], 4 * RTF_BEC_Y_th[inda]], method='Bounded')
    #     yMin_vals.append(sol_U.x)
    #     Uopt_offset_vals.append(sol_U.fun)

    # print(yMin_vals)
    # print(Uopt_offset_vals) 

    # # Finding max value of distribution for each interaction strength (1D)

    # yMax_vals = []  # location of maximum value of thermal distribution
    # fMax_vals = []  # maximum value of thermal distribution

    # for inda, aIBi in enumerate(aIBi_Vals):
    #     sampleParams = {'omegaX_radHz': omega_x_Na, 'omegaY_radHz': omega_Na[inda], 'omegaZ_radHz': omega_z_Na, 'temperature_K': T, 'zTF_MuM': RTF_BEC_Z[inda], 'y0_BEC': y0_BEC_lab[inda], 'y0_ODT1': y0_ODT1_lab[inda], 'omega_Imp_y': omega_Imp_y[inda], 'n0_BEC_m^-3': n0_BEC[inda], 'L_exp2th': L_exp2th,
    #                     'U_opt_offset': U_opt_offset[inda], 'U0_opt_offset': U0_opt_offset[inda],'E_pol_offset': E_pol_offset[inda], 'A_ODT1': A_ODT1_th, 'wx_ODT1': wx_ODT1_th, 'wy_ODT1': wy_ODT1_th, 'A_ODT2': A_ODT2_th, 'wx_ODT2': wx_ODT2_th, 'wz_ODT2': wz_ODT2_th, 'A_TiSa': A_TiSa_th[inda], 'wx_TiSa': wx_TiSa_th, 'wy_TiSa': wy_TiSa_th}
    #     cParams = {'aIBi': aIBi_Vals[inda]}
    #     sParams = [mI, mB, n0[inda], gBB]
    #     mu_th = mu_div_hbar_K[inda] / T_exp2th  # converts chemical potential in rad*Hz to theory units

    #     # def f(y): return -1*pf_dynamic_sph.f_thermal(0, y, 0, 0, beta_th, mu_th, kgrid, cParams, sParams, sampleParams)
    #     def f(y): return -1*pf_dynamic_sph.f_thermal_true2D(0, y, 0, 0, 0, beta_th, mu_th, kgrid, cParams, sParams, sampleParams)
    #     sol = minimize_scalar(f, bounds=[-2 * RTF_BEC_Y_th[inda], 2 * RTF_BEC_Y_th[inda]], method='Bounded')
    #     # sol = minimize_scalar(f, bounds=[-0.99 * RTF_BEC_Y_th[inda], 0.99 * RTF_BEC_Y_th[inda]], method='Bounded')
    #     yMax_vals.append(sol.x)
    #     fMax_vals.append(-1*sol.fun)

    # print(yMax_vals)
    # print(fMax_vals) 

    # # Finding max value of distribution for each interaction strength (2D)

    # xMax_vals = []  # location of maximum value of thermal distribution
    # yMax_vals = []  # location of maximum value of thermal distribution
    # fMax_vals = []  # maximum value of thermal distribution

    # for inda, aIBi in enumerate(aIBi_Vals):
    #     sampleParams = {'omegaX_radHz': omega_x_Na, 'omegaY_radHz': omega_Na[inda], 'omegaZ_radHz': omega_z_Na, 'temperature_K': T, 'zTF_MuM': RTF_BEC_Z[inda], 'y0_BEC': y0_BEC_lab[inda], 'y0_ODT1': y0_ODT1_lab[inda], 'omega_Imp_y': omega_Imp_y[inda], 'n0_BEC_m^-3': n0_BEC[inda], 'L_exp2th': L_exp2th,
    #                     'U_opt_offset': U_opt_offset[inda], 'U0_opt_offset': U0_opt_offset[inda],'E_pol_offset': E_pol_offset[inda], 'A_ODT1': A_ODT1_th, 'wx_ODT1': wx_ODT1_th, 'wy_ODT1': wy_ODT1_th, 'A_ODT2': A_ODT2_th, 'wx_ODT2': wx_ODT2_th, 'wz_ODT2': wz_ODT2_th, 'A_TiSa': A_TiSa_th[inda], 'wx_TiSa': wx_TiSa_th, 'wy_TiSa': wy_TiSa_th}
    #     cParams = {'aIBi': aIBi_Vals[inda]}
    #     sParams = [mI, mB, n0[inda], gBB]
    #     mu_th = mu_div_hbar_K[inda] / T_exp2th  # converts chemical potential in rad*Hz to theory units

    #     def f(x): return -1*pf_dynamic_sph.f_thermal_true2D(x[0], x[1], 0, 0, 0, beta_th, mu_th, kgrid, cParams, sParams, sampleParams)
    #     if aIBi < 0:
    #         x0 = np.array([0, 0])
    #     else:
    #         x0 = np.array([-1 * RTF_BEC_X_th[inda], -1 * RTF_BEC_Y_th[inda]])
    #     print(aIBexp_Vals[inda])
    #     print(x0)
    #     # sol = minimize(f, x0=x0, bounds=((-2 * RTF_BEC_X_th[inda], 2 * RTF_BEC_X_th[inda]),(-2 * RTF_BEC_Y_th[inda], 2 * RTF_BEC_Y_th[inda])), method='SLSQP')
    #     sol = minimize(f, x0=x0, method='Nelder-Mead')
    #     xMax_vals.append(sol.x[0])
    #     yMax_vals.append(sol.x[1])
    #     fMax_vals.append(-1*sol.fun)

    # print(xMax_vals)
    # print(yMax_vals)
    # print(fMax_vals) 

    # # Testing

    # true2D = True

    # xVals = np.linspace(-2 * RTF_BEC_X_th[inda], 2 * RTF_BEC_X_th[inda], 100)
    # yVals = np.linspace(-2 * RTF_BEC_Y_th[inda], 2 * RTF_BEC_Y_th[inda], 100)
    # # yVals = np.linspace(-1.5 * RTF_BEC_Y_th[inda], 1.5 * RTF_BEC_Y_th[inda], 100)
    # pVals = np.linspace(-1*mI * nu[inda], mI * nu[inda], 100)

    # x_des = 50
    # y_des = 50
    # # testVals = np.zeros((xVals.size, yVals.size, pVals.size))
    # testVals = np.zeros((xVals.size, yVals.size))
    # # testVals = np.zeros((yVals.size, pVals.size))

    # for indx, x in enumerate(xVals):
    #     # if indx != x_des:
    #     #     continue
    #     # # print(x)
    #     EPol = np.zeros(yVals.size)
    #     UOpt = np.zeros(yVals.size)
    #     Vharm = np.zeros(yVals.size)
    #     start = timer()
    #     for indy, y in enumerate(yVals):
    #         # if indy != y_des:
    #         #     continue
    #         # print(y)
    #         # start = timer()
    #         testVals[indx, indy] = pf_dynamic_sph.f_thermal_true2D(x, y, 0, 0, 0, beta_th, mu_th, kgrid, cParams, sParams, sampleParams)
    #         # for indp, p in enumerate(pVals):
    #         #     if true2D:
    #         #         testVals[indy, indp] = pf_dynamic_sph.f_thermal_true2D(0, y, 0, 0, p, beta_th, mu_th, kgrid, cParams, sParams, sampleParams)
    #         #     else:
    #         #         testVals[indy, indp] = pf_dynamic_sph.f_thermal(0, y, 0, p, beta_th, mu_th, kgrid, cParams, sParams, sampleParams)
    #         #         # testVals[indx, indy, indp] = pf_dynamic_sph.f_thermal(0, y0_imp[inda], 0, p, beta_th, mu_th, kgrid, cParams, sParams, sampleParams)
    #         # # testVals[indx, indy] = pf_dynamic_sph.f_thermal(0, y, 0, p0, beta_th, mu_th, kgrid, cParams, sParams, sampleParams)
    #         # # EPol[indy] = pf_dynamic_sph.E_Pol_gs(0, y, 0, p0, kgrid, cParams, sParams, sampleParams)
    #         # # UOpt[indy] = pf_dynamic_sph.U_tot_opt(0, y, 0, sampleParams)
    #         # # Vharm[indy] = 0.5 * mI * (omega_Imp_y[inda]**2) * (y + sampleParams['y0_BEC'])**2
    #         # # # Vharm[indy] = 0.5 * mI * (omega_Imp_y[inda]**2) * y**2
    #         # print(timer() - start)
    #     print(timer() - start)
    #     # ymean = np.sum(testVals[indx,:] * yVals) / np.sum(testVals)
    #     # ftot = simpson(testVals[indx,:],yVals)
    #     # ymean2 = simpson(testVals[indx,:] * yVals,yVals)/ftot
    #     # print(ftot)
    #     # print(ymean2 * 1e6 / L_exp2th,  y0_imp[inda] * 1e6 / L_exp2th)

    # # ptot = simpson(testVals[x_des,y_des,:],pVals)
    # # pmean = simpson(testVals[x_des,y_des,:] * pVals,pVals)/ptot
    # # print(p0, pmean)

    # # fig, ax = plt.subplots()
    # # # ax.plot(pVals, testVals[x_des,y_des,:])
    # # ax.plot(yVals, testVals[x_des,:])
    # # # ax.plot(yVals, EPol)
    # # # ax.plot(yVals, UOpt)
    # # # ax.plot(yVals, Vharm)
    # # plt.show()

    # xg, yg = np.meshgrid(xVals, yVals, indexing='ij')
    # fig, ax = plt.subplots()
    # c = ax.pcolormesh(1e6 * xg / L_exp2th, 1e6 * yg / L_exp2th, testVals, cmap='RdBu')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # fig.colorbar(c, ax=ax)
    # plt.show()

    # yg, pg = np.meshgrid(yVals, pVals, indexing='ij')
    # fig, ax = plt.subplots()
    # c = ax.pcolormesh(1e6 * yg / L_exp2th, pg/(mI * nu[inda]), testVals,cmap='RdBu')
    # ax.set_xlabel('y')
    # ax.set_ylabel('p')
    # fig.colorbar(c, ax=ax)
    # plt.show()

    # # Create dicts

    # jobList = []
    # for ind, aIBi in enumerate(aIBi_Vals):
    #     sParams = [mI, mB, n0[ind], gBB]
    #     den_tck = np.load('zwData/densitySplines/nBEC_aIB_{0}a0.npy'.format(aIBexp_Vals[ind]), allow_pickle=True)
    #     trapParams = {'nBEC_tck': den_tck, 'RTF_BEC': RTF_BEC_Y_th[ind], 'omega_BEC_osc': omega_BEC_osc[ind], 'gamma_BEC_osc': gamma_BEC_osc[ind], 'phi_BEC_osc': phi_BEC_osc[ind], 'amp_BEC_osc': amp_BEC_osc[ind], 'omega_Imp_x': omega_Imp_x[ind], 'gaussian_amp': gaussian_amp[ind], 'gaussian_width': gaussian_width, 'X0': x0_imp[ind], 'P0': P0_scaleFactor[ind] * mI * v0_imp[ind]}
    #     jobList.append((aIBi, sParams, trapParams))
