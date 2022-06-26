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
    n0_BEC = np.array([5.50743315e+19, 5.03889459e+19, 6.04081899e+19, 5.61903369e+19, 6.19914061e+19, 7.11346218e+19, 6.73466436e+19, 6.51920977e+19, 5.73665093e+19, 6.38326341e+19, 5.98486416e+19, 6.11450398e+19, 6.16486935e+19, 5.94439691e+19, 6.08352926e+19, 6.35042149e+19, 5.51802931e+19, 5.93638236e+19])
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

    # Optical trap experimental parameters (for K gas)

    A_ODT1_uK = -4.25; A_ODT2_uK = -3.44; A_TiSa_uK = -6.9  # Amplitude of each gaussian beam in uK
    A_ODT1_Hz = kB * A_ODT1_uK * 1e-6 / hbar / (2 * np.pi); A_ODT2_Hz = kB * A_ODT2_uK * 1e-6 / hbar / (2 * np.pi); A_TiSa_Hz = kB * A_TiSa_uK * 1e-6 / hbar / (2 * np.pi)  # Amplitude of each gaussian beam in Hz
    wx_ODT1 = 82; wy_ODT1 = 82  # beam waists of ODT1 in um
    ep_ODT2 = 4  # ellipticity of ODT2 beam
    wx_ODT2 = 144; wz_ODT2 = wx_ODT2 / ep_ODT2  # beam waists of ODT2 in um
    wx_TiSa = 95; wy_TiSa = 95  # beam waists of TiSa in um

    A_ODT1_Na_uK = -1.66; A_ODT2_Na_uK = -1.35; A_TiSa_Na_uK = -1.39  # Amplitude of each gaussian beam in uK (for Na)
    A_ODT1_Na_Hz = kB * A_ODT1_Na_uK * 1e-6 / hbar / (2 * np.pi); A_ODT2_Na_Hz = kB * A_ODT2_Na_uK * 1e-6 / hbar / (2 * np.pi); A_TiSa_Na_Hz = kB * A_TiSa_Na_uK * 1e-6 / hbar / (2 * np.pi)  # Amplitude of each gaussian beam in Hz

    scale_fac = np.array([0.9191250336816127, 1.0820215784890415, 1.0045367702210797, 1.059151023960805, 0.998027347459867, 1.0322275524975513, 1.015715140919877, 1.0344684463876583, 1.0511737903469414, 1.0290662994361741, 1.0260064676580842, 1.027145606565003, 0.9632500076336702, 0.9001633188311512, 0.9772252948388005, 0.9854926080339469, 0.9642499474231473, 0.8952239304967515])  # fitting the optical trap depth to the K oscillation frequencies
    # scale_fac = np.array([0.9724906988829373, 0.8896535250404035, 0.9569791653444897, 1.0382627423131257, 0.903699258873818, 0.971557704172668, 0.9505698121511785, 0.9689281438661883, 0.9716029799261697, 0.9970638982528262, 0.9970576906401939, 1.00494970248614, 0.9280067282060056, 0.9078274542675744, 0.9781498957395225, 0.9939697862098377, 1.0630900294469736, 0.9278113852604492])  # fitting the optical trap depth to the Na oscillation frequencies
    A_TiSa_Hz_scaled = A_TiSa_Hz * scale_fac
    A_TiSa_Na_Hz_scaled = A_TiSa_Na_Hz * scale_fac

    ODT1_displacement = np.array([39.9734769508128, 37.20726134699691, 29.022743967492712, 32.85605962371015, 23.00479821032066, 30.475997313212293, 27.49539761274011, 30.277006179531572, 30.746034106569127, 31.517392916389632, 34.17496197024173, 29.467112794532262, 28.46260872772458, 13.428923709748158, 25.777101525763207, 36.645281366522546, 37.56837023644184, 42.51753230100077])  # initial position of the ODT1 beam (in um) before it is turned off -> assumes that lab frame origin is the center of the TiSa beam (to the left of the ODT1 beam)

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
    gaussian_width_y = wy_TiSa * 1e-6 * L_exp2th
    gaussian_width_x = wx_TiSa * 1e-6 * L_exp2th

    a0_exp = 5.29e-11  # Bohr radius (m)
    aIBi_Vals = 1 / (aIBexp_Vals * a0_exp * L_exp2th)

    P0_scaleFactor = np.array([1.2072257743801846, 1.1304777274446096, 1.53, 1.0693683091221053, 1.0349159867400886, 0.95, 2.0, 1.021253131703251, 0.9713134192266438, 0.9781007832739641, 1.0103135855263197, 1.0403095335853234, 0.910369833990368, 0.9794720983829749, 1.0747443076336567, 0.79, 1.0, 0.8127830658214898])  # scales the momentum of the initial polaron state (scaleFac * mI * v0) so that the initial impurity velocity matches the experiment. aIB= -125a0 and +750a0 have ~8% and ~20% error respectively
    P0_imp = P0_scaleFactor * mI * v0_imp

    # Quantities for sampling

    A_TiSa_th = 2 * np.pi * A_TiSa_Hz_scaled / T_exp2th  # equivalent to gaussian_amp above
    A_ODT1_th = 2 * np.pi * A_ODT1_Hz / T_exp2th
    # A_ODT1_th = 0
    A_ODT2_th = 2 * np.pi * A_ODT2_Hz / T_exp2th
    wx_ODT1_th = wx_ODT1 * 1e-6 * L_exp2th; wy_ODT1_th = wy_ODT1 * 1e-6 * L_exp2th; wx_ODT2_th = wx_ODT2 * 1e-6 * L_exp2th; wz_ODT2_th = wz_ODT2 * 1e-6 * L_exp2th; wx_TiSa_th = wx_TiSa * 1e-6 * L_exp2th; wy_TiSa_th = wy_TiSa * 1e-6 * L_exp2th
    beta_exp = 1 / (kB * T / hbar)  # inverse temperature in units of 1/(rad*Hz) = s/rad
    beta_th = beta_exp * T_exp2th

    U_opt_offset = np.array([-28.60814655, -31.21901016, -30.87067425, -31.3245341, -31.22762206, -31.15045883, -31.15850962, -31.20071966, -31.40204218, -31.0106113, -30.71828868, -31.16363115, -30.31259895, -30.27470958, -30.72544145, -29.89234401, -29.49573856, -28.0016819])  # constant energy offset (theory units) of U_tot_opt to make sure the minimum value U_tot_opt(0,ymin,0) = 0. This offset is determined by numerically determining min(U_tot_opt(0,y,0))
    U0_opt_offset = A_ODT1_th + A_ODT2_th + A_TiSa_th  # constant energy offset (theory units) of U_tot_opt to make sure the minimum value U_tot_opt(0,ymin,0) = 0 when AODT1 = 0 (the ODT1 beam is turned off)
    E_pol_offset = np.array([-1.18662877e+00, -8.41179486e-01, -6.87998107e-01, -4.88436591e-01, -3.64301316e-01, -2.12254441e-01, -9.73809669e-02, -3.16003180e-02, -8.62356415e-36, 3.11130148e-02, 7.32152511e-02, 1.88938399e-01, 2.68558712e-01, 3.73587857e-01, 5.83872838e-01, 8.28556730e-01, 1.11273234e+00, 1.66368733e+00])

    f_thermal_maxVals = np.array([0.7367184970676869, 0.7023880636086897, 0.7459548034386865, 0.79242602587235, 0.8340965891576568, 0.8912926562637322, 0.8949215118953942, 0.8698744612202944, 0.8675373653753994, 0.8801537515385071, 0.885519273327338, 0.9098398084573248, 0.9212560518600181, 0.8316673727452937, 0.9746952226616081, 0.997698493028504, 0.9991709720238214, 0.9999781206212883])  # Numerically determined maximum value of f_thermal for each interaction (occurs at x=0, y=yMax, z=0, p=0 where yMax is given below)
    y_thermal_maxVals = np.array([-12.358738440529192, -15.142125013161545, -13.144260996614438, -16.589728903343367, -13.283761232947477, -21.15224812723023, -20.742902013227198, -24.997818462639284, -26.498895342912757, -28.469193967012014, -33.970099935372986, -33.66816313625496, -36.2296199300486, -18.44950460158087, -55.65474817181998, -56.536254230789034, -50.99473895410588, -56.758005648192395])  # the y values y=yMax at which f_thermal has a maximum

    mu_th = mu_div_hbar_K / T_exp2th  # converts chemical potential in rad*Hz to theory units

    # Load data

    inda = 3
    aIB_exp = aIBexp_Vals[inda]; aIBi = aIBi_Vals[inda]
    print('aIB: {0}'.format(aIB_exp))

    cParams = {'aIBi': aIBi_Vals[inda]}
    sParams = [mI, mB, n0[inda], gBB]

    # samples = loadmat('zwData/samples/posMu/aIB_{0}a0.mat'.format(aIB_exp))['samples']  # loads matrix representing samples of initial conditions: each row is a different initial condition and the columns represent (x0, y0, p0) in theory units
    # samples = loadmat('zwData/samples/aIB_{0}a0.mat'.format(aIB_exp))['samples']  # loads matrix representing samples of initial conditions: each row is a different initial condition and the columns represent (x0, y0, p0) in theory units
    samples = loadmat('zwData/samples/aIB_{0}a0_P_P0.mat'.format(aIB_exp))['samples']  # loads matrix representing samples of initial conditions: each row is a different initial condition and the columns represent (x0, y0, p0) in theory units
    # samples = loadmat('zwData/samples/aIB_{0}a0_P_P0_Y_Y0.mat'.format(aIB_exp))['samples']  # loads matrix representing samples of initial conditions: each row is a different initial condition and the columns represent (x0, y0, p0) in theory units
    Ns = samples.shape[0]
    xmean = np.mean(samples[:, 0]); ymean = np.mean(samples[:, 1]); pmean = np.mean(samples[:, 2])
    print(xmean, 0, ymean, y0_imp[inda], pmean, P0_imp[inda])

    # sampleParams = {'omegaX_radHz': omega_x_Na, 'omegaY_radHz': omega_Na[inda], 'omegaZ_radHz': omega_z_Na, 'temperature_K': T, 'zTF_MuM': RTF_BEC_Z[inda], 'y0_BEC': y0_BEC_lab[inda], 'y0_ODT1': y0_ODT1_lab[inda], 'omega_Imp_y': omega_Imp_y[inda], 'n0_BEC_m^-3': n0_BEC[inda], 'L_exp2th': L_exp2th,
    #                 'U_opt_offset': U_opt_offset[inda], 'U0_opt_offset': U0_opt_offset[inda], 'E_pol_offset': E_pol_offset[inda], 'A_ODT1': A_ODT1_th, 'wx_ODT1': wx_ODT1_th, 'wy_ODT1': wy_ODT1_th, 'A_ODT2': A_ODT2_th, 'wx_ODT2': wx_ODT2_th, 'wz_ODT2': wz_ODT2_th, 'A_TiSa': A_TiSa_th[inda], 'wx_TiSa': wx_TiSa_th, 'wy_TiSa': wy_TiSa_th}

    # Create dicts

    toggleDict = {'Location': 'cluster', 'Dynamics': 'real', 'Interaction': 'on', 'InitCS': 'steadystate', 'InitCS_datapath': '', 'Coupling': 'twophonon', 'Grid': 'spherical',
                  'F_ext': 'off', 'PosScat': 'off', 'BEC_density': 'on', 'BEC_density_osc': 'on', 'Imp_trap': 'on', 'ImpTrap_Type': 'gaussian', 'CS_Dyn': 'on', 'Polaron_Potential': 'on', 'PP_Type': 'smarter'}

    # ---- SET OUTPUT DATA FOLDER ----

    if toggleDict['Location'] == 'personal':
        datapath = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/2021'
    elif toggleDict['Location'] == 'cluster':
        datapath = '/n/holyscratch01/jaffe_lab/Everyone/kis/data/ZwierleinExp_data/2021'

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
    for inds in np.arange(Ns):
        x0 = samples[inds, 0]; y0 = samples[inds, 1]; p0 = samples[inds, 2]
        trapParams = {'RTF_BEC': RTF_BEC_Y_th[inda], 'omega_BEC_osc': omega_BEC_osc[inda], 'gamma_BEC_osc': gamma_BEC_osc[inda], 'phi_BEC_osc': phi_BEC_osc[inda], 'amp_BEC_osc': amp_BEC_osc[inda], 'omega_Imp_y': omega_Imp_y[inda], 'gaussian_amp': gaussian_amp[inda], 'gaussian_width_x': gaussian_width_x, 'gaussian_width_y': gaussian_width_y, 'X0': x0, 'Y0': y0, 'P0': p0, 'L_exp2th': L_exp2th, 'omegaX_radHz': omega_x_Na, 'omegaY_radHz': omega_Na[inda], 'omegaZ_radHz': omega_z_Na, 'temperature_K': T, 'zTF_MuM': RTF_BEC_Z[inda]}
        filepath = innerdatapath + '/initdist/aIB_{0}a0_{1}.nc'.format(aIB_exp, inds)
        jobList.append((aIBi, sParams, trapParams, filepath))

    print(len(jobList))

    # # ---- COMPUTE DATA ON COMPUTER ----

    # runstart = timer()
    # for ind, tup in enumerate(jobList):
    #     if ind != 3:
    #         continue
    #     loopstart = timer()
    #     (aIBi, sParams, trapParams, filepath) = tup
    #     cParams = {'aIBi': aIBi}
    #     ds = pf_dynamic_sph.zw2021_quenchDynamics_2D(cParams, gParams, sParams, trapParams, toggleDict)
    #     ds.to_netcdf(filepath)

    #     loopend = timer()
    #     print('Sample: {:d}, Time: {:.2f}'.format(ind, loopend - loopstart))
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
    ds = pf_dynamic_sph.zw2021_quenchDynamics_2D(cParams, gParams, sParams, trapParams, toggleDict)
    ds.to_netcdf(filepath)

    end = timer()
    print('Task ID: {:d}, Time: {:.2f}'.format(taskID, end - runstart))
