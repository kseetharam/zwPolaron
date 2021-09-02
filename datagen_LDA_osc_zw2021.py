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

    # tMax = 6000; dt = 0.5
    # tMax = 600; dt = 0.5
    tMax = 0.5; dt = 0.5
    tgrid = np.arange(0, tMax + dt, dt)

    gParams = [xgrid, kgrid, tgrid]
    NGridPoints = kgrid.size()

    print('Total time steps: {0}'.format(tgrid.size))
    print('UV cutoff: {0}'.format(k_max))
    print('dk: {0}'.format(dk))
    print('NGridPoints: {0}'.format(NGridPoints))

    # Experimental params

    aIBexp_Vals = np.array([-1000, -750, -500, -375, -250, -125, -60, -20, 0, 20, 50, 125, 175, 250, 375, 500, 750, 1000])

    n0_BEC = np.array([5.51533197e+19, 5.04612835e+19, 6.04947525e+19, 5.62709096e+19, 6.20802175e+19, 7.12364194e+19, 6.74430590e+19, 6.52854564e+19, 5.74487521e+19, 6.39240612e+19, 5.99344093e+19, 6.12326489e+19, 6.17370181e+19, 5.95291621e+19, 6.09224617e+19, 6.35951755e+19, 5.52594316e+19, 5.94489028e+19])  # peak BEC density (given in in m^(-3))
    RTF_BEC_Y = np.array([11.4543973014280, 11.4485027292274, 12.0994087866866, 11.1987472415996, 12.6147755284164, 13.0408759297917, 12.8251948079726, 12.4963915490121, 11.6984708883771, 12.1884624646191, 11.7981246004719, 11.8796464214276, 12.4136593404667, 12.3220325703494, 12.0104329130883, 12.1756670927480, 10.9661042681457, 12.1803009563806])  # Thomas-Fermi radius of BEC in direction of oscillation (given in um)

    Na_displacement = np.array([26.2969729628679, 22.6668334850173, 18.0950989598699, 20.1069898676222, 14.3011351453467, 18.8126473489499, 17.0373115356076, 18.6684373282353, 18.8357213162278, 19.5036039713438, 21.2438389441807, 18.2089748680659, 18.0433963046778, 8.62940156299093, 16.2007030552903, 23.2646987822343, 24.1115616621798, 28.4351972435186])  # initial position of the BEC (in um)
    K_displacement_raw = np.array([0.473502276902047, 0.395634326123081, 8.66936929134637, 11.1470221226478, 9.34778274195669, 16.4370036199872, 19.0938486958001, 18.2135041439547, 21.9211790347041, 20.6591098913628, 19.7281375591975, 17.5425503131171, 17.2460344933717, 11.7179407507981, 12.9845862662090, 9.18113956217101, 11.9396846941782, 4.72461841775226])  # initial position of the impurity (in um)
    K_displacement_scale = np.mean(K_displacement_raw[6:11] / Na_displacement[6:11])
    K_displacement = deepcopy(K_displacement_raw); K_displacement[0:6] = K_displacement_scale * Na_displacement[0:6]; K_displacement[11::] = K_displacement_scale * Na_displacement[11::]   # in um
    K_relPos = K_displacement - Na_displacement   # in um

    omega_Na = np.array([465.418650581347, 445.155256942448, 461.691943131414, 480.899902898451, 448.655522184374, 465.195338759998, 460.143258369460, 464.565377197007, 465.206177963899, 471.262139163205, 471.260672147216, 473.122081065092, 454.649394420577, 449.679107889662, 466.770887179217, 470.530355145510, 486.615655444221, 454.601540658640])   # in rad*Hz
    omega_K_raw = np.array([764.649207995890, 829.646158322623, 799.388442120805, 820.831266284088, 796.794204312379, 810.331402280747, 803.823888714144, 811.210511844489, 817.734286423120, 809.089608774626, 807.885837386121, 808.334196591376, 782.788534907910, 756.720677755942, 788.446619623011, 791.774719564856, 783.194731826180, 754.641677886382])   # in rad*Hz
    omega_K_scale = np.mean(omega_K_raw[6:11] / omega_Na[6:11])
    omega_K = deepcopy(omega_K_raw); omega_K[0:6] = omega_K_scale * omega_Na[0:6]; omega_K[11::] = omega_K_scale * omega_Na[11::]  # in rad*Hz

    K_relVel = np.array([1.56564660488838, 1.31601642026105, 0.0733613860991014, 1.07036861258786, 1.22929932184982, -13.6137940945403, 0.0369377794311800, 1.61258456681232, -1.50457700049200, -1.72583008593939, 4.11884512615162, 1.04853747806043, -0.352830359266360, -4.00683426531578, 0.846101589896479, -0.233660196108278, 4.82122627459411, -1.04341939663180])  # in um/ms

    phi_Na = np.array([-0.2888761, -0.50232022, -0.43763589, -0.43656233, -0.67963017, -0.41053479, -0.3692152, -0.40826816, -0.46117853, -0.41393032, -0.53483635, -0.42800711, -0.3795508, -0.42279337, -0.53760432, -0.4939509, -0.47920687, -0.51809527])  # phase of the BEC oscillation in rad
    gamma_Na = np.array([4.97524294, 14.88208436, 4.66212187, 6.10297397, 7.77264927, 4.5456649, 4.31293083, 7.28569606, 8.59578888, 3.30558254, 8.289436, 4.14485229, 7.08158476, 4.84228082, 9.67577823, 11.5791718, 3.91855863, 10.78070655])  # decay rate of the BEC oscillation in Hz

    # Basic parameters

    expParams = pf_dynamic_sph.Zw_expParams_2021()
    L_exp2th, M_exp2th, T_exp2th = pf_dynamic_sph.unitConv_exp2th(expParams['n0_BEC_scale'], expParams['mB'])

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

    # Convert experimental parameters to theory parameters

    x0_imp = K_relPos * 1e-6 * L_exp2th  # initial positions of impurity in BEC frame (relative to the BEC)
    v0_imp = K_relVel * (1e-6 / 1e-3) * (L_exp2th / T_exp2th)  # initial velocities of impurity in BEC frame (relative to BEC)
    RTF_BEC = RTF_BEC_Y * 1e-6 * L_exp2th  # BEC oscillation amplitude (carries units of position)

    omega_BEC_osc = omega_Na / T_exp2th
    gamma_BEC_osc = gamma_Na / T_exp2th
    phi_BEC_osc = phi_Na
    amp_BEC_osc = (Na_displacement * 1e-6 * L_exp2th) / np.cos(phi_Na)  # BEC oscillation amplitude (carries units of position)

    omega_Imp_x = omega_K / T_exp2th

    a0_exp = 5.29e-11  # Bohr radius (m)
    aIBi_Vals = 1 / (aIBexp_Vals * a0_exp * L_exp2th)

    P0_scaleFactor = np.array([1.2072257743801846, 1.1304777274446096, 1.53, 1.0693683091221053, 1.0349159867400886, 0.95, 2.0, 1.021253131703251, 0.9713134192266438, 0.9781007832739641, 1.0103135855263197, 1.0403095335853234, 0.910369833990368, 0.9794720983829749, 1.0747443076336567, 0.79, 1.0, 0.8127830658214898])  # scales the momentum of the initial polaron state (scaleFac * mI * v0) so that the initial impurity velocity matches the experiment. aIB= -125a0 and +750a0 have ~8% and ~20% error respectively

    # Create dicts

    toggleDict = {'Location': 'cluster', 'Dynamics': 'real', 'Interaction': 'on', 'InitCS': 'steadystate', 'InitCS_datapath': '', 'Coupling': 'twophonon', 'Grid': 'spherical',
                  'F_ext': 'off', 'PosScat': 'off', 'BEC_density': 'on', 'BEC_density_osc': 'on', 'Imp_trap': 'on', 'CS_Dyn': 'on', 'Polaron_Potential': 'off'}

    # ---- SET OUTPUT DATA FOLDER ----

    if toggleDict['Location'] == 'personal':
        datapath = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/2021'
    elif toggleDict['Location'] == 'cluster':
        datapath = '/n/holyscratch01/demler_lab/kis/data/ZwierleinExp_data/2021'

    # if toggleDict['BEC_density'] == 'off':
    #     innerdatapath = innerdatapath + '/HomogBEC'
    #     toggleDict['Polaron_Potential'] = 'off'

    if toggleDict['Polaron_Potential'] == 'off':
        toggleDict['Polaron_Potential'] = 'off'
        innerdatapath = datapath + '/NoPolPot'
    else:
        innerdatapath = datapath + '/PolPot'

    jobList = []
    for ind, aIBi in enumerate(aIBi_Vals):
        sParams = [mI, mB, n0[ind], gBB]
        den_tck = np.load('zwData/densitySplines/nBEC_aIB_{0}a0.npy'.format(aIBexp_Vals[ind]), allow_pickle=True)
        trapParams = {'nBEC_tck': den_tck, 'RTF_BEC': RTF_BEC[ind], 'omega_BEC_osc': omega_BEC_osc[ind], 'gamma_BEC_osc': gamma_BEC_osc[ind], 'phi_BEC_osc': phi_BEC_osc[ind], 'amp_BEC_osc': amp_BEC_osc[ind], 'omega_Imp_x': omega_Imp_x[ind], 'X0': x0_imp[ind], 'P0': P0_scaleFactor[ind] * mI * v0_imp[ind]}
        filepath = innerdatapath + '/aIB_{0}a0.nc'.format(aIBexp_Vals[ind])
        jobList.append((aIBi, sParams, trapParams, filepath))

    # print(len(jobList))

    # # ---- COMPUTE DATA ON COMPUTER ----

    # runstart = timer()
    # for ind, tup in enumerate(jobList):
    #     if ind != 4:
    #         continue
    #     print('aIB: {0}a0'.format(aIBexp_Vals[ind]))
    #     loopstart = timer()
    #     (aIBi, sParams, trapParams, filepath) = tup
    #     cParams = {'aIBi': aIBi}
    #     ds = pf_dynamic_sph.zw2021_quenchDynamics(cParams, gParams, sParams, trapParams, toggleDict)
    #     ds.to_netcdf(filepath)

    #     # v0 = ds['V'].values[0] * (T_exp2th / L_exp2th) * (1e6 / 1e3)
    #     # print(v0_imp[ind], K_relVel[ind], v0, K_relVel[ind] / v0)

    #     loopend = timer()
    #     print('aIBi: {:.2f}, Time: {:.2f}'.format(aIBi, loopend - loopstart))
    # end = timer()
    # print('Total Time: {:.2f}'.format(end - runstart))

    # ---- COMPUTE DATA ON CLUSTER ----

    runstart = timer()

    # taskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    # taskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    taskCount = len(jobList); taskID = 4

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
