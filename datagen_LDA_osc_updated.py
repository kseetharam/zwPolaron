import numpy as np
import pandas as pd
import xarray as xr
import Grid
import pf_dynamic_sph
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

    tMax = 5000; dt = 0.5
    tgrid = np.arange(0, tMax + dt, dt)

    gParams = [xgrid, kgrid, tgrid]
    NGridPoints = kgrid.size()

    print('Total time steps: {0}'.format(tgrid.size))
    print('UV cutoff: {0}'.format(k_max))
    print('dk: {0}'.format(dk))
    print('NGridPoints: {0}'.format(NGridPoints))

    # Basic parameters

    expParams = pf_dynamic_sph.Zw_expParams_updated()
    L_exp2th, M_exp2th, T_exp2th = pf_dynamic_sph.unitConv_exp2th(expParams['n0_BEC_scale'], expParams['mB'])

    n0 = expParams['n0_BEC'] / (L_exp2th**3)  # should ~ 1
    mB = expParams['mB'] * M_exp2th  # should = 1
    mI = expParams['mI'] * M_exp2th
    aBB = expParams['aBB'] * L_exp2th
    gBB = (4 * np.pi / mB) * aBB

    sParams = [mI, mB, n0, gBB]

    # Trap parameters

    n0_TF = expParams['n0_TF'] / (L_exp2th**3)
    n0_thermal = expParams['n0_thermal'] / (L_exp2th**3)
    RTF_BEC_X = expParams['RTF_BEC_X'] * L_exp2th; RTF_BEC_Y = expParams['RTF_BEC_Y'] * L_exp2th; RTF_BEC_Z = expParams['RTF_BEC_Z'] * L_exp2th
    RG_BEC_X = expParams['RG_BEC_X'] * L_exp2th; RG_BEC_Y = expParams['RG_BEC_Y'] * L_exp2th; RG_BEC_Z = expParams['RG_BEC_Z'] * L_exp2th
    omega_BEC_osc = expParams['omega_BEC_osc'] / T_exp2th
    omega_Imp_x = expParams['omega_Imp_x'] / T_exp2th

    # Derived quantities

    nu = pf_dynamic_sph.nu(mB, n0, gBB)
    xi = (8 * np.pi * n0 * aBB)**(-1 / 2)
    Fscale = 2 * np.pi * (nu / xi**2)
    vI_init = expParams['vI_init'] * L_exp2th / T_exp2th
    PI_init = mI * vI_init
    tscale = xi / nu
    To = 2 * np.pi / omega_BEC_osc
    print(To, To / tscale)
    print(80 * 1e-3 * T_exp2th, 1e3 * 5000 / T_exp2th)
    print(1 / L_exp2th, expParams['RTF_BEC_X'], RTF_BEC_X / L_exp2th)
    print(0.75 * RTF_BEC_X / xi, omega_BEC_osc * tscale)
    print(RTF_BEC_X * (omega_BEC_osc / 8) / nu)

    print(RTF_BEC_X * (omega_BEC_osc / 2) / nu)
    print('c_BEC: {:.2E}'.format(nu * T_exp2th / L_exp2th))
    print(mI * nu)

    # Experimental params

    aIBexp_Vals = np.array([-1000, -750, -500, -375, -250, -125, -60, -20, 0, 20, 50, 125, 175, 250, 375, 500, 750, 1000])

    Na_displacement = np.array([26.2969729628679, 22.6668334850173, 18.0950989598699, 20.1069898676222, 14.3011351453467, 18.8126473489499, 17.0373115356076, 18.6684373282353, 18.8357213162278, 19.5036039713438, 21.2438389441807, 18.2089748680659, 18.0433963046778, 8.62940156299093, 16.2007030552903, 23.2646987822343, 24.1115616621798, 28.4351972435186])
    K_displacement_raw = np.array([0.473502276902047, 0.395634326123081, 8.66936929134637, 11.1470221226478, 9.34778274195669, 16.4370036199872, 19.0938486958001, 18.2135041439547, 21.9211790347041, 20.6591098913628, 19.7281375591975, 17.5425503131171, 17.2460344933717, 11.7179407507981, 12.9845862662090, 9.18113956217101, 11.9396846941782, 4.72461841775226])
    K_displacement_scale = np.mean(K_displacement_raw[6:11] / Na_displacement[6:11])
    K_displacement = deepcopy(K_displacement_raw); K_displacement[0:6] = K_displacement_scale * Na_displacement[0:6]; K_displacement[11::] = K_displacement_scale * Na_displacement[11::]
    K_relPos = K_displacement - Na_displacement

    omega_Na = np.array([465.418650581347, 445.155256942448, 461.691943131414, 480.899902898451, 448.655522184374, 465.195338759998, 460.143258369460, 464.565377197007, 465.206177963899, 471.262139163205, 471.260672147216, 473.122081065092, 454.649394420577, 449.679107889662, 466.770887179217, 470.530355145510, 486.615655444221, 454.601540658640])
    omega_K_raw = np.array([764.649207995890, 829.646158322623, 799.388442120805, 820.831266284088, 796.794204312379, 810.331402280747, 803.823888714144, 811.210511844489, 817.734286423120, 809.089608774626, 807.885837386121, 808.334196591376, 782.788534907910, 756.720677755942, 788.446619623011, 791.774719564856, 783.194731826180, 754.641677886382])
    omega_K_scale = np.mean(omega_K_raw[6:11] / omega_Na[6:11])
    omega_K = deepcopy(omega_K_raw); omega_K[0:6] = omega_K_scale * omega_Na[0:6]; omega_K[11::] = omega_K_scale * omega_Na[11::]

    K_relVel = np.array([1.56564660488838, 1.31601642026105, 0.0733613860991014, 1.07036861258786, 1.22929932184982, -13.6137940945403, 0.0369377794311800, 1.61258456681232, -1.50457700049200, -1.72583008593939, 4.11884512615162, 1.04853747806043, -0.352830359266360, -4.00683426531578, 0.846101589896479, -0.233660196108278, 4.82122627459411, -1.04341939663180])

    # # ---- SET OSC PARAMS ----
    # x0 = round(pf_dynamic_sph.x_BEC_osc(0, omega_BEC_osc, RTF_BEC_X, 0.5), 1)
    # print('X0: {0}, Tosc: {1}'.format(x0, To))

    # oscParams_List = [{'X0': 0.0, 'P0': 0.4, 'a_osc': expParams['a_osc']}]

    # TTList = []
    # for oscParams in oscParams_List:

    #     toggleDict = {'Location': 'cluster', 'Dynamics': 'real', 'Interaction': 'on', 'InitCS': 'steadystate', 'InitCS_datapath': '', 'Coupling': 'twophonon', 'Grid': 'spherical',
    #                   'F_ext': 'off', 'PosScat': 'off', 'BEC_density': 'on', 'BEC_density_osc': 'on', 'Imp_trap': 'on', 'CS_Dyn': 'on', 'Polaron_Potential': 'off'}

    #     trapParams = {'n0_TF_BEC': n0_TF, 'RTF_BEC_X': RTF_BEC_X, 'RTF_BEC_Y': RTF_BEC_Y, 'RTF_BEC_Z': RTF_BEC_Z, 'n0_thermal_BEC': n0_thermal, 'RG_BEC_X': RG_BEC_X, 'RG_BEC_Y': RG_BEC_Y, 'RG_BEC_Z': RG_BEC_Z,
    #                   'omega_Imp_x': omega_Imp_x, 'omega_BEC_osc': omega_BEC_osc, 'X0': oscParams['X0'], 'P0': oscParams['P0'], 'a_osc': oscParams['a_osc']}

    #     if trapParams['P0'] >= 1.1 * mI * nu:
    #         toggleDict['InitCS'] = 'file'

    #     if trapParams['a_osc'] == 0.0:
    #         toggleDict['BEC_density_osc'] = 'off'

    #     if toggleDict['BEC_density_osc'] == 'off':
    #         trapParams['a_osc'] = 0.0

    #     if toggleDict['Imp_trap'] == 'off':
    #         trapParams['omega_Imp_x'] = 0.0

    #     # ---- SET OUTPUT DATA FOLDER ----

    #     if toggleDict['Location'] == 'personal':
    #         datapath = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/BEC_osc'.format(aBB, NGridPoints_cart)
    #     elif toggleDict['Location'] == 'cluster':
    #         datapath = '/n/scratchlfs02/demler_lab/kis/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/BEC_osc'.format(aBB, NGridPoints_cart)

    #     if toggleDict['PosScat'] == 'on':
    #         innerdatapath = datapath + '/PosScat'
    #     else:
    #         innerdatapath = datapath + '/NegScat'

    #     if toggleDict['BEC_density'] == 'off':
    #         innerdatapath = innerdatapath + '/HomogBEC'
    #         toggleDict['Polaron_Potential'] = 'off'

    #     if toggleDict['Polaron_Potential'] == 'off':
    #         innerdatapath = innerdatapath + '/NoPolPot'
    #     else:
    #         innerdatapath = innerdatapath + '/PolPot'

    #     if toggleDict['CS_Dyn'] == 'off':
    #         innerdatapath = innerdatapath + '_NoCSDyn'
    #     else:
    #         innerdatapath = innerdatapath + '_CSDyn'

    #     innerdatapath = innerdatapath + '/fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(int(np.ceil(expParams['omega_BEC_osc'] / (2 * np.pi))), int(np.ceil(expParams['omega_Imp_x'] / (2 * np.pi))), trapParams['a_osc'], trapParams['X0'], trapParams['P0'])

    #     if toggleDict['InitCS'] == 'file':
    #         toggleDict['InitCS_datapath'] = datapath + '/PolGS_spherical'
    #     else:
    #         toggleDict['InitCS_datapath'] = 'InitCS ERROR'

    #     TTList.append((toggleDict, trapParams, innerdatapath))

    # # # # ---- CREATE EXTERNAL DATA FOLDERS  ----

    # # if os.path.isdir(datapath) is False:
    # #     os.mkdir(datapath)
    # #     os.mkdir(datapath + '/BEC_osc')

    # # # ---- CREATE OUTPUT DATA FOLDERS  ----

    # # for tup in TTList:
    # #     (toggleDict, trapParams, innerdatapath) = tup
    # #     if os.path.isdir(innerdatapath) is False:
    # #         os.mkdir(innerdatapath)

    # # ---- SET CPARAMS (RANGE OVER MULTIPLE aIBi) ----

    # a0_exp = 5.29e-11  # Bohr radius (m)

    # aIBexp_Vals = np.array([-1000, -750, -500, -375, -250, -125, -60, -20, 0, 20, 50, 125, 175, 250, 375, 500, 750, 1000]) * a0_exp
    # # aIBexp_Vals = np.array([-1000, -750, -500, -375, -250, -125, -60, -20, 0]) * a0_exp

    # aIBi_Vals = 1 / (aIBexp_Vals * L_exp2th)

    # metaList = []
    # for tup in TTList:
    #     (toggleDict, trapParams, innerdatapath) = tup
    #     for aIBi in aIBi_Vals:
    #         metaList.append((toggleDict, trapParams, innerdatapath, aIBi))

    # print(len(metaList))

    # # ---- COMPUTE DATA ON COMPUTER ----

    # runstart = timer()
    # for tup in metaList:
    #     loopstart = timer()
    #     (toggleDict, trapParams, innerdatapath, aIBi) = tup
    #     cParams = {'aIBi': aIBi}
    #     fParams = {'dP_ext': 0, 'Fext_mag': 0}
    #     filepath = innerdatapath + '/aIBi_{:.2f}.nc'.format(aIBi)
    #     if aIBi == 0.1:
    #         filepath = innerdatapath + '/aIBi_{:.2f}.nc'.format(-0.1)
    #     ds = pf_dynamic_sph.LDA_quenchDynamics_DataGeneration(cParams, gParams, sParams, fParams, trapParams, toggleDict)
    #     # Obs_ds = ds[['Pph', 'Nph', 'P', 'X', 'XLab', 'Energy']]; Obs_ds.attrs = ds.attrs; Obs_ds.to_netcdf(filepath)
    #     loopend = timer()
    #     print('X0: {:.2f}, P0: {:.2f}, a_osc: {:.2f}, aIBi: {:.2f}, Time: {:.2f}'.format(trapParams['X0'], trapParams['P0'], trapParams['a_osc'], aIBi, loopend - loopstart))
    # end = timer()
    # print('Total Time: {:.2f}'.format(end - runstart))

    # # ---- COMPUTE DATA ON CLUSTER ----

    # print(innerdatapath)

    # runstart = timer()

    # taskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    # taskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    # # taskCount = len(metaList); taskID = 72

    # if(taskCount > len(metaList)):
    #     print('ERROR: TASK COUNT MISMATCH')
    #     sys.exit()
    # else:
    #     tup = metaList[taskID]
    #     (toggleDict, trapParams, innerdatapath, aIBi) = tup

    # cParams = {'aIBi': aIBi}
    # fParams = {'dP_ext': 0, 'Fext_mag': 0}
    # filepath = innerdatapath + '/aIBi_{:.2f}.nc'.format(aIBi)
    # if aIBi == 0.1:
    #     filepath = innerdatapath + '/aIBi_{:.2f}.nc'.format(-0.1)
    # ds = pf_dynamic_sph.LDA_quenchDynamics_DataGeneration(cParams, gParams, sParams, fParams, trapParams, toggleDict)
    # # Obs_ds = ds[['Pph', 'Nph', 'P', 'X', 'XLab', 'Energy']]; Obs_ds.attrs = ds.attrs; Obs_ds.to_netcdf(filepath)
    # ds.to_netcdf(filepath)

    # end = timer()
    # print('Task ID: {:d}, X0: {:.2f}, P0: {:.2f}, a_osc: {:.2f}, aIBi: {:.2f}, Time: {:.2f}'.format(taskID, trapParams['X0'], trapParams['P0'], trapParams['a_osc'], aIBi, end - runstart))
