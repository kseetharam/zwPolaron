import numpy as np
import pandas as pd
import xarray as xr
import Grid
import pf_dynamic_sph
import os
from timeit import default_timer as timer
import sys
# import matplotlib
# import matplotlib.pyplot as plt

if __name__ == "__main__":

    start = timer()

    # ---- INITIALIZE GRIDS ----

    # (Lx, Ly, Lz) = (30, 30, 30)
    (Lx, Ly, Lz) = (20, 20, 20)
    # (Lx, Ly, Lz) = (10, 10, 10)
    (dx, dy, dz) = (0.2, 0.2, 0.2)

    xgrid = Grid.Grid('CARTESIAN_3D')
    xgrid.initArray('x', -Lx, Lx, dx); xgrid.initArray('y', -Ly, Ly, dy); xgrid.initArray('z', -Lz, Lz, dz)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    NGridPoints_desired = (1 + 2 * Lx / dx) * (1 + 2 * Lz / dz)
    Ntheta = 50
    Nk = np.ceil(NGridPoints_desired / Ntheta)

    theta_max = np.pi
    thetaArray, dtheta = np.linspace(0, theta_max, Ntheta, retstep=True)

    # k_max = np.sqrt((np.pi / dx)**2 + (np.pi / dy)**2 + (np.pi / dz)**2)
    k_max = ((2 * np.pi / dx)**3 / (4 * np.pi / 3))**(1 / 3)

    k_min = 1e-5
    kArray, dk = np.linspace(k_min, k_max, Nk, retstep=True)
    if dk < k_min:
        print('k ARRAY GENERATION ERROR')

    kgrid = Grid.Grid("SPHERICAL_2D")
    kgrid.initArray_premade('k', kArray)
    kgrid.initArray_premade('th', thetaArray)

    tMax = 20; dt = 0.01
    tgrid = np.arange(0, tMax + dt, dt)

    gParams = [xgrid, kgrid, tgrid]
    NGridPoints = kgrid.size()

    print('Total time steps: {0}'.format(tgrid.size))
    print('UV cutoff: {0}'.format(k_max))
    print('dk: {0}'.format(dk))
    print('NGridPoints: {0}'.format(NGridPoints))

    # Basic parameters

    exp_params = pf_dynamic_sph.Zw_exp_params()
    L_exp2th, M_exp2th, T_exp2th = pf_dynamic_sph.unitConv_th_exp(exp_params['n0'], exp_params['mB'])

    n0 = 1
    mB = 1
    mI = exp_params['mI'] * M_exp2th
    aBB = exp_params['aBB'] * L_exp2th
    gBB = (4 * np.pi / mB) * aBB

    sParams = [mI, mB, n0, gBB]

    # Trap parameters

    n0_thermal = exp_params['n0_thermal'] / (L_exp2th**3)
    RTF_BEC_X = exp_params['RTF_BEC_X'] * L_exp2th; RTF_BEC_Y = exp_params['RTF_BEC_Y'] * L_exp2th; RTF_BEC_Z = exp_params['RTF_BEC_Z'] * L_exp2th
    RG_BEC_X = exp_params['RG_BEC_X'] * L_exp2th; RG_BEC_Y = exp_params['RG_BEC_Y'] * L_exp2th; RG_BEC_Z = exp_params['RG_BEC_Z'] * L_exp2th

    trapParams = {'n0_BEC': n0, 'RTF_BEC_X': RTF_BEC_X, 'RTF_BEC_Y': RTF_BEC_Y, 'RTF_BEC_Z': RTF_BEC_Z, 'n0_thermal_BEC': n0_thermal, 'RG_BEC_X': RG_BEC_X, 'RG_BEC_Y': RG_BEC_Y, 'RG_BEC_Z': RG_BEC_Z}

    # Derived quantities

    nu = pf_dynamic_sph.nu(mB, n0, gBB)
    xi = (8 * np.pi * n0 * aBB)**(-1 / 2)
    Fscale = 2 * np.pi * (nu / xi**2)

    # Toggle parameters

    toggleDict = {'Location': 'home', 'Dynamics': 'real', 'Interaction': 'on', 'InitCS': 'steadystate', 'InitCS_datapath': '', 'Coupling': 'twophonon', 'Grid': 'spherical',
                  'F_ext': 'on', 'BEC_density': 'on'}

    # ---- SET OUTPUT DATA FOLDER ----

    if toggleDict['Location'] == 'home':
        datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/LDA'.format(aBB, NGridPoints_cart)
    elif toggleDict['Location'] == 'work':
        datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/LDA'.format(aBB, NGridPoints_cart)
    elif toggleDict['Location'] == 'cluster':
        datapath = '/n/regal/demler_lab/kis/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/LDA'.format(aBB, NGridPoints_cart)

    if toggleDict['Dynamics'] == 'real':
        innerdatapath = datapath + '/redyn'
    elif toggleDict['Dynamics'] == 'imaginary':
        innerdatapath = datapath + '/imdyn'

    if toggleDict['Grid'] == 'cartesian':
        innerdatapath = innerdatapath + '_cart_extForce'
    elif toggleDict['Grid'] == 'spherical':
        innerdatapath = innerdatapath + '_spherical_extForce'

    if toggleDict['Coupling'] == 'frohlich':
        innerdatapath = innerdatapath + '_froh'
    elif toggleDict['Coupling'] == 'twophonon':
        innerdatapath = innerdatapath

    if toggleDict['InitCS'] == 'file':
        toggleDict['InitCS_datapath'] = datapath + '/P_0_PolStates'
        innerdatapath = innerdatapath + '_ImDynStart'
    elif toggleDict['InitCS'] == 'steadystate':
        toggleDict['InitCS_datapath'] = 'InitCS ERROR'
        innerdatapath = innerdatapath + '_SteadyStart_P_0.1'
    elif toggleDict['InitCS'] == 'default':
        toggleDict['InitCS_datapath'] = 'InitCS ERROR'

    if toggleDict['Interaction'] == 'off':
        innerdatapath = innerdatapath + '_nonint'
    elif toggleDict['Interaction'] == 'on':
        innerdatapath = innerdatapath

    # if os.path.isdir(datapath) is False:
    #     os.mkdir(datapath)

    # if os.path.isdir(innerdatapath) is False:
    #     os.mkdir(innerdatapath)

    # ---- SINGLE FUNCTION RUN ----

    runstart = timer()
    aIBi = (exp_params['aIB'] * L_exp2th)**(-1)
    dP = 0.5 * mI * nu
    F = 0.1 * Fscale
    print('mI: {0}, mB:{1}, aBB: {2}, aIBi: {3}'.format(mI, mB, aBB, aIBi))
    print('TF: {0}'.format(dP / F))

    cParams = [aIBi]
    fParams = [dP, F]

    ds = pf_dynamic_sph.LDA_quenchDynamics_DataGeneration(cParams, gParams, sParams, fParams, trapParams, toggleDict)
    # Obs_ds = ds[['Pph', 'Nph', 'P', 'X']]; Obs_ds.attrs = ds.attrs; Obs_ds.to_netcdf(innerdatapath + '/F_{:.3f}_aIBi_{:.2f}.nc'.format(F, aIBi))

    end = timer()
    print('Time: {:.2f}'.format(end - runstart))

    # # ---- SET CPARAMS (RANGE OVER MULTIPLE aIBi) ----

    # cParams_List = []

    # # aIBi_Vals = np.array([-5.0, -1.17, -0.5, -0.05, 0.1])
    # # F_Vals = np.linspace(0.1 * Fscale, 35 * Fscale, 20)

    # aIBi_Vals = np.array([-5.0, -1.24, -0.5, -0.05, 0.1])
    # F_Vals = np.linspace(0.1 * Fscale, 35 * Fscale, 20)
    # # print(dP / F_Vals)

    # for ind, aIBi in enumerate(aIBi_Vals):
    #     for F in F_Vals:
    #         cParams_List.append([F, aIBi])

    # # ---- COMPUTE DATA ON COMPUTER ----

    # runstart = timer()

    # for ind, cParams in enumerate(cParams_List):
    #     loopstart = timer()
    #     [F, aIBi] = cParams
    #     ds = pf_dynamic_sph.LDA_quenchDynamics_DataGeneration(cParams, gParams, sParams, fParams, toggleDict)
    #     Obs_ds = ds[['Pph', 'Nph', 'P', 'X']]; Obs_ds.attrs = ds.attrs; Obs_ds.to_netcdf(innerdatapath + '/F_{:.3f}_aIBi_{:.2f}.nc'.format(F, aIBi))

    #     loopend = timer()
    #     print('Index: {:d}, F: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(ind, F, aIBi, loopend - loopstart))

    # end = timer()
    # print('Total Time: {:.2f}'.format(end - runstart))

    # # ---- COMPUTE DATA ON CLUSTER ----

    # runstart = timer()

    # taskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    # taskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    # if(taskCount > len(cParams_List)):
    #     print('ERROR: TASK COUNT MISMATCH')
    #     P = float('nan')
    #     aIBi = float('nan')
    #     sys.exit()
    # else:
    #     cParams = cParams_List[taskID]
    #     [F, aIBi] = cParams

    # ds = pf_dynamic_sph.LDA_quenchDynamics_DataGeneration(cParams, gParams, sParams, fParams, toggleDict)
    # Obs_ds = ds[['Pph', 'Nph', 'P', 'X']]; Obs_ds.attrs = ds.attrs; Obs_ds.to_netcdf(innerdatapath + '/F_{:.3f}_aIBi_{:.2f}.nc'.format(F, aIBi))

    # end = timer()
    # print('Task ID: {:d}, F: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(taskID, F, aIBi, end - runstart))
