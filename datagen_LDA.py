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

    # # (Lx, Ly, Lz) = (30, 30, 30)
    (Lx, Ly, Lz) = (20, 20, 20)
    # # (Lx, Ly, Lz) = (10, 10, 10)
    (dx, dy, dz) = (0.2, 0.2, 0.2)

    # (Lx, Ly, Lz) = (21, 21, 21)
    # (dx, dy, dz) = (0.375, 0.375, 0.375)

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

    tMax = 25; dt = 0.01
    tgrid = np.arange(0, tMax + dt, dt)

    gParams = [xgrid, kgrid, tgrid]
    NGridPoints = kgrid.size()

    print('Total time steps: {0}'.format(tgrid.size))
    print('UV cutoff: {0}'.format(k_max))
    print('dk: {0}'.format(dk))
    print('NGridPoints: {0}'.format(NGridPoints))

    # Basic parameters

    expParams = pf_dynamic_sph.Zw_expParams()
    # L_exp2th, M_exp2th, T_exp2th = pf_dynamic_sph.unitConv_exp2th(expParams['n0_BEC'], expParams['mB'])
    L_exp2th, M_exp2th, T_exp2th = pf_dynamic_sph.unitConv_exp2th(expParams['n0_BEC_scale'], expParams['mB'])

    n0 = expParams['n0_BEC'] / (L_exp2th**3)  # should ~ 1
    mB = expParams['mB'] * M_exp2th  # should = 1
    mI = expParams['mI'] * M_exp2th
    aBB = expParams['aBB'] * L_exp2th
    gBB = (4 * np.pi / mB) * aBB

    sParams = [mI, mB, n0, gBB]

    # # Basic parameters

    # mI = 1
    # mB = 1
    # n0 = 1
    # aBB = 0.05
    # gBB = (4 * np.pi / mB) * aBB

    # sParams = [mI, mB, n0, gBB]

    # Trap parameters

    n0_TF = expParams['n0_TF'] / (L_exp2th**3)
    n0_thermal = expParams['n0_thermal'] / (L_exp2th**3)
    RTF_BEC_X = expParams['RTF_BEC_X'] * L_exp2th; RTF_BEC_Y = expParams['RTF_BEC_Y'] * L_exp2th; RTF_BEC_Z = expParams['RTF_BEC_Z'] * L_exp2th
    RG_BEC_X = expParams['RG_BEC_X'] * L_exp2th; RG_BEC_Y = expParams['RG_BEC_Y'] * L_exp2th; RG_BEC_Z = expParams['RG_BEC_Z'] * L_exp2th
    omega_BEC_osc = expParams['omega_BEC_osc'] / T_exp2th

    trapParams = {'n0_TF_BEC': n0_TF, 'RTF_BEC_X': RTF_BEC_X, 'RTF_BEC_Y': RTF_BEC_Y, 'RTF_BEC_Z': RTF_BEC_Z, 'n0_thermal_BEC': n0_thermal, 'RG_BEC_X': RG_BEC_X, 'RG_BEC_Y': RG_BEC_Y, 'RG_BEC_Z': RG_BEC_Z, 'omega_BEC_osc': omega_BEC_osc}

    # Derived quantities

    nu = pf_dynamic_sph.nu(mB, n0, gBB)
    xi = (8 * np.pi * n0 * aBB)**(-1 / 2)
    Fscale = 2 * np.pi * (nu / xi**2)
    vI_init = expParams['vI_init'] * L_exp2th / T_exp2th
    PI_init = mI * vI_init

    # Toggle parameters

    toggleDict = {'Location': 'work', 'Dynamics': 'real', 'Interaction': 'on', 'InitCS': 'steadystate', 'InitCS_datapath': '', 'Coupling': 'twophonon', 'Grid': 'spherical',
                  'F_ext': 'off', 'BEC_density': 'on', 'BEC_density_osc': 'on'}

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
        innerdatapath = innerdatapath + '_cart'
    elif toggleDict['Grid'] == 'spherical':
        innerdatapath = innerdatapath + '_spherical'

    if toggleDict['F_ext'] == 'on':
        innerdatapath = innerdatapath + '_extForce'
    elif toggleDict['F_ext'] == 'off':
        innerdatapath = innerdatapath

    if toggleDict['BEC_density'] == 'on':
        innerdatapath = innerdatapath + '_BECden'
    elif toggleDict['BEC_density'] == 'off':
        innerdatapath = innerdatapath

    if toggleDict['BEC_density_osc'] == 'on':
        innerdatapath = innerdatapath + '_BECosc'
    elif toggleDict['BEC_density_osc'] == 'off':
        innerdatapath = innerdatapath

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

    # # ---- SINGLE FUNCTION RUN ----

    # runstart = timer()
    # aIBi = -1.3
    # dP = 0.5 * mI * nu
    # F = 0.1 * Fscale
    # filepath = innerdatapath + '/aIBi_{:.2f}_dP_{:.2f}mIc_F_{:.2f}.nc'.format(aIBi, dP / (mI * nu), F)
    # if toggleDict['F_ext'] == 'off':
    #     dP = 0; F = 0; filepath = innerdatapath + '/aIBi_{:.2f}.nc'.format(aIBi)
    # print('mI: {:.2f}, mB:{:.1f}, aBB: {:.3f}, aIBi: {:.2f}, n0: {:.1f}'.format(mI, mB, aBB, aIBi, n0))
    # # print('TF: {0}'.format(dP / F))

    # cParams = {'aIBi': aIBi}
    # fParams = {'dP_ext': dP, 'Fext_mag': F}

    # ds = pf_dynamic_sph.LDA_quenchDynamics_DataGeneration(cParams, gParams, sParams, fParams, trapParams, toggleDict)
    # Obs_ds = ds[['Pph', 'Nph', 'P', 'X']]; Obs_ds.attrs = ds.attrs; Obs_ds.to_netcdf(filepath)

    # end = timer()
    # print('Time: {:.2f}'.format(end - runstart))

    # ---- SET CPARAMS (RANGE OVER MULTIPLE aIBi) ----

    cFParams_List = []

    # aIBi_Vals = np.array([-10, -5, -2, -1])
    # dP_Vals = np.array([0.5 * mI * nu, 1.3 * mI * nu, 3 * mI * nu])
    # F_Vals = np.array([0.2 * Fscale, 10 * Fscale, 35 * Fscale])

    # aIBi_Vals = np.array([-5.0, -1.3, -0.5, -0.05, 0.1])
    # dP_Vals = np.array([0.5 * mI * nu, PI_init, 3 * mI * nu])
    # F_Vals = np.array([0.2 * Fscale, 10 * Fscale, 35 * Fscale])

    aIBi_Vals = np.array([-5.0, -1.3, -0.5, -0.05, 0.1])
    dP_Vals = np.array([0])
    F_Vals = np.array([0])

    for aIBi in aIBi_Vals:
        for dP in dP_Vals:
            for Fext_mag in F_Vals:
                cFParams = {'aIBi': aIBi, 'dP': dP, 'Fext_mag': Fext_mag}
                cFParams_List.append(cFParams)

    # ---- COMPUTE DATA ON COMPUTER ----

    runstart = timer()

    for ind, cFParams in enumerate(cFParams_List):
        loopstart = timer()
        aIBi = cFParams['aIBi']; dP = cFParams['dP']; F = cFParams['Fext_mag']
        cParams = {'aIBi': aIBi}
        fParams = {'dP_ext': dP, 'Fext_mag': F}
        filepath = innerdatapath + '/aIBi_{:.2f}_dP_{:.2f}mIc_F_{:.2f}.nc'.format(aIBi, dP / (mI * nu), F)
        if toggleDict['F_ext'] == 'off':
            dP = 0; F = 0; filepath = innerdatapath + '/aIBi_{:.2f}.nc'.format(aIBi)
        ds = pf_dynamic_sph.LDA_quenchDynamics_DataGeneration(cParams, gParams, sParams, fParams, trapParams, toggleDict)
        Obs_ds = ds[['Pph', 'Nph', 'P', 'X']]; Obs_ds.attrs = ds.attrs; Obs_ds.to_netcdf(filepath)

        loopend = timer()
        print('Index: {:d}, F: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(ind, F, aIBi, loopend - loopstart))

    end = timer()
    print('Total Time: {:.2f}'.format(end - runstart))

    # # ---- COMPUTE DATA ON CLUSTER ----

    # runstart = timer()

    # taskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    # taskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    # if(taskCount > len(cFParams_List)):
    #     print('ERROR: TASK COUNT MISMATCH')
    #     P = float('nan')
    #     aIBi = float('nan')
    #     sys.exit()
    # else:
    #     cFParams = cFParams_List[taskID]
    #     aIBi = cFParams['aIBi']; dP = cFParams['dP']; F = cFParams['Fext_mag']
    #     cParams = {'aIBi': aIBi}
    #     fParams = {'dP_ext': dP, 'Fext_mag': F}

    # filepath = innerdatapath + '/aIBi_{:.2f}_dP_{:.2f}mIc_F_{:.2f}.nc'.format(aIBi, dP / (mI * nu), F)
    # if toggleDict['F_ext'] == 'off':
    #     dP = 0; F = 0; filepath = innerdatapath + '/aIBi_{:.2f}.nc'.format(aIBi)

    # ds = pf_dynamic_sph.LDA_quenchDynamics_DataGeneration(cParams, gParams, sParams, fParams, trapParams, toggleDict)
    # Obs_ds = ds[['Pph', 'Nph', 'P', 'X']]; Obs_ds.attrs = ds.attrs; Obs_ds.to_netcdf(filepath)

    # end = timer()
    # print('Task ID: {:d}, F: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(taskID, F, aIBi, end - runstart))
