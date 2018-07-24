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

    # (Lx, Ly, Lz) = (30, 30, 30)
    # (dx, dy, dz) = (0.2, 0.2, 0.2)

    # (Lx, Ly, Lz) = (30, 30, 30)
    # (dx, dy, dz) = (0.25, 0.25, 0.25)

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

    # tMax = 400; dt = 1
    # tMax = 240; dt = 0.1
    tMax = 480; dt = 0.1
    # tMax = 1; dt = 0.2
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
    print(1 / L_exp2th, expParams['RTF_BEC_X'], RTF_BEC_X / L_exp2th)
    print(0.75 * RTF_BEC_X / xi, omega_BEC_osc * tscale)
    print(RTF_BEC_X * (omega_BEC_osc / 8) / nu)

    # ---- SET OSC PARAMS ----
    x0 = round(pf_dynamic_sph.x_BEC_osc(0, omega_BEC_osc, RTF_BEC_X, 0.5), 1)
    print('X0: {0}, Tosc: {1}'.format(x0, To))

    oscParams_List = [{'X0': 0.0, 'P0': 0.6, 'a_osc': 0.5}]

    # oscParams_List = [{'X0': 0.0, 'P0': 0.1, 'a_osc': 0.5},
    #                   {'X0': 0.0, 'P0': 0.6, 'a_osc': 0.5}]

    # oscParams_List = [{'X0': 0.0, 'P0': 1.8, 'a_osc': 0.5},
    #                   {'X0': 0.0, 'P0': 0.1, 'a_osc': 0.0},
    #                   {'X0': 0.0, 'P0': 0.6, 'a_osc': 0.0},
    #                   {'X0': 0.0, 'P0': 1.8, 'a_osc': 0.0}]

    TTList = []
    for oscParams in oscParams_List:

        toggleDict = {'Location': 'cluster', 'Dynamics': 'real', 'Interaction': 'on', 'InitCS': 'steadystate', 'InitCS_datapath': '', 'Coupling': 'twophonon', 'Grid': 'spherical',
                      'F_ext': 'off', 'BEC_density': 'on', 'BEC_density_osc': 'on', 'Imp_trap': 'on'}

        trapParams = {'n0_TF_BEC': n0_TF, 'RTF_BEC_X': RTF_BEC_X, 'RTF_BEC_Y': RTF_BEC_Y, 'RTF_BEC_Z': RTF_BEC_Z, 'n0_thermal_BEC': n0_thermal, 'RG_BEC_X': RG_BEC_X, 'RG_BEC_Y': RG_BEC_Y, 'RG_BEC_Z': RG_BEC_Z,
                      'omega_Imp_x': omega_Imp_x, 'omega_BEC_osc': omega_BEC_osc, 'X0': oscParams['X0'], 'P0': oscParams['P0'], 'a_osc': oscParams['a_osc']}

        if trapParams['P0'] >= 1.1 * mI * nu:
            toggleDict['InitCS'] = 'file'

        if trapParams['a_osc'] == 0.0:
            toggleDict['BEC_density_osc'] = 'off'

        if toggleDict['BEC_density_osc'] == 'off':
            trapParams['a_osc'] = 0.0

        if toggleDict['Imp_trap'] == 'off':
            trapParams['omega_Imp_x'] = 0.0

        # ---- SET OUTPUT DATA FOLDER ----

        if toggleDict['Location'] == 'home':
            datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}'.format(aBB, NGridPoints_cart)
        elif toggleDict['Location'] == 'work':
            datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}'.format(aBB, NGridPoints_cart)
        elif toggleDict['Location'] == 'cluster':
            datapath = '/n/regal/demler_lab/kis/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}'.format(aBB, NGridPoints_cart)
        innerdatapath = datapath + '/BEC_osc/fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(int(np.ceil(expParams['omega_BEC_osc'] / (2 * np.pi))), int(np.ceil(expParams['omega_Imp_x'] / (2 * np.pi))), trapParams['a_osc'], trapParams['X0'], trapParams['P0'])
        if toggleDict['InitCS'] == 'file':
            toggleDict['InitCS_datapath'] = datapath + '/PolGS_spherical'
        else:
            toggleDict['InitCS_datapath'] = 'InitCS ERROR'

        TTList.append((toggleDict, trapParams, innerdatapath))

    # # # ---- CREATE EXTERNAL DATA FOLDERS  ----

    # if os.path.isdir(datapath) is False:
    #     os.mkdir(datapath)
    #     os.mkdir(datapath + '/BEC_osc')

    # # ---- CREATE OUTPUT DATA FOLDERS  ----

    # for tup in TTList:
    #     (toggleDict, trapParams, innerdatapath) = tup
    #     if os.path.isdir(innerdatapath) is False:
    #         os.mkdir(innerdatapath)

    # # ---- SINGLE FUNCTION RUN ----

    # runstart = timer()
    # (toggleDict, trapParams, innerdatapath0, innerdatapath) = TTList[0]
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

    # aIBi_Vals = np.array([-1000.0, -20.0, -5.0, -1.3, -0.05])
    # aIBi_Vals = np.array([-1000.0, -20.0, -5.0, -0.05])
    # aIBi_Vals = np.concatenate((np.linspace(-1000, -10, 100), np.linspace(-5, -.05, 5)))
    aIBi_Vals = np.concatenate((np.array([-150, -140, -130, -120, -110]), np.linspace(-100, -1, 199)))
    # aIBi_Vals = [-0.25]

    metaList = []
    for tup in TTList:
        (toggleDict, trapParams, innerdatapath) = tup
        for aIBi in aIBi_Vals:
            metaList.append((toggleDict, trapParams, innerdatapath, aIBi))

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
    #     Obs_ds = ds[['Pph', 'Nph', 'P', 'X', 'XLab', 'Energy']]; Obs_ds.attrs = ds.attrs; Obs_ds.to_netcdf(filepath)
    #     loopend = timer()
    #     print('X0: {:.2f}, P0: {:.2f}, a_osc: {:.2f}, aIBi: {:.2f}, Time: {:.2f}'.format(trapParams['X0'], trapParams['P0'], trapParams['a_osc'], aIBi, loopend - loopstart))
    # end = timer()
    # print('Total Time: {:.2f}'.format(end - runstart))

    # ---- COMPUTE DATA ON CLUSTER ----

    runstart = timer()

    taskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    taskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    if(taskCount > len(metaList)):
        print('ERROR: TASK COUNT MISMATCH')
        sys.exit()
    else:
        tup = metaList[taskID]
        (toggleDict, trapParams, innerdatapath, aIBi) = tup

    cParams = {'aIBi': aIBi}
    fParams = {'dP_ext': 0, 'Fext_mag': 0}
    filepath = innerdatapath + '/aIBi_{:.2f}.nc'.format(aIBi)
    if aIBi == 0.1:
        filepath = innerdatapath + '/aIBi_{:.2f}.nc'.format(-0.1)
    ds = pf_dynamic_sph.LDA_quenchDynamics_DataGeneration(cParams, gParams, sParams, fParams, trapParams, toggleDict)
    Obs_ds = ds[['Pph', 'Nph', 'P', 'X', 'XLab', 'Energy']]; Obs_ds.attrs = ds.attrs; Obs_ds.to_netcdf(filepath)

    end = timer()
    print('Task ID: {:d}, X0: {:.2f}, P0: {:.2f}, a_osc: {:.2f}, aIBi: {:.2f}, Time: {:.2f}'.format(taskID, trapParams['X0'], trapParams['P0'], trapParams['a_osc'], aIBi, end - runstart))
