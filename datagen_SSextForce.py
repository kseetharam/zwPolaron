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

    tMax = 25; dt = 0.1
    tgrid = np.arange(0, tMax + dt, dt)

    gParams = [xgrid, kgrid, tgrid]
    NGridPoints = kgrid.size()

    print('Total time steps: {0}'.format(tgrid.size))
    print('UV cutoff: {0}'.format(k_max))
    print('NGridPoints: {0}'.format(NGridPoints))

    # Basic parameters

    # Basic parameters

    mI = 1.7
    mB = 1
    n0 = 1
    aBB = 0.016
    gBB = (4 * np.pi / mB) * aBB
    nu = pf_dynamic_sph.nu(mB, n0, gBB)
    xi = (8 * np.pi * n0 * aBB)**(-1 / 2)

    sParams = [mI, mB, n0, gBB]

    Fscale = 2 * np.pi * (nu / xi**2)

    # LDA functions

    def F_ext_func(t, F, dP):
        TF = dP / F
        if t <= TF:
            return F
        else:
            return 0

    def F_Vconf_func(X):
        return 0

    def F_Vden_func(X):
        return 0

    LDA_funcs = [F_ext_func, F_Vconf_func, F_Vden_func]

    # Toggle parameters

    toggleDict = {'Location': 'home', 'Dynamics': 'real', 'Interaction': 'on', 'InitCS': 'file', 'InitCS_datapath': '', 'LastTimeStepOnly': 'no', 'Coupling': 'twophonon', 'Grid': 'spherical'}

    # ---- SET OUTPUT DATA FOLDER ----

    if toggleDict['Location'] == 'home':
        datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/NGridPoints_{:.2E}/LDA/supersonic'.format(NGridPoints_cart)
    elif toggleDict['Location'] == 'work':
        datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/NGridPoints_{:.2E}/LDA/supersonic'.format(NGridPoints_cart)
    elif toggleDict['Location'] == 'cluster':
        datapath = '/n/regal/demler_lab/kis/ZwierleinExp_data/NGridPoints_{:.2E}/LDA/supersonic'.format(NGridPoints_cart)

    if toggleDict['Dynamics'] == 'real':
        innerdatapath = datapath + '/redyn'
    elif toggleDict['Dynamics'] == 'imaginary':
        innerdatapath = datapath + '/imdyn'

    if toggleDict['Grid'] == 'cartesian':
        innerdatapath = innerdatapath + '_cart'
    elif toggleDict['Grid'] == 'spherical':
        innerdatapath = innerdatapath + '_spherical'

    if toggleDict['Coupling'] == 'frohlich':
        innerdatapath = innerdatapath + '_froh'
    elif toggleDict['Coupling'] == 'twophonon':
        innerdatapath = innerdatapath

    if toggleDict['InitCS'] == 'file':
        toggleDict['InitCS_datapath'] = datapath[:-11] + '/P_0_PolStates'
    elif toggleDict['InitCS'] == 'default':
        toggleDict['InitCS_datapath'] = 'InitCS ERROR'

    if toggleDict['Interaction'] == 'off':
        innerdatapath = innerdatapath + '_nonint'
    elif toggleDict['Interaction'] == 'on':
        innerdatapath = innerdatapath

    if os.path.isdir(datapath) is False:
        os.mkdir(datapath)

    if os.path.isdir(innerdatapath) is False:
        os.mkdir(innerdatapath)

    # # ---- SINGLE FUNCTION RUN ----

    # runstart = timer()
    # F = 0.1 * Fscale
    # aIBi = -0.05
    # dP = 0.9 * mI * nu

    # cParams = [F, aIBi]
    # fParams = [dP]
    # print('TF: {0}'.format(dP / F))

    # ds = pf_dynamic_sph.LDA_quenchDynamics_DataGeneration(cParams, gParams, sParams, fParams, LDA_funcs, toggleDict)
    # Obs_ds = ds[['Pph', 'Nph', 'P', 'X']]; Obs_ds.attrs = ds.attrs; Obs_ds.to_netcdf(innerdatapath + '/dP_{:.3f}_F_{:.3f}_aIBi_{:.2f}.nc'.format(dP, F, aIBi))

    # end = timer()
    # print('Time: {:.2f}'.format(end - runstart))

    # ---- SET CPARAMS (RANGE OVER MULTIPLE aIBi) ----

    cfParams_List = []

    aIBi_Vals = np.array([-1.17, -0.5, -0.05])
    F_Vals = np.linspace(0.1 * Fscale, 35 * Fscale, 20)
    dP_Vals = np.array([0.3 * mI * nu, 0.9 * mI * nu])

    for aIBi in aIBi_Vals:
        for F in F_Vals:
            for dP in dP_Vals:
                cfParams_List.append([F, aIBi, dP])

    # ---- COMPUTE DATA ON COMPUTER ----

    runstart = timer()

    for ind, cParams in enumerate(cfParams_List):
        loopstart = timer()
        [F, aIBi, dP] = cfParams_List[ind]
        cParams = [F, aIBi]
        fParams = [dP]
        ds = pf_dynamic_sph.LDA_quenchDynamics_DataGeneration(cParams, gParams, sParams, fParams, LDA_funcs, toggleDict)
        Obs_ds = ds[['Pph', 'Nph', 'P', 'X']]; Obs_ds.attrs = ds.attrs; Obs_ds.to_netcdf(innerdatapath + '/dP_{:.3f}_F_{:.3f}_aIBi_{:.2f}.nc'.format(dP, F, aIBi))
        loopend = timer()
        print('Index: {:d}, dP: {:.2f}, F: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(ind, dP, F, aIBi, loopend - loopstart))

    end = timer()
    print('Total Time: {:.2f}'.format(end - runstart))

    # # ---- COMPUTE DATA ON CLUSTER ----

    # runstart = timer()

    # taskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    # taskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    # if(taskCount > len(cfParams_List)):
    #     print('ERROR: TASK COUNT MISMATCH')
    #     sys.exit()
    # else:
    #     [F, aIBi, dP] = cfParams_List[taskID]
    #     cParams = [F, aIBi]
    #     fParams = [dP]
    # ds = pf_dynamic_sph.LDA_quenchDynamics_DataGeneration(cParams, gParams, sParams, fParams, LDA_funcs, toggleDict)
    # Obs_ds = ds[['Pph', 'Nph', 'P', 'X']]; Obs_ds.attrs = ds.attrs; Obs_ds.to_netcdf(innerdatapath + '/dP_{:.3f}_F_{:.3f}_aIBi_{:.2f}.nc'.format(dP, F, aIBi))

    # end = timer()
    # print('Task ID: {:d}, dP: {:.2f}, F: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(taskID, dP, F, aIBi, end - runstart))
