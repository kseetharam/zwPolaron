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

    # tMax = 1000; dt = 100
    tMax = 25; dt = 0.2
    tgrid = np.arange(0, tMax + dt, dt)

    gParams = [xgrid, kgrid, tgrid]
    NGridPoints = kgrid.size()

    print('Total time steps: {0}'.format(tgrid.size))
    print('UV cutoff: {0}'.format(k_max))
    print('dk: {0}'.format(dk))
    print('NGridPoints: {0}'.format(NGridPoints))

    # # Basic parameters

    # mI = 1.7
    # mB = 1
    # n0 = 1
    # aBB = 0.016
    # gBB = (4 * np.pi / mB) * aBB
    # nu = pf_dynamic_sph.nu(mB, n0, gBB)

    # sParams = [mI, mB, n0, gBB]

    # Basic parameters

    expParams = pf_dynamic_sph.Zw_expParams()
    L_exp2th, M_exp2th, T_exp2th = pf_dynamic_sph.unitConv_exp2th(expParams['n0_BEC'], expParams['mB'])

    n0 = expParams['n0_BEC'] / (L_exp2th**3)  # should = 1
    mB = expParams['mB'] * M_exp2th  # should = 1
    mI = expParams['mI'] * M_exp2th
    aBB = expParams['aBB'] * L_exp2th
    gBB = (4 * np.pi / mB) * aBB

    vI_init = expParams['vI_init'] * L_exp2th / T_exp2th
    PI_init = mI * vI_init

    sParams = [mI, mB, n0, gBB]

    # Toggle parameters

    # toggleDict = {'Location': 'work', 'Dynamics': 'imaginary', 'Interaction': 'on', 'InitCS': 'none', 'InitCS_datapath': '', 'LastTimeStepOnly': 'yes', 'Coupling': 'twophonon', 'Grid': 'spherical'}
    toggleDict = {'Location': 'work', 'Dynamics': 'real', 'Interaction': 'off', 'InitCS': 'file', 'InitCS_datapath': '', 'LastTimeStepOnly': 'no', 'Coupling': 'twophonon', 'Grid': 'spherical'}

    # ---- SET OUTPUT DATA FOLDER ----

    if toggleDict['Location'] == 'home':
        datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}'.format(aBB, NGridPoints_cart)
    elif toggleDict['Location'] == 'work':
        datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}'.format(aBB, NGridPoints_cart)
    elif toggleDict['Location'] == 'cluster':
        datapath = '/n/regal/demler_lab/kis/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}'.format(aBB, NGridPoints_cart)

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
        toggleDict['InitCS_datapath'] = datapath + '/imdyn_spherical'
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

    P = 0.1
    aIBi = -1.77

    cParams = [P, aIBi]

    dynsph_ds = pf_dynamic_sph.quenchDynamics_DataGeneration(cParams, gParams, sParams, toggleDict)

    if toggleDict['LastTimeStepOnly'] == 'yes':
        CSAmp_ds = dynsph_ds[['Real_CSAmp', 'Imag_CSAmp', 'Phase']].isel(t=-1); CSAmp_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))  # imag time evolution to get polaron state
    else:
        # CSAmp_ds = dynsph_ds[['Real_CSAmp', 'Imag_CSAmp', 'Phase']]; CSAmp_ds.attrs = dynsph_ds.attrs; CSAmp_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))  # real time evolution to get direct S(t)
        DynOv_ds = pf_dynamic_sph.dirRF(dynsph_ds, kgrid, cParams, sParams); DynOv_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))  # real time evolution to get direct S(t)

    end = timer()
    print('Time: {:.2f}'.format(end - runstart))

    # ---- SET CPARAMS (RANGE OVER MULTIPLE aIBi, P VALUES) ----

    cParams_List = []

    # aIBi_Vals = (n0*6 * np.pi**2)**(1 / 3) * np.array([-0.3])
    # aIBi_Vals = np.array([-1.17, -0.5, 0.1, 0.7])
    # P_Vals = np.linspace(0.1, mI * nu, 30)

    aIBi_Vals = np.array([-5.0, -1.77, -0.5, -0.05, 0.1])
    P_Vals = np.array([0.1])

    for ind, aIBi in enumerate(aIBi_Vals):
        # gIB = pf_dynamic_sph.g(kgrid, aIBi, mI, mB, n0, gBB)
        # print(gIB)
        for P in P_Vals:
            cParams_List.append([P, aIBi])

    # # ---- COMPUTE DATA ON COMPUTER ----

    # runstart = timer()

    # for ind, cParams in enumerate(cParams_List):
    #     loopstart = timer()
    #     [P, aIBi] = cParams
    #     dynsph_ds = pf_dynamic_sph.quenchDynamics_DataGeneration(cParams, gParams, sParams, toggleDict)

    #     if toggleDict['LastTimeStepOnly'] == 'yes':
    #         CSAmp_ds = dynsph_ds[['Real_CSAmp', 'Imag_CSAmp', 'Phase']].isel(t=-1); CSAmp_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))  # imag time evolution to get polaron state
    #     else:
    #         # CSAmp_ds = dynsph_ds[['Real_CSAmp', 'Imag_CSAmp', 'Phase']]; CSAmp_ds.attrs = dynsph_ds.attrs; CSAmp_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))  # real time evolution to get direct S(t)
    #         DynOv_ds = pf_dynamic_sph.dirRF(dynsph_ds, kgrid, cParams, sParams); DynOv_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))  # real time evolution to get direct S(t)

    #     loopend = timer()
    #     print('Index: {:d}, P: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(ind, P, aIBi, loopend - loopstart))

    # end = timer()
    # print('Total Time: {:.2f}'.format(end - runstart))

    # # ---- COMPUTE DATA ON CLUSTER ----

    # runstart = timer()

    # taskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    # taskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    # if(taskCount != len(cParams_List)):
    #     print('ERROR: TASK COUNT MISMATCH')
    #     P = float('nan')
    #     aIBi = float('nan')
    #     sys.exit()
    # else:
    #     cParams = cParams_List[taskID]
    #     [P, aIBi] = cParams

    # dynsph_ds = pf_dynamic_sph.quenchDynamics_DataGeneration(cParams, gParams, sParams, toggleDict)

    # if toggleDict['LastTimeStepOnly'] == 'yes':
    #     CSAmp_ds = dynsph_ds[['Real_CSAmp', 'Imag_CSAmp', 'Phase']].isel(t=-1); CSAmp_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))  # imag time evolution to get polaron state
    # else:
    #     # CSAmp_ds = dynsph_ds[['Real_CSAmp', 'Imag_CSAmp', 'Phase']]; CSAmp_ds.attrs = dynsph_ds.attrs; CSAmp_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))  # real time evolution to get direct S(t)
    #     DynOv_ds = pf_dynamic_sph.dirRF(dynsph_ds, kgrid, cParams, sParams); DynOv_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))  # real time evolution to get direct S(t)

    # end = timer()
    # print('Task ID: {:d}, P: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(taskID, P, aIBi, end - runstart))
