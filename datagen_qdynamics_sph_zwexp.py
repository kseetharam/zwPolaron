import numpy as np
import pandas as pd
import xarray as xr
import Grid
import pf_dynamic_sph
import os
from timeit import default_timer as timer
import sys


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

    # tMax = 1000; dt = 500
    tMax = 50; dt = 0.2
    tgrid = np.arange(0, tMax + dt, dt)

    gParams = [xgrid, kgrid, tgrid]
    NGridPoints = kgrid.size()

    print('Total time steps: {0}'.format(tgrid.size))
    print('UV cutoff: {0}'.format(k_max))
    print('NGridPoints: {0}'.format(NGridPoints))

    # Basic parameters

    mI = 1.7
    mB = 1
    n0 = 1
    aBB = 0.062
    gBB = (4 * np.pi / mB) * aBB
    nu = pf_dynamic_sph.nu(gBB)

    sParams = [mI, mB, n0, gBB]

    def dirRF(dataset, kgrid):
        CSAmp = dynsph_ds['Real_CSAmp'] + 1j * dynsph_ds['Imag_CSAmp']
        dVk = kgrid.dV()
        tgrid = CSAmp.coords['t'].values
        CSA0 = CSAmp.isel(t=0).values; CSA0 = CSA0.reshape(CSA0.size)
        DynOv_Vec = np.zeros(tgrid.size, dtype=complex)
        for tind, t in enumerate(tgrid):
            CSAt = CSAmp.sel(t=t).values; CSAt = CSAt.reshape(CSAt.size)
            exparg = np.dot(np.abs(CSAt)**2 + np.abs(CSA0)**2 + 2 * CSA0.conjugate() * CSAt, dVk)
            DynOv_Vec[tind] = np.exp((-1 / 2) * exparg)

        ReDynOv_da = xr.DataArray(np.real(DynOv_Vec), coords=[tgrid], dims=['t'])
        ImDynOv_da = xr.DataArray(np.imag(DynOv_Vec), coords=[tgrid], dims=['t'])
        dirRF_ds = xr.Dataset({'Real_DynOv': ReDynOv_da, 'Imag_DynOv': ImDynOv_da}, coords={'t': tgrid}, attrs=dataset.attrs)
        return dirRF_ds

    # ---- SET OUTPUT DATA FOLDER ----

    datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    # datapath = '/n/regal/demler_lab/kis/ZwierleinExp_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)

    innerdatapath = datapath + '/imdyn_spherical'
    nonintdatapath = datapath + '/redyn_nonint'

    # if os.path.isdir(datapath) is False:
    #     os.mkdir(datapath)

    # if os.path.isdir(innerdatapath) is False:
    #     os.mkdir(innerdatapath)

    # # # ---- SINGLE FUNCTION RUN ----

    # runstart = timer()

    # P = 0.1
    # aIBi = -0.32

    # cParams = [P, aIBi]

    # dynsph_ds = pf_dynamic_sph.quenchDynamics_DataGeneration(cParams, gParams, sParams)

    # # CSAmp_ds = dynsph_ds[['Real_CSAmp', 'Imag_CSAmp']].isel(t=-1); CSAmp_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi)) # imag time evolution to get polaron state
    # dirRF_ds = dirRF(dynsph_ds, kgrid); dirRF_ds.to_netcdf(nonintdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))  # real time evolution to get direct S(t)

    # end = timer()
    # print('Time: {:.2f}'.format(end - runstart))

    # ---- SET CPARAMS (RANGE OVER MULTIPLE aIBi, P VALUES) ----

    cParams_List = []

    # aIBi_Vals = (6 * np.pi**2)**(1 / 3) * np.linspace(-4, 4, 5)
    aIBi_Vals = np.array([-0.32])

    P_Vals = np.linspace(0.1, mI * nu, 30)

    for ind, aIBi in enumerate(aIBi_Vals):
        for P in P_Vals:
            cParams_List.append([P, aIBi])

    # ---- COMPUTE DATA ON COMPUTER ----

    runstart = timer()

    for ind, cParams in enumerate(cParams_List):
        loopstart = timer()
        [P, aIBi] = cParams
        dynsph_ds = pf_dynamic_sph.quenchDynamics_DataGeneration(cParams, gParams, sParams)

        # CSAmp_ds = dynsph_ds[['Real_CSAmp', 'Imag_CSAmp']].isel(t=-1); CSAmp_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi)) # imag time evolution to get polaron state
        dirRF_ds = dirRF(dynsph_ds, kgrid); dirRF_ds.to_netcdf(nonintdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))  # real time evolution to get direct S(t)

        loopend = timer()
        print('Index: {:d}, P: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(ind, P, aIBi, loopend - loopstart))

    end = timer()
    print('Total Time: {:.2f}'.format(end - runstart))

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

    # dynsph_ds = pf_dynamic_sph.quenchDynamics_DataGeneration(cParams, gParams, sParams)

    # # CSAmp_ds = dynsph_ds[['Real_CSAmp', 'Imag_CSAmp']].isel(t=-1); CSAmp_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi)) # imag time evolution to get polaron state
    # dirRF_ds = dirRF(dynsph_ds, kgrid); dirRF_ds.to_netcdf(nonintdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))  # real time evolution to get direct S(t)

    # end = timer()
    # print('Task ID: {:d}, P: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(taskID, P, aIBi, end - runstart))
