import numpy as np
import pandas as pd
import xarray as xr
import Grid
import pf_dynamic_cart
import os
import sys
from timeit import default_timer as timer
# import pf_static_cart


if __name__ == "__main__":

    start = timer()

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (21, 21, 21)
    (dx, dy, dz) = (0.375, 0.375, 0.375)

    xgrid = Grid.Grid('CARTESIAN_3D')
    xgrid.initArray('x', -Lx, Lx, dx); xgrid.initArray('y', -Ly, Ly, dy); xgrid.initArray('z', -Lz, Lz, dz)

    (Nx, Ny, Nz) = (len(xgrid.getArray('x')), len(xgrid.getArray('y')), len(xgrid.getArray('z')))

    kxfft = np.fft.fftfreq(Nx) * 2 * np.pi / dx; kyfft = np.fft.fftfreq(Nx) * 2 * np.pi / dy; kzfft = np.fft.fftfreq(Nx) * 2 * np.pi / dz

    kgrid = Grid.Grid('CARTESIAN_3D')
    kgrid.initArray_premade('kx', np.fft.fftshift(kxfft)); kgrid.initArray_premade('ky', np.fft.fftshift(kyfft)); kgrid.initArray_premade('kz', np.fft.fftshift(kzfft))

    kx = kgrid.getArray('kx')

    tMax = 1000
    dt = 10
    tgrid = np.arange(0, tMax + dt, dt)
    print(tgrid)

    gParams = [xgrid, kgrid, tgrid]

    # NGridPoints = (2 * Lx / dx) * (2 * Ly / dy) * (2 * Lz / dz)
    NGridPoints = xgrid.size()

    kx = kgrid.getArray('kx'); ky = kgrid.getArray('ky'); kz = kgrid.getArray('kz')
    k_max = np.sqrt(np.max(kx)**2 + np.max(ky)**2 + np.max(kz)**2)

    print('Total time steps: {0}'.format(tgrid.size))
    print('UV cutoff: {0}'.format(k_max))
    print('NGridPoints: {0}'.format(NGridPoints))

    # Basic parameters

    mI = 1
    mB = 1
    n0 = 1
    gBB = (4 * np.pi / mB) * 0.05

    sParams = [mI, mB, n0, gBB]

    # ---- SET OUTPUT DATA FOLDER ----
    # datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints)
    # datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints)
    datapath = '/n/regal/demler_lab/kis/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints)

    # innerdatapath = datapath + '/redyn_cart'
    # innerdatapath = datapath + '/imdyn_cart'
    innerdatapath = datapath + '/imdyn_cart_long'

    # if os.path.isdir(datapath) is False:
    #     os.mkdir(datapath)

    # if os.path.isdir(innerdatapath) is False:
    #     os.mkdir(innerdatapath)

    # # ---- SINGLE FUNCTION RUN ----

    # runstart = timer()

    # P = 2.4
    # aIBi = -5
    # cParams = [P, aIBi]

    # dyncart_ds = pf_dynamic_cart.quenchDynamics_DataGeneration(cParams, gParams, sParams)
    # dyncart_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))

    # end = timer()
    # print('Time: {:.2f}'.format(end - runstart))

    # ---- SET CPARAMS (RANGE OVER MULTIPLE aIBi, P VALUES) ----

    cParams_List = []
    # aIBi_Vals = np.array([-5.0, -2.0, -0.1])
    # P_Vals = np.array([5.0, 10.0])
    aIBi_Vals = np.array([-10.0, -5.0, -2.0])
    P_Vals = np.array([0.1, 0.4, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0, 2.4, 2.7, 3.0, 4.0, 5.0])

    for ind, aIBi in enumerate(aIBi_Vals):
        for P in P_Vals:
            cParams_List.append([P, aIBi])

    # ---- COMPUTE DATA ON CLUSTER ----

    runstart = timer()

    taskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    taskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    if(taskCount != len(cParams_List)):
        print('ERROR: TASK COUNT MISMATCH')
        P = float('nan')
        aIBi = float('nan')
        sys.exit()
    else:
        cParams = cParams_List[taskID]
        [P, aIBi] = cParams

    dyncart_ds = pf_dynamic_cart.quenchDynamics_DataGeneration(cParams, gParams, sParams)
    dyncart_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))

    end = timer()
    print('Task ID: {:d}, P: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(taskID, P, aIBi, end - runstart))
