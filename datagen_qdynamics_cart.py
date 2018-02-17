import numpy as np
import pandas as pd
import xarray as xr
import Grid
import pf_dynamic_cart
import os
from timeit import default_timer as timer


if __name__ == "__main__":

    start = timer()

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (21, 21, 21)
    (dx, dy, dz) = (0.375, 0.375, 0.375)

    # (Lx, Ly, Lz) = (20, 20, 20)
    # (dx, dy, dz) = (0.5, 0.5, 0.5)

    xgrid = Grid.Grid('CARTESIAN_3D')
    xgrid.initArray('x', -Lx, Lx, dx); xgrid.initArray('y', -Ly, Ly, dy); xgrid.initArray('z', -Lz, Lz, dz)

    (Nx, Ny, Nz) = (len(xgrid.getArray('x')), len(xgrid.getArray('y')), len(xgrid.getArray('z')))

    kxfft = np.fft.fftfreq(Nx) * 2 * np.pi / dx; kyfft = np.fft.fftfreq(Nx) * 2 * np.pi / dy; kzfft = np.fft.fftfreq(Nx) * 2 * np.pi / dz

    kgrid = Grid.Grid('CARTESIAN_3D')
    kgrid.initArray_premade('kx', np.fft.fftshift(kxfft)); kgrid.initArray_premade('ky', np.fft.fftshift(kyfft)); kgrid.initArray_premade('kz', np.fft.fftshift(kzfft))

    kx = kgrid.getArray('kx')

    tMax = 99
    dt = 0.2
    tgrid = np.arange(0, tMax + dt, dt)

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
    datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints)
    # datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints)
    # datapath = '/n/regal/demler_lab/kis/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints)

    if os.path.isdir(datapath) is False:
        os.mkdir(datapath)

    # ---- SINGLE FUNCTION RUN ----

    runstart = timer()

    P = 1.8 * pf_dynamic_cart.nu(gBB)
    aIBi = -2
    cParams = [P, aIBi]

    dyncart_ds = pf_dynamic_cart.quenchDynamics_DataGeneration(cParams, gParams, sParams)
    dyncart_ds.to_netcdf(datapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))

    end = timer()
    print('Time: {:.2f}'.format(end - runstart))


# # ---- SET CPARAMS (RANGE OVER MULTIPLE aIBi, P VALUES) ----

    # cParams_List = []
    # aIBi_Vals = np.array([-5, -2, -0.1])
    # Pcrit_Vals = pf_static_cart.PCrit_grid(kxg, kyg, kzg, dVk, aIBi_Vals, mI, mB, n0, gBB)
    # Pcrit_max = np.max(Pcrit_Vals)
    # Pcrit_submax = np.max(Pcrit_Vals[Pcrit_Vals <= 10])
    # P_Vals_max = np.concatenate((np.linspace(0.01, Pcrit_submax, 50), np.linspace(Pcrit_submax, .95 * Pcrit_max, 10)))

    # for ind, aIBi in enumerate(aIBi_Vals):
    #     Pcrit = Pcrit_Vals[ind]
    #     P_Vals = P_Vals_max[P_Vals_max <= Pcrit]
    #     for P in P_Vals:
    #         cParams_List.append([P, aIBi])

    cParams_List = []
    aIBi_Vals = np.array([-5.0, -2.0])
    P_Vals = np.array([0.1, 1.0])
    for ind, aIBi in enumerate(aIBi_Vals):
        for P in P_Vals:
            cParams_List.append([P, aIBi])

#     # # ---- COMPUTE DATA ON CLUSTER ----

#     runstart = timer()

#     datapath = '/n/regal/demler_lab/kis/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints)
#     # if os.path.isdir(datapath) is False:
#     #     os.mkdir(datapath)

#     taskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
#     taskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

#     if(taskCount != len(cParams_List)):
#         print('ERROR: TASK COUNT MISMATCH')
#         P = float('nan')
#         aIBi = float('nan')
#     else:
#         cParams = cParams_List[taskID]
#         [P, aIBi] = cParams
#         innerdatapath = datapath + '/P_{:.3f}_aIBi_{:.2f}'.format(P, aIBi)
#         if os.path.isdir(innerdatapath) is False:
#             os.mkdir(innerdatapath)

#         time_grids, metrics_data, pos_xyz_data, mom_xyz_data, cont_xyz_data, mom_mag_data = pf_dynamic_cart.quenchDynamics_DataGeneration(cParams, gParams, sParams)

#         with open(innerdatapath + '/time_grids.pickle', 'wb') as f:
#             pickle.dump(time_grids, f)
#         with open(innerdatapath + '/metrics_data.pickle', 'wb') as f:
#             pickle.dump(metrics_data, f)
#         with open(innerdatapath + '/pos_xyz_data.pickle', 'wb') as f:
#             pickle.dump(pos_xyz_data, f)
#         with open(innerdatapath + '/mom_xyz_data.pickle', 'wb') as f:
#             pickle.dump(mom_xyz_data, f)
#         with open(innerdatapath + '/cont_xyz_data.pickle', 'wb') as f:
#             pickle.dump(cont_xyz_data, f)
#         with open(innerdatapath + '/mom_mag_data.pickle', 'wb') as f:
#             pickle.dump(mom_mag_data, f)

#     end = timer()
#     print('Task ID: {:d}, P: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(taskID, P, aIBi, end - runstart))
