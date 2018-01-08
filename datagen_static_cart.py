import numpy as np
import Grid
import pf_static_cart
import os
from timeit import default_timer as timer


if __name__ == "__main__":

    start = timer()

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (20, 20, 20)
    (dx, dy, dz) = (5e-01, 5e-01, 5e-01)

    xgrid = Grid.Grid('CARTESIAN_3D')
    xgrid.initArray('x', -Lx, Lx, dx); xgrid.initArray('y', -Ly, Ly, dy); xgrid.initArray('z', -Lz, Lz, dz)

    (Nx, Ny, Nz) = (len(xgrid.getArray('x')), len(xgrid.getArray('y')), len(xgrid.getArray('z')))
    kxfft = np.fft.fftfreq(Nx) * 2 * np.pi / dx; kyfft = np.fft.fftfreq(Nx) * 2 * np.pi / dy; kzfft = np.fft.fftfreq(Nx) * 2 * np.pi / dz

    kgrid = Grid.Grid('CARTESIAN_3D')
    kgrid.initArray_premade('kx', np.fft.fftshift(kxfft)); kgrid.initArray_premade('ky', np.fft.fftshift(kyfft)); kgrid.initArray_premade('kz', np.fft.fftshift(kzfft))

    # gParams = [xgrid, kgrid, kFgrid]
    gParams = [xgrid, kgrid]

    # NGridPoints = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    NGridPoints = xgrid.size()

    # Basic parameters

    mI = 1
    mB = 1
    n0 = 1
    gBB = (4 * np.pi / mB) * 0.05

    # Interpolation

    kxg, kyg, kzg = np.meshgrid(kgrid.getArray('kx'), kgrid.getArray('ky'), kgrid.getArray('kz'), indexing='ij', sparse=True)
    dVk = kgrid.arrays_diff['kx'] * kgrid.arrays_diff['ky'] * kgrid.arrays_diff['kz'] / ((2 * np.pi)**3)

    # Nsteps = 1e2
    # pf_static_cart.createSpline_grid(Nsteps, kxg, kyg, kzg, dVk, mI, mB, n0, gBB)

    aSi_tck = np.load('aSi_spline_cart.npy')
    PBint_tck = np.load('PBint_spline_cart.npy')

    sParams = [mI, mB, n0, gBB, aSi_tck, PBint_tck]

    # ---- SET OUTPUT DATA FOLDER ----

    datapath = os.path.dirname(os.path.realpath(__file__)) + '/data_static' + '/cart' + '/NGridPoints_{:.2E}'.format(NGridPoints)
    if os.path.isdir(datapath) is False:
        os.mkdir(datapath)

    # ---- SINGLE FUNCTION RUN ----

    runstart = timer()

    P = 1.4 * pf_static_cart.nu(gBB)
    aIBi = -2
    cParams = [P, aIBi]

    innerdatapath = datapath + '/P_{:.3f}_aIBi_{:.2f}'.format(P, aIBi)
    if os.path.isdir(innerdatapath) is False:
        os.mkdir(innerdatapath)

    metrics_string, metrics_data, pos_xyz_string, pos_xyz_data, mom_xyz_string, mom_xyz_data, cont_xyz_string, cont_xyz_data, mom_mag_string, mom_mag_data = pf_static_cart.static_DataGeneration(cParams, gParams, sParams)
    with open(innerdatapath + '/metrics_string.txt', 'w') as f:
        f.write(metrics_string)
    with open(innerdatapath + '/pos_xyz_string.txt', 'w') as f:
        f.write(pos_xyz_string)
    with open(innerdatapath + '/mom_xyz_string.txt', 'w') as f:
        f.write(mom_xyz_string)
    with open(innerdatapath + '/mom_mag_string.txt', 'w') as f:
        f.write(mom_mag_string)
    np.savetxt(innerdatapath + '/metrics.dat', metrics_data)
    np.savetxt(innerdatapath + '/pos_xyz.dat', pos_xyz_data)
    np.savetxt(innerdatapath + '/mom_xyz.dat', mom_xyz_data)
    np.savetxt(innerdatapath + '/mag.dat', mom_mag_data)

    end = timer()
    print('Time: {:.2f}'.format(end - runstart))

    # # ---- SET CPARAMS (RANGE OVER MULTIPLE aIBi, P VALUES) ----

    # cParams_List = []
    # aIBi_Vals = np.array([-5, -3, -1, 1, 3, 5, 7])
    # # aIBi_Vals = np.linspace(-5, 7, 10)
    # Pcrit_Vals = pf_static_cart.PCrit_grid(kxg, kyg, kzg, dVk, aIBi_Vals, mI, mB, n0, gBB)
    # Pcrit_max = np.max(Pcrit_Vals)
    # Pcrit_submax = np.max(Pcrit_Vals[Pcrit_Vals <= 10])
    # P_Vals_max = np.concatenate((np.linspace(0.01, Pcrit_submax, 50), np.linspace(Pcrit_submax, .95 * Pcrit_max, 10)))

    # for ind, aIBi in enumerate(aIBi_Vals):
    #     Pcrit = Pcrit_Vals[ind]
    #     P_Vals = P_Vals_max[P_Vals_max <= Pcrit]
    #     for P in P_Vals:
    #         cParams_List.append([P, aIBi])

    # # ---- COMPUTE DATA ON COMPUTER ----

    # runstart = timer()

    # for ind, cParams in enumerate(cParams_List):
    #     loopstart = timer()
    #     [P, aIBi] = cParams
    #     innerdatapath = datapath + '/P_{:.3f}_aIBi_{:.2f}'.format(P, aIBi)
    #     if os.path.isdir(innerdatapath) is False:
    #         os.mkdir(innerdatapath)
    #     metrics_string, metrics_data, pos_xyz_string, pos_xyz_data, mom_xyz_string, mom_xyz_data, cont_xyz_string, cont_xyz_data, mom_mag_string, mom_mag_data = pf_static_cart.static_DataGeneration(cParams, gParams, sParams)
    #     with open(innerdatapath + '/metrics_string.txt', 'w') as f:
    #         f.write(metrics_string)
    #     with open(innerdatapath + '/pos_xyz_string.txt', 'w') as f:
    #         f.write(pos_xyz_string)
    #     with open(innerdatapath + '/mom_xyz_string.txt', 'w') as f:
    #         f.write(mom_xyz_string)
    #     with open(innerdatapath + '/mom_mag_string.txt', 'w') as f:
    #         f.write(mom_mag_string)
    #     np.savetxt(innerdatapath + '/metrics.dat', metrics_data)
    #     np.savetxt(innerdatapath + '/pos_xyz.dat', pos_xyz_data)
    #     np.savetxt(innerdatapath + '/mom_xyz.dat', mom_xyz_data)
    #     np.savetxt(innerdatapath + '/mag.dat', mom_mag_data)

    #     loopend = timer()
    #     print('Index: {:d}, P: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(ind, P, aIBi, loopend - loopstart))

    # end = timer()
    # print('Total Time: {.2f}'.format(end - runstart))

    # # ---- COMPUTE DATA ON CLUSTER ----

    # runstart = timer()

    # taskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    # taskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    # if(taskCount != len(cParams_List)):
    #     print('ERROR: TASK COUNT MISMATCH')
    #     P = float('nan')
    #     aIBi = float('nan')
    # else:
    #     cParams = cParams_List[taskID]
    #     [P, aIBi] = cParams
    #     innerdatapath = datapath + '/P_{:.3f}_aIBi_{:.2f}'.format(P, aIBi)
    #     if os.path.isdir(innerdatapath) is False:
    #         os.mkdir(innerdatapath)
    #     metrics_string, metrics_data, pos_xyz_string, pos_xyz_data, mom_xyz_string, mom_xyz_data, cont_xyz_string, cont_xyz_data, mom_mag_string, mom_mag_data = pf_static_cart.static_DataGeneration(cParams, gParams, sParams)
    #     with open(innerdatapath + '/metrics_string.txt', 'w') as f:
    #         f.write(metrics_string)
    #     with open(innerdatapath + '/pos_xyz_string.txt', 'w') as f:
    #         f.write(pos_xyz_string)
    #     with open(innerdatapath + '/mom_xyz_string.txt', 'w') as f:
    #         f.write(mom_xyz_string)
    #     with open(innerdatapath + '/mom_mag_string.txt', 'w') as f:
    #         f.write(mom_mag_string)
    #     np.savetxt(innerdatapath + '/metrics.dat', metrics_data)
    #     np.savetxt(innerdatapath + '/pos_xyz.dat', pos_xyz_data)
    #     np.savetxt(innerdatapath + '/mom_xyz.dat', mom_xyz_data)
    #     np.savetxt(innerdatapath + '/mag.dat', mom_mag_data)

    # end = timer()
    # print('Task ID: {:d}, P: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(taskID, P, aIBi, end - runstart))
