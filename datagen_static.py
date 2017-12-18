import numpy as np
import Grid
import pf_static_cart
import os
from timeit import default_timer as timer


if __name__ == "__main__":

    start = timer()

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (25, 25, 25)
    (dx, dy, dz) = (2.5e-01, 2.5e-01, 2.5e-01)

    xgrid = Grid.Grid('CARTESIAN_3D')
    xgrid.initArray('x', -Lx, Lx, dx); xgrid.initArray('y', -Ly, Ly, dy); xgrid.initArray('z', -Lz, Lz, dz)

    (Nx, Ny, Nz) = (len(xgrid.getArray('x')), len(xgrid.getArray('y')), len(xgrid.getArray('z')))

    kxfft = np.fft.fftfreq(Nx) * 2 * np.pi / dx; kyfft = np.fft.fftfreq(Nx) * 2 * np.pi / dy; kzfft = np.fft.fftfreq(Nx) * 2 * np.pi / dz
    kFgrid = Grid.Grid('CARTESIAN_3D')
    kFgrid.initArray_premade('kx', kxfft); kFgrid.initArray_premade('ky', kyfft); kFgrid.initArray_premade('kz', kzfft)

    kgrid = Grid.Grid('CARTESIAN_3D')
    kgrid.initArray_premade('kx', np.fft.fftshift(kxfft)); kgrid.initArray_premade('ky', np.fft.fftshift(kyfft)); kgrid.initArray_premade('kz', np.fft.fftshift(kzfft))

    gParams = [xgrid, kgrid, kFgrid]

    NGridPoints = (Lx / dx) * (Ly / dy) * (Lz / dz)

    # Basic parameters

    mI = 1
    mB = 1
    n0 = 1
    gBB = (4 * np.pi / mB) * 0.05

    # Interpolation

    kxFg, kyFg, kzFg = np.meshgrid(kFgrid.getArray('kx'), kFgrid.getArray('ky'), kFgrid.getArray('kz'), indexing='ij', sparse=True)
    dVk = kgrid.arrays_diff['kx'] * kgrid.arrays_diff['ky'] * kgrid.arrays_diff['kz']

    # Nsteps = 1e2
    # pf_static_cart.createSpline_grid(Nsteps, kxFg, kyFg, kzFg, dVk, mI, mB, n0, gBB)

    aSi_tck = np.load('aSi_spline.npy')
    PBint_tck = np.load('PBint_spline.npy')

    sParams = [mI, mB, n0, gBB, aSi_tck, PBint_tck]

    # ---- SET OUTPUT DATA FOLDER ----

    datapath = os.path.dirname(os.path.realpath(__file__)) + '/data_static' + '/NGridPoints_{:.2E}'.format(NGridPoints)
    if os.path.isdir(datapath) is False:
        os.mkdir(datapath)

    # # ---- SINGLE FUNCTION RUN ----

    # runstart = timer()

    # P = 1.4 * pf_static_cart.nu(gBB)
    # aIBi = -2
    # cParams = [P, aIBi]

    # innerdatapath = datapath + '/P_{:.3f}_aIBi_{:.2f}'.format(P, aIBi)
    # if os.path.isdir(innerdatapath) is False:
    #     os.mkdir(innerdatapath)

    # metrics_string, metrics_data, xyz_data, mag_data = pf_static_cart.staticDataGeneration(cParams, gParams, sParams)
    # with open(innerdatapath + '/metrics_string.txt', 'w') as f:
    #     f.write(metrics_string)
    # np.savetxt(innerdatapath + '/metrics.dat', metrics_data)
    # np.savetxt(innerdatapath + '/xyz.dat', xyz_data)
    # np.savetxt(innerdatapath + '/mag.dat', mag_data)

    # end = timer()
    # print('Time: {:.2f}'.format(end - runstart))

    # ---- SET CPARAMS (RANGE OVER MULTIPLE aIBi, P VALUES) ----

    cParams_List = []
    aIBi_Vals = [-5, -2, -1, 1, 2, 5]
    NPVals = 50

    for aIBi in aIBi_Vals:
        Pcrit = pf_static_cart.PCrit_grid(kxFg, kyFg, kzFg, dVk, aIBi, mI, mB, n0, gBB)
        PVals = np.linspace(0.1, .95 * Pcrit, NPVals)
        for P in PVals:
            cParams_List.append([P, aIBi])

    # ---- COMPUTE DATA ON COMPUTER ----

    runstart = timer()

    for ind, cParams in enumerate(cParams_List):
        [P, aIBi] = cParams
        innerdatapath = datapath + '/P_{:.3f}_aIBi_{:.2f}'.format(P, aIBi)
        if os.path.isdir(innerdatapath) is False:
            os.mkdir(innerdatapath)
        metrics_string, metrics_data, xyz_data, mag_data = pf_static_cart.staticDataGeneration(cParams, gParams, sParams)
        with open(innerdatapath + '/metrics_string.txt', 'w') as f:
            f.write(metrics_string)
        np.savetxt(innerdatapath + '/metrics.dat', metrics_data)
        np.savetxt(innerdatapath + '/xyz.dat', xyz_data)
        np.savetxt(innerdatapath + '/mag.dat', mag_data)

        runend = timer()
        print('Index: {:d}, P: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(ind, P, aIBi, runend - runstart))

    end = timer()
    print('Total Time: {.2f}'.format(end - runstart))

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
    #     metrics_string, metrics_data, xyz_data, mag_data = pf_static_cart.staticDataGeneration(cParams, gParams, sParams)
    #     with open(innerdatapath + '/metrics_string.txt', 'w') as f:
    #         f.write(metrics_string)
    #     np.savetxt(innerdatapath + '/metrics.dat', metrics_data)
    #     np.savetxt(innerdatapath + '/xyz.dat', xyz_data)
    #     np.savetxt(innerdatapath + '/mag.dat', mag_data)

    # end = timer()
    # print('Task ID: {:d}, P: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(taskID, P, aIBi, end - runstart))
