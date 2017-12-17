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

    # Nsteps = 1e2
    # createSpline_grid(Nsteps, kxFg, kyFg, kzFg, dVk, mI, mB, n0, gBB)
    # kxFg, kyFg, kzFg = np.meshgrid(kFgrid.getArray('kx'), kFgrid.getArray('ky'), kFgrid.getArray('kz'), indexing='ij', sparse=True)
    # dVk = kgrid.arrays_diff['kx'] * kgrid.arrays_diff['ky'] * kgrid.arrays_diff['kz']

    aSi_tck = np.load('aSi_spline.npy')
    PBint_tck = np.load('PBint_spline.npy')

    sParams = [mI, mB, n0, gBB, aSi_tck, PBint_tck]

    # ---- SINGLE FUNCTION RUN ----

    P = 1.4 * pf_static_cart.nu(gBB)
    aIBi = -2
    cParams = [P, aIBi]

    datapath = os.path.dirname(os.path.realpath(__file__)) + '/data_static' + '/NGridPoints_{:.2E}'.format(NGridPoints)
    if os.path.isdir(datapath) is False:
        os.mkdir(datapath)
    innerdatapath = datapath + '/P_{:.3f}_aIBi_{:.2f}'.format(P, aIBi)
    if os.path.isdir(innerdatapath) is False:
        os.mkdir(innerdatapath)

    metrics_string, metrics_data, xyz_data, mag_data = pf_static_cart.staticDataGeneration(cParams, gParams, sParams)
    with open(innerdatapath + '/metrics_string.txt', 'w') as f:
        f.write(metrics_string)
    np.savetxt(innerdatapath + '/metrics.dat', metrics_data)
    np.savetxt(innerdatapath + '/xyz.dat', xyz_data)
    np.savetxt(innerdatapath + '/mag.dat', mag_data)

    # # ---- SET CPARAMS (RANGE OVER MULTIPLE P VALUES) ----

    # aIBi = -2
    # g = pf.g(kgrid, 0, aIBi, mI, mB, n0, gBB)
    # Pg = pf.PCrit_grid(kgrid, 0, aIBi, mI, mB, n0, gBB)
    # Pc = pf.PCrit_inf(kcutoff, aIBi, mI, mB, n0, gBB)

    # # NPVals = 40
    # # PVals = np.linspace(0, 3 * Pc, NPVals)
    # NPVals = 4
    # PVals = np.linspace(0.1, .95 * Pg, NPVals)
    # cParams_List = [[P, aIBi] for P in PVals]
    # print('Pc_inf: %.3f, Pc_g: %.3f' % (Pc, Pg))

    # # ---- SET OUTPUT DATA FOLDER, CREATE PARAMETER SET, AND SAVE PARAMETER INFO FILE ----

    # dirpath = os.path.dirname(os.path.realpath(__file__))
    # outer_datapath = dirpath + '/clusterdata/aIBi_%.2f' % aIBi
    # datapath = outer_datapath + '/NGridPoints_%.2E' % NGridPoints

    # # if os.path.isdir(outer_datapath) is False:
    # #     os.mkdir(outer_datapath)
    # # if os.path.isdir(datapath) is False:
    # #     os.mkdir(datapath)
    # #     os.mkdir(datapath + '/Dist')

    # paramInfo = 'kcutoff - {:.2f}, dk - {:.3f}, Ntheta - {:d}, NGridPoints - {:.2E}, tMax - {:.1f}, dt1 - {:.3f}, dt2 - {:.3f} NtPoints - {:d}\nmI - {:.1f}, mB - {:.1f}, n0 - {:0.1f}, gBB - {:0.3f}\naIBi - {:.2f}, gIB - {:0.3f}, PCrit_grid - {:.5f}, PCrit_true - {:0.5f}, NPVals - {:d}'.format(kcutoff, dk, Ntheta, NGridPoints, tMax, dt1, dt2, tGrid.size, mI, mB, n0, gBB, aIBi, g, Pg, Pc, NPVals)
    # # with open(datapath + '/paramInfo.txt', 'w') as f:
    # #     f.write(paramInfo)

    # # # ---- COMPUTE DATA ----

    # # taskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    # # taskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    # # if(taskCount != NPVals):
    # #     print('ERROR: TASK COUNT MISMATCH')
    # # else:
    # #     os.mkdir(datapath + '/Dist/P_%.2f' % PVals[taskID])
    # #     pf.quenchDynamics(cParams_List[taskID], gParams, sParams, datapath)

    # # end = timer()
    # # print('Task ID: {:d}, P: {:.2f}, Time: {:.2f}'.format(taskID, PVals[taskID], end - start))
