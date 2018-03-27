import numpy as np
import Grid
import pf_static_sph
import os
from timeit import default_timer as timer


if __name__ == "__main__":

    start = timer()

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (21, 21, 21)
    (dx, dy, dz) = (0.375, 0.375, 0.375)

    # (Lx, Ly, Lz) = (21, 21, 21)
    # (dx, dy, dz) = (0.25, 0.25, 0.25)

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

    gParams = [kgrid]

    NGridPoints = kgrid.size()

    print('dk: {0}'.format(dk))
    print('UV cutoff: {0}'.format(k_max))
    print('NGridPoints: {0}'.format(NGridPoints))

    # print('Perfect \int ep: {0}'.format(k_max**5 / (5 * (2 * np.pi)**2)))

    # Basic parameters

    mI = 1
    mB = 1
    n0 = 1
    gBB = (4 * np.pi / mB) * 0.05

    # Interpolation

    Nsteps = 1e2
    pf_static_sph.createSpline_grid(Nsteps, kgrid, mI, mB, n0, gBB)

    aSi_tck = np.load('aSi_spline_sph.npy')
    PBint_tck = np.load('PBint_spline_sph.npy')

    sParams = [mI, mB, n0, gBB, aSi_tck, PBint_tck]

    # ---- SET OUTPUT DATA FOLDER ----

    datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    # datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    # datapath = '/n/regal/demler_lab/kis/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)

    innerdatapath = datapath + '/steadystate_spherical'

    if os.path.isdir(datapath) is False:
        os.mkdir(datapath)

    if os.path.isdir(innerdatapath) is False:
        os.mkdir(innerdatapath)

    # # # ---- SINGLE FUNCTION RUN ----

    # runstart = timer()

    # P = 0.1
    # aIBi = -5
    # cParams = [P, aIBi]

    # stsph_ds = pf_static_sph.static_DataGeneration(cParams, gParams, sParams)
    # stsph_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))

    # end = timer()
    # print('Time: {:.2f}'.format(end - runstart))

    # ---- SET CPARAMS (RANGE OVER MULTIPLE aIBi, P VALUES) ----

    cParams_List = []
    # aIBi_Vals = np.array([-1, -0.8])
    # aIBi_Vals = np.array([-10.0, -5.0, -2.0, -1.0])
    aIBi_Vals = np.arange(-10, 10.25, 0.25)
    # P_Vals = np.array([0.1, 1.0, 2.0, 3.0])
    P_Vals = np.linspace(0.1, 5.0, 100)

    for ind, aIBi in enumerate(aIBi_Vals):
        for P in P_Vals:
            cParams_List.append([P, aIBi])

    # ---- COMPUTE DATA ON COMPUTER ----

    runstart = timer()

    for ind, cParams in enumerate(cParams_List):
        loopstart = timer()
        [P, aIBi] = cParams
        stsph_ds = pf_static_sph.static_DataGeneration(cParams, gParams, sParams)
        stsph_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))
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

    # stsph_ds = pf_static_sph.static_DataGeneration(cParams, gParams, sParams)
    # stsph_ds.to_netcdf(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))

    # end = timer()
    # print('Task ID: {:d}, P: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(taskID, P, aIBi, end - runstart))
