import numpy as np
import Grid
import pf_dynamic_sph
import os
from timeit import default_timer as timer

import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":

    start = timer()

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (300, 300, 300)
    (dx, dy, dz) = (5, 5, 5)

    xgrid = Grid.Grid('CARTESIAN_3D')
    xgrid.initArray('x', -Lx, Lx, dx); xgrid.initArray('y', -Ly, Ly, dy); xgrid.initArray('z', -Lz, Lz, dz)

    # NGridPoints_desired = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    NGridPoints_desired = (1 + 2 * Lx / dx) * (1 + 2 * Lz / dz)
    Ntheta = 50
    Nk = np.ceil(NGridPoints_desired / Ntheta)

    theta_max = np.pi
    thetaArray, dtheta = np.linspace(0, theta_max, Ntheta, retstep=True)

    k_max = np.sqrt((np.pi / dx)**2 + (np.pi / dy)**2 + (np.pi / dz)**2)
    k_min = 1e-5
    kArray, dk = np.linspace(k_min, k_max, Nk, retstep=True)
    if dk < k_min:
        print('k ARRAY GENERATION ERROR')

    kgrid = Grid.Grid("SPHERICAL_2D")
    kgrid.initArray_premade('k', kArray)
    kgrid.initArray_premade('th', thetaArray)

    tMax = 99
    dt = 1
    tgrid = np.arange(0, tMax + dt, dt)

    gParams = [xgrid, kgrid, tgrid]
    NGridPoints = kgrid.size()

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

    datapath = os.path.dirname(os.path.realpath(__file__)) + '/data_qdynamics' + '/sph' + '/NGridPoints_{:.2E}'.format(NGridPoints)
    if os.path.isdir(datapath) is False:
        os.mkdir(datapath)

    # ---- SINGLE FUNCTION RUN ----

    runstart = timer()

    P = 0.1 * pf_dynamic_sph.nu(gBB)
    aIBi = -2
    cParams = [P, aIBi]

    innerdatapath = datapath + '/P_{:.3f}_aIBi_{:.2f}'.format(P, aIBi)
    if os.path.isdir(innerdatapath) is False:
        os.mkdir(innerdatapath)

    time_grid, metrics_data = pf_dynamic_sph.quenchDynamics_DataGeneration(cParams, gParams, sParams)

    end = timer()
    print('Time: {:.2f}'.format(end - runstart))

    # TEMP DATA CHECK

    [NGridPoints, k_max, P, aIBi, mI, mB, n0, gBB, nu_const, gIB, PB_tVec, NB_tVec, DynOv_tVec, Phase_tVec] = metrics_data
    print(k_max, P, aIBi, mI, mB, n0, gBB, nu_const, gIB)
    # print(np.abs(DynOv_tVec))
    # print(PB_tVec)
    # print(np.abs(DynOv_tVec)[-1])
    # print(NB_tVec[-1])
    # print(Phase_tVec)

    ob_data = np.concatenate((tgrid[:, np.newaxis], np.abs(DynOv_tVec)[:, np.newaxis], NB_tVec[:, np.newaxis], PB_tVec[:, np.newaxis], Phase_tVec[:, np.newaxis]), axis=1)
    np.savetxt(innerdatapath + '/ob.dat', ob_data)

    staticdatapath = os.path.dirname(os.path.realpath(__file__)) + '/data_static/sph/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}/metrics.dat'.format(NGridPoints, P, aIBi)
    NGridPoints_s, k_max_s, P_s, aIBi_s, mI_s, mB_s, n0_s, gBB_s, nu_const_s, gIB_s, Pcrit_s, aSi_s, DP_s, PB_Val_s, En_s, eMass_s, Nph_s, Z_factor_s = np.loadtxt(staticdatapath, unpack=True)

    print('|S(t) - Z|: {0}'.format(np.abs(np.abs(DynOv_tVec[-1]) - Z_factor_s)))
    print('|N(t)-2*Npol|: {0}'.format(np.abs(NB_tVec[-1] - 2 * Nph_s)))

    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].plot(time_grid, np.abs(np.abs(DynOv_tVec) - Z_factor_s))
    # ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    ax[1].plot(time_grid, np.abs(NB_tVec - 2 * Nph_s))
    # ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    # ax[1].plot(time_grid, np.abs(DynOv_tVec * np.exp(-4.881080635697411019 * time_grid * 1j)))
    plt.show()

    # !!!! HAVE TO EDIT THE MULTIPLE FUNCTION RUN SCRIPTS BELOW ONCE SINGLE FUNCTION RUN IS FINALIZED

    # # ---- SET CPARAMS (RANGE OVER MULTIPLE aIBi, P VALUES) ----

    # cParams_List = []
    # aIBi_Vals = np.array([-5, -3, -1, 1, 3, 5, 7])
    # # aIBi_Vals = np.linspace(-5, 7, 10)
    # Pcrit_Vals = pf_static_cart.PCrit_grid(kxFg, kyFg, kzFg, dVk, aIBi_Vals, mI, mB, n0, gBB)
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
    #     metrics_string, metrics_data, xyz_string, xyz_data, mag_string, mag_data = pf_static_cart.static_DataGeneration(cParams, gParams, sParams)
    #     with open(innerdatapath + '/metrics_string.txt', 'w') as f:
    #         f.write(metrics_string)
    #     with open(innerdatapath + '/xyz_string.txt', 'w') as f:
    #         f.write(xyz_string)
    #     with open(innerdatapath + '/mag_string.txt', 'w') as f:
    #         f.write(mag_string)
    #     np.savetxt(innerdatapath + '/metrics.dat', metrics_data)
    #     np.savetxt(innerdatapath + '/xyz.dat', xyz_data)
    #     np.savetxt(innerdatapath + '/mag.dat', mag_data)

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
    #     metrics_string, metrics_data, xyz_string, xyz_data, mag_string, mag_data = pf_static_cart.static_DataGeneration(cParams, gParams, sParams)
    #     with open(innerdatapath + '/metrics_string.txt', 'w') as f:
    #         f.write(metrics_string)
    #     with open(innerdatapath + '/xyz_string.txt', 'w') as f:
    #         f.write(xyz_string)
    #     with open(innerdatapath + '/mag_string.txt', 'w') as f:
    #         f.write(mag_string)
    #     np.savetxt(innerdatapath + '/metrics.dat', metrics_data)
    #     np.savetxt(innerdatapath + '/xyz.dat', xyz_data)
    #     np.savetxt(innerdatapath + '/mag.dat', mag_data)

    # end = timer()
    # print('Task ID: {:d}, P: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(taskID, P, aIBi, end - runstart))
