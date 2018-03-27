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

    (Lx, Ly, Lz) = (120, 120, 120)
    (dx, dy, dz) = (4, 4, 4)

    xgrid = Grid.Grid('CARTESIAN_3D')
    xgrid.initArray('x', -Lx, Lx, dx); xgrid.initArray('y', -Ly, Ly, dy); xgrid.initArray('z', -Lz, Lz, dz)

    # NGridPoints_desired = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
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

    datapath = os.path.dirname(os.path.realpath(__file__)) + '/dyn_stat_discrepancy/data/sph/imagtime' + '/NGridPoints_{:.2E}'.format(NGridPoints)

    # datapath = os.path.dirname(os.path.realpath(__file__)) + '/data_qdynamics' + '/sph/realtime' + '/NGridPoints_{:.2E}'.format(NGridPoints)
    # datapath = os.path.dirname(os.path.realpath(__file__)) + '/data_qdynamics' + '/sph/realtime' + '/time_NGridPoints_{:.2E}'.format(NGridPoints)
    # datapath = os.path.dirname(os.path.realpath(__file__)) + '/data_qdynamics' + '/sph/imagtime' + '/NGridPoints_{:.2E}'.format(NGridPoints)
    # datapath = os.path.dirname(os.path.realpath(__file__)) + '/data_qdynamics' + '/sph/imagtime' + '/time_NGridPoints_{:.2E}'.format(NGridPoints)

    # datapath = os.path.dirname(os.path.realpath(__file__)) + '/data_qdynamics' + '/sph/frolich/realtime' + '/NGridPoints_{:.2E}'.format(NGridPoints)
    # datapath = os.path.dirname(os.path.realpath(__file__)) + '/data_qdynamics' + '/sph/frolich/realtime' + '/time_NGridPoints_{:.2E}'.format(NGridPoints)
    # datapath = os.path.dirname(os.path.realpath(__file__)) + '/data_qdynamics' + '/sph/frolich/imagtime' + '/NGridPoints_{:.2E}'.format(NGridPoints)
    # datapath = os.path.dirname(os.path.realpath(__file__)) + '/data_qdynamics' + '/sph/frolich/imagtime' + '/time_NGridPoints_{:.2E}'.format(NGridPoints)

    if os.path.isdir(datapath) is False:
        os.mkdir(datapath)

    # ---- SINGLE FUNCTION RUN ----

    runstart = timer()

    P = 0.1 * pf_dynamic_sph.nu(gBB)
    aIBi = -2
    cParams = [P, aIBi]

    innerdatapath = datapath + '/P_{:.3f}_aIBi_{:.2f}'.format(P, aIBi)
    # innerdatapath = datapath + '/{0}'.format(tMax)
    if os.path.isdir(innerdatapath) is False:
        os.mkdir(innerdatapath)

    time_grid, metrics_data = pf_dynamic_sph.quenchDynamics_DataGeneration(cParams, gParams, sParams)

    end = timer()
    print('Time: {:.2f}'.format(end - runstart))

    # TEMP DATA CHECK

    [NGridPoints, k_max, P, aIBi, mI, mB, n0, gBB, nu_const, gIB, PB_tVec, NB_tVec, DynOv_tVec, Phase_tVec] = metrics_data
    print(k_max, P, aIBi, mI, mB, n0, gBB, nu_const, gIB)

    ob_data = np.concatenate((tgrid[:, np.newaxis], np.abs(DynOv_tVec)[:, np.newaxis], NB_tVec[:, np.newaxis], PB_tVec[:, np.newaxis], Phase_tVec[:, np.newaxis]), axis=1)
    np.savetxt(innerdatapath + '/ob.dat', ob_data)

    ob_string = 't, |S(t)|, Nph(t), PB(t), Phi(t)'
    with open(innerdatapath + '/ob_string.txt', 'w') as f:
        f.write(ob_string)

    # staticdatapath = os.path.dirname(os.path.realpath(__file__)) + '/data_static/sph/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}/metrics.dat'.format(NGridPoints, P, aIBi)
    # NGridPoints_s, k_max_s, dk_s, P_s, aIBi_s, mI_s, mB_s, n0_s, gBB_s, nu_const_s, gIB_s, Pcrit_s, aSi_s, DP_s, PB_Val_s, En_s, eMass_s, Nph_s, Z_factor_s = np.loadtxt(staticdatapath, unpack=True)

    # # print('|S(t) - Z|: {0}'.format(np.abs(np.abs(DynOv_tVec[-1]) - Z_factor_s) / Z_factor_s))
    # # print('|N(t)-2*Npol|: {0}'.format(np.abs(NB_tVec[-1] - 2 * Nph_s) / (2 * Nph_s)))

    # print('|S(t)|: {0}'.format(np.abs(DynOv_tVec[-1])**2))
    # print('N(t): {0}'.format(2 * NB_tVec[-1]))

    # fig, ax = plt.subplots(nrows=1, ncols=2)

    # ax[0].plot(time_grid, np.abs(DynOv_tVec)**2)
    # ax[0].plot(time_grid, Z_factor_s * np.ones(len(time_grid)))
    # # ax[0].set_xscale('log')
    # # ax[0].set_yscale('log')

    # ax[1].plot(time_grid, 2 * NB_tVec)
    # ax[1].plot(time_grid, 2 * Nph_s * np.ones(len(time_grid)))
    # # ax[1].set_xscale('log')
    # # ax[1].set_yscale('log')

    # plt.show()
