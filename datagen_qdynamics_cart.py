import numpy as np
import Grid
import pf_dynamic_cart
import os
from timeit import default_timer as timer
import pickle

# import matplotlib
# import matplotlib.pyplot as plt

if __name__ == "__main__":

    start = timer()

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (20, 20, 20)
    (dx, dy, dz) = (0.5, 0.5, 0.5)

    xgrid = Grid.Grid('CARTESIAN_3D')
    xgrid.initArray('x', -Lx, Lx, dx); xgrid.initArray('y', -Ly, Ly, dy); xgrid.initArray('z', -Lz, Lz, dz)

    (Nx, Ny, Nz) = (len(xgrid.getArray('x')), len(xgrid.getArray('y')), len(xgrid.getArray('z')))

    kxfft = np.fft.fftfreq(Nx) * 2 * np.pi / dx; kyfft = np.fft.fftfreq(Nx) * 2 * np.pi / dy; kzfft = np.fft.fftfreq(Nx) * 2 * np.pi / dz

    kgrid = Grid.Grid('CARTESIAN_3D')
    kgrid.initArray_premade('kx', np.fft.fftshift(kxfft)); kgrid.initArray_premade('ky', np.fft.fftshift(kyfft)); kgrid.initArray_premade('kz', np.fft.fftshift(kzfft))

    kx = kgrid.getArray('kx')

    tMax = 1
    dt = 0.1
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

    # # ---- SET OUTPUT DATA FOLDER ----

    # dirpath = os.path.dirname(os.path.realpath(__file__))

    # # datapath = dirpath + '/dyn_stat_discrepancy/data/cart/realtime' + '/NGridPoints_{:.2E}'.format(NGridPoints)

    # datapath = dirpath + '/data_qdynamics' + '/cart/realtime' + '/NGridPoints_{:.2E}'.format(NGridPoints)
    # # datapath = dirpath + '/data_qdynamics' + '/cart/realtime' + '/time_NGridPoints_{:.2E}'.format(NGridPoints)
    # # datapath = dirpath + '/data_qdynamics' + '/cart/imagtime' + '/NGridPoints_{:.2E}'.format(NGridPoints)
    # # datapath = dirpath + '/data_qdynamics' + '/cart/imagtime' + '/time_NGridPoints_{:.2E}'.format(NGridPoints)

    # # datapath = dirpath + '/data_qdynamics' + '/cart/frolich/realtime' + '/NGridPoints_{:.2E}'.format(NGridPoints)
    # # datapath = dirpath + '/data_qdynamics' + '/cart/frolich/realtime' + '/time_NGridPoints_{:.2E}'.format(NGridPoints)
    # # datapath = dirpath + '/data_qdynamics' + '/cart/frolich/imagtime' + '/NGridPoints_{:.2E}'.format(NGridPoints)
    # # datapath = dirpath + '/data_qdynamics' + '/cart/frolich/imagtime' + '/time_NGridPoints_{:.2E}'.format(NGridPoints)

    # if os.path.isdir(datapath) is False:
    #     os.mkdir(datapath)

    # # ---- SINGLE FUNCTION RUN ----

    # runstart = timer()

    # # P = .07926654595212022369
    # P = 1.8 * pf_dynamic_cart.nu(gBB)
    # aIBi = -2
    # cParams = [P, aIBi]

    # innerdatapath = datapath + '/P_{:.3f}_aIBi_{:.2f}'.format(P, aIBi)
    # if os.path.isdir(innerdatapath) is False:
    #     os.mkdir(innerdatapath)

    # time_grids, metrics_data, pos_xyz_data, mom_xyz_data, cont_xyz_data, mom_mag_data = pf_dynamic_cart.quenchDynamics_DataGeneration(cParams, gParams, sParams)

    # with open(innerdatapath + '/time_grids.pickle', 'wb') as f:
    #     pickle.dump(time_grids, f)
    # with open(innerdatapath + '/metrics_data.pickle', 'wb') as f:
    #     pickle.dump(metrics_data, f)
    # with open(innerdatapath + '/pos_xyz_data.pickle', 'wb') as f:
    #     pickle.dump(pos_xyz_data, f)
    # with open(innerdatapath + '/mom_xyz_data.pickle', 'wb') as f:
    #     pickle.dump(mom_xyz_data, f)
    # with open(innerdatapath + '/cont_xyz_data.pickle', 'wb') as f:
    #     pickle.dump(cont_xyz_data, f)
    # with open(innerdatapath + '/mom_mag_data.pickle', 'wb') as f:
    #     pickle.dump(mom_mag_data, f)

    # end = timer()
    # print('Time: {:.2f}'.format(end - runstart))

    # # TEMP DATA CHECK

    # [tgrid, tgrid_coarse] = time_grids
    # [NGridPoints, k_max, P, aIBi, mI, mB, n0, gBB, nu_const, gIB, PB_tVec, NB_tVec, DynOv_tVec, Phase_tVec] = metrics_data
    # print(k_max, P, aIBi, mI, mB, n0, gBB, nu_const, gIB)

    # ob_data = np.concatenate((tgrid[:, np.newaxis], np.abs(DynOv_tVec)[:, np.newaxis], NB_tVec[:, np.newaxis], PB_tVec[:, np.newaxis], Phase_tVec[:, np.newaxis]), axis=1)
    # np.savetxt(innerdatapath + '/ob.dat', ob_data)

    # ob_string = 't, |S(t)|, Nph(t), PB(t), Phi(t)'
    # with open(innerdatapath + '/ob_string.txt', 'w') as f:
    #     f.write(ob_string)

    # staticdatapath = os.path.dirname(os.path.realpath(__file__)) + '/data_static/cart/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}/metrics.dat'.format(NGridPoints, P, aIBi)
    # NGridPoints_s, k_max_s, P_s, aIBi_s, mI_s, mB_s, n0_s, gBB_s, nu_const_s, gIB_s, Pcrit_s, aSi_s, DP_s, PB_Val_s, En_s, eMass_s, Nph_s, Nph_xyz_s, Z_factor_s, nxyz_Tot, nPB_Tot, nPBm_Tot, nPIm_Tot, nPB_Mom1, beta2_kz_Mom1, nPB_deltaK0, FWHM = np.loadtxt(staticdatapath, unpack=True)

    # # # print('|S(t) - Z|: {0}'.format(np.abs(np.abs(DynOv_tVec[-1]) - Z_factor_s) / Z_factor_s))
    # # # print('|N(t)-2*Npol|: {0}'.format(np.abs(NB_tVec[-1] - 2 * Nph_s) / (2 * Nph_s)))

    # print('|S(t)|: {0}'.format(np.abs(DynOv_tVec[-1])))
    # print('N(t): {0}'.format(NB_tVec[-1]))

    # fig, ax = plt.subplots(nrows=1, ncols=2)

    # ax[0].plot(tgrid, np.abs(DynOv_tVec))
    # ax[0].plot(tgrid, Z_factor_s * np.ones(len(tgrid)))
    # # ax[0].set_xscale('log')
    # # ax[0].set_yscale('log')

    # ax[1].plot(tgrid, NB_tVec)
    # ax[1].plot(tgrid, 2 * Nph_s * np.ones(len(tgrid)))
    # # ax[1].set_xscale('log')
    # # ax[1].set_yscale('log')

    # plt.show()

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

    # # ---- COMPUTE DATA ON CLUSTER ----

    runstart = timer()

    datapath = '/n/regal/demler_lab/kis/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints)
    # if os.path.isdir(datapath) is False:
    #     os.mkdir(datapath)

    taskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    taskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    if(taskCount != len(cParams_List)):
        print('ERROR: TASK COUNT MISMATCH')
        P = float('nan')
        aIBi = float('nan')
    else:
        cParams = cParams_List[taskID]
        [P, aIBi] = cParams
        innerdatapath = datapath + '/P_{:.3f}_aIBi_{:.2f}'.format(P, aIBi)
        if os.path.isdir(innerdatapath) is False:
            os.mkdir(innerdatapath)

        time_grids, metrics_data, pos_xyz_data, mom_xyz_data, cont_xyz_data, mom_mag_data = pf_dynamic_cart.quenchDynamics_DataGeneration(cParams, gParams, sParams)

        with open(innerdatapath + '/time_grids.pickle', 'wb') as f:
            pickle.dump(time_grids, f)
        with open(innerdatapath + '/metrics_data.pickle', 'wb') as f:
            pickle.dump(metrics_data, f)
        with open(innerdatapath + '/pos_xyz_data.pickle', 'wb') as f:
            pickle.dump(pos_xyz_data, f)
        with open(innerdatapath + '/mom_xyz_data.pickle', 'wb') as f:
            pickle.dump(mom_xyz_data, f)
        with open(innerdatapath + '/cont_xyz_data.pickle', 'wb') as f:
            pickle.dump(cont_xyz_data, f)
        with open(innerdatapath + '/mom_mag_data.pickle', 'wb') as f:
            pickle.dump(mom_mag_data, f)

    end = timer()
    print('Task ID: {:d}, P: {:.2f}, aIBi: {:.2f} Time: {:.2f}'.format(taskID, P, aIBi, end - runstart))
