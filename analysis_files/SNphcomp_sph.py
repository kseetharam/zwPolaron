import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pf_dynamic_sph

if __name__ == "__main__":

    # # Initialization

    matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    mI = 1
    mB = 1
    n0 = 1
    gBB = (4 * np.pi / mB) * 0.05

    # cParams

    P = 0.1 * pf_dynamic_sph.nu(gBB)
    aIBi = -2

    # gParams
    (Lx, Ly, Lz) = (75, 75, 75)
    (dx, dy, dz) = (5, 5, 5)

    NGridPoints_desired = (1 + 2 * Lx / dx)**2
    Ntheta = 50
    Nk = np.ceil(NGridPoints_desired / Ntheta)
    theta_max = np.pi
    thetaArray, dtheta = np.linspace(0, theta_max, Ntheta, retstep=True)
    k_max = np.sqrt(3 * (np.pi / dx)**2)
    k_min = 1e-5
    kArray, dk = np.linspace(k_min, k_max, Nk, retstep=True)
    NGridPoints = len(kArray) * len(thetaArray)

    dirpath = os.path.dirname(os.path.realpath(__file__))
    staticdatapath = dirpath + '/data_static/sph/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}/metrics.dat'.format(NGridPoints, P, aIBi)
    rqd_datapath = dirpath + '/data_qdynamics' + '/sph/realtime' + '/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}/ob.dat'.format(NGridPoints, P, aIBi)
    iqd_datapath = dirpath + '/data_qdynamics' + '/sph/imagtime' + '/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}/ob.dat'.format(NGridPoints, P, aIBi)

    NGridPoints_s, k_max_s, P_s, aIBi_s, mI_s, mB_s, n0_s, gBB_s, nu_const_s, gIB_s, Pcrit_s, aSi_s, DP_s, PB_Val_s, En_s, eMass_s, Nph_s, Z_factor_s = np.loadtxt(staticdatapath, unpack=True)
    tgrid, S_tVec, Nph_tVec, PB_tVec, Phase_tVec = np.loadtxt(rqd_datapath, unpack=True)
    itgrid, iS_tVec, iNph_tVec, iPB_tVec, iPhase_tVec = np.loadtxt(iqd_datapath, unpack=True)

    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].plot(tgrid, np.abs(S_tVec), label='RD')
    ax[0].plot(tgrid, np.abs(iS_tVec)**2, label='ID')
    ax[0].plot(tgrid, Z_factor_s * np.ones(len(tgrid)), label='S')
    ax[0].plot(tgrid, 0.5 * (np.abs(iS_tVec)**2 + Z_factor_s * np.ones(len(tgrid))), label='ave')
    ax[0].set_title('S(t)')
    # ax[0].set_xscale('log')
    # ax[0].set_yscale('log')
    ax[0].legend()

    ax[1].plot(tgrid, Nph_tVec, label='RD')
    ax[1].plot(tgrid, 2 * iNph_tVec, label='ID')
    ax[1].plot(tgrid, 2 * Nph_s * np.ones(len(tgrid)), label='S')
    ax[1].plot(tgrid, 0.5 * (2 * iNph_tVec + 2 * Nph_s * np.ones(len(tgrid))), label='ave')
    ax[1].set_title('Nph')
    # ax[1].set_xscale('log')
    # ax[1].set_yscale('log')
    ax[1].legend()

    plt.show()
