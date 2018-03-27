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
    (Lx, Ly, Lz) = (120, 120, 120)
    (dx, dy, dz) = (4, 4, 4)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    NGridPoints_desired = (1 + 2 * Lx / dx)**2
    Ntheta = 50
    Nk = np.ceil(NGridPoints_desired / Ntheta)
    theta_max = np.pi
    thetaArray, dtheta = np.linspace(0, theta_max, Ntheta, retstep=True)
    k_max = ((2 * np.pi / dx)**3 / (4 * np.pi / 3))**(1 / 3)
    k_min = 1e-5
    kArray, dk = np.linspace(k_min, k_max, Nk, retstep=True)
    NGridPoints_sph = len(kArray) * len(thetaArray)

    dirpath = os.path.dirname(os.path.realpath(__file__))
    static_datapath_cart = dirpath + '/dyn_stat_discrepancy/data/cart/static/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}/metrics.dat'.format(NGridPoints_cart, P, aIBi)
    rqd_datapath_cart = dirpath + '/dyn_stat_discrepancy/data/cart/realtime/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}/ob.dat'.format(NGridPoints_cart, P, aIBi)
    iqd_datapath_cart = dirpath + '/dyn_stat_discrepancy/data/cart/imagtime/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}/ob.dat'.format(NGridPoints_cart, P, aIBi)
    static_datapath_sph = dirpath + '/dyn_stat_discrepancy/data/sph/static/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}/metrics.dat'.format(NGridPoints_sph, P, aIBi)
    rqd_datapath_sph = dirpath + '/dyn_stat_discrepancy/data/sph/realtime/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}/ob.dat'.format(NGridPoints_sph, P, aIBi)
    iqd_datapath_sph = dirpath + '/dyn_stat_discrepancy/data/sph/imagtime/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}/ob.dat'.format(NGridPoints_sph, P, aIBi)

    NGridPoints_s, k_max_s, dk_s, P_s, aIBi_s, mI_s, mB_s, n0_s, gBB_s, nu_const_s, gIB_s, Pcrit_s, aSi_s, DP_s, PB_Val_s, En_s, eMass_s, Nph_s, Z_factor_s = np.loadtxt(static_datapath_sph, unpack=True)
    tgrid_sph, S_tVec_sph, Nph_tVec_sph, PB_tVec_sph, Phase_tVec_sph = np.loadtxt(rqd_datapath_sph, unpack=True)
    itgrid_sph, iS_tVec_sph, iNph_tVec_sph, iPB_tVec_sph, iPhase_tVec_sph = np.loadtxt(iqd_datapath_sph, unpack=True)

    NGridPoints_c, k_max_c, P_c, aIBi_c, mI_c, mB_c, n0_c, gBB_c, nu_const_c, gIB_c, Pcrit_c, aSi_c, DP_c, PB_Val_c, En_c, eMass_c, Nph_c, Nph_xyz_c, Z_factor_c, nxyz_Tot, nPB_Tot, nPBm_Tot, nPIm_Tot, nPB_Mom1, beta2_kz_Mom1, nPB_deltaK0, FWHM = np.loadtxt(static_datapath_cart, unpack=True)
    tgrid_cart, S_tVec_cart, Nph_tVec_cart, PB_tVec_cart, Phase_tVec_cart = np.loadtxt(rqd_datapath_cart, unpack=True)
    itgrid_cart, iS_tVec_cart, iNph_tVec_cart, iPB_tVec_cart, iPhase_tVec_cart = np.loadtxt(iqd_datapath_cart, unpack=True)

    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].plot(tgrid_sph, np.abs(S_tVec_sph), label='RD (sph)')
    ax[0].plot(tgrid_sph, np.abs(iS_tVec_sph)**2, label='ID (sph)')
    ax[0].plot(tgrid_sph, Z_factor_s * np.ones(len(tgrid_sph)), label='Stat (sph)')
    ax[0].plot(tgrid_cart, np.abs(S_tVec_cart), label='RD (cart)')
    ax[0].plot(tgrid_cart, np.abs(iS_tVec_cart)**2, label='ID (cart)')
    ax[0].plot(tgrid_cart, Z_factor_c * np.ones(len(tgrid_cart)), label='Stat (cart)')
    ax[0].set_title('S(t)')
    ax[0].legend()

    ax[1].plot(tgrid_sph, Nph_tVec_sph, label='RD (sph)')
    ax[1].plot(tgrid_sph, 2 * iNph_tVec_sph, label='ID (sph)')
    ax[1].plot(tgrid_sph, 2 * Nph_s * np.ones(len(tgrid_sph)), label='Stat (sph)')
    ax[1].plot(tgrid_cart, Nph_tVec_cart, label='RD (cart)')
    ax[1].plot(tgrid_cart, 2 * iNph_tVec_cart, label='ID (cart)')
    ax[1].plot(tgrid_cart, 2 * Nph_c * np.ones(len(tgrid_cart)), label='Stat (cart)')
    ax[1].set_title('Nph')
    ax[1].legend()

    plt.show()
