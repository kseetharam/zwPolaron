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
    dx = 20e-01
    Lx_List = [50, 100, 200, 300]
    NG_Vec = np.zeros(len(Lx_List))
    for ind, Lx in enumerate(Lx_List):
        NGridPoints_desired = (1 + 2 * Lx / dx)**2
        Ntheta = 50
        Nk = np.ceil(NGridPoints_desired / Ntheta)

        theta_max = np.pi
        thetaArray, dtheta = np.linspace(0, theta_max, Ntheta, retstep=True)

        k_max = np.sqrt(3 * (np.pi / dx)**2)
        k_min = 1e-5
        kArray, dk = np.linspace(k_min, k_max, Nk, retstep=True)

        NG_Vec[ind] = len(kArray) * len(thetaArray)

    print(NG_Vec)
    dirpath = os.path.dirname(os.path.realpath(__file__))

    # # REAL TIME DYNAMICS

    # Grid resolution variation

    Nph_s_Vec = np.zeros(len(NG_Vec))
    Z_Vec = np.zeros(len(NG_Vec))
    S_tail_Vec = np.zeros(len(NG_Vec))
    Nph_qd_tail_Vec = np.zeros(len(NG_Vec))
    iS_tail_Vec = np.zeros(len(NG_Vec))
    iNph_qd_tail_Vec = np.zeros(len(NG_Vec))

    for ind, NG in enumerate(NG_Vec):
        s_datapath = dirpath + '/data_static/sph/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}/metrics.dat'.format(NG, P, aIBi)
        NGridPoints_s, k_max_s, P_s, aIBi_s, mI_s, mB_s, n0_s, gBB_s, nu_const_s, gIB_s, Pcrit_s, aSi_s, DP_s, PB_Val_s, En_s, eMass_s, Nph_s_Vec[ind], Z_Vec[ind] = np.loadtxt(s_datapath, unpack=True)

        rqd_datapath = dirpath + '/data_qdynamics/sph/realtime/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}/ob.dat'.format(NG, P, aIBi)
        tgrid, S_tVec, Nph_tVec, PB_tVec, Phase_tVec = np.loadtxt(rqd_datapath, unpack=True)
        # S_tail_Vec[ind] = np.average(S_tVec[-10:])
        # Nph_qd_tail_Vec[ind] = np.average(Nph_tVec[-10:])
        S_tail_Vec[ind] = S_tVec[-1]
        Nph_qd_tail_Vec[ind] = Nph_tVec[-1]

        iqd_datapath = dirpath + '/data_qdynamics/sph/imagtime/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}/ob.dat'.format(NG, P, aIBi)
        itgrid, iS_tVec, iNph_tVec, iPB_tVec, iPhase_tVec = np.loadtxt(iqd_datapath, unpack=True)
        iS_tail_Vec[ind] = iS_tVec[-1]
        iNph_qd_tail_Vec[ind] = iNph_tVec[-1]

    # max time variation

    NGridPoints = 40450
    tmax_List = [29, 49, 99, 199, 499]
    tmax_Vec = np.array(tmax_List)
    S_inf_Vec = np.zeros(len(tmax_Vec))
    Nph_qd_inf_Vec = np.zeros(len(tmax_Vec))
    iS_inf_Vec = np.zeros(len(tmax_Vec))
    iNph_qd_inf_Vec = np.zeros(len(tmax_Vec))

    s_datapath = dirpath + '/data_static/sph/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}/metrics.dat'.format(NGridPoints, P, aIBi)
    NGridPoints_s, k_max_s, P_s, aIBi_s, mI_s, mB_s, n0_s, gBB_s, nu_const_s, gIB_s, Pcrit_s, aSi_s, DP_s, PB_Val_s, En_s, eMass_s, Nph_s, Z = np.loadtxt(s_datapath, unpack=True)

    for ind, tmax in enumerate(tmax_Vec):
        rqd_datapath = dirpath + '/data_qdynamics/sph/realtime/time_NGridPoints_{:.2E}/{:d}/ob.dat'.format(NGridPoints, tmax)
        tgrid, S_tVec, Nph_tVec, PB_tVec, Phase_tVec = np.loadtxt(rqd_datapath, unpack=True)
        S_inf_Vec[ind] = S_tVec[-1]
        Nph_qd_inf_Vec[ind] = Nph_tVec[-1]

        iqd_datapath = dirpath + '/data_qdynamics/sph/imagtime/time_NGridPoints_{:.2E}/{:d}/ob.dat'.format(NGridPoints, tmax)
        itgrid, iS_tVec, iNph_tVec, iPB_tVec, iPhase_tVec = np.loadtxt(iqd_datapath, unpack=True)
        iS_inf_Vec[ind] = iS_tVec[-1]
        iNph_qd_inf_Vec[ind] = iNph_tVec[-1]

    # plotting

    fig1, ax1 = plt.subplots(nrows=1, ncols=2)

    ax1[0].plot(1 / NG_Vec, Nph_qd_tail_Vec, label=r'$N_{ph,r}(t=t_{max})$')
    ax1[0].plot(1 / NG_Vec, 2 * iNph_qd_tail_Vec, label=r'$2*N_{ph,i}(t=t_{max})$')
    ax1[0].plot(1 / NG_Vec, 2 * Nph_s_Vec, label=r'$2*N_{ph,static}$')
    # ax1[0].set_ylim([0.98 * 2 * Nph_s_Vec[-1], 1.18 * 2 * Nph_s_Vec[-1]])
    # ax1[0].set_title('Nph Tail Average')
    ax1[0].set_title('Phonon Number vs. Grid Resolution')
    ax1[0].set_xlabel(r'$\frac{1}{NG}$')
    ax1[0].legend()
    ax1[0].set_xscale('log')
    # ax1[0].set_yscale('log')

    ax1[1].plot(1 / NG_Vec, S_tail_Vec, label=r'$S(t=t_{max})_{r}$')
    ax1[1].plot(1 / NG_Vec, iS_tail_Vec**2, label=r'$|S(t=t_{max})_{i}|^{2}$')
    ax1[1].plot(1 / NG_Vec, Z_Vec, label='Z-factor')
    # ax1[1].set_ylim([0.83 * Z_Vec[-1], 1.03 * Z_Vec[-1]])
    # ax1[1].set_title('S(t) Tail Average')
    ax1[1].set_title('Dynamical Overlap vs. Grid Resolution')
    ax1[1].set_xlabel(r'$\frac{1}{NG}$')
    ax1[1].legend()
    ax1[1].set_xscale('log')
    # ax1[1].set_yscale('log')

    fig2, ax2 = plt.subplots(nrows=1, ncols=2)

    ax2[0].plot(1 / tmax_Vec, Nph_qd_inf_Vec, label=r'$N_{ph,r}(t=t_{max})$')
    ax2[0].plot(1 / tmax_Vec, 2 * iNph_qd_inf_Vec, label=r'$N_{ph,i}(t=t_{max})$')
    ax2[0].plot(1 / tmax_Vec, 2 * Nph_s * np.ones(len(tmax_Vec)), label=r'$2*N_{ph,static}$')
    # ax2[0].set_ylim([0.98 * 2 * Nph_s, 1.18 * 2 * Nph_s])
    ax2[0].set_title('Phonon Number vs. End Time')
    ax2[0].set_xlabel(r'$\frac{1}{t_{max}}$')
    ax2[0].legend()
    ax2[0].set_xscale('log')
    # ax2[0].set_yscale('log')

    ax2[1].plot(1 / tmax_Vec, S_inf_Vec, label=r'$S(t=t_{max})_{r}$')
    ax2[1].plot(1 / tmax_Vec, iS_inf_Vec**2, label=r'$|S(t=t_{max})_{i}|^{2}$')
    ax2[1].plot(1 / tmax_Vec, Z * np.ones(len(tmax_Vec)), label='Z-factor')
    # ax2[1].set_ylim([0.83 * Z, 1.03 * Z])
    ax2[1].set_title('Dynamical Overlap vs. End Time')
    ax2[1].set_xlabel(r'$\frac{1}{t_{max}}$')
    ax2[1].legend()
    ax2[1].set_xscale('log')
    # ax2[1].set_yscale('log')

    # # # IMAGINARY TIME DYNAMICS

    # # Grid resolution variation

    # iNph_s_Vec = np.zeros(len(NG_Vec))
    # iZ_Vec = np.zeros(len(NG_Vec))
    # iS_List = []; iS_tail_Vec = np.zeros(len(NG_Vec))
    # iNph_qd_List = []; iNph_qd_tail_Vec = np.zeros(len(NG_Vec))
    # for ind, NG in enumerate(NG_Vec):
    #     s_datapath = dirpath + '/data_static/sph/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}/metrics.dat'.format(NG, P, aIBi)
    #     qd_datapath = dirpath + '/data_qdynamics/sph/imagtime/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}/ob.dat'.format(NG, P, aIBi)

    #     NGridPoints_s, k_max_s, P_s, aIBi_s, mI_s, mB_s, n0_s, gBB_s, nu_const_s, gIB_s, Pcrit_s, aSi_s, DP_s, PB_Val_s, En_s, eMass_s, iNph_s_Vec[ind], iZ_Vec[ind] = np.loadtxt(s_datapath, unpack=True)
    #     tgrid, S_tVec, Nph_tVec, PB_tVec, Phase_tVec = np.loadtxt(qd_datapath, unpack=True)
    #     iS_List.append(S_tVec)
    #     iNph_qd_List.append(Nph_tVec)
    #     # iS_tail_Vec[ind] = np.average(S_tVec[-10:])
    #     # iNph_qd_tail_Vec[ind] = np.average(Nph_tVec[-10:])
    #     iS_tail_Vec[ind] = S_tVec[-1]
    #     iNph_qd_tail_Vec[ind] = Nph_tVec[-1]

    # # max time variation

    # NGridPoints = 40450
    # tmax_List = [29, 49, 99, 199, 499]
    # tmax_Vec = np.array(tmax_List)
    # iS_inf_Vec = np.zeros(len(tmax_Vec))
    # iNph_qd_inf_Vec = np.zeros(len(tmax_Vec))

    # s_datapath = dirpath + '/data_static/sph/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}/metrics.dat'.format(NGridPoints, P, aIBi)
    # NGridPoints_s, k_max_s, P_s, aIBi_s, mI_s, mB_s, n0_s, gBB_s, nu_const_s, gIB_s, Pcrit_s, aSi_s, DP_s, PB_Val_s, En_s, eMass_s, iNph_s, iZ = np.loadtxt(s_datapath, unpack=True)
    # for ind, tmax in enumerate(tmax_Vec):
    #     qd_datapath = dirpath + '/data_qdynamics/sph/imagtime/time_NGridPoints_{:.2E}/{:d}/ob.dat'.format(NGridPoints, tmax)
    #     tgrid, S_tVec, Nph_tVec, PB_tVec, Phase_tVec = np.loadtxt(qd_datapath, unpack=True)
    #     iS_inf_Vec[ind] = S_tVec[-1]
    #     iNph_qd_inf_Vec[ind] = Nph_tVec[-1]

    # # plotting

    # fig3, ax3 = plt.subplots(nrows=1, ncols=2)

    # ax3[0].plot(1 / NG_Vec, iNph_qd_tail_Vec, label=r'$N_{ph}(t=t_{max})$')
    # ax3[0].plot(1 / NG_Vec, iNph_s_Vec, label=r'$N_{ph,static}$')
    # ax3[0].set_ylim([0.83 * iNph_s_Vec[-1], 1.03 * iNph_s_Vec[-1]])
    # # ax3[0].set_title('Nph Tail Average')
    # ax3[0].set_title('Phonon Number vs. Grid Resolution')
    # ax3[0].set_xlabel(r'$\frac{1}{NG}$')
    # ax3[0].legend()
    # ax3[0].set_xscale('log')
    # # ax3[0].set_yscale('log')

    # ax3[1].plot(1 / NG_Vec, iS_tail_Vec**2, label=r'$|S(t=t_{max})|^{2}$')
    # ax3[1].plot(1 / NG_Vec, iZ_Vec, label='Z-factor')
    # ax3[1].set_ylim([0.98 * iZ_Vec[-1], 1.18 * iZ_Vec[-1]])
    # # ax3[1].set_title('S(t) Tail Average')
    # ax3[1].set_title('Dynamical Overlap vs. Grid Resolution')
    # ax3[1].set_xlabel(r'$\frac{1}{NG}$')
    # ax3[1].legend()
    # ax3[1].set_xscale('log')
    # # ax3[1].set_yscale('log')

    # fig4, ax4 = plt.subplots(nrows=1, ncols=2)

    # ax4[0].plot(1 / tmax_Vec, iNph_qd_inf_Vec, label=r'$N_{ph}(t=t_{max})$')
    # ax4[0].plot(1 / tmax_Vec, iNph_s * np.ones(len(tmax_Vec)), label=r'$N_{ph,static}$')
    # ax4[0].set_ylim([0.83 * iNph_s, 1.03 * iNph_s])
    # ax4[0].set_title('Phonon Number vs. End Time')
    # ax4[0].set_xlabel(r'$\frac{1}{t_{max}}$')
    # ax4[0].legend()
    # ax4[0].set_xscale('log')
    # # ax4[0].set_yscale('log')

    # ax4[1].plot(1 / tmax_Vec, iS_inf_Vec**2, label=r'$|S(t=t_{max})|^{2}$')
    # ax4[1].plot(1 / tmax_Vec, iZ * np.ones(len(tmax_Vec)), label='Z-factor')
    # ax4[1].set_ylim([0.98 * iZ, 1.18 * iZ])
    # ax4[1].set_title('Dynamical Overlap vs. End Time')
    # ax4[1].set_xlabel(r'$\frac{1}{t_{max}}$')
    # ax4[1].legend()
    # ax4[1].set_xscale('log')
    # # ax4[1].set_yscale('log')

    plt.show()
