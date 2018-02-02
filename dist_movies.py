import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pf_dynamic_cart
import pickle

if __name__ == "__main__":

    # # Initialization

    matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    mI = 1
    mB = 1
    n0 = 1
    gBB = (4 * np.pi / mB) * 0.05

    # cParams

    P = 0.1 * pf_dynamic_cart.nu(gBB)
    aIBi = -2

    # gParams
    (Lx, Ly, Lz) = (120, 120, 120)
    (dx, dy, dz) = (2, 2, 2)

    NGridPoints = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    # Load dynamics data

    dirpath = os.path.dirname(os.path.realpath(__file__))
    datapath = dirpath + '/data_qdynamics' + '/cart/realtime' + '/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}'.format(NGridPoints, P, aIBi)

    with open(datapath + '/time_grids.pickle', 'rb') as f:
        time_grids = pickle.load(f)
    with open(datapath + '/metrics_data.pickle', 'rb') as f:
        metrics_data = pickle.load(f)
    with open(datapath + '/pos_xyz_data.pickle', 'rb') as f:
        pos_xyz_data = pickle.load(f)
    with open(datapath + '/mom_xyz_data.pickle', 'rb') as f:
        mom_xyz_data = pickle.load(f)
    with open(datapath + '/cont_xyz_data.pickle', 'rb') as f:
        cont_xyz_data = pickle.load(f)
    with open(datapath + '/mom_mag_data.pickle', 'rb') as f:
        mom_mag_data = pickle.load(f)

    [tgrid, tgrid_coarse] = time_grids
    [NGridPoints, k_max, P, aIBi, mI, mB, n0, gBB, nu_const, gIB, PB_tVec, NB_tVec, DynOv_tVec, Phase_tVec] = metrics_data
    [x, y, z, nxyz_x_ctVec, nxyz_y_ctVec, nxyz_z_ctVec, nxyz_x_slice_ctVec, nxyz_y_slice_ctVec, nxyz_z_slice_ctVec] = pos_xyz_data
    [PB_x, PB_y, PB_z, nPB_x_ctVec, nPB_y_ctVec, nPB_z_ctVec, nPB_x_slice_ctVec, nPB_y_slice_ctVec, nPB_z_slice_ctVec, PI_x, PI_y, PI_z, nPI_x_ctVec, nPI_y_ctVec, nPI_z_ctVec, nPI_x_slice_ctVec, nPI_y_slice_ctVec, nPI_z_slice_ctVec, phonon_mom_k0deltapeak_ctVec] = mom_xyz_data
    [nxyz_xz_slice_ctVec, nxyz_xy_slice_ctVec, nPB_xz_slice_ctVec, nPB_xy_slice_ctVec, nPI_xz_slice_ctVec, nPI_xy_slice_ctVec] = cont_xyz_data
    [PBm, nPBm_ctVec, PIm, nPIm_ctVec] = mom_mag_data

    nPBm_Tot_ctVec = np.zeros(len(tgrid_coarse))
    nPIm_Tot_ctVec = np.zeros(len(tgrid_coarse))
    dPBm = PBm[1] - PBm[0]
    dPIm = PIm[1] - PIm[0]
    for ind, t in enumerate(tgrid_coarse):
        nPBm_Tot_ctVec[ind] = np.sum(nPBm_ctVec[ind] * dPBm) + phonon_mom_k0deltapeak_ctVec[ind]
        nPIm_Tot_ctVec[ind] = np.sum(nPIm_ctVec[ind] * dPIm) + phonon_mom_k0deltapeak_ctVec[ind]

    # Load static data

    static_datapath = dirpath + '/data_static' + '/cart' + '/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}'.format(NGridPoints, P, aIBi)

    NGridPoints_st, k_max_st, P_st, aIBi_st, mI_st, mB_st, n0_st, gBB_st, nu_stonst_st, gIB_st, Pcrit_st, aSi_st, DP_st, PB_Val_st, En_st, eMass_st, Nph_st, Nph_xyz_st, Z_factor_st, nxyz_Tot_st, nPB_Tot_st, nPBm_Tot_st, nPIm_Tot_st, nPB_Mom1_st, beta2_kz_Mom1_st, nPB_deltaK0_st, FWHM_st = np.loadtxt(static_datapath + '/metrics.dat', unpack=True)
    x_st, y_st, z_st, nxyz_x_st, nxyz_y_st, nxyz_z_st, nxyz_x_slice_st, nxyz_y_slice_st, nxyz_z_slice_st = np.loadtxt(static_datapath + '/pos_xyz.dat', unpack=True)
    PB_x_st, PB_y_st, PB_z_st, nPB_x_st, nPB_y_st, nPB_z_st, nPB_x_slice_st, nPB_y_slice_st, nPB_z_slice_st, PI_x_st, PI_y_st, PI_z_st, nPI_x_st, nPI_y_st, nPI_z_st, nPI_x_slice_st, nPI_y_slice_st, nPI_z_slice_st = np.loadtxt(static_datapath + '/mom_xyz.dat', unpack=True)
    with open(static_datapath + '/cont_xyz_data.pickle', 'rb') as f:
        [nxyz_xz_slice_st, nxyz_xy_slice_st, nPB_xz_slice_st, nPB_xy_slice_st, nPI_xz_slice_st, nPI_xy_slice_st] = pickle.load(f)
    PBm_Vec_st, nPBm_Vec_st, PIm_Vec_st, nPIm_Vec_st = np.loadtxt(static_datapath + '/mom_mag.dat', unpack=True)

    # Set animation datapath and max end time

    figdatapath = dirpath + '/data_qdynamics' + '/cart/realtime' + '/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}'.format(NGridPoints, P, aIBi) + '/figures'
    if os.path.isdir(figdatapath) is False:
        os.mkdir(figdatapath)

    tmax = 15
    tmax_ind = (np.abs(tgrid_coarse - tmax)).argmin()

    # # MAGNITUDE ANIMATION

    # nPIm

    fig1, ax = plt.subplots()
    curve = ax.plot(PIm, nPIm_ctVec[0], color='g', lw=2)[0]
    line = ax.plot(P * np.ones(len(PIm)), np.linspace(0, phonon_mom_k0deltapeak_ctVec[0], len(PIm)), 'go')[0]
    time_text = ax.text(0.85, 0.9, 't: {:.2f}'.format(tgrid_coarse[0]), transform=ax.transAxes, color='r')
    norm_text = ax.text(0.02, 0.9, r'$\int n_{|\vec{P_{I}}|}(t) d|\vec{P_{I}}| = $' + '{:.3f}'.format(nPIm_Tot_ctVec[0]), transform=ax.transAxes, color='g')

    ax.plot(PIm_Vec_st, nPIm_Vec_st, 'b')
    ax.plot(P * np.ones(len(PIm_Vec_st)), np.linspace(0, nPB_deltaK0_st, len(PIm_Vec_st)), 'b--')
    norm_st_text = ax.text(0.02, 0.8, r'$\int n_{|\vec{P_{I}}|,st} d|\vec{P_{I}}| = $' + '{:.3f}'.format(nPIm_Tot_st), transform=ax.transAxes, color='b')

    ax.set_xlim([-0.01, np.max(PIm)])
    ax.set_ylim([0, 1])
    ax.set_title('Impurity Momentum Magnitude Distribution')
    ax.set_ylabel(r'$n_{|\vec{P_{I}}|}$')
    ax.set_xlabel(r'$|\vec{P_{I}}|$')

    def animate1(i):
        curve.set_ydata(nPIm_ctVec[i])
        line.set_ydata(np.linspace(0, phonon_mom_k0deltapeak_ctVec[i], len(PIm)))

        time_text.set_text('t: {:.2f}'.format(tgrid_coarse[i]))
        norm_text.set_text(r'$\int n_{|\vec{P_{I}}|}(t) d|\vec{P_{I}}| = $' + '{:.3f}'.format(nPIm_Tot_ctVec[i]))

    anim1 = FuncAnimation(fig1, animate1, interval=300, frames=range(tmax_ind + 1))
    anim1.save(figdatapath + '/nPIm.gif', writer='imagemagick')

    # plt.draw()
    # plt.show()

    # nPBm

    fig2, ax = plt.subplots()
    curve = ax.plot(PBm, nPBm_ctVec[0], color='g', lw=2)[0]
    line = ax.plot(np.zeros(len(PBm)), np.linspace(0, phonon_mom_k0deltapeak_ctVec[0], len(PBm)), 'go')[0]
    time_text = ax.text(0.85, 0.9, 't: {:.2f}'.format(tgrid_coarse[0]), transform=ax.transAxes, color='r')
    norm_text = ax.text(0.02, 0.9, r'$\int n_{|\vec{P_{B}}|}(t) d|\vec{P_{B}}| = $' + '{:.3f}'.format(nPBm_Tot_ctVec[0]), transform=ax.transAxes, color='g')

    ax.plot(PBm_Vec_st, nPBm_Vec_st, 'b')
    ax.plot(np.zeros(len(PBm_Vec_st)), np.linspace(0, nPB_deltaK0_st, len(PBm_Vec_st)), 'b--')
    norm_st_text = ax.text(0.02, 0.8, r'$\int n_{|\vec{P_{B}}|,st} d|\vec{P_{B}}| = $' + '{:.3f}'.format(nPBm_Tot_st), transform=ax.transAxes, color='b')

    ax.set_xlim([-0.01, np.max(PBm)])
    ax.set_ylim([0, 1])
    ax.set_title('Phonon Momentum Magnitude Distribution')
    ax.set_ylabel(r'$n_{|\vec{P_{B}}|}$')
    ax.set_xlabel(r'$|\vec{P_{B}}|$')

    def animate2(i):
        curve.set_ydata(nPBm_ctVec[i])
        line.set_ydata(np.linspace(0, phonon_mom_k0deltapeak_ctVec[i], len(PBm)))

        time_text.set_text('t: {:.2f}'.format(tgrid_coarse[i]))
        norm_text.set_text(r'$\int n_{|\vec{P_{B}}|}(t) d|\vec{P_{B}}| = $' + '{:.3f}'.format(nPBm_Tot_ctVec[i]))

    anim2 = FuncAnimation(fig2, animate2, interval=300, frames=range(tmax_ind + 1))
    anim2.save(figdatapath + '/nPBm.gif', writer='imagemagick')

    # # plt.draw()
    # # plt.show()

    # # MAGNITUDE 2D DYNAMICAL PLOT

    tgrid_coarse_short = tgrid_coarse[:tmax_ind + 1]
    tc_g, PIm_g = np.meshgrid(tgrid_coarse_short, PIm, indexing='ij')
    nPIm_g = np.zeros((len(tgrid_coarse_short), len(PIm)))
    for ind, nPIm_Vec in enumerate(nPIm_ctVec[:tmax_ind + 1]):
        nPIm_g[ind, :] = nPIm_Vec[:]

    tc_g, PBm_g = np.meshgrid(tgrid_coarse_short, PBm, indexing='ij')
    nPBm_g = np.zeros((len(tgrid_coarse_short), len(PBm)))
    for ind, nPBm_Vec in enumerate(nPBm_ctVec[:tmax_ind + 1]):
        nPBm_g[ind, :] = nPBm_Vec[:]

    fig1, ax1 = plt.subplots()
    q1 = ax1.pcolormesh(tc_g, PIm_g, nPIm_g)
    ax1.set_title('Impurity Momentum Magnitude Distribution ' + r'($n_{|\vec{P_{I}}|}(t)$)')
    ax1.set_ylabel(r'$|\vec{P_{I}}|$')
    ax1.set_xlabel(r'$t$')
    fig1.colorbar(q1, ax=ax1, extend='both')

    fig2, ax2 = plt.subplots()
    q2 = ax2.pcolormesh(tc_g, PBm_g, nPBm_g)
    ax2.set_title('Phonon Momentum Magnitude Distribution ' + r'($n_{|\vec{P_{B}}|}(t)$)')
    ax2.set_ylabel(r'$|\vec{P_{B}}|$')
    ax2.set_xlabel(r'$t$')
    fig2.colorbar(q2, ax=ax2, extend='both')

    fig1.savefig(figdatapath + '/nPIm_2D.pdf')
    fig2.savefig(figdatapath + '/nPBm_2D.pdf')

    # plt.show()

    # # 2D SLICE ANIMATION

    # nPI

    PI_x_g, PI_z_g = np.meshgrid(PI_x, PI_z, indexing='ij')
    fig1, ax1 = plt.subplots()

    # vmin = np.min(nPI_xz_slice_st)
    # vmax = np.max(nPI_xz_slice_st)

    vmin = 1
    vmax = 0
    for ind, vec in enumerate(nPI_xz_slice_ctVec):
        if np.min(vec) < vmin:
            vmin = np.min(vec)
        if np.max(vec) > vmax:
            vmax = np.max(vec)

    quad = ax1.pcolormesh(PI_x_g, PI_z_g, nPI_xz_slice_ctVec[1][:-1, :-1], vmin=vmin, vmax=vmax)
    time_text = ax1.text(0.85, 0.9, 't: {:.2f}'.format(tgrid_coarse[0]), transform=ax1.transAxes, color='r')

    ax1.set_title('Impurity Longitudinal Momentum Distribution ' + r'($n_{P_{I,x},P_{I,z}}(t)$)')
    ax1.set_ylabel(r'$P_{I,z}$')
    ax1.set_xlabel(r'$P_{I,x}$')
    ax1.grid(True, linewidth=0.5)
    fig1.colorbar(quad, ax=ax1, extend='both')

    def animate1(i):
        quad.set_array(nPI_xz_slice_ctVec[i][:-1, :-1].ravel())
        time_text.set_text('t: {:.2f}'.format(tgrid_coarse[i]))

    anim1 = FuncAnimation(fig1, animate1, interval=300, frames=range(tmax_ind + 1), blit=False)
    anim1.save(figdatapath + '/nPI_xz.gif', writer='imagemagick')

    #

    PI_x_g, PI_y_g = np.meshgrid(PI_x, PI_y, indexing='ij')
    fig2, ax2 = plt.subplots()

    # vmin = np.min(nPI_xy_slice_st)
    # vmax = np.max(nPI_xy_slice_st)

    vmin = 1
    vmax = 0
    for ind, vec in enumerate(nPI_xy_slice_ctVec):
        if np.min(vec) < vmin:
            vmin = np.min(vec)
        if np.max(vec) > vmax:
            vmax = np.max(vec)

    quad = ax2.pcolormesh(PI_x_g, PI_y_g, nPI_xy_slice_ctVec[1][:-1, :-1], vmin=vmin, vmax=vmax)
    time_text = ax2.text(0.85, 0.9, 't: {:.2f}'.format(tgrid_coarse[0]), transform=ax2.transAxes, color='r')

    ax2.set_title('Impurity Transverse Momentum Distribution ' + r'($n_{P_{I,x},P_{I,y}}(t)$)')
    ax2.set_ylabel(r'$P_{I,y}$')
    ax2.set_xlabel(r'$P_{I,x}$')
    ax2.grid(True, linewidth=0.5)
    fig2.colorbar(quad, ax=ax2, extend='both')

    def animate2(i):
        quad.set_array(nPI_xy_slice_ctVec[i][:-1, :-1].ravel())
        time_text.set_text('t: {:.2f}'.format(tgrid_coarse[i]))

    anim2 = FuncAnimation(fig2, animate2, interval=300, frames=range(tmax_ind + 1), blit=False)
    anim2.save(figdatapath + '/nPI_xy.gif', writer='imagemagick')

    # nPB

    PB_x_g, PB_z_g = np.meshgrid(PB_x, PB_z, indexing='ij')
    fig3, ax3 = plt.subplots()

    # vmin = np.min(nPB_xz_slice_st)
    # vmax = np.max(nPB_xz_slice_st)

    vmin = 1
    vmax = 0
    for ind, vec in enumerate(nPB_xz_slice_ctVec):
        if np.min(vec) < vmin:
            vmin = np.min(vec)
        if np.max(vec) > vmax:
            vmax = np.max(vec)

    quad = ax3.pcolormesh(PB_x_g, PB_z_g, nPB_xz_slice_ctVec[1][:-1, :-1], vmin=vmin, vmax=vmax)
    time_text = ax3.text(0.85, 0.9, 't: {:.2f}'.format(tgrid_coarse[0]), transform=ax3.transAxes, color='r')

    ax3.set_title('Phonon Longitudinal Momentum Distribution ' + r'($n_{P_{B,x},P_{B,z}}(t)$)')
    ax3.set_ylabel(r'$P_{B,z}$')
    ax3.set_xlabel(r'$P_{B,x}$')
    ax3.grid(True, linewidth=0.5)
    fig3.colorbar(quad, ax=ax3, extend='both')

    def animate3(i):
        quad.set_array(nPB_xz_slice_ctVec[i][:-1, :-1].ravel())
        time_text.set_text('t: {:.2f}'.format(tgrid_coarse[i]))

    anim3 = FuncAnimation(fig3, animate3, interval=300, frames=range(tmax_ind + 1), blit=False)
    anim3.save(figdatapath + '/nPB_xz.gif', writer='imagemagick')

    #

    PB_x_g, PB_y_g = np.meshgrid(PB_x, PB_y, indexing='ij')
    fig4, ax4 = plt.subplots()

    # vmin = np.min(nPB_xy_slice_st)
    # vmax = np.max(nPB_xy_slice_st)

    vmin = 1
    vmax = 0
    for ind, vec in enumerate(nPB_xy_slice_ctVec):
        if np.min(vec) < vmin:
            vmin = np.min(vec)
        if np.max(vec) > vmax:
            vmax = np.max(vec)

    quad = ax4.pcolormesh(PB_x_g, PB_y_g, nPB_xy_slice_ctVec[1][:-1, :-1], vmin=vmin, vmax=vmax)
    time_text = ax4.text(0.85, 0.9, 't: {:.2f}'.format(tgrid_coarse[0]), transform=ax4.transAxes, color='r')

    ax4.set_title('Phonon Transverse Momentum Distribution ' + r'($n_{P_{B,x},P_{B,y}}(t)$)')
    ax4.set_ylabel(r'$P_{B,y}$')
    ax4.set_xlabel(r'$P_{B,x}$')
    ax4.grid(True, linewidth=0.5)
    fig4.colorbar(quad, ax=ax4, extend='both')

    def animate4(i):
        quad.set_array(nPB_xy_slice_ctVec[i][:-1, :-1].ravel())
        time_text.set_text('t: {:.2f}'.format(tgrid_coarse[i]))

    anim4 = FuncAnimation(fig4, animate4, interval=300, frames=range(tmax_ind + 1), blit=False)
    anim4.save(figdatapath + '/nPB_xy.gif', writer='imagemagick')

    # plt.draw()
    # plt.show()

    # # 2D SLICE STATIC

    PI_x_g, PI_z_g = np.meshgrid(PI_x_st, PI_z_st, indexing='ij')
    fig1, ax1 = plt.subplots()
    q1 = ax1.pcolormesh(PI_x_g, PI_z_g, nPI_xz_slice_st)
    ax1.set_title('Static Impurity Longitudinal Momentum Distribution ' + r'($n_{P_{I,x},P_{I,z},st}$)')
    ax1.set_ylabel(r'$P_{I,z}$')
    ax1.set_xlabel(r'$P_{I,x}$')
    ax1.grid(True, linewidth=0.5)
    fig1.colorbar(q1, ax=ax1, extend='both')

    PI_x_g, PI_y_g = np.meshgrid(PI_x_st, PI_y_st, indexing='ij')
    fig2, ax2 = plt.subplots()
    q2 = ax2.pcolormesh(PI_x_g, PI_y_g, nPI_xy_slice_st)
    ax2.set_title('Static Impurity Transverse Momentum Distribution ' + r'($n_{P_{I,x},P_{I,y},st}$)')
    ax2.set_ylabel(r'$P_{I,y}$')
    ax2.set_xlabel(r'$P_{I,x}$')
    ax2.grid(True, linewidth=0.5)
    fig2.colorbar(q2, ax=ax2, extend='both')

    PB_x_g, PB_z_g = np.meshgrid(PB_x_st, PB_z_st, indexing='ij')
    fig3, ax3 = plt.subplots()
    q3 = ax3.pcolormesh(PB_x_g, PB_z_g, nPB_xz_slice_st)
    ax3.set_title('Static Phonon Longitudinal Momentum Distribution ' + r'($n_{P_{B,x},P_{B,z},st}$)')
    ax3.set_ylabel(r'$P_{B,z}$')
    ax3.set_xlabel(r'$P_{B,x}$')
    ax3.grid(True, linewidth=0.5)
    fig3.colorbar(q3, ax=ax3, extend='both')

    PB_x_g, PB_y_g = np.meshgrid(PB_x_st, PB_y_st, indexing='ij')
    fig4, ax4 = plt.subplots()
    q4 = ax4.pcolormesh(PB_x_g, PB_y_g, nPB_xy_slice_st)
    ax4.set_title('Static Phonon Transverse Momentum Distribution ' + r'($n_{P_{B,x},P_{B,y},st}$)')
    ax4.set_ylabel(r'$P_{B,y}$')
    ax4.set_xlabel(r'$P_{B,x}$')
    ax4.grid(True, linewidth=0.5)
    fig4.colorbar(q4, ax=ax4, extend='both')

    fig1.savefig(figdatapath + '/nPI_xz_st.pdf')
    fig2.savefig(figdatapath + '/nPI_xy_st.pdf')
    fig3.savefig(figdatapath + '/nPB_xz_st.pdf')
    fig4.savefig(figdatapath + '/nPB_xy_st.pdf')

    # plt.show()
