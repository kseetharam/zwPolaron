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
    (dx, dy, dz) = (4, 4, 4)

    NGridPoints = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

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

    # fig, ax = plt.subplots()
    # line = ax.plot([], [], color='k', lw=2)[0]
    # time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    # ax.set_xlim([np.min(PIm), np.max(PIm)])
    # ax.set_ylim([0, 0.0025])
    # ax.set_title('Impurity Momentum Distribution (Magnitude)')
    # ax.set_ylabel(r'$n_{|\vec{P_{I}}|}$')
    # ax.set_xlabel(r'$|\vec{P_{I}}|$')

    # def init():
    #     line.set_data([], [])
    #     time_text.set_text('')
    #     return line,

    # def animate(i):
    #     line.set_data(PIm, nPIm_ctVec[i])
    #     time_text.set_text('t: {0}'.format(tgrid_coarse[i]))
    #     return tuple([line]) + tuple([time_text])

    # anim = FuncAnimation(fig, animate, init_func=init, interval=500, frames=len(tgrid_coarse) - 1)
    # plt.draw()
    # plt.show()

    fig, ax = plt.subplots()
    line = ax.plot(PIm, nPIm_ctVec[0], color='k', lw=2)[0]
    time_text = ax.text(0.85, 0.9, 't: {0}'.format(tgrid_coarse[0]), transform=ax.transAxes, color='b')
    norm_text = ax.text(0.02, 0.9, r'$\int n_{|\vec{P_{I}}|} |\vec{P_{I}}| = $' + '{0}'.format(tgrid_coarse[0]), transform=ax.transAxes, color='g')
    ax.set_xlim([0, np.max(PIm)])
    ax.set_ylim([0, 0.005])
    ax.set_title('Impurity Momentum Distribution (Magnitude)')
    ax.set_ylabel(r'$n_{|\vec{P_{I}}|}$')
    ax.set_xlabel(r'$|\vec{P_{I}}|$')

    def animate(i):
        line.set_ydata(nPIm_ctVec[i])
        time_text.set_text('t: {0}'.format(tgrid_coarse[i]))
        norm_text.set_text(r'$\int n_{|\vec{P_{I}}|} d|\vec{P_{I}}| = $' + '{:.4f}'.format(nPIm_Tot_ctVec[i]))

    anim = FuncAnimation(fig, animate, interval=500, frames=len(tgrid_coarse) - 1)

    animdatapath = dirpath + '/data_qdynamics' + '/cart/realtime' + '/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}'.format(NGridPoints, P, aIBi) + '/animations'
    if os.path.isdir(animdatapath) is False:
        os.mkdir(animdatapath)

    # anim.save(animdatapath + '/nPIm.gif', writer='imagemagick')

    plt.draw()
    plt.show()
