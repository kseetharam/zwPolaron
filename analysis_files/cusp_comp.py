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

    Lx_List = np.array([120, 120, 120, 80])
    dx_List = np.array([4, 2, 1, 0.5])
    NG_List = []

    for ind, dx in enumerate(dx_List):
        NG_List.append((1 + 2 * Lx_List[ind] / dx)**3)

    dirpath = os.path.dirname(os.path.realpath(__file__))

    fig, ax = plt.subplots()

    for ind, NG in enumerate(NG_List):
        static_datapath = dirpath + '/data_static' + '/cart' + '/NGridPoints_{:.2E}/P_{:.3f}_aIBi_{:.2f}'.format(NG, P, aIBi)
        NGridPoints_st, k_max_st, P_st, aIBi_st, mI_st, mB_st, n0_st, gBB_st, nu_stonst_st, gIB_st, Pcrit_st, aSi_st, DP_st, PB_Val_st, En_st, eMass_st, Nph_st, Nph_xyz_st, Z_factor_st, nxyz_Tot_st, nPB_Tot_st, nPBm_Tot_st, nPIm_Tot_st, nPB_Mom1_st, beta2_kz_Mom1_st, nPB_deltaK0_st, FWHM_st = np.loadtxt(static_datapath + '/metrics.dat', unpack=True)
        x_st, y_st, z_st, nxyz_x_st, nxyz_y_st, nxyz_z_st, nxyz_x_slice_st, nxyz_y_slice_st, nxyz_z_slice_st = np.loadtxt(static_datapath + '/pos_xyz.dat', unpack=True)
        PB_x_st, PB_y_st, PB_z_st, nPB_x_st, nPB_y_st, nPB_z_st, nPB_x_slice_st, nPB_y_slice_st, nPB_z_slice_st, PI_x_st, PI_y_st, PI_z_st, nPI_x_st, nPI_y_st, nPI_z_st, nPI_x_slice_st, nPI_y_slice_st, nPI_z_slice_st = np.loadtxt(static_datapath + '/mom_xyz.dat', unpack=True)
        with open(static_datapath + '/cont_xyz_data.pickle', 'rb') as f:
            [nxyz_xz_slice_st, nxyz_xy_slice_st, nPB_xz_slice_st, nPB_xy_slice_st, nPI_xz_slice_st, nPI_xy_slice_st] = pickle.load(f)
        PBm_Vec_st, nPBm_Vec_st, PIm_Vec_st, nPIm_Vec_st = np.loadtxt(static_datapath + '/mom_mag.dat', unpack=True)
        ax.plot(PIm_Vec_st, nPIm_Vec_st, label=r'$|k|_{max}=$' + '{:.2f}'.format(k_max_st))

    ax.legend()
    ax.set_title('Static Impurity Momentum Magnitude Distribution')
    ax.set_ylabel(r'$n_{|\vec{P_{I}}|}$')
    ax.set_xlabel(r'$|\vec{P_{I}}|$')
    plt.show()
