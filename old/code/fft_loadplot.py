import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os


# # Initialization

matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})
NiceBlue = '#0087BD'
NiceRed = '#C40233'
NiceGreen = '#009F6B'
NiceYellow = '#FFD300'
fontsize = 16
# load data
Color = NiceRed
colorList = [NiceBlue, NiceRed, NiceGreen, NiceYellow, 'c', 'm', 'k', 'g', 'r']

datapath = os.path.dirname(os.path.realpath(__file__)) + '/fftdata'


aIBi_Vals = [-5, -2, -1, 1, 2, 5]
Pc = 0.8925539653008207
PVals = np.linspace(0, 0.95 * Pc, 6)

# aIBi Figure

figA, axA = plt.subplots(nrows=3, ncols=3)

Nph_Vec_aIBi = np.zeros(len(aIBi_Vals))
NphX_Vec_aIBi = np.zeros(len(aIBi_Vals))
nPBM1_Vec_aIBi = np.zeros(len(aIBi_Vals))
b2M1_Vec_aIBi = np.zeros(len(aIBi_Vals))
CTan_Vec_aIBi = np.zeros(len(aIBi_Vals))
FWHM_Vec_aIBi = np.zeros(len(aIBi_Vals))

for ind, aIBi in enumerate(aIBi_Vals):
    P = 0.85
    DP, Nph, Nph_x, nPB_Tot, nPB_Mom1, beta2_kz_Mom1, FWHM, C_Tan, x, y, z, nx_x_norm, nx_y_norm, nx_z_norm, kx, ky, kz, nPB_kx, nPB_ky, nPB_kz, PI_z, nPI_z = np.loadtxt(datapath + '/3Ddist_aIBi_{:.2f}_P_{:.2f}.dat'.format(aIBi, P), unpack=True)

    Nph_Vec_aIBi[ind] = Nph[0]; NphX_Vec_aIBi[ind] = Nph_x[0]; nPBM1_Vec_aIBi[ind] = nPB_Mom1[0]; b2M1_Vec_aIBi[ind] = beta2_kz_Mom1[0]; CTan_Vec_aIBi[ind] = C_Tan[0]; FWHM_Vec_aIBi[ind] = FWHM[0]

    axA[0, 0].plot(kx, nPB_kx, label=r'$a_{IB}^{-1}=$%.2f' % (aIBi), color=colorList[ind], linestyle='-')
    axA[0, 0].plot(np.zeros(len(kx)), np.linspace(0, np.exp(-1 * Nph[0]), len(kx)), color=colorList[ind], linestyle=':')

    axA[0, 1].plot(ky, nPB_ky, label=r'$a_{IB}^{-1}=$%.2f' % (aIBi), color=colorList[ind], linestyle='-')
    axA[0, 1].plot(np.zeros(len(ky)), np.linspace(0, np.exp(-1 * Nph[0]), len(ky)), color=colorList[ind], linestyle=':')

    axA[0, 2].plot(kz, nPB_kz, label=r'$a_{IB}^{-1}=$%.2f' % (aIBi), color=colorList[ind], linestyle='-')
    axA[0, 2].plot(np.zeros(len(kz)), np.linspace(0, np.exp(-1 * Nph[0]), len(kz)), color=colorList[ind], linestyle=':')

    axA[1, 0].plot(x, nx_x_norm, label=r'$a_{IB}^{-1}=$%.2f' % (aIBi), color=colorList[ind], linestyle='-')
    axA[1, 1].plot(y, nx_y_norm, label=r'$a_{IB}^{-1}=$%.2f' % (aIBi), color=colorList[ind], linestyle='-')
    axA[1, 2].plot(z, nx_z_norm, label=r'$a_{IB}^{-1}=$%.2f' % (aIBi), color=colorList[ind], linestyle='-')

axA[2, 0].plot(np.array(aIBi_Vals), Nph_Vec_aIBi, label=r'$|\beta_{\vec{k}}|^2$', color=colorList[0], linestyle='-')
axA[2, 0].plot(np.array(aIBi_Vals), NphX_Vec_aIBi, label=r'$n(\vec{x})$', color=colorList[1], linestyle='-')
axA[2, 1].plot(np.array(aIBi_Vals), b2M1_Vec_aIBi, label=r'$|\beta_{\vec{k}}|^2$', color=colorList[0], linestyle='-')
axA[2, 1].plot(np.array(aIBi_Vals), nPBM1_Vec_aIBi, label=r'$n_{\vec{P_B}}$', color=colorList[1], linestyle='-')
axA[2, 2].plot(np.array(aIBi_Vals), FWHM_Vec_aIBi, label='FWHM', color=colorList[0], linestyle='-')
axA[2, 2].plot(np.array(aIBi_Vals), CTan_Vec_aIBi, label='Contact', color=colorList[1], linestyle='-')

# labels and modifications

axA[0, 0].set_title('Total Phonon Momentum Distribution (x)'); axA[0, 0].legend()
axA[0, 0].set_ylabel(r'$n_{\vec{P_{B,x}}}$'); axA[0, 0].set_xlabel(r'$P_{B,x}$')
axA[0, 0].set_xlim([-10, 10])
# axA[0, 0].set_xscale('log'); axA[0, 0].set_yscale('log')

axA[0, 1].set_title('Total Phonon Momentum Distribution (y)'); axA[0, 1].legend()
axA[0, 1].set_ylabel(r'$n_{\vec{P_{B,y}}}$'); axA[0, 1].set_xlabel(r'$P_{B,y}$')
axA[0, 1].set_xlim([-10, 10])
# axA[0, 1].set_xscale('log'); axA[0, 1].set_yscale('log')

axA[0, 2].set_title('Total Phonon Momentum Distribution (z)'); axA[0, 2].legend()
axA[0, 2].set_ylabel(r'$n_{\vec{P_{B,z}}}$'); axA[0, 2].set_xlabel(r'$P_{B,z}$')
axA[0, 2].set_xlim([-10, 10])
# axA[0, 2].set_xscale('log'); axA[0, 2].set_yscale('log')

axA[1, 0].set_title('Phonon Normalized Density Distribution (x)'); axA[1, 0].legend()
axA[1, 0].set_ylabel(r'$\frac{n(\vec{x})_{x}}{N_{ph}}$'); axA[1, 0].set_xlabel(r'$x$')
# axA[1, 0].set_xlim([-0.5, 0.5])
axA[1, 0].set_xscale('log'); axA[1, 0].set_yscale('log')

axA[1, 1].set_title('Phonon Normalized Density Distribution (y)'); axA[1, 1].legend()
axA[1, 1].set_ylabel(r'$\frac{n(\vec{x})_{y}}{N_{ph}}$'); axA[1, 1].set_xlabel(r'$y$')
# axA[1, 1].set_xlim([-0.5, 0.5])
axA[1, 1].set_xscale('log'); axA[1, 1].set_yscale('log')

axA[1, 2].set_title('Phonon Normalized Density Distribution (z)'); axA[1, 2].legend()
axA[1, 2].set_ylabel(r'$\frac{n(\vec{x})_{z}}{N_{ph}}$'); axA[1, 2].set_xlabel(r'$z$')
# axA[1, 2].set_xlim([-0.5, 0.5])
axA[1, 2].set_xscale('log'); axA[1, 2].set_yscale('log')

axA[2, 0].set_title('Phonon Number'); axA[2, 0].legend()
axA[2, 0].set_ylabel(r'$N_{B}$'); axA[2, 0].set_xlabel(r'$a_{IB}^{-1}$')

axA[2, 1].set_title('Average Total Phonon Momentum (z)'); axA[2, 1].legend()
axA[2, 1].set_ylabel(r'$<P_{B,z}>$'); axA[2, 1].set_xlabel(r'$a_{IB}^{-1}$')

axA[2, 2].set_title('FWHM and Tan Tail Contact (z)'); axA[2, 2].legend()
axA[2, 2].set_xlabel(r'$a_{IB}^{-1}$')

# figA.delaxes(axA[2, 2])
# figA.tight_layout(pad=0.9, w_pad=0.9, h_pad=0.9)
figA.subplots_adjust(wspace=0.2, hspace=0.4)
# figA.savefig(datapath + '/figures/staticDist_P_%.2f.pdf' % (0.85))


# DP Figure

figD, axD = plt.subplots(nrows=3, ncols=3)

Nph_Vec_P = np.zeros(len(PVals))
NphX_Vec_P = np.zeros(len(PVals))
nPBM1_Vec_P = np.zeros(len(PVals))
b2M1_Vec_P = np.zeros(len(PVals))
CTan_Vec_P = np.zeros(len(PVals))
FWHM_Vec_P = np.zeros(len(PVals))

for ind, P in enumerate(PVals):
    aIBi = -2
    DP, Nph, Nph_x, nPB_Tot, nPB_Mom1, beta2_kz_Mom1, FWHM, C_Tan, x, y, z, nx_x_norm, nx_y_norm, nx_z_norm, kx, ky, kz, nPB_kx, nPB_ky, nPB_kz, PI_z, nPI_z = np.loadtxt(datapath + '/3Ddist_aIBi_{:.2f}_P_{:.2f}.dat'.format(aIBi, P), unpack=True)
    Nph_Vec_P[ind] = Nph[0]; NphX_Vec_P[ind] = Nph_x[0]; nPBM1_Vec_P[ind] = nPB_Mom1[0]; b2M1_Vec_P[ind] = beta2_kz_Mom1[0]; CTan_Vec_P[ind] = C_Tan[0]; FWHM_Vec_P[ind] = FWHM[0]

    axD[0, 0].plot(kx, nPB_kx, label=r'$P=$%.2f' % (P), color=colorList[ind], linestyle='-')
    axD[0, 0].plot(np.zeros(len(kx)), np.linspace(0, np.exp(-1 * Nph[0]), len(kx)), color=colorList[ind], linestyle=':')

    axD[0, 1].plot(ky, nPB_ky, label=r'$P=$%.2f' % (P), color=colorList[ind], linestyle='-')
    axD[0, 1].plot(np.zeros(len(ky)), np.linspace(0, np.exp(-1 * Nph[0]), len(ky)), color=colorList[ind], linestyle=':')

    axD[0, 2].plot(kz, nPB_kz, label=r'$P=$%.2f' % (P), color=colorList[ind], linestyle='-')
    axD[0, 2].plot(np.zeros(len(kz)), np.linspace(0, np.exp(-1 * Nph[0]), len(kz)), color=colorList[ind], linestyle=':')

    axD[1, 0].plot(x, nx_x_norm, label=r'$P=$%.2f' % (P), color=colorList[ind], linestyle='-')
    axD[1, 1].plot(y, nx_y_norm, label=r'$P=$%.2f' % (P), color=colorList[ind], linestyle='-')
    axD[1, 2].plot(z, nx_z_norm, label=r'$P=$%.2f' % (P), color=colorList[ind], linestyle='-')

axD[2, 0].plot(PVals, Nph_Vec_P, label=r'$|\beta_{\vec{k}}|^2$', color=colorList[0], linestyle='-')
axD[2, 0].plot(PVals, NphX_Vec_P, label=r'$n(\vec{x})$', color=colorList[1], linestyle='-')
axD[2, 1].plot(PVals, b2M1_Vec_P, label=r'$|\beta_{\vec{k}}|^2$', color=colorList[0], linestyle='-')
axD[2, 1].plot(PVals, nPBM1_Vec_P, label=r'$n_{\vec{P_B}}$', color=colorList[1], linestyle='-')
axD[2, 2].plot(PVals, FWHM_Vec_P, label='FWHM', color=colorList[0], linestyle='-')
axD[2, 2].plot(PVals, CTan_Vec_P, label='Contact', color=colorList[1], linestyle='-')

# labels and modifications

axD[0, 0].set_title('Total Phonon Momentum Distribution (x)'); axD[0, 0].legend()
axD[0, 0].set_ylabel(r'$n_{\vec{P_{B,x}}}$'); axD[0, 0].set_xlabel(r'$P_{B,x}$')
axD[0, 0].set_xlim([-5, 5])
# axD[0, 0].set_xscale('log'); axD[0, 0].set_yscale('log')

axD[0, 1].set_title('Total Phonon Momentum Distribution (y)'); axD[0, 1].legend()
axD[0, 1].set_ylabel(r'$n_{\vec{P_{B,y}}}$'); axD[0, 1].set_xlabel(r'$P_{B,y}$')
axD[0, 1].set_xlim([-5, 5])
# axD[0, 1].set_xscale('log'); axD[0, 1].set_yscale('log')

axD[0, 2].set_title('Total Phonon Momentum Distribution (z)'); axD[0, 2].legend()
axD[0, 2].set_ylabel(r'$n_{\vec{P_{B,z}}}$'); axD[0, 2].set_xlabel(r'$P_{B,z}$')
axD[0, 2].set_xlim([-5, 5])
# axD[0, 2].set_xscale('log'); axD[0, 2].set_yscale('log')

axD[1, 0].set_title('Phonon Normalized Density Distribution (x)'); axD[1, 0].legend()
axD[1, 0].set_ylabel(r'$\frac{n(\vec{x})_{x}}{N_{ph}}$'); axD[1, 0].set_xlabel(r'$x$')
# axD[1, 0].set_xlim([-0.5, 0.5])
axD[1, 0].set_xscale('log'); axD[1, 0].set_yscale('log')

axD[1, 1].set_title('Phonon Normalized Density Distribution (y)'); axD[1, 1].legend()
axD[1, 1].set_ylabel(r'$\frac{n(\vec{x})_{y}}{N_{ph}}$'); axD[1, 1].set_xlabel(r'$y$')
# axD[1, 1].set_xlim([-0.5, 0.5])
axD[1, 1].set_xscale('log'); axD[1, 1].set_yscale('log')

axD[1, 2].set_title('Phonon Normalized Density Distribution (z)'); axD[1, 2].legend()
axD[1, 2].set_ylabel(r'$\frac{n(\vec{x})_{z}}{N_{ph}}$'); axD[1, 2].set_xlabel(r'$z$')
# axD[1, 2].set_xlim([-0.5, 0.5])
axD[1, 2].set_xscale('log'); axD[1, 2].set_yscale('log')

axD[2, 0].set_title('Phonon Number'); axD[2, 0].legend()
axD[2, 0].set_ylabel(r'$N_{B}$'); axD[2, 0].set_xlabel(r'$P$')

axD[2, 1].set_title('Average Total Phonon Momentum (z)'); axD[2, 1].legend()
axD[2, 1].set_ylabel(r'$<P_{B,z}>$'); axD[2, 1].set_xlabel(r'$P$')

axD[2, 2].set_title('FWHM and Tan Tail Contact (z)'); axD[2, 2].legend()
axD[2, 2].set_xlabel(r'$P$')


# figD.delaxes(axD[2, 2])
# figD.tight_layout(pad=0.9, w_pad=0.9, h_pad=0.9)
figD.subplots_adjust(wspace=0.2, hspace=0.4)
# figD.savefig(datapath + '/figures/staticDist_aIBi_%.2f.pdf' % (-2))

#
plt.show()
