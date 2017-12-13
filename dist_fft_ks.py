import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from timeit import default_timer as timer
from polaron_functions_cart import *
import Grid
import polrabi.staticfm as fm
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.special import expit


matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})


def staticDistCalc(gridargs, params, datapath):
    [xgrid, kgrid, kFgrid] = gridargs
    [P, aIBi, aSi, DP, mI, mB, n0, gBB, nuV] = params
    bparams = [aIBi, aSi, DP, mI, mB, n0, gBB]

    # unpack grid args
    x = xgrid.getArray('x'); y = xgrid.getArray('y'); z = xgrid.getArray('z')
    (Nx, Ny, Nz) = (len(x), len(y), len(z))
    dx = xgrid.arrays_diff['x']; dy = xgrid.arrays_diff['y']; dz = xgrid.arrays_diff['z']

    kxF = kFgrid.getArray('kx'); kyF = kFgrid.getArray('ky'); kzF = kFgrid.getArray('kz')

    kx = kgrid.getArray('kx'); ky = kgrid.getArray('ky'); kz = kgrid.getArray('kz')
    dkx = kgrid.arrays_diff['kx']; dky = kgrid.arrays_diff['ky']; dkz = kgrid.arrays_diff['kz']

    # generation
    xg, yg, zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
    kxg, kyg, kzg = np.meshgrid(kx, ky, kz, indexing='ij', sparse=True)
    kxFg, kyFg, kzFg = np.meshgrid(kxF, kyF, kzF, indexing='ij', sparse=True)

    beta2_kxkykz = np.abs(BetaK(kxFg, kyFg, kzFg, *bparams))**2
    mask = np.isnan(beta2_kxkykz); beta2_kxkykz[mask] = 0

    decay_length = 5
    decay_xyz = np.exp(-1 * (xg**2 + yg**2 + zg**2) / (2 * decay_length**2))

    # Fourier transform and slice
    amp_beta_xyz_0 = np.fft.fftn(np.sqrt(beta2_kxkykz))
    amp_beta_xyz = np.fft.fftshift(amp_beta_xyz_0) * dkx * dky * dkz

    # Calculate Nph
    Nph = np.real(np.sum(beta2_kxkykz) * dkx * dky * dkz)
    Nph_x = np.real(np.sum(np.abs(amp_beta_xyz)**2) * dx * dy * dz * (2 * np.pi)**(-3))

    # Fourier transform and slice
    beta2_xyz_preshift = np.fft.fftn(beta2_kxkykz)
    beta2_xyz = np.fft.fftshift(beta2_xyz_preshift) * dkx * dky * dkz
    beta2_z = beta2_xyz[Nx // 2 + 1, Ny // 2 + 1, :]

    # Exponentiate, slice
    fexp = (np.exp(beta2_xyz - Nph) - np.exp(-Nph)) * decay_xyz
    # fexp = np.exp(beta2_xyz - Nph) - np.exp(-Nph)
    fexp_z = fexp[Nx // 2 + 1, Ny // 2 + 1, :]

    # Inverse Fourier transform
    nPB_preshift = np.fft.ifftn(fexp) * 1 / (dkx * dky * dkz)
    nPB = np.fft.fftshift(nPB_preshift)
    nPB_deltaK0 = np.exp(-Nph)

    # Integrating out y and z

    beta2_kz = np.sum(np.abs(beta2_kxkykz), axis=(0, 1)) * dkx * dky
    nPB_kx = np.sum(np.abs(nPB), axis=(1, 2)) * dky * dkz
    nPB_ky = np.sum(np.abs(nPB), axis=(0, 2)) * dkx * dkz
    nPB_kz = np.sum(np.abs(nPB), axis=(0, 1)) * dkx * dky
    nx_x = np.sum(np.abs(amp_beta_xyz)**2, axis=(1, 2)) * dy * dz
    nx_y = np.sum(np.abs(amp_beta_xyz)**2, axis=(0, 2)) * dx * dz
    nx_z = np.sum(np.abs(amp_beta_xyz)**2, axis=(0, 1)) * dx * dy
    nx_x_norm = np.real(nx_x / Nph_x); nx_y_norm = np.real(nx_y / Nph_x); nx_z_norm = np.real(nx_z / Nph_x)

    nPB_Tot = np.sum(np.abs(nPB) * dkx * dky * dkz) + nPB_deltaK0
    nPB_Mom1 = np.dot(np.abs(nPB_kz), kz * dkz)
    beta2_kz_Mom1 = np.dot(np.abs(beta2_kz), kzF * dkz)

    # beta2_kz_Mom1_2 = np.sum(np.abs(beta2_kxkykz) * kzFg) * dkx * dky * dkz
    # print(beta2_kz_Mom1_2)

    # Flipping domain for P_I instead of P_B so now nPB(PI) -> nPI

    PI_x = -1 * kx
    PI_y = -1 * ky
    PI_z = P - kz

    # Calculate FWHM and Tan tail contact parameter

    PI_z_ord = np.flip(PI_z, 0)
    nPI_z = np.flip(np.real(nPB_kz), 0)

    if np.abs(np.max(nPI_z) - np.min(nPI_z)) < 1e-2:
        FWHM = 0
        # C_Tan = 0

    else:
        D = nPI_z - np.max(nPI_z) / 2
        indices = np.where(D > 0)[0]
        FWHM = PI_z_ord[indices[-1]] - PI_z_ord[indices[0]]
    #     tail_dom = PI_z_ord[indices[-1] + 1:]
    #     tail_ran = nPI_z[indices[-1] + 1:]

    #     def Tanfunc(p, C): return C * (p**-4)

    #     def logTanfunc(p, LC): return LC - 4 * np.log(p)

    #     popt, pcov = curve_fit(Tanfunc, tail_dom, tail_ran)
    #     Lpopt, Lpcov = curve_fit(logTanfunc, tail_dom, tail_ran)
    #     C_Tan = popt[0]
    #     LC_Tan = Lpopt[0]
    #     print(C_Tan, np.sqrt(np.diag(pcov))[0])

    # Calculate magnitude distribution nPB(P) and nPI(P) where P_IorB = sqrt(Px^2 + Py^2 + Pz^2)

    PB = np.sqrt(kxg**2 + kyg**2 + kzg**2)
    PI = np.sqrt((-kxg)**2 + (-kyg)**2 + (P - kzg)**2)
    PB_flat = PB.reshape(PB.size)
    PI_flat = PI.reshape(PI.size)
    nPB_flat = nPB.reshape(nPB.size)

    PB_unique, PB_uind, PB_ucounts = np.unique(PB_flat, return_inverse=True, return_counts=True)
    PI_unique, PI_uind, PI_ucounts = np.unique(PI_flat, return_inverse=True, return_counts=True)
    nPBm_unique = np.zeros(PB_unique.size)
    nPIm_unique = np.zeros(PI_unique.size)

    for ind, val in enumerate(np.abs(nPB_flat) * dkx * dky * dkz):
        PB_index = PB_uind[ind]
        PI_index = PI_uind[ind]
        nPBm_unique[PB_index] += val
        nPIm_unique[PI_index] += val

    nPBm_cum = np.zeros(PB_unique.size)
    nPIm_cum = np.zeros(PI_unique.size)
    for ind in range(PB_unique.size):
        if ind == 0:
            continue
        nPBm_cum[ind] = nPBm_cum[ind - 1] + nPBm_unique[ind]

    for ind in range(PI_unique.size):
        if ind == 0:
            continue
        nPIm_cum[ind] = nPIm_cum[ind - 1] + nPIm_unique[ind]

    # nPBm_tck = interpolate.splrep(PB_unique, nPBm_cum, k=3, s=1)
    # nPIm_tck = interpolate.splrep(PI_unique, nPIm_cum, k=3, s=1)

    # PBm_Vec = np.linspace(0, np.max(PB_unique), 100)
    # PIm_Vec = np.linspace(0, np.max(PI_unique), 100)
    # nPBm_cum_Vec = interpolate.splev(PBm_Vec, nPBm_tck, der=0)
    # nPIm_cum_Vec = interpolate.splev(PIm_Vec, nPIm_tck, der=0)

    # nPBm_Vec = interpolate.splev(PBm_Vec, nPBm_tck, der=1)
    # nPIm_Vec = interpolate.splev(PIm_Vec, nPIm_tck, der=1)

    # dPBm = PBm_Vec[1] - PBm_Vec[0]
    # dPIm = PIm_Vec[1] - PIm_Vec[0]
    # nPBm_Tot = np.sum(nPBm_Vec * dPBm) + nPB_deltaK0
    # nPIm_Tot = np.sum(nPIm_Vec * dPIm) + nPB_deltaK0

    # split interpolation

    PB_mask = PB_unique < nuV
    PI_mask = PI_unique < nuV

    PB_unique_S = PB_unique[PB_mask]
    PB_unique_L = PB_unique[np.logical_not(PB_mask)]
    nPBm_cum_S = nPBm_cum[PB_mask]
    nPBm_cum_L = nPBm_cum[np.logical_not(PB_mask)]

    PI_unique_S = PI_unique[PI_mask]
    PI_unique_L = PI_unique[np.logical_not(PI_mask)]
    nPIm_cum_S = nPIm_cum[PI_mask]
    nPIm_cum_L = nPIm_cum[np.logical_not(PI_mask)]

    nPBm_tck_S = interpolate.splrep(PB_unique_S, nPBm_cum_S, k=3, s=1)
    nPBm_tck_L = interpolate.splrep(PB_unique_L, nPBm_cum_L, k=3, s=1)

    nPIm_tck_S = interpolate.splrep(PI_unique_S, nPIm_cum_S, k=3, s=1)
    nPIm_tck_L = interpolate.splrep(PI_unique_L, nPIm_cum_L, k=3, s=1)

    PBm_Vec_S = np.linspace(0, nuV, 50, endpoint=False)
    PBm_Vec_L = np.linspace(nuV, np.max(PB_unique), 50)
    PIm_Vec_S = np.linspace(0, nuV, 50, endpoint=False)
    PIm_Vec_L = np.linspace(nuV, np.max(PI_unique), 50)

    nPBm_cum_Vec_S = interpolate.splev(PBm_Vec_S, nPBm_tck_S, der=0)
    nPBm_cum_Vec_L = interpolate.splev(PBm_Vec_L, nPBm_tck_L, der=0)
    nPIm_cum_Vec_S = interpolate.splev(PIm_Vec_S, nPIm_tck_S, der=0)
    nPIm_cum_Vec_L = interpolate.splev(PIm_Vec_L, nPIm_tck_L, der=0)

    nPBm_Vec_S = interpolate.splev(PBm_Vec_S, nPBm_tck_S, der=1)
    nPBm_Vec_L = interpolate.splev(PBm_Vec_L, nPBm_tck_L, der=1)
    nPIm_Vec_S = interpolate.splev(PIm_Vec_S, nPIm_tck_S, der=1)
    nPIm_Vec_L = interpolate.splev(PIm_Vec_L, nPIm_tck_L, der=1)

    PBm_Vec = np.concatenate((PBm_Vec_S, PBm_Vec_L))
    PIm_Vec = np.concatenate((PIm_Vec_S, PIm_Vec_L))
    nPBm_cum_Vec = np.concatenate((nPBm_cum_Vec_S, nPBm_cum_Vec_L))
    nPIm_cum_Vec = np.concatenate((nPIm_cum_Vec_S, nPIm_cum_Vec_L))
    nPBm_Vec = np.concatenate((nPBm_Vec_S, nPBm_Vec_L))
    nPIm_Vec = np.concatenate((nPIm_Vec_S, nPIm_Vec_L))

    dPBm_L = PBm_Vec[-2] - PBm_Vec[-1]
    dPIm_L = PIm_Vec[-2] - PIm_Vec[-1]
    nPBm_Tot = np.dot(nPBm_Vec, np.ediff1d(nPBm_Vec, to_end=dPBm_L)) + nPB_deltaK0
    nPIm_Tot = np.dot(nPIm_Vec, np.ediff1d(nPIm_Vec, to_end=dPIm_L)) + nPB_deltaK0

    PBm_max = PBm_Vec[np.argmax(nPBm_Vec)]
    PIm_max = PIm_Vec[np.argmax(nPIm_Vec)]

    # logistic sigmoid curve fit

    # def CDF_fit(Pm,d,b,c):
    #     return 1/(1/(d*Pm**2) + 1/(b*expit(c*Pm)))

    # P_mag data save

    # PBm_DistData = np.concatenate((PB_unique[:, np.newaxis], nPBm_cum[:, np.newaxis]), axis=1)
    PIm_DistData = np.concatenate((PI_unique[:, np.newaxis], nPIm_cum[:, np.newaxis]), axis=1)
    # np.savetxt(datapath + '/PBm_Data.dat', PBm_DistData)
    np.savetxt(datapath + '/PIm_Data_P_{:.3f}.dat'.format(P), PIm_DistData)

    # Metrics/consistency checks

    print("FWHM = {0}, Var = {1}".format(FWHM, (FWHM / 2.355)**2))
    print("Nph = \sum b^2 = %f" % (Nph))
    print("Nph_x = %f " % (Nph_x))
    print("\int np dp = %f" % (nPB_Tot))
    print("\int p np dp = %f" % (nPB_Mom1))
    print("\int k beta^2 dk = %f" % (beta2_kz_Mom1))
    print("Exp[-Nph] = %f" % (nPB_deltaK0))
    print("\int n(PB_mag) dPB_mag = %f" % (nPBm_Tot))
    print("\int n(PI_mag) dPI_mag = %f" % (nPIm_Tot))
    print('PB_mag Max = %f' % (PBm_max))
    print('PI_mag Max = %f' % (PIm_max))

    # Save data
    # Dist_data = np.concatenate((DP * np.ones(Nz)[:, np.newaxis], Nph * np.ones(Nz)[:, np.newaxis], Nph_x * np.ones(Nz)[:, np.newaxis], nPB_Tot * np.ones(Nz)[:, np.newaxis], nPB_Mom1 * np.ones(Nz)[:, np.newaxis], beta2_kz_Mom1 * np.ones(Nz)[:, np.newaxis], FWHM * np.ones(Nz)[:, np.newaxis], x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis], nx_x_norm[:, np.newaxis], nx_y_norm[:, np.newaxis], nx_z_norm[:, np.newaxis], kx[:, np.newaxis], ky[:, np.newaxis], kz[:, np.newaxis], np.real(nPB_kx)[:, np.newaxis], np.real(nPB_ky)[:, np.newaxis], np.real(nPB_kz)[:, np.newaxis], PI_z_ord[:, np.newaxis], np.real(nPI_z)[:, np.newaxis]), axis=1)
    # np.savetxt(datapath + '/3Ddist_aIBi_{:.2f}_P_{:.2f}.dat'.format(aIBi, P), Dist_data)

    # Plot
    # fig, ax = plt.subplots(nrows=4, ncols=3)

    # ax[0, 0].plot(kzF, np.abs(beta2_kz))
    # ax[0, 0].set_title(r'$|\beta_{\vec{k}}|^2$')
    # ax[0, 0].set_xlabel(r'$k_{z}$')

    # ax[0, 1].plot(z, np.real(beta2_z))
    # ax[0, 1].set_title(r'Slice $|\beta_{\vec{k}}|^2$')

    # ax[0, 2].plot(z, np.abs(fexp_z))
    # ax[0, 2].set_title(r'Slice $f(x)$')

    # ax[1, 0].plot(kz, np.real(nPB_kz))
    # ax[1, 0].plot(np.zeros(Nz), np.linspace(0, nPB_deltaK0, Nz))
    # ax[1, 0].set_title(r'$n_{\vec{P_B}}$')
    # ax[1, 0].set_xlabel(r'$P_{B,z}$')
    # ax[1, 0].set_xlim([-10, 10])

    # ax[1, 1].plot(x, np.real(nx_z_norm))
    # ax[1, 1].set_title(r'$\frac{n(\vec{x})}{N_{ph}}$')
    # ax[1, 1].set_xlabel(r'$z$')

    # ax[1, 2].plot(PI_z_ord, np.real(nPI_z))
    # ax[1, 2].plot(P * np.ones(Nz), np.linspace(0, nPB_deltaK0, Nz))
    # ax[1, 2].set_title(r'$n_{\vec{P_I}}$')
    # ax[1, 2].set_xlabel(r'$P_{I,z}$')
    # ax[1, 2].set_xlim([-10, 10])
    # # ax[1, 2].plot(tail_dom, Tanfunc(tail_dom, C_Tan))
    # # ax[1, 2].plot(tail_dom_T, Tanfunc(tail_dom_T, C_T))

    # ax[2, 0].plot(PB_unique, nPBm_unique, 'k*')
    # # ax[2, 0].plot(np.zeros(PB_unique.size), np.linspace(0, nPB_deltaK0, PB_unique.size))
    # ax[2, 0].set_title(r'$n_{\vec{P_B}}$')
    # ax[2, 0].set_xlabel(r'$|P_{B}|$')

    # ax[2, 1].plot(PI_unique, nPIm_unique, 'k*')
    # # ax[2, 1].plot(P * np.ones(PI_unique.size), np.linspace(0, nPB_deltaK0, PI_unique.size))
    # ax[2, 1].set_title(r'$n_{\vec{P_I}}$')
    # ax[2, 1].set_xlabel(r'$|P_{I}|$')

    # ax[2, 2].plot(PBm_Vec, nPBm_Vec)
    # ax[2, 2].set_title(r'$n_{\vec{P_B}}$')
    # ax[2, 2].set_xlabel(r'$|P_{B}|$')
    # ax[2, 2].plot(np.zeros(PB_unique.size), np.linspace(0, nPB_deltaK0, PB_unique.size))

    # ax[3, 2].plot(PIm_Vec, nPIm_Vec)
    # ax[3, 2].set_title(r'$n_{\vec{P_I}}$')
    # ax[3, 2].set_xlabel(r'$|P_{I}|$')
    # ax[3, 2].plot(P * np.ones(PI_unique.size), np.linspace(0, nPB_deltaK0, PI_unique.size))

    # ax[3, 0].plot(PB_unique, nPBm_cum, 'k*')
    # ax[3, 0].set_title('Cumulative Distribution Function')
    # ax[3, 0].set_xlabel(r'$|P_{B}|$')
    # ax[3, 0].plot(PBm_Vec, nPBm_cum_Vec, 'r-')

    # ax[3, 1].plot(PI_unique, nPIm_cum, 'k*')
    # ax[3, 1].set_title('Cumulative Distribution Function')
    # ax[3, 1].set_xlabel(r'$|P_{I}|$')
    # ax[3, 1].plot(PIm_Vec, nPIm_cum_Vec, 'r-')

    # fig.tight_layout()
    # plt.show()

    # alt plot

    fig, ax = plt.subplots(nrows=2, ncols=2)

    ax[0, 0].plot(PB_unique, nPBm_cum, 'k*')
    ax[0, 0].set_title('CDF PB')
    ax[0, 0].set_xlabel(r'$|P_{B}|$')
    ax[0, 0].plot(PBm_Vec, nPBm_cum_Vec, 'r-')
    ax[0, 0].plot(nuV * np.ones(PBm_Vec.size), np.linspace(0, np.max(nPBm_cum_Vec), PBm_Vec.size))
    # ax[0, 0].plot(PI_unique, nPIm_cum, 'r*')

    ax[0, 1].plot(PI_unique, nPIm_cum, 'k*')
    ax[0, 1].set_title('CDF PI')
    ax[0, 1].set_xlabel(r'$|P_{I}|$')
    ax[0, 1].plot(PIm_Vec, nPIm_cum_Vec, 'r-')
    ax[0, 1].plot(nuV * np.ones(PIm_Vec.size), np.linspace(0, np.max(nPIm_cum_Vec), PIm_Vec.size))

    ax[1, 0].plot(PBm_Vec, nPBm_cum_Vec, 'b-')
    ax[1, 0].plot(PIm_Vec, nPIm_cum_Vec, 'r-')
    ax[1, 0].set_title('CDF')
    ax[1, 0].set_xlabel(r'$|P|$')

    ax[1, 1].plot(PBm_Vec, nPBm_Vec, 'b-')
    ax[1, 1].plot(PIm_Vec, nPIm_Vec, 'r-')
    ax[1, 1].set_title('PDF')
    ax[1, 1].set_xlabel(r'$|P|$')
    # ax[1, 1].plot(np.zeros(PB_unique.size), np.linspace(0, nPB_deltaK0, PB_unique.size))
    # ax[1, 1].plot(P * np.ones(PI_unique.size), np.linspace(0, nPB_deltaK0, PI_unique.size))

    # fig.delaxes(ax[1, 1])
    fig.tight_layout()
    plt.show()

# Create grids


start = timer()

(Lx, Ly, Lz) = (15, 15, 15)
(dx, dy, dz) = (5e-01, 5e-01, 5e-01)

xgrid = Grid.Grid('CARTESIAN_3D')
xgrid.initArray('x', -Lx, Lx, dx); xgrid.initArray('y', -Ly, Ly, dy); xgrid.initArray('z', -Lz, Lz, dz)

(Nx, Ny, Nz) = (len(xgrid.getArray('x')), len(xgrid.getArray('y')), len(xgrid.getArray('z')))

kxfft = np.fft.fftfreq(Nx) * 2 * np.pi / dx; kyfft = np.fft.fftfreq(Nx) * 2 * np.pi / dy; kzfft = np.fft.fftfreq(Nx) * 2 * np.pi / dz
kFgrid = Grid.Grid('CARTESIAN_3D')
kFgrid.initArray_premade('kx', kxfft); kFgrid.initArray_premade('ky', kyfft); kFgrid.initArray_premade('kz', kzfft)

kgrid = Grid.Grid('CARTESIAN_3D')
kgrid.initArray_premade('kx', np.fft.fftshift(kxfft)); kgrid.initArray_premade('ky', np.fft.fftshift(kyfft)); kgrid.initArray_premade('kz', np.fft.fftshift(kzfft))

gridargs = [xgrid, kgrid, kFgrid]

# precalculate for stuff below

kxFg, kyFg, kzFg = np.meshgrid(kFgrid.getArray('kx'), kFgrid.getArray('ky'), kFgrid.getArray('kz'), indexing='ij', sparse=True)
dVk = kgrid.arrays_diff['kx'] * kgrid.arrays_diff['ky'] * kgrid.arrays_diff['kz']


# Basic parameters
mI = 1
mB = 1
n0 = 1
gBB = (4 * np.pi / mB) * 0.05
nuV = nu(gBB)

datapath = os.path.dirname(os.path.realpath(__file__)) + '/fftdata'

# # INTERPOLATION

# Nsteps = 1e2
# createSpline_grid(Nsteps, kxFg, kyFg, kzFg, dVk, mI, mB, n0, gBB)

aSi_tck = np.load('aSi_spline.npy')
PBint_tck = np.load('PBint_spline.npy')

# Single function run

P = 1.75 * nuV
aIBi = -2

Pc = PCrit_grid(kxFg, kyFg, kzFg, dVk, aIBi, mI, mB, n0, gBB)
DP = DP_interp(0, P, aIBi, aSi_tck, PBint_tck)
aSi = aSi_interp(DP, aSi_tck)

params = [P, aIBi, aSi, DP, mI, mB, n0, gBB, nuV]
print('DP: {0}'.format(DP))
print('aSi: {0}, aSi_fm: {1}'.format(aSi, fm.aSi(DP, gBB, mI, mB, n0)))
# print('Pc: {0}, Pc_fm: {1}'.format(Pc, fm.PCrit(aIBi, gBB, mI, mB, n0)))
print('Pc: {0}'.format(Pc))
print('P: {0}'.format(P))
print('Nu: {0}'.format(nuV))

staticDistCalc(gridargs, params, datapath)

# Multiple function run

# paramsList = []
# aIBi_Vals = [-5, -2, -1, 1, 2, 5]
# Pc = PCrit_grid(kxFg, kyFg, kzFg, dVk, np.amin(aIBi_Vals), mI, mB, n0, gBB)
# print('PCrit: {0}'.format(Pc))
# PVals = np.linspace(0, 0.95 * Pc, 6)
# for aIBi in aIBi_Vals:
#     for P in PVals:
#         DP = DP_interp(0, P, aIBi, aSi_tck, PBint_tck)
#         aSi = aSi_interp(DP, aSi_tck)
#         paramsList.append([P, aIBi, aSi, DP, mI, mB, n0, gBB])


# for p in paramsList:
#     staticDistCalc(gridargs, p, datapath)


# end = timer()
# print(end - start)
