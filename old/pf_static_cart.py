import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import binned_statistic

err = 1e-5
limit = 1e5
alpha = 0.005

# ---- BASIC FUNCTIONS ----


def ur(mI, mB):
    return (mB * mI) / (mB + mI)


def nu(gBB):
    return np.sqrt(gBB)


def epsilon(kx, ky, kz, mB):
    return (kx**2 + ky**2 + kz**2) / (2 * mB)


def omegak(kx, ky, kz, mB, n0, gBB):
    ep = epsilon(kx, ky, kz, mB)
    return np.sqrt(ep * (ep + 2 * gBB * n0))


def Omega(kx, ky, kz, DP, mI, mB, n0, gBB):
    return omegak(kx, ky, kz, mB, n0, gBB) + (kx**2 + ky**2 + kz**2) / (2 * mI) - kz * DP / mI


def Wk(kx, ky, kz, mB, n0, gBB):
    return np.sqrt(epsilon(kx, ky, kz, mB) / omegak(kx, ky, kz, mB, n0, gBB))


def BetaK(kx, ky, kz, aIBi, aSi, DP, mI, mB, n0, gBB):
    old_settings = np.seterr(); np.seterr(all='ignore')
    Bk = -2 * np.pi * np.sqrt(n0) * Wk(kx, ky, kz, mB, n0, gBB) / (ur(mI, mB) * Omega(kx, ky, kz, DP, mI, mB, n0, gBB) * (aIBi - aSi))
    np.seterr(**old_settings)
    return Bk


def Energy(P, PB, aIBi, aSi, mI, mB, n0):
    return ((P**2 - PB**2) / (2 * mI)) + 2 * np.pi * n0 / (ur(mI, mB) * (aIBi - aSi))


def effMass(P, PB, mI):
    m = mI * P / (P - PB)

    if np.isscalar(P):
        if P == 0:
            return 1
        else:
            return m
    else:
        mask = (P == 0)
        m[mask] = 1
        return m


def g(kxg, kyg, kzg, dVk, aIBi, mI, mB, n0, gBB):
    # gives bare interaction strength constant
    old_settings = np.seterr(); np.seterr(all='ignore')
    mR = ur(mI, mB)
    integrand = 2 * mR / (kxg**2 + kyg**2 + kzg**2)
    mask = np.isinf(integrand); integrand[mask] = 0
    np.seterr(**old_settings)
    return 1 / ((mR / (2 * np.pi)) * aIBi - np.sum(integrand) * dVk)


def test_grid(kgrid, mB, n0, gBB):
    kxg, kyg, kzg = np.meshgrid(kgrid.getArray('kx'), kgrid.getArray('ky'), kgrid.getArray('kz'), indexing='ij', sparse=True)
    ep = epsilon(kxg, kyg, kzg, mB).flatten()
    epint = np.dot(ep, kgrid.dV())
    Wkf = Wk(kxg, kyg, kzg, mB, n0, gBB).flatten()
    mask = np.isnan(Wkf); Wkf[mask] = 0
    Wkint = np.dot(Wkf, kgrid.dV())

    print('\int ep: {0}'.format(epint))
    print('\int Wk: {0}'.format(Wkint))


# ---- INTERPOLATION FUNCTIONS ----


def aSi_grid(kxg, kyg, kzg, dVk, DP, mI, mB, n0, gBB):
    old_settings = np.seterr(); np.seterr(all='ignore')
    integrand = 2 * ur(mI, mB) / (kxg**2 + kyg**2 + kzg**2) - (Wk(kxg, kyg, kzg, mB, n0, gBB)**2) / Omega(kxg, kyg, kzg, DP, mI, mB, n0, gBB)
    mask = np.isnan(integrand); integrand[mask] = 0
    np.seterr(**old_settings)
    return (2 * np.pi / ur(mI, mB)) * np.sum(integrand) * dVk


def PB_integral_grid(kxg, kyg, kzg, dVk, DP, mI, mB, n0, gBB):
    Bk_without_aSi = BetaK(kxg, kyg, kzg, 1, 0, DP, mI, mB, n0, gBB)
    integrand = kzg * np.abs(Bk_without_aSi)**2
    mask = np.isnan(integrand); integrand[mask] = 0
    return np.sum(integrand) * dVk


def createSpline_grid(Nsteps, kxg, kyg, kzg, dVk, mI, mB, n0, gBB):
    DP_max = mI * nu(gBB)
    DP_step = DP_max / Nsteps
    DPVals = np.arange(0, DP_max, DP_step)
    aSiVals = np.zeros(DPVals.size)
    PBintVals = np.zeros(DPVals.size)

    for idp, DP in enumerate(DPVals):
        aSiVals[idp] = aSi_grid(kxg, kyg, kzg, dVk, DP, mI, mB, n0, gBB)
        PBintVals[idp] = PB_integral_grid(kxg, kyg, kzg, dVk, DP, mI, mB, n0, gBB)

    aSi_tck = interpolate.splrep(DPVals, aSiVals, s=0)
    PBint_tck = interpolate.splrep(DPVals, PBintVals, s=0)

    np.save('aSi_spline_cart.npy', aSi_tck)
    np.save('PBint_spline_cart.npy', PBint_tck)


def aSi_interp(DP, aSi_tck):
    return 1 * interpolate.splev(DP, aSi_tck, der=0)


def PB_interp(DP, aIBi, aSi_tck, PBint_tck):
    aSi = aSi_interp(DP, aSi_tck)
    return (aIBi - aSi)**(-2) * interpolate.splev(DP, PBint_tck, der=0)


def DP_interp(DPi, P, aIBi, aSi_tck, PBint_tck):
    global err, limit, alpha
    DP_old = DPi
    DP_new = 0
    lim = np.copy(limit)

    while True:

        if lim == 0:
            print('Loop convergence limit reached')
            return -1

        DP_new = DP_old * (1 - alpha) + alpha * np.abs(P - PB_interp(DP_old, aIBi, aSi_tck, PBint_tck))
        # print(DP_old, DP_new)

        if np.abs(DP_new - DP_old) < err:
            break
        else:
            DP_old = np.copy(DP_new)

        lim = lim - 1

    return DP_new


def PCrit_grid(kxg, kyg, kzg, dVk, aIBi, mI, mB, n0, gBB):
    DPc = mI * nu(gBB)
    aSi = aSi_grid(kxg, kyg, kzg, dVk, DPc, mI, mB, n0, gBB)
    PB = (aIBi - aSi)**(-2) * PB_integral_grid(kxg, kyg, kzg, dVk, DPc, mI, mB, n0, gBB)
    return DPc + PB


# ---- DATA GENERATION ----


def static_DataGeneration(cParams, gParams, sParams):
    [P, aIBi] = cParams
    [xgrid, kgrid] = gParams
    [mI, mB, n0, gBB, aSi_tck, PBint_tck] = sParams

    # unpack grid args

    x = xgrid.getArray('x'); y = xgrid.getArray('y'); z = xgrid.getArray('z')
    (Nx, Ny, Nz) = (len(x), len(y), len(z))
    dx = xgrid.arrays_diff['x']; dy = xgrid.arrays_diff['y']; dz = xgrid.arrays_diff['z']

    kx = kgrid.getArray('kx'); ky = kgrid.getArray('ky'); kz = kgrid.getArray('kz')
    dkx = kgrid.arrays_diff['kx']; dky = kgrid.arrays_diff['ky']; dkz = kgrid.arrays_diff['kz']
    dVk = dkx * dky * dkz * (2 * np.pi)**(-3)

    # xg, yg, zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
    # kxg, kyg, kzg = np.meshgrid(kx, ky, kz, indexing='ij', sparse=True)

    xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')
    kxg, kyg, kzg = np.meshgrid(kx, ky, kz, indexing='ij')

    # calculate relevant parameters

    NGridPoints = kgrid.size()
    k_max = np.sqrt(np.max(kx)**2 + np.max(ky)**2 + np.max(kz)**2)

    DP = DP_interp(0, P, aIBi, aSi_tck, PBint_tck)
    aSi = aSi_interp(DP, aSi_tck)
    PB_Val = PB_interp(DP, aIBi, aSi_tck, PBint_tck)
    Pcrit = PCrit_grid(kxg, kyg, kzg, dVk, aIBi, mI, mB, n0, gBB)
    En = Energy(P, PB_Val, aIBi, aSi, mI, mB, n0)
    nu_const = nu(gBB)
    eMass = effMass(P, PB_Val, mI)
    gIB = g(kxg, kyg, kzg, dVk, aIBi, mI, mB, n0, gBB)

    bparams = [aIBi, aSi, DP, mI, mB, n0, gBB]

    # generation

    beta_kxkykz_preshift = BetaK(kxg, kyg, kzg, *bparams)
    beta_kxkykz = np.fft.ifftshift(beta_kxkykz_preshift)
    mask = np.isnan(beta_kxkykz); beta_kxkykz[mask] = 0
    beta2_kxkykz = np.abs(beta_kxkykz)**2

    decay_length = 5
    decay_xyz = np.exp(-1 * (xg**2 + yg**2 + zg**2) / (2 * decay_length**2))

    # Fourier transform
    amp_beta_xyz_preshift = np.fft.ifftn(beta_kxkykz) / (dx * dy * dz)
    amp_beta_xyz = np.fft.fftshift(amp_beta_xyz_preshift)
    nxyz = np.abs(amp_beta_xyz)**2  # this is the unnormalized phonon position distribution in 3D Cartesian coordinates

    # Calculate Nph and Z-factor
    Nph = np.sum(beta2_kxkykz) * dkx * dky * dkz / ((2 * np.pi)**3)
    Nph_xyz = np.sum(nxyz * dx * dy * dz)
    Z_factor = np.exp(-1 * Nph)

    nxyz_norm = nxyz / Nph  # this is the normalized phonon position distribution in 3D Cartesian coordinates

    # Fourier transform
    beta2_xyz_preshift = np.fft.ifftn(beta2_kxkykz) / (dx * dy * dz)
    beta2_xyz = np.fft.fftshift(beta2_xyz_preshift)

    # Exponentiate
    fexp = (np.exp(beta2_xyz - Nph) - np.exp(-Nph)) * decay_xyz

    # Inverse Fourier transform
    nPB_preshift = np.fft.fftn(fexp) * (dx * dy * dz)
    nPB_complex = np.fft.fftshift(nPB_preshift) / ((2 * np.pi)**3)  # this is the phonon momentum distribution in 3D Cartesian coordinates
    nPB = np.abs(nPB_complex)
    nPB_deltaK0 = np.exp(-Nph)

    # Calculate phonon distribution slices
    nPB_x_slice = nPB[:, Ny // 2, Nz // 2]
    nPB_y_slice = nPB[Nx // 2, :, Nz // 2]
    nPB_z_slice = nPB[Nx // 2, Ny // 2, :]

    nPB_xz_slice = nPB[:, Ny // 2, :]
    nPB_xy_slice = nPB[:, :, Nz // 2]

    nxyz_x_slice = nxyz[:, Ny // 2, Nz // 2]
    nxyz_y_slice = nxyz[Nx // 2, :, Nz // 2]
    nxyz_z_slice = nxyz[Nx // 2, Ny // 2, :]

    nxyz_xz_slice = nxyz[:, Ny // 2, :]
    nxyz_xy_slice = nxyz[:, :, Nz // 2]

    # Integrating out certain directions
    beta2_kz = np.sum(np.fft.fftshift(beta2_kxkykz), axis=(0, 1)) * dkx * dky / ((2 * np.pi)**2)
    nPB_x = np.sum(nPB, axis=(1, 2)) * dky * dkz
    nPB_y = np.sum(nPB, axis=(0, 2)) * dkx * dkz
    nPB_z = np.sum(nPB, axis=(0, 1)) * dkx * dky
    nxyz_x = np.sum(nxyz_norm, axis=(1, 2)) * dy * dz
    nxyz_y = np.sum(nxyz_norm, axis=(0, 2)) * dx * dz
    nxyz_z = np.sum(nxyz_norm, axis=(0, 1)) * dx * dy

    nxyz_Tot = np.sum(nxyz_norm * dx * dy * dz)
    nPB_Tot = np.sum(nPB) * dkx * dky * dkz + nPB_deltaK0
    nPB_Mom1 = np.dot(nPB_z, kz * dkz)
    beta2_kz_Mom1 = np.dot(beta2_kz, kz * dkz / (2 * np.pi))

    # Flipping domain for P_I instead of P_B so now nPB(PI) -> nPI: Then calculcate nPI quantities

    PB_x = kx
    PB_y = ky
    PB_z = kz

    PI_x = -1 * PB_x
    PI_y = -1 * PB_y
    PI_z = P - PB_z

    PI_x_ord = np.flip(PI_x, 0)
    PI_y_ord = np.flip(PI_y, 0)
    PI_z_ord = np.flip(PI_z, 0)

    nPI_x = np.flip(nPB_x, 0)
    nPI_y = np.flip(nPB_y, 0)
    nPI_z = np.flip(nPB_z, 0)

    nPI_x_slice = np.flip(nPB_x_slice, 0)
    nPI_y_slice = np.flip(nPB_y_slice, 0)
    nPI_z_slice = np.flip(nPB_z_slice, 0)

    nPI_xz_slice = np.flip(np.flip(nPB_xz_slice, 0), 1)
    nPI_xy_slice = np.flip(np.flip(nPB_xy_slice, 0), 1)

    # Calculate FWHM

    if np.abs(np.max(nPI_z) - np.min(nPI_z)) < 1e-2:
        FWHM = 0
    else:
        D = nPI_z - np.max(nPI_z) / 2
        indices = np.where(D > 0)[0]
        FWHM = PI_z_ord[indices[-1]] - PI_z_ord[indices[0]]

    # Calculate magnitude distribution nPB(P) and nPI(P) where P_IorB = sqrt(Px^2 + Py^2 + Pz^2) - calculate CDF from this

    PB = np.sqrt(kxg**2 + kyg**2 + kzg**2)
    PI = np.sqrt((-kxg)**2 + (-kyg)**2 + (P - kzg)**2)
    PB_flat = PB.reshape(PB.size)
    PI_flat = PI.reshape(PI.size)
    nPB_flat = nPB.reshape(nPB.size)

    PB_series = pd.Series(nPB_flat, index=PB_flat)
    PI_series = pd.Series(nPB_flat, index=PI_flat)

    nPBm_unique = PB_series.groupby(PB_series.index).sum() * dkx * dky * dkz
    nPIm_unique = PI_series.groupby(PI_series.index).sum() * dkx * dky * dkz

    PB_unique = nPBm_unique.keys().values
    PI_unique = nPIm_unique.keys().values

    nPBm_cum = nPBm_unique.cumsum()
    nPIm_cum = nPIm_unique.cumsum()

    # CDF pre-processing by averaging distribution over small regions of Delta_P{B or I}

    PBm_Vec, dPBm = np.linspace(0, np.max(PB_unique), 200, retstep=True)
    PIm_Vec, dPIm = np.linspace(0, np.max(PI_unique), 200, retstep=True)

    nPBm_cum_smooth = nPBm_cum.groupby(pd.cut(x=nPBm_cum.index, bins=PBm_Vec, right=True, include_lowest=True)).mean()
    nPIm_cum_smooth = nPIm_cum.groupby(pd.cut(x=nPIm_cum.index, bins=PIm_Vec, right=True, include_lowest=True)).mean()

    # one less bin than bin edge so consider each bin average to correspond to left bin edge and throw out last (rightmost) edge
    PBm_Vec = PBm_Vec[0:-1]
    PIm_Vec = PIm_Vec[0:-1]

    # smooth data has NaNs in it from bins that don't contain any points - forward fill these holes

    PBmapping = dict(zip(nPBm_cum_smooth.keys(), PBm_Vec))
    PImapping = dict(zip(nPIm_cum_smooth.keys(), PIm_Vec))

    # PBmapping = pd.Series(PBm_Vec, index=nPBm_cum_smooth.keys()) # not sure why using a series as a mapping doesn't work -> it did initially and then broke...
    # PImapping = pd.Series(PIm_Vec, index=nPIm_cum_smooth.keys())

    nPBm_cum_smooth = nPBm_cum_smooth.rename(PBmapping).fillna(method='ffill')
    nPIm_cum_smooth = nPIm_cum_smooth.rename(PImapping).fillna(method='ffill')

    nPBm_Vec = np.gradient(nPBm_cum_smooth, dPBm)
    nPIm_Vec = np.gradient(nPIm_cum_smooth, dPIm)

    nPBm_Tot = np.sum(nPBm_Vec * dPBm) + nPB_deltaK0
    nPIm_Tot = np.sum(nPIm_Vec * dPIm) + nPB_deltaK0

    # # Consistency checks

    # print("FWHM = {0}, Var = {1}".format(FWHM, (FWHM / 2.355)**2))
    # print("Nph = \sum b^2 = %f" % (Nph))
    # print("Nph_xyz = %f " % (Nph_xyz))
    # print("\int nx dx = %f" % (nxyz_Tot))
    # print("\int np dp = %f" % (nPB_Tot))
    # print("\int p np dp = %f" % (nPB_Mom1))
    # print("\int k beta^2 dk = %f" % (beta2_kz_Mom1))
    # print("Exp[-Nph] = %f" % (nPB_deltaK0))
    # print("\int n(PB_mag) dPB_mag = %f" % (nPBm_Tot))
    # print("\int n(PI_mag) dPI_mag = %f" % (nPIm_Tot))

    # Collate data

    metrics_string = 'NGridPoints, |k|_max, P, aIBi, mI, mB, n0, gBB, nu, gIB, Pcrit, aSi, DP, PB, Energy, effMass, Nph, Nph_xyz, Z_factor, nxyz_Tot, nPB_Tot, nPBm_Tot, nPIm_Tot, PB_1stMoment(nPB), PB_1stMoment(Betak^2), nPB(k=0)_DeltaPeak, FWHM'
    metrics_data = np.array([NGridPoints, k_max, P, aIBi, mI, mB, n0, gBB, nu_const, gIB, Pcrit, aSi, DP, PB_Val, En, eMass, Nph, Nph_xyz, Z_factor, nxyz_Tot, nPB_Tot, nPBm_Tot, nPIm_Tot, nPB_Mom1, beta2_kz_Mom1, nPB_deltaK0, FWHM])
    # note that nPI_x and nPI_y can be derived just by plotting nPB_x and nPI_y against -kx and -ky instead of kx and ky

    # xyz_string = 'x, y, z, nxyz_x_norm, nxyz_y_norm, nxyz_z_norm, kx, ky, kz, nPB_x, nPB_y, nPB_z, PI_z, nPI_z'
    # xyz_data = np.concatenate((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis], nxyz_x_norm[:, np.newaxis], nxyz_y_norm[:, np.newaxis], nxyz_z_norm[:, np.newaxis], kx[:, np.newaxis], ky[:, np.newaxis], kz[:, np.newaxis], np.real(nPB_x)[:, np.newaxis], np.real(nPB_y)[:, np.newaxis], np.real(nPB_z)[:, np.newaxis], PI_z_ord[:, np.newaxis], np.real(nPI_z)[:, np.newaxis]), axis=1)
    pos_xyz_string = 'Phonons: x, y, z, nxyz_x, nxyz_y, nxyz_z, nxyz_x_slice, nxyz_y_slice, nxyz_z_slice'
    pos_xyz_data = np.concatenate((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis], nxyz_x[:, np.newaxis], nxyz_y[:, np.newaxis], nxyz_z[:, np.newaxis], nxyz_x_slice[:, np.newaxis], nxyz_y_slice[:, np.newaxis], nxyz_z_slice[:, np.newaxis]), axis=1)

    mom_xyz_string = 'PB_x, PB_y, PB_z, nPB_x, nPB_y, nPB_z, nPB_x_slice, nPB_y_slice, nPB_z_slice, PI_x, PI_y, PI_z, nPI_x, nPI_y, nPI_z, nPI_x_slice, nPI_y_slice, nPI_z_slice'
    mom_xyz_data = np.concatenate((PB_x[:, np.newaxis], PB_y[:, np.newaxis], PB_z[:, np.newaxis], nPB_x[:, np.newaxis], nPB_y[:, np.newaxis], nPB_z[:, np.newaxis], nPB_x_slice[:, np.newaxis], nPB_y_slice[:, np.newaxis], nPB_z_slice[:, np.newaxis], PI_x_ord[:, np.newaxis], PI_y_ord[:, np.newaxis], PI_z_ord[:, np.newaxis], nPI_x[:, np.newaxis], nPI_y[:, np.newaxis], nPI_z[:, np.newaxis], nPI_x_slice[:, np.newaxis], nPI_y_slice[:, np.newaxis], nPI_z_slice[:, np.newaxis]), axis=1)

    cont_xyz_string = 'nxyz_xz_slice, nxyz_xy_slice, nPB_xz_slice, nPB_xy_slice, nPI_xz_slice, nPI_xy_slice'
    cont_xyz_data = [nxyz_xz_slice, nxyz_xy_slice, nPB_xz_slice, nPB_xy_slice, nPI_xz_slice, nPI_xy_slice]

    mom_mag_string = 'PBm_Vec, nPBm_Vec, PIm_Vec, nPIm_Vec'
    mom_mag_data = np.concatenate((PBm_Vec[:, np.newaxis], nPBm_Vec[:, np.newaxis], PIm_Vec[:, np.newaxis], nPIm_Vec[:, np.newaxis]), axis=1)

    # test_grid(kgrid, mB, n0, gBB)

    return metrics_string, metrics_data, pos_xyz_string, pos_xyz_data, mom_xyz_string, mom_xyz_data, cont_xyz_string, cont_xyz_data, mom_mag_string, mom_mag_data
