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
    prefactor = (2 * np.pi)**(-3 / 2)
    np.seterr(**old_settings)
    return prefactor * Bk


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


def g(kx, ky, kz, aIBi, mI, mB, n0, gBB):
    # gives bare interaction strength constant
    k_max = np.sqrt(np.max(kx)**2 + np.max(ky)**2 + np.max(kz)**2)
    mR = ur(mI, mB)
    return 1 / ((mR / (2 * np.pi)) * aIBi - (mR / np.pi**2) * k_max)

# ---- INTERPOLATION FUNCTIONS ----


def aSi_grid(kxFg, kyFg, kzFg, dVk, DP, mI, mB, n0, gBB):
    old_settings = np.seterr(); np.seterr(all='ignore')
    integrand = 2 * ur(mI, mB) / (kxFg**2 + kyFg**2 + kzFg**2) - (Wk(kxFg, kyFg, kzFg, mB, n0, gBB)**2) / Omega(kxFg, kyFg, kzFg, DP, mI, mB, n0, gBB)
    mask = np.isnan(integrand); integrand[mask] = 0
    np.seterr(**old_settings)
    return (2 * np.pi / ur(mI, mB)) * np.sum(integrand) * dVk * (2 * np.pi)**(-3)


def PB_integral_grid(kxFg, kyFg, kzFg, dVk, DP, mI, mB, n0, gBB):
    Bk_without_aSi = BetaK(kxFg, kyFg, kzFg, 1, 0, DP, mI, mB, n0, gBB)
    integrand = kzFg * np.abs(Bk_without_aSi)**2
    mask = np.isnan(integrand); integrand[mask] = 0
    return np.sum(integrand) * dVk


def createSpline_grid(Nsteps, kxFg, kyFg, kzFg, dVk, mI, mB, n0, gBB):
    DP_max = mI * nu(gBB)
    DP_step = DP_max / Nsteps
    DPVals = np.arange(0, DP_max, DP_step)
    aSiVals = np.zeros(DPVals.size)
    PBintVals = np.zeros(DPVals.size)

    for idp, DP in enumerate(DPVals):
        aSiVals[idp] = aSi_grid(kxFg, kyFg, kzFg, dVk, DP, mI, mB, n0, gBB)
        PBintVals[idp] = PB_integral_grid(kxFg, kyFg, kzFg, dVk, DP, mI, mB, n0, gBB)

    aSi_tck = interpolate.splrep(DPVals, aSiVals, s=0)
    PBint_tck = interpolate.splrep(DPVals, PBintVals, s=0)

    np.save('aSi_spline.npy', aSi_tck)
    np.save('PBint_spline.npy', PBint_tck)


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


def PCrit_grid(kxFg, kyFg, kzFg, dVk, aIBi, mI, mB, n0, gBB):
    DPc = mI * nu(gBB)
    aSi = aSi_grid(kxFg, kyFg, kzFg, dVk, DPc, mI, mB, n0, gBB)
    PB = (aIBi - aSi)**(-2) * PB_integral_grid(kxFg, kyFg, kzFg, dVk, DPc, mI, mB, n0, gBB)
    return DPc + PB


# ---- DATA GENERATION ----


def staticDataGeneration(cParams, gParams, sParams):
    [P, aIBi] = cParams
    [xgrid, kgrid, kFgrid] = gParams
    [mI, mB, n0, gBB, aSi_tck, PBint_tck] = sParams

    # unpack grid args

    x = xgrid.getArray('x'); y = xgrid.getArray('y'); z = xgrid.getArray('z')
    (Nx, Ny, Nz) = (len(x), len(y), len(z))
    dx = xgrid.arrays_diff['x']; dy = xgrid.arrays_diff['y']; dz = xgrid.arrays_diff['z']

    kxF = kFgrid.getArray('kx'); kyF = kFgrid.getArray('ky'); kzF = kFgrid.getArray('kz')

    kx = kgrid.getArray('kx'); ky = kgrid.getArray('ky'); kz = kgrid.getArray('kz')
    dkx = kgrid.arrays_diff['kx']; dky = kgrid.arrays_diff['ky']; dkz = kgrid.arrays_diff['kz']
    dVk = dkx * dky * dkz

    xg, yg, zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
    kxg, kyg, kzg = np.meshgrid(kx, ky, kz, indexing='ij', sparse=True)
    kxFg, kyFg, kzFg = np.meshgrid(kxF, kyF, kzF, indexing='ij', sparse=True)

    # calculate relevant parameters

    DP = DP_interp(0, P, aIBi, aSi_tck, PBint_tck)
    aSi = aSi_interp(DP, aSi_tck)
    PB_Val = PB_interp(DP, aIBi, aSi_tck, PBint_tck)
    Pcrit = PCrit_grid(kxFg, kyFg, kzFg, dVk, aIBi, mI, mB, n0, gBB)
    En = Energy(P, PB_Val, aIBi, aSi, mI, mB, n0)
    nu_const = nu(gBB)
    eMass = effMass(P, PB_Val, mI)
    gIB = g(kx, ky, kz, aIBi, mI, mB, n0, gBB)

    bparams = [aIBi, aSi, DP, mI, mB, n0, gBB]

    # generation

    beta2_kxkykz = np.abs(BetaK(kxFg, kyFg, kzFg, *bparams))**2
    mask = np.isnan(beta2_kxkykz); beta2_kxkykz[mask] = 0

    decay_length = 5
    decay_xyz = np.exp(-1 * (xg**2 + yg**2 + zg**2) / (2 * decay_length**2))

    # Fourier transform
    amp_beta_xyz_0 = np.fft.fftn(np.sqrt(beta2_kxkykz))
    amp_beta_xyz = np.fft.fftshift(amp_beta_xyz_0) * dkx * dky * dkz

    # Calculate Nph and Z-factor
    Nph = np.real(np.sum(beta2_kxkykz) * dkx * dky * dkz)
    Nph_x = np.real(np.sum(np.abs(amp_beta_xyz)**2) * dx * dy * dz * (2 * np.pi)**(-3))
    Z_factor = np.exp(-(1 / 2) * Nph)

    # Fourier transform
    beta2_xyz_preshift = np.fft.fftn(beta2_kxkykz)
    beta2_xyz = np.fft.fftshift(beta2_xyz_preshift) * dkx * dky * dkz

    # Exponentiate
    fexp = (np.exp(beta2_xyz - Nph) - np.exp(-Nph)) * decay_xyz

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

    # Flipping domain for P_I instead of P_B so now nPB(PI) -> nPI

    PI_x = -1 * kx
    PI_y = -1 * ky
    PI_z = P - kz

    # Calculate FWHM

    PI_z_ord = np.flip(PI_z, 0)
    nPI_z = np.flip(np.real(nPB_kz), 0)

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
    nPB_flat = np.abs(nPB.reshape(nPB.size))

    PB_series = pd.Series(nPB_flat, index=PB_flat)
    PI_series = pd.Series(nPB_flat, index=PI_flat)

    nPBm_unique = PB_series.groupby(PB_series.index).sum() * dkx * dky * dkz
    nPIm_unique = PI_series.groupby(PI_series.index).sum() * dkx * dky * dkz

    PB_unique = nPBm_unique.keys().values
    PI_unique = nPIm_unique.keys().values

    nPBm_cum = nPBm_unique.cumsum()
    nPIm_cum = nPIm_unique.cumsum()

    # CDF and PDF pre-processing

    PBm_Vec, dPBm = np.linspace(0, np.max(PB_unique), 200, retstep=True)
    PIm_Vec, dPIm = np.linspace(0, np.max(PI_unique), 200, retstep=True)

    nPBm_cum_smooth = nPBm_cum.groupby(pd.cut(x=nPBm_cum.index, bins=PBm_Vec, right=True, include_lowest=True)).mean()
    nPIm_cum_smooth = nPIm_cum.groupby(pd.cut(x=nPIm_cum.index, bins=PIm_Vec, right=True, include_lowest=True)).mean()

    # one less bin than bin edge so consider each bin average to correspond to left bin edge and throw out last (rightmost) edge
    PBm_Vec = PBm_Vec[0:-1]
    PIm_Vec = PIm_Vec[0:-1]

    # smooth data has NaNs in it from bins that don't contain any points - forward fill these holes
    PBmapping = pd.Series(PBm_Vec, index=nPBm_cum_smooth.keys())
    PImapping = pd.Series(PIm_Vec, index=nPIm_cum_smooth.keys())
    nPBm_cum_smooth = nPBm_cum_smooth.rename(PBmapping).fillna(method='ffill')
    nPIm_cum_smooth = nPIm_cum_smooth.rename(PImapping).fillna(method='ffill')

    nPBm_Vec = np.gradient(nPBm_cum_smooth, dPBm)
    nPIm_Vec = np.gradient(nPIm_cum_smooth, dPIm)

    nPBm_Tot = np.sum(nPBm_Vec * dPBm) + nPB_deltaK0
    nPIm_Tot = np.sum(nPIm_Vec * dPIm) + nPB_deltaK0

    # Consistency checks

    # print("FWHM = {0}, Var = {1}".format(FWHM, (FWHM / 2.355)**2))
    # print("Nph = \sum b^2 = %f" % (Nph))
    # print("Nph_x = %f " % (Nph_x))
    # print("\int np dp = %f" % (nPB_Tot))
    # print("\int p np dp = %f" % (nPB_Mom1))
    # print("\int k beta^2 dk = %f" % (beta2_kz_Mom1))
    # print("Exp[-Nph] = %f" % (nPB_deltaK0))
    # print("\int n(PB_mag) dPB_mag = %f" % (nPBm_Tot))
    # print("\int n(PI_mag) dPI_mag = %f" % (nPIm_Tot))

    # Collate data

    metrics_string = 'P, aIBi, mI, mB, n0, gBB, nu, gIB, Pcrit, DP, PB, Energy, effMass, Nph, Nph_x, Z_factor, nPB_Tot, nPBm_Tot, nPIm_Tot, PB_1stMoment(nPB), PB_1stMoment(Betak^2), FWHM'
    metrics_data = np.array([P, aIBi, mI, mB, n0, gBB, nu_const, gIB, Pcrit, aSi, DP, PB_Val, En, eMass, Nph, Nph_x, Z_factor, nPB_Tot, nPBm_Tot, nPIm_Tot, nPB_Mom1, beta2_kz_Mom1, FWHM])
    # note that nPI_x and nPI_y can be derived just by plotting nPB_x and nPI_y against -kx and -ky instead of kx and ky
    xyz_string = 'x, y, z, nx_x_norm, nx_y_norm, nx_z_norm, kx, ky, kz, nPB_kx, nPB_ky, nPB_kz, PI_z, nPI_z'
    xyz_data = np.concatenate((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis], nx_x_norm[:, np.newaxis], nx_y_norm[:, np.newaxis], nx_z_norm[:, np.newaxis], kx[:, np.newaxis], ky[:, np.newaxis], kz[:, np.newaxis], np.real(nPB_kx)[:, np.newaxis], np.real(nPB_ky)[:, np.newaxis], np.real(nPB_kz)[:, np.newaxis], PI_z_ord[:, np.newaxis], np.real(nPI_z)[:, np.newaxis]), axis=1)
    mag_string = 'PBm_Vec, nPBm_Vec, PIm_Vec, nPIm_Vec'
    mag_data = np.concatenate((PBm_Vec[:, np.newaxis], nPBm_Vec[:, np.newaxis], PIm_Vec[:, np.newaxis], nPIm_Vec[:, np.newaxis]), axis=1)
    return metrics_string, metrics_data, xyz_string, xyz_data, mag_string, mag_data
