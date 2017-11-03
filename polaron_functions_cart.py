import numpy as np
from scipy import interpolate

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
    Bk = -2 * np.pi * np.sqrt(n0) * Wk(kx, ky, kz, mB, n0, gBB) / (ur(mI, mB) * Omega(kx, ky, kz, DP, mI, mB, n0, gBB) * (aIBi - aSi))
    prefactor = (2 * np.pi)**(-3 / 2)
    return prefactor * Bk

# ---- INTERPOLATION FUNCTIONS ----


def aSi_grid(kxFg, kyFg, kzFg, dVk, DP, mI, mB, n0, gBB):
    integrand = 2 * ur(mI, mB) / (kxFg**2 + kyFg**2 + kzFg**2) - (Wk(kxFg, kyFg, kzFg, mB, n0, gBB)**2) / Omega(kxFg, kyFg, kzFg, DP, mI, mB, n0, gBB)
    mask = np.isnan(integrand); integrand[mask] = 0
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
    return interpolate.splev(DP, aSi_tck, der=0)


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
