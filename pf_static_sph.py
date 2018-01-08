import numpy as np
from scipy import interpolate

err = 1e-5
limit = 1e5
alpha = 0.005


# ---- HELPER FUNCTIONS ----


def kcos_func(kgrid):
    #
    names = list(kgrid.arrays.keys())
    functions_kcos = [lambda k: k, np.cos]
    return kgrid.function_prod(names, functions_kcos)


def kpow2_func(kgrid):
    #
    names = list(kgrid.arrays.keys())
    functions_kpow2 = [lambda k: k**2, lambda th: 0 * th + 1]
    return kgrid.function_prod(names, functions_kpow2)


# ---- BASIC FUNCTIONS ----


def ur(mI, mB):
    return (mB * mI) / (mB + mI)


def nu(gBB):
    return np.sqrt(gBB)


def epsilon(k, mB):
    return k**2 / (2 * mB)


def omegak(k, mB, n0, gBB):
    ep = epsilon(k, mB)
    # print(ep == 0, k)
    return np.sqrt(ep * (ep + 2 * gBB * n0))


def Omega(kgrid, DP, mI, mB, n0, gBB):
    names = list(kgrid.arrays.keys())  # ***need to have arrays added as k, th when kgrid is created
    if names[0] != 'k':
        print('CREATED kgrid IN WRONG ORDER')
    functions_omega0 = [lambda k: omegak(k, mB, n0, gBB) + (k**2 / (2 * mI)), lambda th: 0 * th + 1]
    omega0 = kgrid.function_prod(names, functions_omega0)
    return omega0 - kcos_func(kgrid) * DP / mI


def Wk(kgrid, mI, mB, n0, gBB):
    names = list(kgrid.arrays.keys())
    functions_Wk = [lambda k: np.sqrt(epsilon(k, mB) / omegak(k, mB, n0, gBB)), lambda th: 0 * th + 1]
    return kgrid.function_prod(names, functions_Wk)


def BetaK(kgrid, aIBi, aSi, DP, mI, mB, n0, gBB):
    return -2 * np.pi * np.sqrt(n0) * Wk(kgrid, mI, mB, n0, gBB) / (ur(mI, mB) * Omega(kgrid, DP, mI, mB, n0, gBB) * (aIBi - aSi))


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


def g(kgrid, P, aIBi, mI, mB, n0, gBB):
    # gives bare interaction strength constant
    k_max = kgrid.getArray('k')[-1]
    mR = ur(mI, mB)
    return 1 / ((mR / (2 * np.pi)) * aIBi - (mR / np.pi**2) * k_max)


def num_phonons(kgrid, aIBi, aSi, DP, mI, mB, n0, gBB):
    integrand = kpow2_func(kgrid) * np.abs(BetaK(kgrid, aIBi, aSi, DP, mI, mB, n0, gBB))**2
    return np.dot(integrand, kgrid.dV())


def qp_residue(kgrid, aIBi, aSi, DP, mI, mB, n0, gBB):
    exparg = -1 * (1 / 2) * num_phonons(kgrid, aIBi, aSi, DP, mI, mB, n0, gBB)
    return np.exp(exparg)


# ---- INTERPOLATION FUNCTIONS ----


def aSi_grid(kgrid, DP, mI, mB, n0, gBB):
    integrand = 2 * ur(mI, mB) / kpow2_func(kgrid) - Wk(kgrid, mI, mB, n0, gBB)**2 / Omega(kgrid, DP, mI, mB, n0, gBB)
    return (2 * np.pi / ur(mI, mB)) * np.dot(integrand, kgrid.dV())


def PB_integral_grid(kgrid, DP, mI, mB, n0, gBB):
    Bk_without_aSi = BetaK(kgrid, 1, 0, DP, mI, mB, n0, gBB)
    integrand = kcos_func(kgrid) * np.abs(Bk_without_aSi)**2
    return np.dot(integrand, kgrid.dV())


def createSpline_grid(Nsteps, kgrid, mI, mB, n0, gBB):
    DP_max = mI * nu(gBB)
    DP_step = DP_max / Nsteps
    DPVals = np.arange(0, DP_max, DP_step)
    aSiVals = np.zeros(DPVals.size)
    PBintVals = np.zeros(DPVals.size)

    for idp, DP in enumerate(DPVals):
        aSiVals[idp] = aSi_grid(kgrid, DP, mI, mB, n0, gBB)
        PBintVals[idp] = PB_integral_grid(kgrid, DP, mI, mB, n0, gBB)

    aSi_tck = interpolate.splrep(DPVals, aSiVals, s=0)
    PBint_tck = interpolate.splrep(DPVals, PBintVals, s=0)

    np.save('aSi_spline_sph.npy', aSi_tck)
    np.save('PBint_spline_sph.npy', PBint_tck)


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


def PCrit_grid(kgrid, aIBi, mI, mB, n0, gBB):
    DPc = mI * nu(gBB)
    aSi = aSi_grid(kgrid, DPc, mI, mB, n0, gBB)
    PB = (aIBi - aSi)**(-2) * PB_integral_grid(kgrid, DPc, mI, mB, n0, gBB)
    return DPc + PB


# ---- DATA GENERATION ----


def static_DataGeneration(cParams, gParams, sParams):
    [P, aIBi] = cParams
    [kgrid] = gParams
    [mI, mB, n0, gBB, aSi_tck, PBint_tck] = sParams

    # calculate relevant parameters

    NGridPoints = kgrid.size()
    k_max = kgrid.getArray('k')[-1]

    DP = DP_interp(0, P, aIBi, aSi_tck, PBint_tck)
    aSi = aSi_interp(DP, aSi_tck)
    PB_Val = PB_interp(DP, aIBi, aSi_tck, PBint_tck)
    Pcrit = PCrit_grid(kgrid, aIBi, mI, mB, n0, gBB)
    En = Energy(P, PB_Val, aIBi, aSi, mI, mB, n0)
    nu_const = nu(gBB)
    eMass = effMass(P, PB_Val, mI)
    gIB = g(kgrid, P, aIBi, mI, mB, n0, gBB)
    Nph = num_phonons(kgrid, aIBi, aSi, DP, mI, mB, n0, gBB)
    Z_factor = qp_residue(kgrid, aIBi, aSi, DP, mI, mB, n0, gBB)

    # Collate data

    metrics_string = 'NGridPoints, |k|_max, P, aIBi, mI, mB, n0, gBB, nu, gIB, Pcrit, aSi, DP, PB, Energy, effMass, Nph, Z_factor'
    metrics_data = np.array([NGridPoints, k_max, P, aIBi, mI, mB, n0, gBB, nu_const, gIB, Pcrit, aSi, DP, PB_Val, En, eMass, Nph, Z_factor])

    return metrics_string, metrics_data
