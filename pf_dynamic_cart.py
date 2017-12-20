import numpy as np
import pandas as pd
import Grid

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


def g(kx, ky, kz, aIBi, mI, mB, n0, gBB):
    # gives bare interaction strength constant
    k_max = np.sqrt(np.max(kx)**2 + np.max(ky)**2 + np.max(kz)**2)
    mR = ur(mI, mB)
    return 1 / ((mR / (2 * np.pi)) * aIBi - (mR / np.pi**2) * k_max)


# ---- CALCULATION HELPER FUNCTIONS ----


def ImpMomGrid_from_PhononMomGrid(kgrid, P):
    kx = kgrid.getArray('kx'); ky = kgrid.getArray('ky'); kz = kgrid.getArray('kz')
    PI_x = -1 * kx; PI_y = -1 * ky; PI_z = P - kz
    PIgrid = Grid.Grid('CARTESIAN_3D')
    PIgrid.initArray_premade('kx', PI_x); PIgrid.initArray_premade('ky', PI_y); PIgrid.initArray_premade('kz', PI_z)
    return PIgrid


def FWHM(x, f):
    # f is function of x -> f(x)
    if np.abs(np.max(f) - np.min(f)) < 1e-2:
        return 0
    else:
        D = f - np.max(f) / 2
        indices = np.where(D > 0)[0]
        return x[indices[-1]] - x[indices[0]]


def xyzDist_To_magDist(kgrid, nPB, P):
    # kgrid is the Cartesian grid upon which the 3D matrix nPB is defined -> nPB is the phonon momentum distribution in kx,ky,kz
    kxg, kyg, kzg = np.meshgrid(kgrid.getArray('kx'), kgrid.getArray('ky'), kgrid.getArray('kz'), indexing='ij', sparse=True)  # can optimize speed by taking this from the coherent_state precalculation
    dVk = kgrid.dV()

    PB = np.sqrt(kxg**2 + kyg**2 + kzg**2)
    PI = np.sqrt((-kxg)**2 + (-kyg)**2 + (P - kzg)**2)
    PB_flat = PB.reshape(PB.size)
    PI_flat = PI.reshape(PI.size)
    nPB_flat = np.abs(nPB.reshape(nPB.size))

    PB_series = pd.Series(nPB_flat, index=PB_flat)
    PI_series = pd.Series(nPB_flat, index=PI_flat)

    nPBm_unique = PB_series.groupby(PB_series.index).sum() * dVk
    nPIm_unique = PI_series.groupby(PI_series.index).sum() * dVk

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
    PBmapping = dict(zip(nPBm_cum_smooth.keys(), PBm_Vec))
    PImapping = dict(zip(nPIm_cum_smooth.keys(), PIm_Vec))
    # PBmapping = pd.Series(PBm_Vec, index=nPBm_cum_smooth.keys())
    # PImapping = pd.Series(PIm_Vec, index=nPIm_cum_smooth.keys())
    nPBm_cum_smooth = nPBm_cum_smooth.rename(PBmapping).fillna(method='ffill')
    nPIm_cum_smooth = nPIm_cum_smooth.rename(PImapping).fillna(method='ffill')

    nPBm_Vec = np.gradient(nPBm_cum_smooth, dPBm)
    nPIm_Vec = np.gradient(nPIm_cum_smooth, dPIm)

    mag_dist_List = [PBm_Vec, nPBm_Vec, PIm_Vec, nPIm_Vec]

    return mag_dist_List


# ---- DATA GENERATION ----


def quenchDynamics_DataGeneration(cParams, gParams, sParams):
    #
    # do not run this inside CoherentState or PolaronHamiltonian
    import CoherentState
    import PolaronHamiltonian
    # takes parameters, performs dynamics, and outputs desired observables
    [P, aIBi] = cParams
    [xgrid, kgrid, tgrid] = gParams
    [mI, mB, n0, gBB] = sParams

    # Initialization CoherentState
    cs = CoherentState.CoherentState(kgrid, xgrid)
    # Initialization PolaronHamiltonian
    Params = [P, aIBi, mI, mB, n0, gBB]
    ham = PolaronHamiltonian.PolaronHamiltonian(cs, Params)
    # Other book-keeping
    PIgrid = ImpMomGrid_from_PhononMomGrid(kgrid, P)

    # Time evolution
    PB_Vec = np.zeros(tgrid.size, dtype=float)
    NB_Vec = np.zeros(tgrid.size, dtype=float)
    DynOv_Vec = np.zeros(tgrid.size, dtype=complex)
    Phase_Vec = np.zeros(tgrid.size, dtype=float)

    for ind, t in enumerate(tgrid):
        if ind == 0:
            dt = t
            cs.evolve(dt, ham)
        else:
            dt = t - tgrid[ind - 1]
            cs.evolve(dt, ham)
        print('t: {:.2f}, cst: {:.2f}, dt:{:.3f}'.format(t, cs.time, dt))
        PB_Vec[ind] = cs.get_PhononMomentum()
        NB_Vec[ind] = cs.get_PhononNumber()
        DynOv_Vec[ind] = cs.get_DynOverlap()
        Phase_Vec[ind] = cs.get_Phase()

        # save distribution data every 10 time values
        if t != 0 and ind % int(tgrid.size / 10) == 0:

            phonon_pos_dist, nPB = cs.get_PhononDistributions()
            [PBm_Vec, nPBm_Vec, PIm_Vec, nPIm_Vec] = xyzDist_To_magDist(cs.kgrid, nPB, P)
            # ***pick out appropriate slices, etc. that we want to see

    # Save Data

    observables_data = [PB_Vec, NB_Vec, np.real(DynOv_Vec), np.imag(DynOv_Vec), Phase_Vec]
    distribution_data = 0
    return observables_data, distribution_data
