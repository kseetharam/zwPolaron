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
    # old_settings = np.seterr(); np.seterr(all='ignore')
    output = np.sqrt(epsilon(kx, ky, kz, mB) / omegak(kx, ky, kz, mB, n0, gBB))
    # np.seterr(**old_settings)
    return output


def g(kx, ky, kz, aIBi, mI, mB, n0, gBB):
    # gives bare interaction strength constant
    k_max = np.sqrt(np.max(kx)**2 + np.max(ky)**2 + np.max(kz)**2)
    mR = ur(mI, mB)
    return 1 / ((mR / (2 * np.pi)) * aIBi - (mR / np.pi**2) * k_max)


# ---- CALCULATION HELPER FUNCTIONS ----


def ImpMomGrid_from_PhononMomGrid(kgrid, P):
    kx = kgrid.getArray('kx'); ky = kgrid.getArray('ky'); kz = kgrid.getArray('kz')
    PI_x = -1 * kx; PI_y = -1 * ky; PI_z = P - kz
    PI_x_ord = np.flip(PI_x, 0); PI_y_ord = np.flip(PI_y, 0); PI_z_ord = np.flip(PI_z, 0)
    PIgrid = Grid.Grid('CARTESIAN_3D')
    PIgrid.initArray_premade('kx', PI_x_ord); PIgrid.initArray_premade('ky', PI_y_ord); PIgrid.initArray_premade('kz', PI_z_ord)
    return PIgrid


def FWHM(x, f):
    # f is function of x -> f(x)
    if np.abs(np.max(f) - np.min(f)) < 1e-2:
        return 0
    else:
        D = f - np.max(f) / 2
        indices = np.where(D > 0)[0]
        return x[indices[-1]] - x[indices[0]]


def xyzDist_ProjSlices(phonon_pos_dist, phonon_mom_dist):
    nxyz = phonon_pos_dist
    nPB = phonon_mom_dist

    # slice directions
    nPB_x_slice = np.real(np.abs(nPB[:, Ny // 2 + 1, Nz // 2 + 1]))
    nPB_y_slice = np.real(np.abs(nPB[Nx // 2 + 1, :, Nz // 2 + 1]))
    nPB_z_slice = np.real(np.abs(nPB[Nx // 2 + 1, Ny // 2 + 1, :]))

    nxyz_x_slice = np.real(nxyz[:, Ny // 2 + 1, Nz // 2 + 1])
    nxyz_y_slice = np.real(nxyz[Nx // 2 + 1, :, Nz // 2 + 1])
    nxyz_z_slice = np.real(nxyz[Nx // 2 + 1, Ny // 2 + 1, :])

    nPI_x_slice = np.flip(nPB_x_slice, 0)
    nPI_y_slice = np.flip(nPB_y_slice, 0)
    nPI_z_slice = np.flip(nPB_z_slice, 0)

    pos_slices = nxyz_x_slice, nxyz_y_slice, nxyz_z_slice
    mom_slices = nPB_x_slice, nPB_y_slice, nPB_z_slice, nPI_x_slice, nPI_y_slice, nPI_z_slice

    # integrate directions
    nPB_x = np.sum(np.abs(nPB), axis=(1, 2)) * dky * dkz
    nPB_y = np.sum(np.abs(nPB), axis=(0, 2)) * dkx * dkz
    nPB_z = np.sum(np.abs(nPB), axis=(0, 1)) * dkx * dky

    nxyz_x = np.sum(nxyz, axis=(1, 2)) * dy * dz
    nxyz_y = np.sum(nxyz, axis=(0, 2)) * dx * dz
    nxyz_z = np.sum(nxyz, axis=(0, 1)) * dx * dy

    nPI_x = np.flip(nPB_x, 0)
    nPI_y = np.flip(nPB_y, 0)
    nPI_z = np.flip(nPB_z, 0)

    pos_integration = nxyz_x, nxyz_y, nxyz_z
    mom_integration = nPB_x, nPB_y, nPB_z, nPI_x, nPI_y, nPI_z
    return pos_slices, mom_slices, pos_integration, mom_integration  # !!!FIGURE OUT ARGS NEEDED FOR NX,NY,NZ,DX...


def xyzDist_To_magDist(kgrid, phonon_mom_dist, P):
    nPB = phonon_mom_dist
    # kgrid is the Cartesian grid upon which the 3D matrix nPB is defined -> nPB is the phonon momentum distribution in kx,ky,kz
    kxg, kyg, kzg = np.meshgrid(kgrid.getArray('kx'), kgrid.getArray('ky'), kgrid.getArray('kz'), indexing='ij', sparse=True)  # can optimize speed by taking this from the coherent_state precalculation
    dVk_const = kgrid.dV()[0]

    PB = np.sqrt(kxg**2 + kyg**2 + kzg**2)
    PI = np.sqrt((-kxg)**2 + (-kyg)**2 + (P - kzg)**2)
    PB_flat = PB.reshape(PB.size)
    PI_flat = PI.reshape(PI.size)
    nPB_flat = np.abs(nPB.reshape(nPB.size))

    PB_series = pd.Series(nPB_flat, index=PB_flat)
    PI_series = pd.Series(nPB_flat, index=PI_flat)

    nPBm_unique = PB_series.groupby(PB_series.index).sum() * dVk_const
    nPIm_unique = PI_series.groupby(PI_series.index).sum() * dVk_const

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
    PBgrid = kgrid
    PIgrid = ImpMomGrid_from_PhononMomGrid(kgrid, P)
    PB_x = PBgrid.getArray('kx'); PB_y = PBgrid.getArray('ky'); PB_z = PBgrid.getArray('kz')
    PI_x = PIgrid.getArray('kx'); PI_y = PIgrid.getArray('ky'); PI_z = PIgrid.getArray('kz')

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
            # calculate distribution information
            phonon_pos_dist, phonon_mom_dist, phonon_mom_k0deltapeak = cs.get_PhononDistributions()
            pos_slices, mom_slices, pos_integration, mom_integration = xyzDist_ProjSlices(phonon_pos_dist, phonon_mom_dist)
            [PBm_Vec, nPBm_Vec, PIm_Vec, nPIm_Vec] = xyzDist_To_magDist(cs.kgrid, phonon_mom_dist, P)

            # unpack above calculations
            nxyz_x_slice, nxyz_y_slice, nxyz_z_slice = pos_slices
            nPB_x_slice, nPB_y_slice, nPB_z_slice, nPI_x_slice, nPI_y_slice, nPI_z_slice = mom_slices
            nxyz_x, nxyz_y, nxyz_z = pos_integration
            nPB_x, nPB_y, nPB_z, nPI_x, nPI_y, nPI_z = mom_integration
            # !!!! SAVE DIST DATA

    # Save Data

    observables_data = [PB_Vec, NB_Vec, np.real(DynOv_Vec), np.imag(DynOv_Vec), Phase_Vec]
    distribution_data = 0
    return observables_data, distribution_data
