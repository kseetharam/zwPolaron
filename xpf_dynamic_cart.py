import numpy as np
import pandas as pd
import xarray as xr
import Grid
from timeit import default_timer as timer

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


def g(kxg, kyg, kzg, dVk, aIBi, mI, mB, n0, gBB):
    # gives bare interaction strength constant
    old_settings = np.seterr(); np.seterr(all='ignore')
    mR = ur(mI, mB)
    integrand = 2 * mR / (kxg**2 + kyg**2 + kzg**2)
    mask = np.isinf(integrand); integrand[mask] = 0
    np.seterr(**old_settings)
    return 1 / ((mR / (2 * np.pi)) * aIBi - np.sum(integrand) * dVk)


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


def xyzDist_ProjSlices(phonon_pos_dist, phonon_mom_dist, grid_size_args, grid_diff_args):
    nxyz = phonon_pos_dist
    nPB = phonon_mom_dist
    Nx, Ny, Nz = grid_size_args
    dx, dy, dz, dkx, dky, dkz = grid_diff_args

    # slice directions
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

    nPI_x_slice = np.flip(nPB_x_slice, 0)
    nPI_y_slice = np.flip(nPB_y_slice, 0)
    nPI_z_slice = np.flip(nPB_z_slice, 0)

    nPI_xz_slice = np.flip(np.flip(nPB_xz_slice, 0), 1)
    nPI_xy_slice = np.flip(np.flip(nPB_xy_slice, 0), 1)

    pos_slices = nxyz_x_slice, nxyz_y_slice, nxyz_z_slice
    mom_slices = nPB_x_slice, nPB_y_slice, nPB_z_slice, nPI_x_slice, nPI_y_slice, nPI_z_slice
    cont_slices = nxyz_xz_slice, nxyz_xy_slice, nPB_xz_slice, nPB_xy_slice, nPI_xz_slice, nPI_xy_slice

    # integrate directions
    nPB_x = np.sum(nPB, axis=(1, 2)) * dky * dkz
    nPB_y = np.sum(nPB, axis=(0, 2)) * dkx * dkz
    nPB_z = np.sum(nPB, axis=(0, 1)) * dkx * dky

    nxyz_x = np.sum(nxyz, axis=(1, 2)) * dy * dz
    nxyz_y = np.sum(nxyz, axis=(0, 2)) * dx * dz
    nxyz_z = np.sum(nxyz, axis=(0, 1)) * dx * dy

    nPI_x = np.flip(nPB_x, 0)
    nPI_y = np.flip(nPB_y, 0)
    nPI_z = np.flip(nPB_z, 0)

    pos_integration = nxyz_x, nxyz_y, nxyz_z
    mom_integration = nPB_x, nPB_y, nPB_z, nPI_x, nPI_y, nPI_z
    return pos_slices, mom_slices, cont_slices, pos_integration, mom_integration


# @profile
def xyzDist_To_magDist(kgrid, phonon_mom_dist, P):
    nPB = phonon_mom_dist
    # kgrid is the Cartesian grid upon which the 3D matrix nPB is defined -> nPB is the phonon momentum distribution in kx,ky,kz
    kxg, kyg, kzg = np.meshgrid(kgrid.getArray('kx'), kgrid.getArray('ky'), kgrid.getArray('kz'), indexing='ij', sparse=True)  # can optimize speed by taking this from the coherent_state precalculation
    dVk_const = kgrid.dV()[0] * (2 * np.pi)**(3)

    PB = np.sqrt(kxg**2 + kyg**2 + kzg**2)
    PI = np.sqrt((-kxg)**2 + (-kyg)**2 + (P - kzg)**2)
    PB_flat = PB.reshape(PB.size)
    PI_flat = PI.reshape(PI.size)
    nPB_flat = nPB.reshape(nPB.size)

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
# @profile
def quenchDynamics_DataGeneration(cParams, gParams, sParams):
    #
    # do not run this inside CoherentState or PolaronHamiltonian
    import CoherentState
    import PolaronHamiltonian
    # takes parameters, performs dynamics, and outputs desired observables
    [P, aIBi] = cParams
    [xgrid, kgrid, tgrid] = gParams
    [mI, mB, n0, gBB] = sParams

    # grid unpacking
    x = xgrid.getArray('x'); y = xgrid.getArray('y'); z = xgrid.getArray('z')
    dx = xgrid.arrays_diff['x']; dy = xgrid.arrays_diff['y']; dz = xgrid.arrays_diff['z']
    Nx, Ny, Nz = len(xgrid.getArray('x')), len(xgrid.getArray('y')), len(xgrid.getArray('z'))
    kx = kgrid.getArray('kx'); ky = kgrid.getArray('ky'); kz = kgrid.getArray('kz')
    dkx = kgrid.arrays_diff['kx']; dky = kgrid.arrays_diff['ky']; dkz = kgrid.arrays_diff['kz']

    grid_size_args = Nx, Ny, Nz
    grid_diff_args = dx, dy, dz, dkx, dky, dkz

    NGridPoints = xgrid.size()
    k_max = np.sqrt(np.max(kx)**2 + np.max(ky)**2 + np.max(kz)**2)

    # Initialization CoherentState
    cs = CoherentState.CoherentState(kgrid, xgrid)
    # Initialization PolaronHamiltonian
    Params = [P, aIBi, mI, mB, n0, gBB]
    ham = PolaronHamiltonian.PolaronHamiltonian(cs, Params)
    # calculate some parameters
    nu_const = nu(gBB)
    gIB = g(cs.kxg, cs.kyg, cs.kzg, cs.dVk[0], aIBi, mI, mB, n0, gBB)
    # Other book-keeping
    PIgrid = ImpMomGrid_from_PhononMomGrid(kgrid, P)
    PB_x = kx; PB_y = ky; PB_z = kz
    PI_x = PIgrid.getArray('kx'); PI_y = PIgrid.getArray('ky'); PI_z = PIgrid.getArray('kz')

    # Time evolution

    # Choose coarsegrain step size
    maxfac = 1
    largest_coarsegrain = 2
    for f in range(1, largest_coarsegrain + 1, 1):
        if tgrid.size % f == 0:
            maxfac = f
    tgrid_coarse = np.zeros(int(tgrid.size / maxfac), dtype=float)
    cind = 0
    print('Time grid size: {0}, Coarse grain step: {1}'.format(tgrid.size, maxfac))

    # Initialize observable data vectors
    PB_tVec = np.zeros(tgrid.size, dtype=float)
    NB_tVec = np.zeros(tgrid.size, dtype=float)
    DynOv_tVec = np.zeros(tgrid.size, dtype=complex)
    Phase_tVec = np.zeros(tgrid.size, dtype=float)
    # Initialize distribution data vectors
    nxyz_x_slice_ctVec = np.empty(tgrid_coarse.size, dtype=np.object); nxyz_y_slice_ctVec = np.empty(tgrid_coarse.size, dtype=np.object); nxyz_z_slice_ctVec = np.empty(tgrid_coarse.size, dtype=np.object)
    nPB_x_slice_ctVec = np.empty(tgrid_coarse.size, dtype=np.object); nPB_y_slice_ctVec = np.empty(tgrid_coarse.size, dtype=np.object); nPB_z_slice_ctVec = np.empty(tgrid_coarse.size, dtype=np.object)
    nPI_x_slice_ctVec = np.empty(tgrid_coarse.size, dtype=np.object); nPI_y_slice_ctVec = np.empty(tgrid_coarse.size, dtype=np.object); nPI_z_slice_ctVec = np.empty(tgrid_coarse.size, dtype=np.object)
    nxyz_xz_slice_ctVec = np.empty(tgrid_coarse.size, dtype=np.object); nxyz_xy_slice_ctVec = np.empty(tgrid_coarse.size, dtype=np.object)
    nPB_xz_slice_ctVec = np.empty(tgrid_coarse.size, dtype=np.object); nPB_xy_slice_ctVec = np.empty(tgrid_coarse.size, dtype=np.object)
    nPI_xz_slice_ctVec = np.empty(tgrid_coarse.size, dtype=np.object); nPI_xy_slice_ctVec = np.empty(tgrid_coarse.size, dtype=np.object)
    nxyz_x_ctVec = np.empty(tgrid_coarse.size, dtype=np.object); nxyz_y_ctVec = np.empty(tgrid_coarse.size, dtype=np.object); nxyz_z_ctVec = np.empty(tgrid_coarse.size, dtype=np.object)
    nPB_x_ctVec = np.empty(tgrid_coarse.size, dtype=np.object); nPB_y_ctVec = np.empty(tgrid_coarse.size, dtype=np.object); nPB_z_ctVec = np.empty(tgrid_coarse.size, dtype=np.object)
    nPI_x_ctVec = np.empty(tgrid_coarse.size, dtype=np.object); nPI_y_ctVec = np.empty(tgrid_coarse.size, dtype=np.object); nPI_z_ctVec = np.empty(tgrid_coarse.size, dtype=np.object)
    phonon_mom_k0deltapeak_ctVec = np.zeros(tgrid_coarse.size, dtype=float)
    nPBm_ctVec = np.empty(tgrid_coarse.size, dtype=np.object)
    nPIm_ctVec = np.empty(tgrid_coarse.size, dtype=np.object)
    PBm = 0
    PIm = 0

    start = timer()
    for ind, t in enumerate(tgrid):
        if ind == 0:
            dt = t
            cs.evolve(dt, ham)
        else:
            dt = t - tgrid[ind - 1]
            cs.evolve(dt, ham)

        PB_tVec[ind] = cs.get_PhononMomentum()
        NB_tVec[ind] = cs.get_PhononNumber()
        DynOv_tVec[ind] = cs.get_DynOverlap()
        Phase_tVec[ind] = cs.get_Phase()

        # save distribution data every 10 time values
        # if t != 0 and (ind + 1) % maxfac == 0:
        if (ind + 1) % maxfac == 0:
            # calculate distribution information
            phonon_pos_dist, phonon_mom_dist, phonon_mom_k0deltapeak_ctVec[cind] = cs.get_PhononDistributions()
            pos_slices, mom_slices, cont_slices, pos_integration, mom_integration = xyzDist_ProjSlices(phonon_pos_dist, phonon_mom_dist, grid_size_args, grid_diff_args)
            [PBm_Vec, nPBm_ctVec[cind], PIm_Vec, nPIm_ctVec[cind]] = xyzDist_To_magDist(cs.kgrid, phonon_mom_dist, P)

            # unpack above calculations and store data
            nxyz_x_slice_ctVec[cind], nxyz_y_slice_ctVec[cind], nxyz_z_slice_ctVec[cind] = pos_slices
            nPB_x_slice_ctVec[cind], nPB_y_slice_ctVec[cind], nPB_z_slice_ctVec[cind], nPI_x_slice_ctVec[cind], nPI_y_slice_ctVec[cind], nPI_z_slice_ctVec[cind] = mom_slices
            nxyz_xz_slice_ctVec[cind], nxyz_xy_slice_ctVec[cind], nPB_xz_slice_ctVec[cind], nPB_xy_slice_ctVec[cind], nPI_xz_slice_ctVec[cind], nPI_xy_slice_ctVec[cind] = cont_slices
            nxyz_x_ctVec[cind], nxyz_y_ctVec[cind], nxyz_z_ctVec[cind] = pos_integration
            nPB_x_ctVec[cind], nPB_y_ctVec[cind], nPB_z_ctVec[cind], nPI_x_ctVec[cind], nPI_y_ctVec[cind], nPI_z_ctVec[cind] = mom_integration
            tgrid_coarse[cind] = t
            cind += 1

        end = timer()
        print('t: {:.2f}, cst: {:.2f}, dt: {:.3f}, cind: {:d}, runtime: {:.3f}'.format(t, cs.time, dt, cind, end - start))
        start = timer()

    PBm = PBm_Vec
    PIm = PIm_Vec

    # Save Data - note: _tVec means depends on tgrid, _ctVec means depends on tgrid_coarse, everything else not time-dependent

    time_grids = [tgrid, tgrid_coarse]
    metrics_data = [NGridPoints, k_max, P, aIBi, mI, mB, n0, gBB, nu_const, gIB, PB_tVec, NB_tVec, DynOv_tVec, Phase_tVec]
    pos_xyz_data = [x, y, z, nxyz_x_ctVec, nxyz_y_ctVec, nxyz_z_ctVec, nxyz_x_slice_ctVec, nxyz_y_slice_ctVec, nxyz_z_slice_ctVec]
    mom_xyz_data = [PB_x, PB_y, PB_z, nPB_x_ctVec, nPB_y_ctVec, nPB_z_ctVec, nPB_x_slice_ctVec, nPB_y_slice_ctVec, nPB_z_slice_ctVec, PI_x, PI_y, PI_z, nPI_x_ctVec, nPI_y_ctVec, nPI_z_ctVec, nPI_x_slice_ctVec, nPI_y_slice_ctVec, nPI_z_slice_ctVec, phonon_mom_k0deltapeak_ctVec]
    cont_xyz_data = [nxyz_xz_slice_ctVec, nxyz_xy_slice_ctVec, nPB_xz_slice_ctVec, nPB_xy_slice_ctVec, nPI_xz_slice_ctVec, nPI_xy_slice_ctVec]
    mom_mag_data = [PBm, nPBm_ctVec, PIm, nPIm_ctVec]

    return time_grids, metrics_data, pos_xyz_data, mom_xyz_data, cont_xyz_data, mom_mag_data
