import numpy as np
import pandas as pd
import xarray as xr
from scipy.integrate import quad
from timeit import default_timer as timer
import os

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
    return np.sqrt(ep * (ep + 2 * gBB * n0))


def Omega(kgrid, DP, mI, mB, n0, gBB):
    names = list(kgrid.arrays.keys())  # ***need to have arrays added as k, th when kgrid is created
    if names[0] != 'k':
        print('CREATED kgrid IN WRONG ORDER')
    functions_omega0 = [lambda k: omegak(k, mB, n0, gBB) + (k**2 / (2 * mI)), lambda th: 0 * th + 1]
    omega0 = kgrid.function_prod(names, functions_omega0)
    return omega0 - kcos_func(kgrid) * DP / mI


def Wk(kgrid, mB, n0, gBB):
    names = list(kgrid.arrays.keys())
    functions_Wk = [lambda k: np.sqrt(epsilon(k, mB) / omegak(k, mB, n0, gBB)), lambda th: 0 * th + 1]
    return kgrid.function_prod(names, functions_Wk)


def g(kgrid, aIBi, mI, mB, n0, gBB):
    # gives bare interaction strength constant
    k_max = kgrid.getArray('k')[-1]
    mR = ur(mI, mB)
    return 1 / ((mR / (2 * np.pi)) * aIBi - (mR / np.pi**2) * k_max)


# ---- OTHER HELPER FUNCTIONS AND DYNAMICS ----


# def PCrit_inf(kcutoff, aIBi, mI, mB, n0, gBB):
#     #
#     DP = mI * nu(gBB)  # condition for critical momentum is P-PB = mI*nu where nu is the speed of sound
#     # non-grid helper function

#     def Wk(k, gBB, mB, n0):
#         return np.sqrt(eB(k, mB) / w(k, gBB, mB, n0))

#     # calculate aSi
#     def integrand(k): return (4 * ur(mI, mB) / (k**2) - ((Wk(k, gBB, mB, n0)**2) / (DP * k / mI)) * np.log((w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)) / (w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)))) * (k**2)
#     val, abserr = quad(integrand, 0, kcutoff, epsabs=0, epsrel=1.49e-12)
#     aSi = (1 / (2 * np.pi * ur(mI, mB))) * val
#     # calculate PB (phonon momentum)

#     def integrand(k): return ((2 * (w(k, gBB, mB, n0) + (k**2) / (2 * mI)) * (DP * k / mI) + (w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)) * (w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)) * np.log((w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)) / (w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)))) / ((w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)) * (w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)) * (DP * k / mI)**2)) * (Wk(k, gBB, mB, n0)**2) * (k**3)
#     val, abserr = quad(integrand, 0, kcutoff, epsabs=0, epsrel=1.49e-12)
#     PB = n0 / (ur(mI, mB)**2 * (aIBi - aSi)**2) * val

#     return DP + PB

def dirRF(dataset, kgrid):
    CSAmp = dataset['Real_CSAmp'] + 1j * dataset['Imag_CSAmp']
    Phase = dataset['Phase']
    dVk = kgrid.dV()
    tgrid = CSAmp.coords['t'].values
    CSA0 = CSAmp.isel(t=0).values; CSA0 = CSA0.reshape(CSA0.size)
    Phase0 = Phase.isel(t=0).values
    DynOv_Vec = np.zeros(tgrid.size, dtype=complex)

    for tind, t in enumerate(tgrid):
        CSAt = CSAmp.sel(t=t).values; CSAt = CSAt.reshape(CSAt.size)
        Phaset = Phase.sel(t=t).values
        exparg = np.dot(np.abs(CSAt)**2 + np.abs(CSA0)**2 - 2 * CSA0.conjugate() * CSAt, dVk)
        DynOv_Vec[tind] = np.exp(-1j * (Phaset - Phase0)) * np.exp((-1 / 2) * exparg)

    ReDynOv_da = xr.DataArray(np.real(DynOv_Vec), coords=[tgrid], dims=['t'])
    ImDynOv_da = xr.DataArray(np.imag(DynOv_Vec), coords=[tgrid], dims=['t'])
    dirRF_ds = xr.Dataset({'Real_DynOv': ReDynOv_da, 'Imag_DynOv': ImDynOv_da}, coords={'t': tgrid}, attrs=dataset.attrs)
    return dirRF_ds


def spectFunc(t_Vec, S_Vec, tdecay):
    # spectral function (Fourier Transform of dynamical overlap)
    dt = t_Vec[1] - t_Vec[0]
    Nt = t_Vec.size
    domega = 2 * np.pi / (Nt * dt)
    decayFactor = np.exp(-1 * t_Vec / tdecay)
    Sarg = np.fft.ifftshift(S_Vec * decayFactor)
    sf_preshift = 2 * np.real((1 / domega) * np.fft.ifft(Sarg))
    sf = np.fft.fftshift(sf_preshift)
    omega = np.fft.fftshift((2 * np.pi / dt) * np.fft.fftfreq(Nt))
    return omega, sf

# ---- DYNAMICS ----


def quenchDynamics_DataGeneration(cParams, gParams, sParams, toggleDict):
    #
    # do not run this inside CoherentState or PolaronHamiltonian
    import CoherentState
    import PolaronHamiltonian
    # takes parameters, performs dynamics, and outputs desired observables
    [P, aIBi] = cParams
    [xgrid, kgrid, tgrid] = gParams
    [mI, mB, n0, gBB] = sParams

    NGridPoints = kgrid.size()
    k_max = kgrid.getArray('k')[-1]
    kVec = kgrid.getArray('k')
    thVec = kgrid.getArray('th')

    # calculate some parameters
    nu_const = nu(gBB)
    gIB = g(kgrid, aIBi, mI, mB, n0, gBB)

    # Initialization CoherentState
    cs = CoherentState.CoherentState(kgrid, xgrid)

    # Initialization PolaronHamiltonian
    Params = [P, aIBi, mI, mB, n0, gBB]
    ham = PolaronHamiltonian.PolaronHamiltonian(cs, Params, toggleDict)

    # Change initialization of CoherentState and PolaronHamiltonian for Direct RF Real-time evolution in the non-interacting state
    if toggleDict['InitCS'] == 'file':
        ds = xr.open_dataset(toggleDict['InitCS_datapath'] + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))
        CSAmp = (ds['Real_CSAmp'] + 1j * ds['Imag_CSAmp']).values
        cs.amplitude_phase[0:-1] = CSAmp.reshape(CSAmp.size)  # this is the initial condition for quenching the impurity from the interacting state to the non-interacting state
        cs.amplitude_phase[-1] = ds['Phase'].values

    if toggleDict['Interaction'] == 'off':
        ham.gnum = 0

    # Time evolution

    # Initialize observable Data Arrays
    PB_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    NB_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    ReDynOv_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    ImDynOv_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    Phase_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    ReAmp_da = xr.DataArray(np.full((tgrid.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid, kVec, thVec], dims=['t', 'k', 'th'])
    ImAmp_da = xr.DataArray(np.full((tgrid.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid, kVec, thVec], dims=['t', 'k', 'th'])
    ReDeltaAmp_da = xr.DataArray(np.full((tgrid.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid, kVec, thVec], dims=['t', 'k', 'th'])
    ImDeltaAmp_da = xr.DataArray(np.full((tgrid.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid, kVec, thVec], dims=['t', 'k', 'th'])

    start = timer()
    for ind, t in enumerate(tgrid):
        if ind == 0:
            dt = t
            cs.evolve(dt, ham)
        else:
            dt = t - tgrid[ind - 1]
            cs.evolve(dt, ham)

        PB_da[ind] = cs.get_PhononMomentum()
        NB_da[ind] = cs.get_PhononNumber()
        DynOv = cs.get_DynOverlap()
        ReDynOv_da[ind] = np.real(DynOv)
        ImDynOv_da[ind] = np.imag(DynOv)
        Phase_da[ind] = cs.get_Phase()
        Amp = cs.get_Amplitude().reshape(len(kVec), len(thVec))
        ReAmp_da[ind] = np.real(Amp)
        ImAmp_da[ind] = np.imag(Amp)

        amplitude = cs.get_Amplitude()
        PB = np.dot(ham.kz * np.abs(amplitude)**2, cs.dVk)
        betaSum = amplitude + np.conjugate(amplitude)
        xp = 0.5 * np.dot(ham.Wk_grid, betaSum * cs.dVk)
        betaDiff = amplitude - np.conjugate(amplitude)
        xm = 0.5 * np.dot(ham.Wki_grid, betaDiff * cs.dVk)

        damp = -1j * (ham.gnum * np.sqrt(n0) * ham.Wk_grid +
                      amplitude * (ham.Omega0_grid - ham.kz * (P - PB) / mI) +
                      ham.gnum * (ham.Wk_grid * xp + ham.Wki_grid * xm))

        DeltaAmp = damp.reshape(len(kVec), len(thVec))
        ReDeltaAmp_da[ind] = np.real(DeltaAmp)
        ImDeltaAmp_da[ind] = np.imag(DeltaAmp)

        end = timer()
        print('t: {:.2f}, cst: {:.2f}, dt: {:.3f}, runtime: {:.3f}'.format(t, cs.time, dt, end - start))
        start = timer()

    # Create Data Set

    data_dict = {'Pph': PB_da, 'Nph': NB_da, 'Real_DynOv': ReDynOv_da, 'Imag_DynOv': ImDynOv_da, 'Phase': Phase_da, 'Real_CSAmp': ReAmp_da, 'Imag_CSAmp': ImAmp_da, 'Real_Delta_CSAmp': ReDeltaAmp_da, 'Imag_Delta_CSAmp': ImDeltaAmp_da}
    coords_dict = {'t': tgrid}
    attrs_dict = {'NGridPoints': NGridPoints, 'k_mag_cutoff': k_max, 'P': P, 'aIBi': aIBi, 'mI': mI, 'mB': mB, 'n0': n0, 'gBB': gBB, 'nu': nu_const, 'gIB': gIB}

    dynsph_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)

    return dynsph_ds


def LDA_quenchDynamics_DataGeneration(cParams, gParams, sParams, fParams, LDA_funcs, toggleDict):
    #
    # do not run this inside CoherentState or PolaronHamiltonian
    import LDA_CoherentState
    import LDA_PolaronHamiltonian
    # takes parameters, performs dynamics, and outputs desired observables
    [Fext_mag, aIBi] = cParams
    [xgrid, kgrid, tgrid] = gParams
    [mI, mB, n0, gBB] = sParams
    [dP] = fParams

    NGridPoints = kgrid.size()
    k_max = kgrid.getArray('k')[-1]
    kVec = kgrid.getArray('k')
    thVec = kgrid.getArray('th')

    # calculate some parameters
    nu_const = nu(gBB)
    gIB = g(kgrid, aIBi, mI, mB, n0, gBB)
    aBB = (mB / (4 * np.pi)) * gBB
    xi = np.sqrt(8 * np.pi * n0 * aBB)

    # Initialization CoherentState
    cs = LDA_CoherentState.LDA_CoherentState(kgrid, xgrid)

    # Initialization PolaronHamiltonian
    Params = [aIBi, mI, mB, n0, gBB]
    LDA_Params = [Fext_mag, dP]
    ham = LDA_PolaronHamiltonian.LDA_PolaronHamiltonian(cs, Params, LDA_funcs, LDA_Params, toggleDict)

    # Change initialization of CoherentState and PolaronHamiltonian for Direct RF Real-time evolution in the non-interacting state
    if toggleDict['InitCS'] == 'file':
        ds = xr.open_dataset(toggleDict['InitCS_datapath'] + '/initPolState_aIBi_{:.2f}.nc'.format(aIBi))
        CSAmp = (ds['Real_CSAmp'] + 1j * ds['Imag_CSAmp']).values
        CSPhase = ds['Phase'].values
        cs.set_initState(amplitude=CSAmp.reshape(CSAmp.size), phase=CSPhase, P=0, X=0)

    if toggleDict['Interaction'] == 'off':
        ham.gnum = 0

    # Time evolution

    # Initialize observable Data Arrays
    Pph_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    Nph_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    Phase_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    ReAmp_da = xr.DataArray(np.full((tgrid.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid, kVec, thVec], dims=['t', 'k', 'th'])
    ImAmp_da = xr.DataArray(np.full((tgrid.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid, kVec, thVec], dims=['t', 'k', 'th'])

    P_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    X_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])

    start = timer()
    for ind, t in enumerate(tgrid):
        if ind == 0:
            dt = t
            cs.evolve(dt, ham)
        else:
            dt = t - tgrid[ind - 1]
            cs.evolve(dt, ham)

        Pph_da[ind] = cs.get_PhononMomentum()
        Nph_da[ind] = cs.get_PhononNumber()
        Phase_da[ind] = cs.get_Phase()
        Amp = cs.get_Amplitude().reshape(len(kVec), len(thVec))
        ReAmp_da[ind] = np.real(Amp)
        ImAmp_da[ind] = np.imag(Amp)
        P_da[ind] = cs.get_totMom()
        X_da[ind] = cs.get_impPos()

        end = timer()
        print('t: {:.2f}, cst: {:.2f}, dt: {:.3f}, runtime: {:.3f}'.format(t, cs.time, dt, end - start))
        start = timer()

    # Create Data Set

    data_dict = {'Pph': Pph_da, 'Nph': Nph_da, 'Phase': Phase_da, 'Real_CSAmp': ReAmp_da, 'Imag_CSAmp': ImAmp_da, 'P': P_da, 'X': X_da}
    coords_dict = {'t': tgrid}
    attrs_dict = {'NGridPoints': NGridPoints, 'k_mag_cutoff': k_max, 'aIBi': aIBi, 'mI': mI, 'mB': mB, 'n0': n0, 'gBB': gBB, 'nu': nu_const, 'gIB': gIB, 'xi': xi, 'Fext_mag': Fext_mag, 'TF': dP / Fext_mag}

    dynsph_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)

    return dynsph_ds
