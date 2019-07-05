import numpy as np
import pandas as pd
import xarray as xr
from scipy.integrate import quad
import scipy.interpolate as spi
import pf_static_sph
from scipy import interpolate
from timeit import default_timer as timer

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


def nu(mB, n0, gBB):
    return np.sqrt(n0 * gBB / mB)


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


# ---- SPECTRUM RELATED FUNCTIONS ----


# def PCrit_inf(kcutoff, aIBi, mI, mB, n0, gBB):
#     #
#     DP = mI * nu(mB, n0, gBB)  # condition for critical momentum is P-PB = mI*nu where nu is the speed of sound
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

def dirRF(dataset, kgrid, cParams, sParams):
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

    # calculate polaron energy (energy of initial state CSA0)
    [P, aIBi] = cParams
    [mI, mB, n0, gBB] = sParams
    dVk = kgrid.dV()
    kzg_flat = kcos_func(kgrid)
    gIB = g(kgrid, aIBi, mI, mB, n0, gBB)
    PB0 = np.dot(kzg_flat * np.abs(CSA0)**2, dVk).real.astype(float)
    DP0 = P - PB0
    Energy0 = (P**2 - PB0**2) / (2 * mI) + np.dot(Omega(kgrid, DP0, mI, mB, n0, gBB) * np.abs(CSA0)**2, dVk) + gIB * (np.dot(Wk(kgrid, mB, n0, gBB) * CSA0, dVk) + np.sqrt(n0))**2

    # calculate full dynamical overlap
    DynOv_Vec = np.exp(1j * Energy0) * DynOv_Vec
    ReDynOv_da = xr.DataArray(np.real(DynOv_Vec), coords=[tgrid], dims=['t'])
    ImDynOv_da = xr.DataArray(np.imag(DynOv_Vec), coords=[tgrid], dims=['t'])
    # DynOv_ds = xr.Dataset({'Real_DynOv': ReDynOv_da, 'Imag_DynOv': ImDynOv_da}, coords={'t': tgrid}, attrs=dataset.attrs)
    DynOv_ds = dataset[['Real_CSAmp', 'Imag_CSAmp', 'Phase']]; DynOv_ds['Real_DynOv'] = ReDynOv_da; DynOv_ds['Imag_DynOv'] = ImDynOv_da; DynOv_ds.attrs = dataset.attrs
    return DynOv_ds


# def spectFunc(t_Vec, S_Vec, tdecay):
#     # spectral function (Fourier Transform of dynamical overlap) using convention A(omega) = 2*Re[\int {S(t)*e^(-i*omega*t)}]
#     dt = t_Vec[1] - t_Vec[0]
#     Nt = t_Vec.size
#     decayFactor = np.exp(-1 * t_Vec / tdecay)
#     Sarg = S_Vec * decayFactor
#     sf_preshift = 2 * np.real(dt * np.fft.fft(Sarg))
#     sf = np.fft.fftshift(sf_preshift)
#     omega = np.fft.fftshift((2 * np.pi / dt) * np.fft.fftfreq(Nt))
#     return omega, sf


def spectFunc(t_Vec, S_Vec, tdecay):
    # spectral function (Fourier Transform of dynamical overlap) using convention A(omega) = 2*Re[\int {S(t)*e^(i*omega*t)}]
    dt = t_Vec[1] - t_Vec[0]
    Nt = t_Vec.size
    domega = 2 * np.pi / (Nt * dt)
    decayFactor = np.exp(-1 * t_Vec / tdecay)
    Sarg = S_Vec * decayFactor
    sf_preshift = np.real((2 * np.pi / domega) * np.fft.ifft(Sarg))
    # sf_preshift = 2 * np.real((2 * np.pi / domega) * np.fft.ifft(Sarg))
    sf = np.fft.fftshift(sf_preshift)
    omega = np.fft.fftshift((2 * np.pi / dt) * np.fft.fftfreq(Nt))
    return omega, sf


def Energy(CSAmp, kgrid, P, aIBi, mI, mB, n0, gBB):
    dVk = kgrid.dV()
    kzg_flat = kcos_func(kgrid)
    Wk_grid = Wk(kgrid, mB, n0, gBB)
    Wki_grid = 1 / Wk_grid

    amplitude = CSAmp.reshape(CSAmp.size)
    PB = np.dot(kzg_flat * np.abs(amplitude)**2, dVk).real.astype(float)
    DP = P - PB
    Omega_grid = Omega(kgrid, DP, mI, mB, n0, gBB)
    gnum = g(kgrid, aIBi, mI, mB, n0, gBB)

    xp = 0.5 * np.dot(Wk_grid, amplitude * dVk)
    xm = 0.5 * np.dot(Wki_grid, amplitude * dVk)
    En = ((P**2 - PB**2) / (2 * mI) +
          np.dot(dVk * Omega_grid, np.abs(amplitude)**2) +
          gnum * (2 * np.real(xp) + np.sqrt(n0))**2 -
          gnum * (2 * np.imag(xm))**2)

    return En.real.astype(float)

# ---- LDA/FORCE FUNCTIONS ----


def n_thomasFermi(X, Y, Z, n0_TF, RTF_X, RTF_Y, RTF_Z):
    # returns 3D density using Thomas-Fermi approximation where n0 is the central density and RTF is the Thomas-Fermi radius in each direction
    # nTF = (n0 * 15 / (8 * np.pi * RTF_X * RTF_Y * RTF_Z)) * (1 - X**2 / RTF_X**2 - Y**2 / RTF_Y**2 - Z**2 / RTF_Z**2)
    nTF = n0_TF * (1 - X**2 / RTF_X**2 - Y**2 / RTF_Y**2 - Z**2 / RTF_Z**2)
    if np.isscalar(nTF):
        if nTF > 0:
            return nTF
        else:
            return 0
    else:
        nTF[nTF < 0] = 0
        return nTF


def n_thermal(X, Y, Z, n0_thermal, RG_X, RG_Y, RG_Z):
    # returns thermal correction to density assuming Gaussian profile given Gaussian waists and n0_thermal which is central thermal density
    # return (n0_thermal / (RG_X * RG_Y * RG_Z * np.pi**(1.5))) * np.exp(-1 * (X**2 / RG_X**2 + Y**2 / RG_Y**2 + Z**2 / RG_Z**2))
    return n0_thermal * np.exp(-1 * (X**2 / RG_X**2 + Y**2 / RG_Y**2 + Z**2 / RG_Z**2))


def n_BEC(X, Y, Z, n0_TF, n0_thermal, RTF_X, RTF_Y, RTF_Z, RG_X, RG_Y, RG_Z):
    return n_thomasFermi(X, Y, Z, n0_TF, RTF_X, RTF_Y, RTF_Z) + n_thermal(X, Y, Z, n0_thermal, RG_X, RG_Y, RG_Z)


def V_Pol_interp(kgrid, X_Vals, cParams, sParams, trapParams):
    # returns an interpolation function for the polaron energy
    [mI, mB, n0, gBB] = sParams
    aIBi = cParams['aIBi']
    n0_TF = trapParams['n0_TF_BEC']; RTF_X = trapParams['RTF_BEC_X']; RTF_Y = trapParams['RTF_BEC_Y']; RTF_Z = trapParams['RTF_BEC_Z']
    n0_thermal = trapParams['n0_thermal_BEC']; RG_X = trapParams['RG_BEC_X']; RG_Y = trapParams['RG_BEC_Y']; RG_Z = trapParams['RG_BEC_Z']

    # ASSUMING FIXED ABB, KIB, ASSUMING POTENTIAL IS FOR ZERO MOMENTUM, AND ASSUMING PARTICLE IS IN CENTER OF TRAP IN Y AND Z DIRECTIONS
    DP = 0
    P = 0

    EpVals = np.zeros(X_Vals.size)
    for ind, X in enumerate(X_Vals):
        n = n_BEC(X, 0, 0, n0_TF, n0_thermal, RTF_X, RTF_Y, RTF_Z, RG_X, RG_Y, RG_Z)
        aSi = pf_static_sph.aSi_grid(kgrid, DP, mI, mB, n, gBB)
        PB = pf_static_sph.PB_integral_grid(kgrid, DP, mI, mB, n, gBB)
        EpVals[ind] = pf_static_sph.Energy(P, PB, aIBi, aSi, mI, mB, n)

    E_Pol_tck = interpolate.splrep(X_Vals, EpVals, s=0)
    return E_Pol_tck


def F_pol(X, E_Pol_tck):
    return -1 * interpolate.splev(X, E_Pol_tck, der=1)


def F_ext(t, F, dP):
    TF = dP / F
    if t <= TF:
        return F
    else:
        return 0


def x_BEC_osc(t, omega_BEC_osc, RTF_X, a_osc):
    # returns function describing oscillation of BEC (peak) over time
    return a_osc * RTF_X * np.cos(omega_BEC_osc * t)


def F_BEC_osc(t, omega_BEC_osc, RTF_X, a_osc, mI):
    # returns function describing oscillation of BEC (peak) over time
    return -1 * mI * (omega_BEC_osc**2) * x_BEC_osc(t, omega_BEC_osc, RTF_X, a_osc)


def F_Imp_trap(X, omega_Imp_x, mI):
    # returns function describing force on impurity from harmonic potential confining impurity in the direction of motion
    return -1 * mI * (omega_Imp_x**2) * X


# ---- OTHER FUNCTIONS ----


def unitConv_exp2th(n0_exp_scale, mB_exp):
    # Theory scale is set by experimental order of magnitude of total peak BEC density n0=1 (length), boson mass mB=1 (mass), hbar = 1 (time) -> note that n0_exp_scale isn't the actual peak BEC density, just a fixed order of magnitude (here ~1e14 cm^(-3))
    # This function takes experimental values for these quantities in SI units
    # The output are conversion factors for length, mass, and time from experiment (SI units) to theory
    # For example, if I have a quantity aBB_exp in meters, then aBB_th = aBB_exp * L_exp2th gives the quantity in theory units

    # Constants (SI units)
    hbar = 1.0555e-34  # reduced Planck's constant (J*s/rad)

    # Choice of theoretical scale
    n0_th = 1
    mB_th = 1
    hbar_th = 1

    # Conversion factors for length, mass, time
    L_exp2th = n0_th**(-1 / 3) / n0_exp_scale**(-1 / 3)
    M_exp2th = mB_th / mB_exp
    T_exp2th = (mB_th * (n0_th)**(-2 / 3) / (2 * np.pi * hbar_th)) / (mB_exp * (n0_exp_scale)**(-2 / 3) / (2 * np.pi * hbar))

    return L_exp2th, M_exp2th, T_exp2th


# def Zw_expParams():
#     # Constants (SI units)
#     a0 = 5.29e-11  # Bohr radius (m)
#     u = 1.661e-27  # atomic mass unit (kg)
#     params = {}

#     # Experimental parameters (SI units)
#     params['aIB'] = -3000 * a0
#     params['aBB'] = 52 * a0
#     params['n0_TF'] = 6e13 * 1e6  # BEC TF peak density in m^(-3)
#     params['n0_thermal'] = 0.9e13 * 1e6  # BEC thermal Gaussian peak density in m^(-3)
#     params['n0_BEC'] = params['n0_TF'] + params['n0_thermal']  # Total BEC peak (central) density in m^(-3)
#     params['nI'] = 1.4e11 * 1e6  # impurity peak density
#     params['n0_BEC_scale'] = 1e14 * 1e6  # order of magnitude scale of peak BEC density in m^(-3)
#     params['omega_BEC_x'] = 2 * np.pi * 101; params['omega_BEC_y'] = 2 * np.pi * 41; params['omega_BEC_z'] = 2 * np.pi * 13  # BEC trapping frequencies in rad*Hz ***THESE DON'T MATCH UP TO THE TF RADII...
#     params['RTF_BEC_X'] = 103e-6; params['RTF_BEC_Y'] = 32e-6; params['RTF_BEC_Z'] = 13e-6  # BEC density Thomas-Fermi radii in each direction (m) assuming shallowest trap is direction of propagation X and second shallowest direction is Y
#     params['RG_BEC_X'] = 95e-6; params['RG_BEC_Y'] = 29e-6; params['RG_BEC_Z'] = 12e-6  # BEC density thermal Gaussian waists in each direction (m)
#     params['mI'] = 39.96 * u
#     params['mB'] = 22.99 * u
#     params['vI_init'] = 7 * 1e-3  # average initial velocity of impurities (m/s)
#     # params['omega_Imp_x'] = 2 * np.pi * 500  # Impurity trapping frequency in rad*Hz
#     params['omega_Imp_x'] = 2 * np.pi * 1000  # Impurity trapping frequency in rad*Hz
#     params['omega_BEC_osc'] = 2 * np.pi * 500  # BEC oscillation frequency in rad*Hz
#     # params['omega_BEC_osc'] = 2 * np.pi * 1.25e3  # BEC oscillation frequency in rad*Hz
#     params['a_osc'] = 0.5

#     return params


def Zw_expParams():
    # Constants (SI units)
    a0 = 5.29e-11  # Bohr radius (m)
    u = 1.661e-27  # atomic mass unit (kg)
    params = {}

    # Experimental parameters (SI units)
    params['aIB'] = -2600 * a0
    params['aBB'] = 52 * a0
    params['n0_TF'] = 6e13 * 1e6  # BEC TF peak density in m^(-3)
    params['n0_thermal'] = 0.9e13 * 1e6  # BEC thermal Gaussian peak density in m^(-3)
    params['n0_BEC'] = params['n0_TF'] + params['n0_thermal']  # Total BEC peak (central) density in m^(-3)
    params['nI'] = 1.4e11 * 1e6  # impurity peak density
    params['n0_BEC_scale'] = 1e14 * 1e6  # order of magnitude scale of peak BEC density in m^(-3)
    params['omega_BEC_x'] = 2 * np.pi * 80; params['omega_BEC_y'] = 2 * np.pi * 100; params['omega_BEC_z'] = 2 * np.pi * 12  # BEC trapping frequencies in rad*Hz ***THESE DON'T MATCH UP TO THE TF RADII...
    params['RTF_BEC_X'] = 103e-6; params['RTF_BEC_Y'] = 32e-6; params['RTF_BEC_Z'] = 13e-6  # BEC density Thomas-Fermi radii in each direction (m)
    params['RG_BEC_X'] = 95e-6; params['RG_BEC_Y'] = 29e-6; params['RG_BEC_Z'] = 12e-6  # BEC density thermal Gaussian waists in each direction (m)
    params['mI'] = 39.96 * u
    params['mB'] = 22.99 * u
    params['vI_init'] = 7 * 1e-3  # average initial velocity of impurities (m/s)
    params['omega_Imp_x'] = 2 * np.pi * 150  # Impurity trapping frequency in rad*Hz
    params['omega_BEC_osc'] = 2 * np.pi * 75  # BEC oscillation frequency in rad*Hz
    params['a_osc'] = 10e-6 / params['RTF_BEC_X']  # Initial displacement of BEC (m) divided by TF radius in the direction of displacement (x-direction)

    return params


def xinterp2D(xdataset, coord1, coord2, mult):
    # xdataset is the desired xarray dataset with the desired plotting quantity already selected
    # coord1 and coord2 are the two coordinates making the 2d plot
    # mul is the multiplicative factor by which you wish to increase the resolution of the grid
    # e.g. xdataset = qds['nPI_xz_slice'].sel(P=P,aIBi=aIBi).dropna('PI_z'), coord1 = 'PI_x', coord2 = 'PI_z'
    # returns meshgrid values for C1_interp and C2_interp as well as the function value on this 2D grid -> these are ready to plot
    C1 = xdataset.coords[coord1].values
    C2 = xdataset.coords[coord2].values
    C1g, C2g = np.meshgrid(C1, C2, indexing='ij')
    C1_interp = np.linspace(np.min(C1), np.max(C1), mult * C1.size)
    C2_interp = np.linspace(np.min(C2), np.max(C2), mult * C2.size)
    C1g_interp, C2g_interp = np.meshgrid(C1_interp, C2_interp, indexing='ij')
    interp_vals = spi.griddata((C1g.flatten(), C2g.flatten()), xdataset.values.flatten(), (C1g_interp, C2g_interp), method='cubic')
    return interp_vals, C1g_interp, C2g_interp

# ---- DYNAMICS ----


def LDA_quenchDynamics_DataGeneration(cParams, gParams, sParams, fParams, trapParams, toggleDict):
    #
    # do not run this inside CoherentState or PolaronHamiltonian
    import LDA_CoherentState
    import LDA_PolaronHamiltonian
    # takes parameters, performs dynamics, and outputs desired observables
    aIBi = cParams['aIBi']
    [xgrid, kgrid, tgrid] = gParams
    [mI, mB, n0, gBB] = sParams
    dP = fParams['dP_ext']; Fext_mag = fParams['Fext_mag']
    P0 = trapParams['P0']
    X0 = trapParams['X0']

    NGridPoints = kgrid.size()
    k_max = kgrid.getArray('k')[-1]
    kVec = kgrid.getArray('k')
    thVec = kgrid.getArray('th')

    # calculate some parameters
    nu_const = nu(mB, n0, gBB)
    gIB = g(kgrid, aIBi, mI, mB, n0, gBB)
    aBB = (mB / (4 * np.pi)) * gBB
    xi = (8 * np.pi * n0 * aBB)**(-1 / 2)

    # LDA Force functions
    LDA_funcs = {}
    if toggleDict['F_ext'] == 'on':
        LDA_funcs['F_ext'] = F_ext
        TF = dP / Fext_mag
    else:
        LDA_funcs['F_ext'] = lambda t, F, dP: 0
        TF = 0
        dP = 0
    if toggleDict['BEC_density'] == 'on':
        # assuming we only have a particle in the center of the trap that travels in the direction of largest Thomas Fermi radius (easy to generalize this)
        X_Vals = np.linspace(-1 * trapParams['RTF_BEC_X'] * 0.99, trapParams['RTF_BEC_X'] * 0.99, 100)
        E_Pol_tck = V_Pol_interp(kgrid, X_Vals, cParams, sParams, trapParams)
        LDA_funcs['F_pol'] = lambda X: F_pol(X, E_Pol_tck)
    else:
        LDA_funcs['F_pol'] = lambda X: 0
        omega_BEC_osc = 0
        a_osc = 0
    if toggleDict['BEC_density_osc'] == 'on':
        omega_BEC_osc = trapParams['omega_BEC_osc']
        a_osc = trapParams['a_osc']
        LDA_funcs['F_BEC_osc'] = lambda t: F_BEC_osc(t, omega_BEC_osc, trapParams['RTF_BEC_X'], a_osc, mI)
    else:
        omega_BEC_osc = 0
        a_osc = 0
        LDA_funcs['F_BEC_osc'] = lambda t: 0

    if toggleDict['Imp_trap'] == 'on':
        omega_Imp_x = trapParams['omega_Imp_x']
        LDA_funcs['F_Imp_trap'] = lambda X: F_Imp_trap(X, omega_Imp_x, mI)
    else:
        omega_Imp_x = 0
        LDA_funcs['F_Imp_trap'] = lambda X: 0

    # Initialization CoherentState
    cs = LDA_CoherentState.LDA_CoherentState(kgrid, xgrid)

    # Initialization PolaronHamiltonian
    Params = [aIBi, mI, mB, n0, gBB]
    ham = LDA_PolaronHamiltonian.LDA_PolaronHamiltonian(cs, Params, LDA_funcs, fParams, trapParams, toggleDict)

    # Change initialization of CoherentState and PolaronHamiltonian
    if toggleDict['InitCS'] == 'file':
        # ds = xr.open_dataset(toggleDict['InitCS_datapath'] + '/initPolState_aIBi_{:.2f}.nc'.format(aIBi))
        ds = xr.open_dataset(toggleDict['InitCS_datapath'] + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P0, aIBi))
        CSAmp = (ds['Real_CSAmp'] + 1j * ds['Imag_CSAmp']).values
        CSPhase = ds['Phase'].values
        cs.set_initState(amplitude=CSAmp.reshape(CSAmp.size), phase=CSPhase, P=P0, X=X0)
    elif toggleDict['InitCS'] == 'steadystate':
        Nsteps = 1e2
        aSi_tck, PBint_tck = pf_static_sph.createSpline_grid(Nsteps, kgrid, mI, mB, n0, gBB)
        DP = pf_static_sph.DP_interp(0, P0, aIBi, aSi_tck, PBint_tck)
        aSi = pf_static_sph.aSi_interp(DP, aSi_tck)
        CSAmp = pf_static_sph.BetaK(kgrid, aIBi, aSi, DP, mI, mB, n0, gBB)
        cs.set_initState(amplitude=CSAmp, phase=0, P=P0, X=X0)

    if toggleDict['Interaction'] == 'off':
        ham.gnum = 0

    # Time evolution

    # Initialize observable Data Arrays
    Pph_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    Nph_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    Phase_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    ReAmp_da = xr.DataArray(np.full((tgrid.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid, kVec, thVec], dims=['t', 'k', 'th'])
    ImAmp_da = xr.DataArray(np.full((tgrid.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid, kVec, thVec], dims=['t', 'k', 'th'])
    Energy_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])

    P_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    X_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    XLab_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])

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
        XLab_da[ind] = cs.get_impPos() + x_BEC_osc(t, omega_BEC_osc, trapParams['RTF_BEC_X'], a_osc)

        if toggleDict['BEC_density'] == 'on':
            n = n_BEC(cs.get_impPos(), 0, 0, trapParams['n0_TF_BEC'], trapParams['n0_thermal_BEC'], trapParams['RTF_BEC_X'], trapParams['RTF_BEC_Y'], trapParams['RTF_BEC_Z'], trapParams['RG_BEC_X'], trapParams['RG_BEC_Y'], trapParams['RG_BEC_Z'])  # ASSUMING PARTICLE IS IN CENTER OF TRAP IN Y AND Z DIRECTIONS
        else:
            n = n0
        Energy_da[ind] = Energy(Amp, kgrid, cs.get_totMom(), aIBi, mI, mB, n, gBB)

        end = timer()
        print('t: {:.2f}, cst: {:.2f}, dt: {:.3f}, runtime: {:.3f}'.format(t, cs.time, dt, end - start))
        start = timer()

    # Create Data Set

    data_dict = {'Pph': Pph_da, 'Nph': Nph_da, 'Phase': Phase_da, 'Real_CSAmp': ReAmp_da, 'Imag_CSAmp': ImAmp_da, 'P': P_da, 'X': X_da, 'XLab': XLab_da, 'Energy': Energy_da}
    coords_dict = {'t': tgrid}
    attrs_dict = {'NGridPoints': NGridPoints, 'k_mag_cutoff': k_max, 'aIBi': aIBi, 'mI': mI, 'mB': mB, 'n0': n0, 'gBB': gBB, 'nu': nu_const, 'gIB': gIB, 'xi': xi, 'Fext_mag': Fext_mag, 'TF': TF, 'Delta_P': dP, 'omega_BEC_osc': omega_BEC_osc, 'X0': X0, 'P0': P0, 'a_osc': a_osc, 'omega_Imp_x': omega_Imp_x}

    dynsph_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)

    return dynsph_ds

# def quenchDynamics_DataGeneration(cParams, gParams, sParams, toggleDict):
#     #
#     # do not run this inside CoherentState or PolaronHamiltonian
#     import CoherentState
#     import PolaronHamiltonian
#     # takes parameters, performs dynamics, and outputs desired observables
#     [P, aIBi] = cParams
#     [xgrid, kgrid, tgrid] = gParams
#     [mI, mB, n0, gBB] = sParams

#     NGridPoints = kgrid.size()
#     k_max = kgrid.getArray('k')[-1]
#     kVec = kgrid.getArray('k')
#     thVec = kgrid.getArray('th')

#     # calculate some parameters
#     nu_const = nu(mB, n0, gBB)
#     gIB = g(kgrid, aIBi, mI, mB, n0, gBB)

#     # Initialization CoherentState
#     cs = CoherentState.CoherentState(kgrid, xgrid)

#     # Initialization PolaronHamiltonian
#     Params = [P, aIBi, mI, mB, n0, gBB]
#     ham = PolaronHamiltonian.PolaronHamiltonian(cs, Params, toggleDict)

#     # Change initialization of CoherentState and PolaronHamiltonian for Direct RF Real-time evolution in the non-interacting state
#     if toggleDict['InitCS'] == 'file':
#         ds = xr.open_dataset(toggleDict['InitCS_datapath'] + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))
#         CSAmp = (ds['Real_CSAmp'] + 1j * ds['Imag_CSAmp']).values
#         cs.amplitude_phase[0:-1] = CSAmp.reshape(CSAmp.size)  # this is the initial condition for quenching the impurity from the interacting state to the non-interacting state
#         cs.amplitude_phase[-1] = ds['Phase'].values

#     if toggleDict['Interaction'] == 'off':
#         ham.gnum = 0

#     # Time evolution

#     # Initialize observable Data Arrays
#     PB_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
#     NB_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
#     ReDynOv_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
#     ImDynOv_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
#     Phase_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
#     ReAmp_da = xr.DataArray(np.full((tgrid.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid, kVec, thVec], dims=['t', 'k', 'th'])
#     ImAmp_da = xr.DataArray(np.full((tgrid.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid, kVec, thVec], dims=['t', 'k', 'th'])
#     ReDeltaAmp_da = xr.DataArray(np.full((tgrid.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid, kVec, thVec], dims=['t', 'k', 'th'])
#     ImDeltaAmp_da = xr.DataArray(np.full((tgrid.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid, kVec, thVec], dims=['t', 'k', 'th'])

#     start = timer()
#     for ind, t in enumerate(tgrid):
#         if ind == 0:
#             dt = t
#             cs.evolve(dt, ham)
#         else:
#             dt = t - tgrid[ind - 1]
#             cs.evolve(dt, ham)

#         PB_da[ind] = cs.get_PhononMomentum()
#         NB_da[ind] = cs.get_PhononNumber()
#         DynOv = np.exp(1j * P**2 / (2 * mI)) * cs.get_DynOverlap()
#         ReDynOv_da[ind] = np.real(DynOv)
#         ImDynOv_da[ind] = np.imag(DynOv)
#         Phase_da[ind] = cs.get_Phase()
#         Amp = cs.get_Amplitude().reshape(len(kVec), len(thVec))
#         ReAmp_da[ind] = np.real(Amp)
#         ImAmp_da[ind] = np.imag(Amp)

#         amplitude = cs.get_Amplitude()
#         PB = np.dot(ham.kz * np.abs(amplitude)**2, cs.dVk)
#         betaSum = amplitude + np.conjugate(amplitude)
#         xp = 0.5 * np.dot(ham.Wk_grid, betaSum * cs.dVk)
#         betaDiff = amplitude - np.conjugate(amplitude)
#         xm = 0.5 * np.dot(ham.Wki_grid, betaDiff * cs.dVk)

#         damp = -1j * (ham.gnum * np.sqrt(n0) * ham.Wk_grid +
#                       amplitude * (ham.Omega0_grid - ham.kz * (P - PB) / mI) +
#                       ham.gnum * (ham.Wk_grid * xp + ham.Wki_grid * xm))

#         DeltaAmp = damp.reshape(len(kVec), len(thVec))
#         ReDeltaAmp_da[ind] = np.real(DeltaAmp)
#         ImDeltaAmp_da[ind] = np.imag(DeltaAmp)

#         end = timer()
#         print('t: {:.2f}, cst: {:.2f}, dt: {:.3f}, runtime: {:.3f}'.format(t, cs.time, dt, end - start))
#         start = timer()

#     # Create Data Set

#     data_dict = {'Pph': PB_da, 'Nph': NB_da, 'Real_DynOv': ReDynOv_da, 'Imag_DynOv': ImDynOv_da, 'Phase': Phase_da, 'Real_CSAmp': ReAmp_da, 'Imag_CSAmp': ImAmp_da, 'Real_Delta_CSAmp': ReDeltaAmp_da, 'Imag_Delta_CSAmp': ImDeltaAmp_da}
#     coords_dict = {'t': tgrid}
#     attrs_dict = {'NGridPoints': NGridPoints, 'k_mag_cutoff': k_max, 'P': P, 'aIBi': aIBi, 'mI': mI, 'mB': mB, 'n0': n0, 'gBB': gBB, 'nu': nu_const, 'gIB': gIB}

#     dynsph_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)

#     return dynsph_ds