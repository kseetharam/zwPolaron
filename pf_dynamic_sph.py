import numpy as np
import pandas as pd
import xarray as xr
from scipy.integrate import quad
import scipy.interpolate as spi
import pf_static_sph
from scipy import interpolate
from timeit import default_timer as timer
import mpmath as mpm
import numdifftools as nd

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


def omegak_grid(kgrid, mB, n0, gBB):
    names = list(kgrid.arrays.keys())
    functions_Wk = [lambda k: omegak(k, mB, n0, gBB), lambda th: 0 * th + 1]
    return kgrid.function_prod(names, functions_Wk)


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


def den_BEC(x, y, z, xTF, yTF, zTF):
    return 15 / (8 * np.pi * xTF * yTF * zTF) * (1 - x ** 2 / xTF ** 2 - y ** 2 / yTF ** 2 - z ** 2 / zTF ** 2) * np.heaviside(1 - x ** 2 / xTF ** 2 - y ** 2 / yTF ** 2 - z ** 2 / zTF ** 2, 1 / 2)


def den_ThermU(x, y, z, omegaX, omegaY, omegaZ, mNa, beta, chemicalPotential):
    arg = mpm.polylog(1.5, np.exp(-beta * np.abs(chemicalPotential - 1 / 2 * mNa * 10 ** (-12) * (omegaX ** 2 * x ** 2 + omegaY ** 2 * y ** 2 + omegaZ ** 2 * z ** 2))))
    return np.real(complex(arg))


def becdensity_zw2021(x, y, z, omegaX, omegaY, omegaZ, temperature, zTF):

    # Note: Follows convention of experiment where z is the widest direction of the cloud, y is the oscillation direction, and x is the narrowest direction (slightly narrower than y) that is integrated out by the camera
    # (x, y, z) provided in uM; (omegaX, omegaY, omegeaZ) provided in rad*Hz; temperature provided in K; zTF provided in um
    # density is output in m^-3
    # Taken and slightly modified from MATLAB function provided by Zwierlein group.

    a0 = 5.29e-11  # Bohr radius (m)
    hbar = 1.0555e-34  # reduced Planck's constant (J*s/rad)
    PlanckConst = 2 * np.pi * hbar
    kB = 1.38064852e-23
    mNeutron = 1.674927471e-27
    mProton = 1.672621898e-27
    mNa = 12 * mNeutron + 11 * mProton
    # mK = 19 * mProton + 21 * mNeutron
    # mReduced = mK * mNa / (mK + mNa)

    xTF = zTF * omegaZ / omegaX
    yTF = zTF * omegaZ / omegaY

    chemicalPotential_kHz = 1 / 2 * mNa * omegaZ ** 2 * (zTF / 10 ** 6) ** 2 / (PlanckConst * 1000)

    chemicalPotential = chemicalPotential_kHz * 1000 * PlanckConst

    omegabar = (omegaX * omegaY * omegaZ) ** (1 / 3)
    a_s = 52 * a0  # Sodium F=1 scattering length, away from FB resonance
    # prefactorEn = hbar ** 2 / 4 / mReduced * (6 * np.pi ** 2) ** (2 / 3)  # prefer REDUCED MASS, not Boson mass
    Nc = (2 * chemicalPotential) ** (2.5) / (15 * hbar ** 2 * np.sqrt(mNa) * omegabar ** 3 * a_s)
    beta = 1 / kB / temperature
    lambdaDB = PlanckConst / np.sqrt(2 * np.pi * mNa * kB * temperature) * 10 ** 6  # lambda is de Broglie wavelength in um

    # normBose = 8 * integral3(den_ThermU, 0, xTF * 10, 0, yTF * 10, 0, zTF * 10, 'RelTol', 1e-3)
    # NThermalBosons = normBose / lambdaDB ** 3

    # # ToverTcGlobal=(1-Nc/(Nc+NThermalBosons))**(1/3)
    # Tc = .94 * PlanckConst / 2 / np.pi * omegabar / kB * (Nc + NThermalBosons) ** (1 / 3)
    # ToverTcGlobal = temperature / Tc

    # params = {}
    # params['ToverTc'] = ToverTcGlobal
    # params['nc0'] = Nc * den_BEC(0, 0, 0) * (1e6) ** 3  # m**-3
    # params['nthermal0'] = NThermalBosons * (den_ThermU(0, 0, 0) / normBose) * (1e6) ** 3  # m**-3
    # params['chemicalPotential'] = chemicalPotential / PlanckConst * 1e-3  # kHz
    # densityArray = (NThermalBosons * (den_ThermU(x, y, z, omegaX, omegaY, omegaZ, mNa, beta, chemicalPotential) / normBose) + Nc * den_BEC(x, y, z, xTF, yTF, zTF)) * (1e6) ** 3  # m**-3

    density = ((den_ThermU(x, y, z, omegaX, omegaY, omegaZ, mNa, beta, chemicalPotential) / lambdaDB**3) + Nc * den_BEC(x, y, z, xTF, yTF, zTF)) * (1e6) ** 3  # m**-3

    return density


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


def V_Pol_interp_dentck(kgrid, X_Vals, cParams, sParams, trapParams):
    # returns an interpolation function for the polaron energy
    [mI, mB, n0, gBB] = sParams
    aIBi = cParams['aIBi']
    nBEC_tck = trapParams['nBEC_tck']

    # ASSUMING FIXED ABB, KIB, ASSUMING POTENTIAL IS FOR ZERO MOMENTUM, AND ASSUMING PARTICLE IS IN CENTER OF TRAP IN Y AND Z DIRECTIONS
    DP = 0
    P = 0

    EpVals = np.zeros(X_Vals.size)
    for ind, X in enumerate(X_Vals):
        n = interpolate.splev(X, nBEC_tck)  # ASSUMING PARTICLE IS IN CENTER OF TRAP IN Y AND Z DIRECTIONS
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


def x_BEC_osc_zw2021(t, omega_BEC_osc, gamma_BEC_osc, phi_BEC_osc, amp_BEC_osc):
    # returns function describing position of the BEC COM over time
    return amp_BEC_osc * np.exp(-1 * gamma_BEC_osc * t) * np.cos(omega_BEC_osc * t + phi_BEC_osc)


def v_BEC_osc_zw2021(t, omega_BEC_osc, gamma_BEC_osc, phi_BEC_osc, amp_BEC_osc):
    # returns function describing the velocty of the BEC COM over time
    return -1 * amp_BEC_osc * np.exp(-1 * gamma_BEC_osc * t) * (gamma_BEC_osc * np.cos(omega_BEC_osc * t + phi_BEC_osc) + omega_BEC_osc * np.sin(omega_BEC_osc * t + phi_BEC_osc))


def a_BEC_osc_zw2021(t, omega_BEC_osc, gamma_BEC_osc, phi_BEC_osc, amp_BEC_osc):
    # returns function describing the acceleration of the BEC COM over time
    return amp_BEC_osc * np.exp(-1 * gamma_BEC_osc * t) * ((gamma_BEC_osc**2 - omega_BEC_osc**2) * np.cos(omega_BEC_osc * t + phi_BEC_osc) + 2 * gamma_BEC_osc * omega_BEC_osc * np.sin(omega_BEC_osc * t + phi_BEC_osc))


def F_BEC_osc_zw2021(t, omega_BEC_osc, gamma_BEC_osc, phi_BEC_osc, amp_BEC_osc, mI):
    # returns function describing fictional force on the impurity in the BECs accelerating frame of motion
    return mI * a_BEC_osc_zw2021(t, omega_BEC_osc, gamma_BEC_osc, phi_BEC_osc, amp_BEC_osc)


def F_Imp_trap(X, omega_Imp_x, mI):
    # returns function describing force on impurity from harmonic potential confining impurity in the direction of motion
    return -1 * mI * (omega_Imp_x**2) * X


def F_Imp_trap_harmonic(X, omega_Imp_x, mI):
    # returns function describing force on impurity from harmonic potential confining impurity in the direction of motion
    return -1 * mI * (omega_Imp_x**2) * X


def F_Imp_trap_gaussian(X, gaussian_amp, gaussian_width):
    # returns function describing force on impurity from harmonic potential confining impurity in the direction of motion
    return 4 * (gaussian_amp / (gaussian_width**2)) * X * np.exp(-2 * (X / gaussian_width)**2)

def F_Imp_trap_gaussian_2D(X, Y, gaussian_amp, gaussian_width_x, gaussian_width_y):
    # returns function describing force on impurity from harmonic potential confining impurity in the direction of motion
    return 4 * (Y / (gaussian_width_y**2)) * U_TiSa(X, Y, gaussian_amp, gaussian_width_x, gaussian_width_y)


def U_ODT1(x, y, A_ODT1, wx_ODT1, wy_ODT1):
    return A_ODT1 * np.exp(-2 * (x / wx_ODT1)**2) * np.exp(-2 * (y / wy_ODT1)**2)


def U_ODT2(x, z, A_ODT2, wx_ODT2, wz_ODT2):
    return A_ODT2 * np.exp(-2 * (x / wx_ODT2)**2) * np.exp(-2 * (z / wz_ODT2)**2)


def U_TiSa(x, y, A_TiSa, wx_TiSa, wy_TiSa):
    return A_TiSa * np.exp(-2 * (x / wx_TiSa)**2) * np.exp(-2 * (y / wy_TiSa)**2)


def U_tot_opt(x, y, z, sampleParams):
    # if sampleParams['A_ODT1'] == 0:
    #     return -1 * sampleParams['U0_opt_offset'] + U_ODT1(x, y - (sampleParams['y0_ODT1'] - sampleParams['y0_BEC']), sampleParams['A_ODT1'], sampleParams['wx_ODT1'], sampleParams['wy_ODT1']) + U_ODT2(x, z, sampleParams['A_ODT2'], sampleParams['wx_ODT2'], sampleParams['wz_ODT2']) + U_TiSa(x, y + sampleParams['y0_BEC'], sampleParams['A_TiSa'], sampleParams['wx_TiSa'], sampleParams['wy_TiSa'])
    # else:
    #     return -1 * sampleParams['U_opt_offset'] + U_ODT1(x, y - (sampleParams['y0_ODT1'] - sampleParams['y0_BEC']), sampleParams['A_ODT1'], sampleParams['wx_ODT1'], sampleParams['wy_ODT1']) + U_ODT2(x, z, sampleParams['A_ODT2'], sampleParams['wx_ODT2'], sampleParams['wz_ODT2']) + U_TiSa(x, y + sampleParams['y0_BEC'], sampleParams['A_TiSa'], sampleParams['wx_TiSa'], sampleParams['wy_TiSa'])
    return -1 * sampleParams['U_opt_offset'] + U_ODT1(x, y - (sampleParams['y0_ODT1'] - sampleParams['y0_BEC']), sampleParams['A_ODT1'], sampleParams['wx_ODT1'], sampleParams['wy_ODT1']) + U_ODT2(x, z, sampleParams['A_ODT2'], sampleParams['wx_ODT2'], sampleParams['wz_ODT2']) + U_TiSa(x, y + sampleParams['y0_BEC'], sampleParams['A_TiSa'], sampleParams['wx_TiSa'], sampleParams['wy_TiSa'])


def E_Pol_gs(x, y, z, P, kgrid, cParams, sParams, sampleParams):
    # returns an interpolation function for the polaron energy
    [mI, mB, n0, gBB] = sParams
    aIBi = cParams['aIBi']

    # nBEC_tck = sampleParams['nBEC_tck']
    # n = interpolate.splev(x, y, z, nBEC_tck)

    L_exp2th = sampleParams['L_exp2th']
    x_MuM = x * (1e6) / L_exp2th; y_MuM = y * (1e6) / L_exp2th; z_MuM = z * (1e6) / L_exp2th  # convert positions in theory units to um
    n_exp = becdensity_zw2021(x_MuM, y_MuM, z_MuM, sampleParams['omegaX_radHz'], sampleParams['omegaY_radHz'], sampleParams['omegaZ_radHz'], sampleParams['temperature_K'], sampleParams['zTF_MuM'])  # computes BEC density in experimental units
    n = n_exp / (L_exp2th**3)  # converts density in SI units to theory units

    # DPi = pf_static_sph.DP_interp_grid(DP, P, aIBi, kgrid, mI, mB, n, gBB)  # computing DP self-consistently requires ~500-1000 evaluations of aSi and PB --> better to do this with a spline using 100 calls to the grid integration functions rather than 500-1000 calls to these functions

    Nsteps = 1e2

    aSi_tck, PBint_tck = pf_static_sph.createSpline_grid(Nsteps, kgrid, mI, mB, n, gBB)
    DP = pf_static_sph.DP_interp(0, P, aIBi, aSi_tck, PBint_tck)
    aSi = pf_static_sph.aSi_interp(DP, aSi_tck)
    PB = pf_static_sph.PB_interp(DP, aIBi, aSi_tck, PBint_tck)
    E_gs = pf_static_sph.Energy(P, PB, aIBi, aSi, mI, mB, n)

    # print(P, DP, np.abs(DP-(P-PB)))

    # n0_exp = sampleParams['n0_BEC_m^-3']  # peak BEC density
    # n0 = n0_exp / (L_exp2th**3)

    # aSi0_tck = sampleParams['aSi0_tck']; PBint0_tck = sampleParams['PBint0_tck']
    # DP0 = pf_static_sph.DP_interp(0, 0, aIBi, aSi0_tck, PBint0_tck)
    # aSi0 = pf_static_sph.aSi_interp(DP0, aSi0_tck)
    # PB0 = pf_static_sph.PB_interp(DP0, aIBi, aSi0_tck, PBint0_tck)
    # E_gs_0 = pf_static_sph.Energy(0, PB0, aIBi, aSi0, mI, mB, n0)
    # # print(E_gs - E_gs_0)
    # # return -1 * E_gs_0 + E_gs

    return -1 * sampleParams['E_pol_offset'] + E_gs


def E_tot_gs(x, y, z, P, kgrid, cParams, sParams, sampleParams):
    # return E_Pol_gs(x, y, z, P, kgrid, cParams, sParams, sampleParams) + 0.5 * sParams[0] * (sampleParams['omega_Imp_y']**2) * (y + sampleParams['y0_BEC'])**2
    return E_Pol_gs(x, y, z, P, kgrid, cParams, sParams, sampleParams) + U_tot_opt(x, y, z, sampleParams)


def f_thermal(x, y, z, P, Beta, mu, kgrid, cParams, sParams, sampleParams):
    # Computes the ground state occupation probability of the initial polaron having coordinates (x,y,z,P) at temperature Beta = 1/(kB*T) and chemical potential mu
    # Note: (x,y,z) are all given in the frame of the BEC (whose center is shifted by sampleParams['y0_BEC'] in the y-direction w.r.t. the lab frame origin; the other two directions have the same origin)
    # Note: all quantities are given in theory units except for the conversion that is done to compute the BEC density (used in E_Pol_gs)

    return 1 / (1 + np.exp(Beta * E_tot_gs(x, y, z, P, kgrid, cParams, sParams, sampleParams) - Beta * mu))

def f_thermal_true2D(x, y, z, Px, Py, Beta, mu, kgrid, cParams, sParams, sampleParams):
    # Computes the ground state occupation probability of the initial polaron having coordinates (x,y,z,Py) at temperature Beta = 1/(kB*T) and chemical potential mu. Adds a 'classical' kinetic energy Px^2/2mI term.
    # Note: (x,y,z) are all given in the frame of the BEC (whose center is shifted by sampleParams['y0_BEC'] in the y-direction w.r.t. the lab frame origin; the other two directions have the same origin)
    # Note: all quantities are given in theory units except for the conversion that is done to compute the BEC density (used in E_Pol_gs)
    return 1 / (1 + np.exp(Beta * (E_tot_gs(x, y, z, Py, kgrid, cParams, sParams, sampleParams) + Px**2/(2*sParams[0])) - Beta * mu))


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


# def Zw_expParams_old():
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
#     params['mu_div_hbar'] = 2 * np.pi * 1.4 * 1e3  # chemical potential divided by hbar (in rad*Hz)
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
    hbar = 1.0555e-34  # reduced Planck's constant (J*s/rad)
    params = {}

    # Note: x-direction is the direciton of oscillation (displacement of the BEC)

    # Experimental parameters (SI units)
    params['mI'] = 39.96 * u
    params['mB'] = 22.99 * u
    params['aIB'] = -2600 * a0
    params['aBB'] = 52 * a0
    params['n0_TF'] = 6e13 * 1e6  # BEC TF peak density in m^(-3)
    params['n0_thermal'] = 0.9e13 * 1e6  # BEC thermal Gaussian peak density in m^(-3)
    params['n0_BEC'] = params['n0_TF'] + params['n0_thermal']  # Total BEC peak (central) density in m^(-3)
    params['nI'] = 2.0e11 * 1e6  # impurity peak density
    params['n0_BEC_scale'] = 1e14 * 1e6  # order of magnitude scale of peak BEC density in m^(-3)
    params['mu_div_hbar'] = 2 * np.pi * 1.5 * 1e3  # chemical potential divided by hbar (in rad*Hz)
    params['omega_BEC_x'] = 2 * np.pi * 80; params['omega_BEC_y'] = 2 * np.pi * 100; params['omega_BEC_z'] = 2 * np.pi * 12  # BEC trapping frequencies in rad*Hz

    params['RG_BEC_X'] = 95e-6; params['RG_BEC_Y'] = 29e-6; params['RG_BEC_Z'] = 12e-6  # BEC density thermal Gaussian waists in each direction (m)

    # # # Use measured TF Radii of BEC
    # params['RTF_BEC_X'] = 103e-6; params['RTF_BEC_Y'] = 32e-6; params['RTF_BEC_Z'] = 13e-6  # BEC density Thomas-Fermi radii in each direction (m)

    # Calculate TF Radii of BEC from Bose gas harmonic trap frequencies (but used explicitly measured thermal values above)
    # params['gBB_Born'] = (4 * np.pi * (hbar**2) / params['mB']) * params['aBB']
    # params['RTF_BEC_X'] = np.sqrt(2 * params['gBB_Born'] * params['n0_TF'] / (params['mB'] * (params['omega_BEC_x']**2))); params['RTF_BEC_Y'] = np.sqrt(2 * params['gBB_Born'] * params['n0_TF'] / (params['mB'] * (params['omega_BEC_y']**2))); params['RTF_BEC_Z'] = np.sqrt(2 * params['gBB_Born'] * params['n0_TF'] / (params['mB'] * (params['omega_BEC_z']**2)))  # BEC density Thomas-Fermi radii in each direction (m)
    params['RTF_BEC_X'] = np.sqrt(2 * hbar * params['mu_div_hbar'] / (params['mB'] * (params['omega_BEC_x']**2))); params['RTF_BEC_Y'] = np.sqrt(2 * hbar * params['mu_div_hbar'] / (params['mB'] * (params['omega_BEC_y']**2))); params['RTF_BEC_Z'] = np.sqrt(2 * hbar * params['mu_div_hbar'] / (params['mB'] * (params['omega_BEC_z']**2)))  # BEC density Thomas-Fermi radii in each direction (m)

    params['vI_init'] = 7 * 1e-3  # average initial velocity of impurities (m/s)
    params['omega_Imp_x'] = 2 * np.pi * 150  # Impurity trapping frequency in rad*Hz
    params['omega_BEC_osc'] = params['omega_BEC_x']  # BEC oscillation frequency in rad*Hz
    params['a_osc'] = 10e-6 / params['RTF_BEC_X']  # Initial displacement of BEC (m) divided by TF radius in the direction of displacement (x-direction)

    return params


def Zw_expParams_2021():
    # Constants (SI units)
    a0 = 5.29e-11  # Bohr radius (m)
    u = 1.661e-27  # atomic mass unit (kg)
    hbar = 1.0555e-34  # reduced Planck's constant (J*s/rad)
    params = {}

    # Note: x-direction is the direciton of oscillation (displacement of the BEC)

    # Experimental parameters (SI units)
    params['mI'] = 39.96 * u
    params['mB'] = 22.99 * u
    params['aBB'] = 52 * a0
    params['n0_BEC_scale'] = 1e20  # order of magnitude scale of peak BEC density in m^(-3). Average of the peak BEC densities across all experimentally measured interaction strengths is ~6e19 m^(-3)

    # params['aIB'] = -2600 * a0
    # params['nI'] = 2.0e11 * 1e6  # impurity peak density
    # params['mu_div_hbar'] = 2 * np.pi * 1.5 * 1e3  # chemical potential divided by hbar (in rad*Hz)
    # params['omega_BEC_x'] = 2 * np.pi * 78; params['omega_BEC_y'] = 2 * np.pi * 100; params['omega_BEC_z'] = 2 * np.pi * 12  # BEC trapping frequencies in rad*Hz
    # params['RG_BEC_X'] = 95e-6; params['RG_BEC_Y'] = 29e-6; params['RG_BEC_Z'] = 12e-6  # BEC density thermal Gaussian waists in each direction (m)
    # params['RTF_BEC_X'] = np.sqrt(2 * hbar * params['mu_div_hbar'] / (params['mB'] * (params['omega_BEC_x']**2))); params['RTF_BEC_Y'] = np.sqrt(2 * hbar * params['mu_div_hbar'] / (params['mB'] * (params['omega_BEC_y']**2))); params['RTF_BEC_Z'] = np.sqrt(2 * hbar * params['mu_div_hbar'] / (params['mB'] * (params['omega_BEC_z']**2)))  # BEC density Thomas-Fermi radii in each direction (m)
    # params['vI_init'] = 7 * 1e-3  # average initial velocity of impurities (m/s)
    # params['a_osc'] = 10e-6 / params['RTF_BEC_X']  # Initial displacement of BEC (m) divided by TF radius in the direction of displacement (x-direction)

    # params['omega_Imp_x'] = 2 * np.pi * 130  # Impurity trapping frequency in rad*Hz
    # params['omega_BEC_osc'] = 2 * np.pi * 75  # BEC oscillation frequency in rad*Hz

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
    interp_vals = spi.griddata((C1g.flatten(), C2g.flatten()), xdataset.values.flatten(), (C1g_interp, C2g_interp), method='linear')
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
        # omega_BEC_osc = 0
        # a_osc = 0
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
    # ReAmp_da = xr.DataArray(np.full((tgrid.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid, kVec, thVec], dims=['t', 'k', 'th'])
    # ImAmp_da = xr.DataArray(np.full((tgrid.size, len(kVec), len(thVec)), np.nan, dtype=float), coords=[tgrid, kVec, thVec], dims=['t', 'k', 'th'])
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
        # ReAmp_da[ind] = np.real(Amp)
        # ImAmp_da[ind] = np.imag(Amp)
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

    # data_dict = {'Pph': Pph_da, 'Nph': Nph_da, 'Phase': Phase_da, 'Real_CSAmp': ReAmp_da, 'Imag_CSAmp': ImAmp_da, 'P': P_da, 'X': X_da, 'XLab': XLab_da, 'Energy': Energy_da}
    data_dict = {'Pph': Pph_da, 'Nph': Nph_da, 'Phase': Phase_da, 'P': P_da, 'X': X_da, 'XLab': XLab_da, 'Energy': Energy_da}
    coords_dict = {'t': tgrid}
    attrs_dict = {'NGridPoints': NGridPoints, 'k_mag_cutoff': k_max, 'aIBi': aIBi, 'mI': mI, 'mB': mB, 'n0': n0, 'gBB': gBB, 'nu': nu_const, 'gIB': gIB, 'xi': xi, 'Fext_mag': Fext_mag, 'TF': TF, 'Delta_P': dP, 'omega_BEC_osc': omega_BEC_osc, 'X0': X0, 'P0': P0, 'a_osc': a_osc, 'omega_Imp_x': omega_Imp_x}

    dynsph_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)

    return dynsph_ds


def zw2021_quenchDynamics(cParams, gParams, sParams, trapParams, toggleDict):
    #
    # do not run this inside CoherentState or PolaronHamiltonian
    import LDA_CoherentState
    import zw2021_PolaronHamiltonian
    # takes parameters, performs dynamics, and outputs desired observables
    aIBi = cParams['aIBi']
    [xgrid, kgrid, tgrid] = gParams
    [mI, mB, n0, gBB] = sParams
    P0 = trapParams['P0']
    X0 = trapParams['X0']

    NGridPoints = kgrid.size()
    k_max = kgrid.getArray('k')[-1]
    # kVec = kgrid.getArray('k')
    # thVec = kgrid.getArray('th')

    # calculate some parameters
    nu_const = nu(mB, n0, gBB)
    gIB = g(kgrid, aIBi, mI, mB, n0, gBB)
    aBB = (mB / (4 * np.pi)) * gBB
    xi = (8 * np.pi * n0 * aBB)**(-1 / 2)

    nBEC_tck = trapParams['nBEC_tck']

    # LDA Force functions
    LDA_funcs = {}

    omega_BEC_osc = trapParams['omega_BEC_osc']; gamma_BEC_osc = trapParams['gamma_BEC_osc']; phi_BEC_osc = trapParams['phi_BEC_osc']; amp_BEC_osc = trapParams['amp_BEC_osc']
    LDA_funcs['F_BEC_osc'] = lambda t: F_BEC_osc_zw2021(t, omega_BEC_osc, gamma_BEC_osc, phi_BEC_osc, amp_BEC_osc, mI)

    omega_Imp_x = trapParams['omega_Imp_x']
    gaussian_amp = trapParams['gaussian_amp']
    gaussian_width = trapParams['gaussian_width']

    # import matplotlib
    # import matplotlib.pyplot as plt
    # print(-1 * mI * (omega_Imp_x**2), (gaussian_amp / (gaussian_width**2)))
    # X_Vals = np.linspace(-1 * trapParams['RTF_BEC'] * 1, trapParams['RTF_BEC'] * 1, 100)
    # fig, ax = plt.subplots()
    # ax.plot(X_Vals, F_Imp_trap_harmonic(X_Vals, omega_Imp_x, mI), label='Harmonic')
    # ax.plot(X_Vals, F_Imp_trap_gaussian(X_Vals, gaussian_amp, gaussian_width), label='Gaussian')
    # ax.legend()
    # plt.show()

    if toggleDict['ImpTrap_Type'] == 'harmonic':
        LDA_funcs['F_Imp_trap'] = lambda X: F_Imp_trap_harmonic(X, omega_Imp_x, mI)
    else:
        LDA_funcs['F_Imp_trap'] = lambda X: F_Imp_trap_gaussian(X, gaussian_amp, gaussian_width)

    if toggleDict['Polaron_Potential'] == 'on':
        # assuming we only have a particle in the center of the trap that travels in the direction of largest Thomas Fermi radius (easy to generalize this)
        # X_Vals = np.linspace(-1 * trapParams['RTF_BEC'] * 0.99, trapParams['RTF_BEC'] * 0.99, 100)
        X_Vals = np.linspace(-1 * trapParams['RTF_BEC'] * 3.99, trapParams['RTF_BEC'] * 3.99, 100)
        E_Pol_tck = V_Pol_interp_dentck(kgrid, X_Vals, cParams, sParams, trapParams)
        LDA_funcs['F_pol_naive'] = lambda X: F_pol(X, E_Pol_tck)

    # Initialization CoherentState
    cs = LDA_CoherentState.LDA_CoherentState(kgrid, xgrid)
    dVk = cs.dVk

    # Initialization PolaronHamiltonian
    Params = [aIBi, mI, mB, n0, gBB]
    ham = zw2021_PolaronHamiltonian.zw2021_PolaronHamiltonian(cs, Params, LDA_funcs, trapParams, toggleDict)
    gnum = ham.gnum

    Nsteps = 1e2
    aSi_tck, PBint_tck = pf_static_sph.createSpline_grid(Nsteps, kgrid, mI, mB, n0, gBB)
    DP = pf_static_sph.DP_interp(0, P0, aIBi, aSi_tck, PBint_tck)
    aSi = pf_static_sph.aSi_interp(DP, aSi_tck)
    CSAmp = pf_static_sph.BetaK(kgrid, aIBi, aSi, DP, mI, mB, n0, gBB)
    cs.set_initState(amplitude=CSAmp, phase=0, P=P0, X=X0)

    # Time evolution

    # Initialize observable Data Arrays
    Pph_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    Nph_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    Phase_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])

    P_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    X_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    XLab_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])

    A_PP_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    F_PP_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    F_impTrap_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])

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
        P_da[ind] = cs.get_totMom()
        X_da[ind] = cs.get_impPos()
        XLab_da[ind] = cs.get_impPos() + x_BEC_osc_zw2021(t, omega_BEC_osc, gamma_BEC_osc, phi_BEC_osc, amp_BEC_osc)

        # Compute time-varying amplitude of 'smarter' polaron potential. Also compute force from smarter polaron potential and bare impurity trap potential at each step
        amplitude = cs.get_Amplitude()
        amp_re = np.real(amplitude); amp_im = np.imag(amplitude)
        n = interpolate.splev(X_da[ind], nBEC_tck)  # ASSUMING PARTICLE IS IN CENTER OF TRAP IN Y AND Z DIRECTIONS
        Wk_grid = Wk(kgrid, mB, n, gBB)
        Wki_grid = 1 / Wk_grid
        Wk2_grid = Wk_grid**2; Wk3_grid = Wk_grid**3; omegak_g = omegak_grid(kgrid, mB, n, gBB)
        eta1 = np.dot(Wk2_grid * np.abs(amplitude)**2, dVk); eta2 = np.dot((Wk3_grid / omegak_g) * amp_re, dVk); eta3 = np.dot((Wk_grid / omegak_g) * amp_im, dVk)
        xp_re = 0.5 * np.dot(Wk_grid * amp_re, dVk); xm_im = 0.5 * np.dot(Wki_grid * amp_im, dVk)
        A_PP_da[ind] = gnum * (1 + 2 * xp_re / n) + gBB * eta1 - gnum * gBB * ((np.sqrt(n) + 2 * xp_re) * eta2 + 2 * xm_im * eta3)
        dndx = interpolate.splev(X_da[ind], nBEC_tck, der=1)
        F_PP_da[ind] = -1 * A_PP_da[ind] * dndx
        F_impTrap_da[ind] = LDA_funcs['F_Imp_trap'](XLab_da[ind])

        end = timer()
        print('t: {:.2f}, cst: {:.2f}, dt: {:.3f}, runtime: {:.3f}'.format(t, cs.time, dt, end - start))
        start = timer()

    V_da = xr.DataArray(np.gradient(X_da.values, tgrid), coords=[tgrid], dims=['t'])

    # Create Data Set

    data_dict = {'Pph': Pph_da, 'Nph': Nph_da, 'Phase': Phase_da, 'P': P_da, 'X': X_da, 'XLab': XLab_da, 'V': V_da, 'A_PP': A_PP_da, 'F_PP': F_PP_da, 'F_impTrap': F_impTrap_da}
    coords_dict = {'t': tgrid}
    attrs_dict = {'NGridPoints': NGridPoints, 'k_mag_cutoff': k_max, 'aIBi': aIBi, 'mI': mI, 'mB': mB, 'n0': n0, 'gBB': gBB, 'nu': nu_const, 'gIB': gIB, 'xi': xi, 'omega_BEC_osc': omega_BEC_osc, 'gamma_BEC_osc': gamma_BEC_osc, 'phi_BEC_osc': phi_BEC_osc, 'amp_BEC_osc': amp_BEC_osc, 'X0': X0, 'P0': P0, 'omega_Imp_x': omega_Imp_x}

    dynsph_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)

    return dynsph_ds


def zw2021_quenchDynamics_2D(cParams, gParams, sParams, trapParams, toggleDict):
    #
    # do not run this inside CoherentState or PolaronHamiltonian
    import LDA_CoherentState
    import zw2021_PolaronHamiltonian
    # takes parameters, performs dynamics, and outputs desired observables
    aIBi = cParams['aIBi']
    [xgrid, kgrid, tgrid] = gParams
    [mI, mB, n0, gBB] = sParams
    P0 = trapParams['P0']; X0 = trapParams['X0']; Y0 = trapParams['Y0']

    NGridPoints = kgrid.size()
    k_max = kgrid.getArray('k')[-1]
    # kVec = kgrid.getArray('k')
    # thVec = kgrid.getArray('th')

    # calculate some parameters
    nu_const = nu(mB, n0, gBB)
    gIB = g(kgrid, aIBi, mI, mB, n0, gBB)
    aBB = (mB / (4 * np.pi)) * gBB
    xi = (8 * np.pi * n0 * aBB)**(-1 / 2)

    yVals = np.linspace(-4 * trapParams['RTF_BEC'], 4 * trapParams['RTF_BEC'], 1000)
    denVals = np.zeros(yVals.size)
    L_exp2th = trapParams['L_exp2th']
    x0_MuM = X0 * (1e6) / L_exp2th; y0_MuM = Y0 * (1e6) / L_exp2th; yVals_MuM = yVals * (1e6) / L_exp2th  # convert positions in theory units to um

    for indy, y in enumerate(yVals_MuM):
        n_exp = becdensity_zw2021(x0_MuM, y, 0, trapParams['omegaX_radHz'], trapParams['omegaY_radHz'], trapParams['omegaZ_radHz'], trapParams['temperature_K'], trapParams['zTF_MuM'])  # computes BEC density in experimental units
        denVals[indy] = n_exp / (L_exp2th**3)  # converts density in SI units to theory units

    nBEC_tck = interpolate.splrep(yVals, denVals, s=0)
    trapParams['nBEC_tck'] = nBEC_tck

    # LDA Force functions
    LDA_funcs = {}

    omega_BEC_osc = trapParams['omega_BEC_osc']; gamma_BEC_osc = trapParams['gamma_BEC_osc']; phi_BEC_osc = trapParams['phi_BEC_osc']; amp_BEC_osc = trapParams['amp_BEC_osc']
    LDA_funcs['F_BEC_osc'] = lambda t: F_BEC_osc_zw2021(t, omega_BEC_osc, gamma_BEC_osc, phi_BEC_osc, amp_BEC_osc, mI)

    omega_Imp_y = trapParams['omega_Imp_y']
    gaussian_amp = trapParams['gaussian_amp']
    gaussian_width_x = trapParams['gaussian_width_x']
    gaussian_width_y = trapParams['gaussian_width_y']

    if toggleDict['ImpTrap_Type'] == 'harmonic':
        LDA_funcs['F_Imp_trap'] = lambda Y: F_Imp_trap_harmonic(Y, omega_Imp_y, mI)
    else:
        # LDA_funcs['F_Imp_trap'] = lambda Y: F_Imp_trap_gaussian(Y, gaussian_amp, gaussian_width)
        LDA_funcs['F_Imp_trap'] = lambda Y: F_Imp_trap_gaussian_2D(X0, Y, gaussian_amp, gaussian_width_x, gaussian_width_y)

    # Initialization CoherentState
    cs = LDA_CoherentState.LDA_CoherentState(kgrid, xgrid)
    dVk = cs.dVk

    # Initialization PolaronHamiltonian
    Params = [aIBi, mI, mB, n0, gBB]
    ham = zw2021_PolaronHamiltonian.zw2021_PolaronHamiltonian(cs, Params, LDA_funcs, trapParams, toggleDict)
    gnum = ham.gnum

    n_exp = becdensity_zw2021(x0_MuM, y0_MuM, 0, trapParams['omegaX_radHz'], trapParams['omegaY_radHz'], trapParams['omegaZ_radHz'], trapParams['temperature_K'], trapParams['zTF_MuM'])  # computes BEC density in experimental units
    n_initial = n_exp / (L_exp2th**3)  # converts density in SI units to theory units

    Nsteps = 1e2
    aSi_tck, PBint_tck = pf_static_sph.createSpline_grid(Nsteps, kgrid, mI, mB, n_initial, gBB)
    DP = pf_static_sph.DP_interp(0, P0, aIBi, aSi_tck, PBint_tck)
    aSi = pf_static_sph.aSi_interp(DP, aSi_tck)
    CSAmp = pf_static_sph.BetaK(kgrid, aIBi, aSi, DP, mI, mB, n_initial, gBB)
    cs.set_initState(amplitude=CSAmp, phase=0, P=P0, X=Y0)

    # Time evolution

    # Initialize observable Data Arrays
    Pph_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    Nph_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    Phase_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])

    P_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    Y_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    YLab_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])

    A_PP_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    F_PP_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    F_impTrap_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])

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
        P_da[ind] = cs.get_totMom()
        Y_da[ind] = cs.get_impPos()
        YLab_da[ind] = cs.get_impPos() + x_BEC_osc_zw2021(t, omega_BEC_osc, gamma_BEC_osc, phi_BEC_osc, amp_BEC_osc)

        # Compute time-varying amplitude of 'smarter' polaron potential. Also compute force from smarter polaron potential and bare impurity trap potential at each step
        amplitude = cs.get_Amplitude()
        amp_re = np.real(amplitude); amp_im = np.imag(amplitude)
        n = interpolate.splev(Y_da[ind], nBEC_tck)  # ASSUMING PARTICLE IS IN CENTER OF TRAP IN Y AND Z DIRECTIONS
        Wk_grid = Wk(kgrid, mB, n, gBB)
        Wki_grid = 1 / Wk_grid
        Wk2_grid = Wk_grid**2; Wk3_grid = Wk_grid**3; omegak_g = omegak_grid(kgrid, mB, n, gBB)
        eta1 = np.dot(Wk2_grid * np.abs(amplitude)**2, dVk); eta2 = np.dot((Wk3_grid / omegak_g) * amp_re, dVk); eta3 = np.dot((Wk_grid / omegak_g) * amp_im, dVk)
        xp_re = 0.5 * np.dot(Wk_grid * amp_re, dVk); xm_im = 0.5 * np.dot(Wki_grid * amp_im, dVk)
        A_PP_da[ind] = gnum * (1 + 2 * xp_re / n) + gBB * eta1 - gnum * gBB * ((np.sqrt(n) + 2 * xp_re) * eta2 + 2 * xm_im * eta3)
        dndx = interpolate.splev(Y_da[ind], nBEC_tck, der=1)
        F_PP_da[ind] = -1 * A_PP_da[ind] * dndx
        F_impTrap_da[ind] = LDA_funcs['F_Imp_trap'](YLab_da[ind])

        end = timer()
        print('t: {:.2f}, cst: {:.2f}, dt: {:.3f}, runtime: {:.3f}'.format(t, cs.time, dt, end - start))
        start = timer()

    V_da = xr.DataArray(np.gradient(Y_da.values, tgrid), coords=[tgrid], dims=['t'])

    # Create Data Set

    data_dict = {'Pph': Pph_da, 'Nph': Nph_da, 'Phase': Phase_da, 'P': P_da, 'Y': Y_da, 'YLab': YLab_da, 'V': V_da, 'A_PP': A_PP_da, 'F_PP': F_PP_da, 'F_impTrap': F_impTrap_da}
    coords_dict = {'t': tgrid}
    attrs_dict = {'NGridPoints': NGridPoints, 'k_mag_cutoff': k_max, 'aIBi': aIBi, 'mI': mI, 'mB': mB, 'n0': n0, 'gBB': gBB, 'nu': nu_const, 'gIB': gIB, 'xi': xi, 'omega_BEC_osc': omega_BEC_osc, 'gamma_BEC_osc': gamma_BEC_osc, 'phi_BEC_osc': phi_BEC_osc, 'amp_BEC_osc': amp_BEC_osc, 'X0': X0, 'Y0': Y0, 'P0': P0, 'omega_Imp_y': omega_Imp_y}

    dynsph_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)

    return dynsph_ds

def zw2021_quenchDynamics_true2D(cParams, gParams, sParams, trapParams, toggleDict):
    #
    # do not run this inside CoherentState or PolaronHamiltonian
    import LDA_CoherentState_2D
    import zw2021_PolaronHamiltonian_2D
    # takes parameters, performs dynamics, and outputs desired observables
    aIBi = cParams['aIBi']
    [xgrid, kgrid, tgrid] = gParams
    [mI, mB, n0, gBB] = sParams
    PX0 = trapParams['PX0']; PY0 = trapParams['PY0']; X0 = trapParams['X0']; Y0 = trapParams['Y0']

    NGridPoints = kgrid.size()
    k_max = kgrid.getArray('k')[-1]
    # kVec = kgrid.getArray('k')
    # thVec = kgrid.getArray('th')

    # calculate some parameters
    nu_const = nu(mB, n0, gBB)
    gIB = g(kgrid, aIBi, mI, mB, n0, gBB)
    aBB = (mB / (4 * np.pi)) * gBB
    xi = (8 * np.pi * n0 * aBB)**(-1 / 2)

    L_exp2th = trapParams['L_exp2th']

    densityFunc = lambda coords: becdensity_zw2021(coords[0] * (1e6) / L_exp2th, coords[1] * (1e6) / L_exp2th, 0,trapParams['omegaX_radHz'], trapParams['omegaY_radHz'], trapParams['omegaZ_radHz'], trapParams['temperature_K'], trapParams['zTF_MuM']) / (L_exp2th**3)  # function that gives the BEC density (expressed in theory units) given a coordinates (x,y) (expressed in theory units)
    densityGradFunc = nd.Gradient(densityFunc, method='central')  # function that gives the gradient of the BEC density (expressed in theory units) given a coordinates (x,y) (expressed in theory units)

    trapParams['densityFunc'] = densityFunc
    trapParams['densityGradFunc'] = densityGradFunc

    # LDA Force functions
    LDA_funcs = {}

    omega_BEC_osc = trapParams['omega_BEC_osc']; gamma_BEC_osc = trapParams['gamma_BEC_osc']; phi_BEC_osc = trapParams['phi_BEC_osc']; amp_BEC_osc = trapParams['amp_BEC_osc']
    LDA_funcs['F_BEC_osc'] = lambda t: F_BEC_osc_zw2021(t, omega_BEC_osc, gamma_BEC_osc, phi_BEC_osc, amp_BEC_osc, mI)

    omega_Imp_y = trapParams['omega_Imp_y']
    gaussian_amp_TiSa = trapParams['gaussian_amp_TiSa']; gaussian_width_x_TiSa = trapParams['gaussian_width_x_TiSa']; gaussian_width_y_TiSa = trapParams['gaussian_width_y_TiSa']
    gaussian_amp_ODT2 = trapParams['gaussian_amp_ODT2']; gaussian_width_x_ODT2 = trapParams['gaussian_width_x_ODT2']; gaussian_width_z_ODT2 = trapParams['gaussian_width_z_ODT2']

    LDA_funcs['F_Imp_trap_Y'] = lambda X, Y: F_Imp_trap_gaussian_2D(X, Y, gaussian_amp_TiSa, gaussian_width_x_TiSa, gaussian_width_y_TiSa)
    LDA_funcs['F_Imp_trap_X'] = lambda X, Y: (4 * X / (gaussian_width_x_TiSa**2) * U_TiSa(X, Y, gaussian_amp_TiSa, gaussian_width_x_TiSa, gaussian_width_y_TiSa) + 4 * X / (gaussian_width_x_ODT2**2) * U_ODT2(X, 0, gaussian_amp_ODT2, gaussian_width_x_ODT2, gaussian_width_z_ODT2))

    # Initialization CoherentState
    cs = LDA_CoherentState_2D.LDA_CoherentState_2D(kgrid, xgrid)
    dVk = cs.dVk

    # Initialization PolaronHamiltonian
    Params = [aIBi, mI, mB, n0, gBB]
    ham = zw2021_PolaronHamiltonian_2D.zw2021_PolaronHamiltonian_2D(cs, Params, LDA_funcs, trapParams, toggleDict)
    gnum = ham.gnum

    n_initial = densityFunc([X0, Y0])

    Nsteps = 1e2
    aSi_tck, PBint_tck = pf_static_sph.createSpline_grid(Nsteps, kgrid, mI, mB, n_initial, gBB)
    DP = pf_static_sph.DP_interp(0, PY0, aIBi, aSi_tck, PBint_tck)
    aSi = pf_static_sph.aSi_interp(DP, aSi_tck)
    CSAmp = pf_static_sph.BetaK(kgrid, aIBi, aSi, DP, mI, mB, n_initial, gBB)
    cs.set_initState(amplitude=CSAmp, phase=0, X=X0, PX=PX0, Y=Y0, PY=PY0)

    # Time evolution

    # Initialize observable Data Arrays
    Pph_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    Nph_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    Phase_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])

    PY_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    Y_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    YLab_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    
    X_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    PX_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])

    A_PP_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    F_PP_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])
    F_impTrap_da = xr.DataArray(np.full(tgrid.size, np.nan, dtype=float), coords=[tgrid], dims=['t'])

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
        PY_da[ind] = cs.get_totMomY()
        Y_da[ind] = cs.get_impPosY()
        YLab_da[ind] = cs.get_impPosY() + x_BEC_osc_zw2021(t, omega_BEC_osc, gamma_BEC_osc, phi_BEC_osc, amp_BEC_osc)
        X_da[ind] = cs.get_impPosX()
        PX_da[ind] = cs.get_totMomX()

        # Compute time-varying amplitude of 'smarter' polaron potential. Also compute force from smarter polaron potential and bare impurity trap potential at each step
        amplitude = cs.get_Amplitude()
        amp_re = np.real(amplitude); amp_im = np.imag(amplitude)
        n = densityFunc([cs.get_impPosX(), cs.get_impPosY()])
        Wk_grid = Wk(kgrid, mB, n, gBB)
        Wki_grid = 1 / Wk_grid
        Wk2_grid = Wk_grid**2; Wk3_grid = Wk_grid**3; omegak_g = omegak_grid(kgrid, mB, n, gBB)
        eta1 = np.dot(Wk2_grid * np.abs(amplitude)**2, dVk); eta2 = np.dot((Wk3_grid / omegak_g) * amp_re, dVk); eta3 = np.dot((Wk_grid / omegak_g) * amp_im, dVk)
        xp_re = 0.5 * np.dot(Wk_grid * amp_re, dVk); xm_im = 0.5 * np.dot(Wki_grid * amp_im, dVk)
        A_PP_da[ind] = gnum * (1 + 2 * xp_re / n) + gBB * eta1 - gnum * gBB * ((np.sqrt(n) + 2 * xp_re) * eta2 + 2 * xm_im * eta3)
        dndy = densityGradFunc([cs.get_impPosX(), cs.get_impPosY()])[1]
        F_PP_da[ind] = -1 * A_PP_da[ind] * dndy
        F_impTrap_da[ind] = LDA_funcs['F_Imp_trap_Y'](X_da[ind], YLab_da[ind])

        end = timer()
        print('t: {:.2f}, cst: {:.2f}, dt: {:.3f}, runtime: {:.3f}'.format(t, cs.time, dt, end - start))
        start = timer()

    V_da = xr.DataArray(np.gradient(Y_da.values, tgrid), coords=[tgrid], dims=['t'])

    # Create Data Set

    data_dict = {'Pph': Pph_da, 'Nph': Nph_da, 'Phase': Phase_da, 'PY': PY_da, 'Y': Y_da, 'YLab': YLab_da, 'V': V_da, 'X': X_da, 'PX': PX_da, 'A_PP': A_PP_da, 'F_PP': F_PP_da, 'F_impTrap': F_impTrap_da}
    coords_dict = {'t': tgrid}
    attrs_dict = {'NGridPoints': NGridPoints, 'k_mag_cutoff': k_max, 'aIBi': aIBi, 'mI': mI, 'mB': mB, 'n0': n0, 'gBB': gBB, 'nu': nu_const, 'gIB': gIB, 'xi': xi, 'omega_BEC_osc': omega_BEC_osc, 'gamma_BEC_osc': gamma_BEC_osc, 'phi_BEC_osc': phi_BEC_osc, 'amp_BEC_osc': amp_BEC_osc, 'X0': X0, 'Y0': Y0, 'PX0': PX0, 'PY0': PY0, 'omega_Imp_y': omega_Imp_y}

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
