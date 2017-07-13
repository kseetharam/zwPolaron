import numpy as np
from scipy.integrate import quad


# ---- BASIC FUNCTIONS ----


def ur(mI, mB):
    return (mB * mI) / (mB + mI)


def eB(k, mB):
    return k**2 / (2 * mB)


def w(k, gBB, mB, n0):
    return np.sqrt(eB(k, mB) * (eB(k, mB) + 2 * gBB * n0))


def nu(gBB):
    return np.sqrt(gBB)


# ---- COMPOSITE FUNCTIONS ----


def g(grid_space, P, aIBi, mI, mB, n0, gBB):
    # gives bare interaction strength constant
    k_max = grid_space.arrays['k'][-1]
    mR = ur(mI, mB)
    return 1 / ((mR / (2 * np.pi)) * aIBi - (mR / np.pi**2) * k_max)


def omega0(grid_space, P, aIBi, mI, mB, n0, gBB):
    #
    names = list(grid_space.arrays.keys())
    functions_omega0 = [lambda k: w(k, gBB, mB, n0) + (k**2 / (2 * mI)), lambda th: 0 * th + 1]
    return grid_space.function_prod(names, functions_omega0)


def Wk(grid_space, P, aIBi, mI, mB, n0, gBB):
    #
    names = list(grid_space.arrays.keys())
    functions_Wk = [lambda k: np.sqrt(eB(k, mB) / w(k, gBB, mB, n0)), lambda th: 0 * th + 1]
    return grid_space.function_prod(names, functions_Wk)


def kcos_func(grid_space):
    #
    names = list(grid_space.arrays.keys())
    functions_kcos = [lambda k: k, np.cos]
    return grid_space.function_prod(names, functions_kcos)


def kpow2_func(grid_space):
    #
    names = list(grid_space.arrays.keys())
    functions_kpow2 = [lambda k: k**2, lambda th: 0 * th + 1]
    return grid_space.function_prod(names, functions_kpow2)


def PCrit_grid(grid_space, P, aIBi, mI, mB, n0, gBB):
    #
    DP = mI * nu(gBB)  # condition for critical momentum is P-PB = mI*nu where nu is the speed of sound
    names = list(grid_space.arrays.keys())
    k2 = kpow2_func(grid_space)
    Wkf = Wk(grid_space, P, aIBi, mI, mB, n0, gBB)
    kcos = kcos_func(grid_space)
    wf = grid_space.function_prod(names, [lambda k: w(k, gBB, mB, n0), lambda th: 0 * th + 1])
    # calculate aSi
    integrand = 2 * ur(mI, mB) / k2 - Wkf**2 / (wf + k2 / (2 * mI) - (DP / mI) * kcos)
    aSi = (2 * np.pi / ur(mI, mB)) * np.dot(integrand, grid_space.dV())
    # calculate PB (phonon momentum)
    integrand = kcos * Wkf**2 / (wf + k2 / (2 * mI) - DP / mI * kcos)**2
    PB = 4 * np.pi**2 * n0 / (ur(mI, mB)**2 * (aIBi - aSi)**2) * np.dot(integrand, grid_space.dV())
    return DP + PB


# ---- OTHER HELPER FUNCTIONS AND DYNAMICS ----


def PCrit_inf(kcutoff, aIBi, mI, mB, n0, gBB):
    #
    DP = mI * nu(gBB)  # condition for critical momentum is P-PB = mI*nu where nu is the speed of sound
    # non-grid helper function

    def Wk(k, gBB, mB, n0):
        return np.sqrt(eB(k, mB) / w(k, gBB, mB, n0))

    # calculate aSi
    integrand = lambda k: (4 * ur(mI, mB) / (k**2) - ((Wk(k, gBB, mB, n0)**2) / (DP * k / mI)) * np.log((w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)) / (w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)))) * (k**2)
    val, abserr = quad(integrand, 0, kcutoff, epsabs=0, epsrel=1.49e-12)
    aSi = (1 / (2 * np.pi * ur(mI, mB))) * val
    # calculate PB (phonon momentum)
    integrand = lambda k: ((2 * (w(k, gBB, mB, n0) + (k**2) / (2 * mI)) * (DP * k / mI) + (w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)) * (w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)) * np.log((w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)) / (w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)))) / ((w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)) * (w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)) * (DP * k / mI)**2)) * (Wk(k, gBB, mB, n0)**2) * (k**3)
    val, abserr = quad(integrand, 0, kcutoff, epsabs=0, epsrel=1.49e-12)
    PB = n0 / (ur(mI, mB)**2 * (aIBi - aSi)**2) * val

    return DP + PB


def spectFunc(t_Vec, S_Vec):
    # spectral function (Fourier Transform of dynamical overlap)
    tstep = t_Vec[1] - t_Vec[0]
    N = t_Vec.size
    tdecay = 3
    decayFactor = np.exp(-1 * t_Vec / tdecay)
    # decayFactor = 1
    sf = 2 * np.real(np.fft.ifft(S_Vec * decayFactor))
    omega = 2 * np.pi * np.fft.fftfreq(N, d=tstep)
    return omega, sf


def quenchDynamics(cParams, gParams, sParams, datapath):
    #
    # do not run this inside CoherentState or PolaronHamiltonian
    import CoherentState
    import PolaronHamiltonian
    # takes parameters, performs dynamics, and outputs desired observables
    [P, aIBi] = cParams
    [grid_space, tGrid] = gParams
    [mI, mB, n0, gBB] = sParams

    # Initialization CoherentState
    cs = CoherentState.CoherentState(grid_space)
    # Initialization PolaronHamiltonian
    Params = [P, aIBi, mI, mB, n0, gBB]
    ham = PolaronHamiltonian.PolaronHamiltonian(cs, Params)
    # Time evolution
    PB_Vec = np.zeros(tGrid.size, dtype=float)
    NB_Vec = np.zeros(tGrid.size, dtype=float)
    DynOv_Vec = np.zeros(tGrid.size, dtype=complex)
    MomDisp_Vec = np.zeros(tGrid.size, dtype=float)
    Phase_Vec = np.zeros(tGrid.size, dtype=float)

    for ind, t in enumerate(tGrid):
        if ind == 0:
            dt = t
            cs.evolve(dt, ham)
        else:
            dt = t - tGrid[ind - 1]
            cs.evolve(dt, ham)
        # print('t: {:.2f}, cst: {:.2f}, dt:{:.3f}'.format(t, cs.time, dt))
        PB_Vec[ind] = cs.get_PhononMomentum()
        NB_Vec[ind] = cs.get_PhononNumber()
        DynOv_Vec[ind] = cs.get_DynOverlap()
        MomDisp_Vec[ind] = cs.get_MomentumDispersion()
        Phase_Vec[ind] = cs.get_Phase()

    # Save Data

    PVec = P * np.ones(tGrid.size)
    # generates data file with columns representing P, t, Phase, Phonon Momentum, Momentum Dispersion, Phonon Number, Re(Dynamical Overlap), Im(Dynamical Overlap)
    data = np.concatenate((PVec[:, np.newaxis], tGrid[:, np.newaxis], Phase_Vec[:, np.newaxis], PB_Vec[:, np.newaxis], MomDisp_Vec[:, np.newaxis], NB_Vec[:, np.newaxis], np.real(DynOv_Vec)[:, np.newaxis], np.imag(DynOv_Vec)[:, np.newaxis]), axis=1)
    np.savetxt(datapath + '/quench_aIBi_%.2f_P_%.2f.dat' % (aIBi, P), data)
