import numpy as np
from scipy.integrate import quad
from scipy.special import iv
import Grid


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


def g(kgrid, P, aIBi, mI, mB, n0, gBB):
    # gives bare interaction strength constant
    k_max = kgrid.arrays['k'][-1]
    mR = ur(mI, mB)
    return 1 / ((mR / (2 * np.pi)) * aIBi - (mR / np.pi**2) * k_max)


def omega0(kgrid, P, aIBi, mI, mB, n0, gBB):
    #
    names = list(kgrid.arrays.keys())
    functions_omega0 = [lambda k: w(k, gBB, mB, n0) + (k**2 / (2 * mI)), lambda th: 0 * th + 1]
    return kgrid.function_prod(names, functions_omega0)


def Wk(kgrid, P, aIBi, mI, mB, n0, gBB):
    #
    names = list(kgrid.arrays.keys())
    functions_Wk = [lambda k: np.sqrt(eB(k, mB) / w(k, gBB, mB, n0)), lambda th: 0 * th + 1]
    return kgrid.function_prod(names, functions_Wk)


def kcos_func(kgrid):
    #
    names = list(kgrid.arrays.keys())
    functions_kcos = [lambda k: k, np.cos]
    return kgrid.function_prod(names, functions_kcos)


def ksin_func(kgrid):
    #
    names = list(kgrid.arrays.keys())
    functions_ksin = [lambda k: k, np.sin]
    return kgrid.function_prod(names, functions_ksin)


def kpow2_func(kgrid):
    #
    names = list(kgrid.arrays.keys())
    functions_kpow2 = [lambda k: k**2, lambda th: 0 * th + 1]
    return kgrid.function_prod(names, functions_kpow2)


def PCrit_grid(kgrid, P, aIBi, mI, mB, n0, gBB):
    #
    DP = mI * nu(gBB)  # condition for critical momentum is P-PB = mI*nu where nu is the speed of sound
    names = list(kgrid.arrays.keys())
    k2 = kpow2_func(kgrid)
    Wkf = Wk(kgrid, P, aIBi, mI, mB, n0, gBB)
    kcos = kcos_func(kgrid)
    wf = kgrid.function_prod(names, [lambda k: w(k, gBB, mB, n0), lambda th: 0 * th + 1])
    # calculate aSi
    integrand = 2 * ur(mI, mB) / k2 - Wkf**2 / (wf + k2 / (2 * mI) - (DP / mI) * kcos)
    aSi = (2 * np.pi / ur(mI, mB)) * np.dot(integrand, kgrid.dV())
    # calculate PB (phonon momentum)
    integrand = kcos * Wkf**2 / (wf + k2 / (2 * mI) - DP / mI * kcos)**2
    PB = 4 * np.pi**2 * n0 / (ur(mI, mB)**2 * (aIBi - aSi)**2) * np.dot(integrand, kgrid.dV())
    return DP + PB


def FTkernel_func(grid1, grid2, exp_complex_pos=True):
    #  order of arguments matters (due to outer product)
    #  assumes grids are SPHERICAL_2D
    #  returns matrix in grid1 coordinates X grid2 coordinates
    #  e.g. kgrid X xgrid will have rows dependent on (k,theta) and columns dependent on (x,theta_prime)
    #  exp_complex_pos is a boolean which controls whether the exponential in the output is positive or negative complex (gets a 1j or a -1j)
    # (by default True)

    functions_rcos = [lambda r: r, np.cos]
    functions_rsin = [lambda r: r, np.sin]

    names1 = list(grid1.arrays.keys())
    names2 = list(grid2.arrays.keys())
    rcos1 = grid1.function_prod(names1, functions_rcos)
    rsin1 = grid1.function_prod(names1, functions_rsin)
    rcos2 = grid2.function_prod(names2, functions_rcos)
    rsin2 = grid2.function_prod(names2, functions_rsin)
    outer_mat_cos = np.outer(rcos1, rcos2)
    outer_mat_sin = np.outer(rsin1, rsin2)
    if exp_complex_pos:
        return np.exp(1j * outer_mat_cos) * iv(0, 1j * outer_mat_sin)
    else:
        return np.exp(-1j * outer_mat_cos) * iv(0, 1j * outer_mat_sin)

# test WSL comment 2 -- addition

# ---- OTHER HELPER FUNCTIONS AND DYNAMICS ----


def PCrit_inf(kcutoff, aIBi, mI, mB, n0, gBB):
    #
    DP = mI * nu(gBB)  # condition for critical momentum is P-PB = mI*nu where nu is the speed of sound
    # non-grid helper function

    def Wk(k, gBB, mB, n0):
        return np.sqrt(eB(k, mB) / w(k, gBB, mB, n0))

    # calculate aSi
    def integrand(k): return (4 * ur(mI, mB) / (k**2) - ((Wk(k, gBB, mB, n0)**2) / (DP * k / mI)) * np.log((w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)) / (w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)))) * (k**2)
    val, abserr = quad(integrand, 0, kcutoff, epsabs=0, epsrel=1.49e-12)
    aSi = (1 / (2 * np.pi * ur(mI, mB))) * val
    # calculate PB (phonon momentum)

    def integrand(k): return ((2 * (w(k, gBB, mB, n0) + (k**2) / (2 * mI)) * (DP * k / mI) + (w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)) * (w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)) * np.log((w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)) / (w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)))) / ((w(k, gBB, mB, n0) + (k**2) / (2 * mI) - (DP * k / mI)) * (w(k, gBB, mB, n0) + (k**2) / (2 * mI) + (DP * k / mI)) * (DP * k / mI)**2)) * (Wk(k, gBB, mB, n0)**2) * (k**3)
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
    [kgrid, xgrid, tGrid] = gParams
    [mI, mB, n0, gBB] = sParams

    # Initialization CoherentState
    cs = CoherentState.CoherentState(kgrid, xgrid)
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

        # save position distribution data every 10 time values
        if ind % int(tGrid.size / 10) == 0:
            # create PB grid for specific P
            PBmax = 40 * P
            dPB = 0.5
            Ntheta = 50
            dtheta = np.pi / (Ntheta - 1)
            PBgrid = Grid.Grid("SPHERICAL_2D")
            PBgrid.initArray('PB', 0, PBmax, dPB)
            PBgrid.initArray('th', dtheta, np.pi, dtheta)

            # dV_PB = (2 * np.pi)**3 * PBgrid.dV()
            dV_PB = PBgrid.dV()
            PBcos = kcos_func(PBgrid)

            # calculate observables
            PD = cs.get_PositionDistribution()
            tVec = t * np.ones(PD.size)
            MD, G0 = cs.get_MomentumDistribution(PBgrid)
            # PD_data = np.concatenate((tVec[:, np.newaxis], cs.xmagVals[:, np.newaxis], cs.xthetaVals[:, np.newaxis], PD[:, np.newaxis], np.real(MD)[:, np.newaxis], np.imag(MD)[:, np.newaxis]), axis=1)
            totMD = np.dot(MD, dV_PB)

            Ppara = np.dot(dV_PB * PBcos, np.real(MD))
            amplitude = cs.amplitude_phase[0:-1]
            Bkave = np.dot(cs.dV * cs.kcos, amplitude * np.conjugate(amplitude))
            relDiff = np.abs(Ppara - Bkave) / Bkave
            print('t: %.2f, P: %.2f, Pph: %.2f, Ppara: %.9f, Bkave: %.9f, relDiff: %.9f, Re(totMD): %.9f, Im(totMD): %.9f' % (t, P, cs.get_PhononMomentum(), Ppara, Bkave, relDiff, np.real(totMD), np.imag(totMD)))
            # print('\n')
            # print(G0)
            # print('\n')
            # np.savetxt(datapath + '/PosSpace/P_%.2f/quench_P_%.2f_t_%.2f.dat' % (P, P, t), PD_data)

    # Save Data

    PVec = P * np.ones(tGrid.size)
    # generates data file with columns representing P, t, Phase, Phonon Momentum, Momentum Dispersion, Phonon Number, Re(Dynamical Overlap), Im(Dynamical Overlap)
    # data = np.concatenate((PVec[:, np.newaxis], tGrid[:, np.newaxis], Phase_Vec[:, np.newaxis], PB_Vec[:, np.newaxis], MomDisp_Vec[:, np.newaxis], NB_Vec[:, np.newaxis], np.real(DynOv_Vec)[:, np.newaxis], np.imag(DynOv_Vec)[:, np.newaxis]), axis=1)
    # np.savetxt(datapath + '/quench_P_%.2f.dat' % P, data)
