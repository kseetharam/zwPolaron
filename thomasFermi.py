import numpy as np
import Grid
import pf_static_sph as pfs
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate


if __name__ == "__main__":

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (20, 20, 20)
    (dx, dy, dz) = (0.2, 0.2, 0.2)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    NGridPoints_desired = (1 + 2 * Lx / dx) * (1 + 2 * Lz / dz)
    Ntheta = 50
    Nk = np.ceil(NGridPoints_desired / Ntheta)

    theta_max = np.pi
    thetaArray, dtheta = np.linspace(0, theta_max, Ntheta, retstep=True)

    k_max = ((2 * np.pi / dx)**3 / (4 * np.pi / 3))**(1 / 3)

    k_min = 1e-5
    kArray, dk = np.linspace(k_min, k_max, Nk, retstep=True)
    if dk < k_min:
        print('k ARRAY GENERATION ERROR')

    kgrid = Grid.Grid("SPHERICAL_2D")
    kgrid.initArray_premade('k', kArray)
    kgrid.initArray_premade('th', thetaArray)

    gParams = [kgrid]

    NGridPoints = kgrid.size()

    print('dk: {0}'.format(dk))
    print('UV cutoff: {0}'.format(k_max))
    print('NGridPoints: {0}'.format(NGridPoints))

    # Functions

    def n_BEC(X, n0, RTF):
        return n0 * (1 - X**2 / RTF**2)

    # Basic parameters

    mI = 1.7
    mB = 1
    n0 = 1
    kb = (6 * np.pi**2)**(1 / 3)
    kn_aBB = 0.0161
    kn_aIB = -1.243

    # Unit Conversion

    hbar = 1.0555e-34  # reduced Planck's constant (J*s/rad)
    u = 1.661e-27  # atomic mass unit (kg)
    n0_exp = 2e14 * 1e6  # BEC peak density
    mB_exp = 22.99 * u
    mB_th = 1
    n0_th = 1
    hbar_th = 1
    L_th_exp = n0_th**(-1 / 3) / n0_exp**(-1 / 3)
    M_th_exp = mB_th / mB_exp
    T_th_exp = (mB_th * (n0_th)**(-2 / 3) / (2 * np.pi * hbar_th)) / (mB_exp * (n0_exp)**(-2 / 3) / (2 * np.pi * hbar))

    E_th_exp = M_th_exp * L_th_exp**2 / T_th_exp**2
    F_th_exp = M_th_exp * L_th_exp / T_th_exp**2

    # Real Space
    RTF_exp = 126e-6  # Thomas-Fermi radius in um
    RTF = RTF_exp * L_th_exp
    X_Vals = np.linspace(-RTF * 0.99, RTF * 0.99, 100)
    X_Vals_m = X_Vals / L_th_exp

    DP = 0
    P = 0

    EpVals = np.zeros(X_Vals.size)
    for ind, X in enumerate(X_Vals):
        n = n_BEC(X, n0, RTF)
        kn = kb * n**(1 / 3)
        aBB = kn_aBB / kn
        aIB = kn_aIB / kn
        gBB = (4 * np.pi / mB) * aBB

        aIBi = aIB**(-1)
        aSi = pfs.aSi_grid(kgrid, DP, mI, mB, n, gBB)
        PB = pfs.PB_integral_grid(kgrid, DP, mI, mB, n, gBB)
        EpVals[ind] = pfs.Energy(P, PB, aIBi, aSi, mI, mB, n)

    EpVals_tck = interpolate.splrep(X_Vals, EpVals, s=0)
    EpVals_interp = 1 * interpolate.splev(X_Vals, EpVals_tck, der=0)
    FpVals_interp = -1 * interpolate.splev(X_Vals, EpVals_tck, der=1)

    X_Vals_poly = np.linspace(-RTF * 0.1, RTF * 0.1, 50)
    EpVals_poly = 1 * interpolate.splev(X_Vals_poly, EpVals_tck, der=0)
    [p2, p1, p0] = np.polyfit(X_Vals_poly, EpVals_poly, deg=2)
    omegap = np.sqrt(2 * p2 / mI)
    EpVals_harm = p2 * X_Vals**2 + p0

    # EpVals_Hz = (2 * np.pi * hbar)**(-1) * EpVals / E_th_exp
    EpVals_Hz = (2 * np.pi * hbar)**(-1) * EpVals_interp / E_th_exp
    FpVals_N = FpVals_interp / F_th_exp
    EpVals_harm_Hz = (2 * np.pi * hbar)**(-1) * EpVals_harm / E_th_exp
    freq_p_Hz = (omegap / (2 * np.pi)) * T_th_exp

    VtB = (gBB * n0 / RTF**2) * X_Vals**2
    VtB_Hz = (2 * np.pi * hbar)**(-1) * VtB / E_th_exp

    print('freq_poly (Hz) = {0}'.format(freq_p_Hz))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    # ax.plot(X_Vals, n_BEC(X_Vals, n0, RTF), 'k-')
    ax[0].plot(X_Vals_m * 1e6, EpVals_Hz * 1e-3, 'r-', label=r'$E_{pol}(n(X))$')
    ax[0].plot(X_Vals_m * 1e6, EpVals_harm_Hz * 1e-3, 'b-', label='Harmonic Fit')
    ax[0].plot(X_Vals_m * 1e6, VtB_Hz * 1e-3 + np.min(EpVals_Hz * 1e-3), 'g-', label='Shifted BEC Trap')
    ax[0].legend()
    ax[0].set_xlabel('X ($\mu$m)')
    ax[0].set_ylabel('Frequency (kHz)')
    ax[0].set_title('Traps')

    ax[1].plot(X_Vals_m * 1e6, FpVals_interp * 1e3, 'g-')
    ax[1].set_xlabel('X ($\mu$m)')
    ax[1].set_ylabel('Force (mN)')
    ax[1].set_title(r'$F_{pol}(n(X))$')
    fig.tight_layout()
    plt.show()