import numpy as np
import Grid
import pf_static_sph as pfs
import pf_dynamic_sph
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

    # # NEW EXPERIMENTAL PARAMS

    # Basic parameters

    expParams = pf_dynamic_sph.Zw_expParams()
    L_exp2th, M_exp2th, T_exp2th = pf_dynamic_sph.unitConv_exp2th(expParams['n0_BEC_scale'], expParams['mB'])
    E_exp2th = M_exp2th * L_exp2th**2 / T_exp2th**2
    F_exp2th = M_exp2th * L_exp2th / T_exp2th**2

    n0 = expParams['n0_BEC'] / (L_exp2th**3)  # should ~ 1
    mB = expParams['mB'] * M_exp2th  # should = 1
    mI = expParams['mI'] * M_exp2th
    aBB = expParams['aBB'] * L_exp2th
    gBB = (4 * np.pi / mB) * aBB

    sParams = [mI, mB, n0, gBB]

    aIBi = (expParams['aIB'] * L_exp2th)**(-1)
    hbar = 1.0555e-34  # reduced Planck's constant (J*s/rad)
    a0_exp = 5.29e-11  # Bohr radius (m)

    aIBexp_Vals = np.concatenate((np.array([-12000, -8000, -7000, -6000, -5000]), np.linspace(-4000, -2000, 20, endpoint=False), np.linspace(-2000, -70, 175, endpoint=False), np.linspace(-70, -20, 5))) * a0_exp
    # aIBexp_Vals = np.concatenate((aIBexp_Vals, -1 * np.flip(aIBexp_Vals)))
    aIBexp_Vals = np.concatenate((aIBexp_Vals, np.linspace(20, 650, 100) * a0_exp))
    aIBi_Vals = 1 / (aIBexp_Vals * L_exp2th)

    mR = pf_dynamic_sph.ur(mI, mB)
    gIB_Vals_LS = 1 / ((mR / (2 * np.pi)) * aIBi_Vals - (mR / np.pi**2) * k_max)
    gIB_Vals_Born = (2 * np.pi / mR) * 1 / aIBi_Vals

    # Trap parameters

    n0_TF = expParams['n0_TF'] / (L_exp2th**3)
    n0_thermal = expParams['n0_thermal'] / (L_exp2th**3)
    RTF_BEC_X = expParams['RTF_BEC_X'] * L_exp2th; RTF_BEC_Y = expParams['RTF_BEC_Y'] * L_exp2th; RTF_BEC_Z = expParams['RTF_BEC_Z'] * L_exp2th
    RG_BEC_X = expParams['RG_BEC_X'] * L_exp2th; RG_BEC_Y = expParams['RG_BEC_Y'] * L_exp2th; RG_BEC_Z = expParams['RG_BEC_Z'] * L_exp2th
    omega_BEC_x = expParams['omega_BEC_x'] / T_exp2th; omega_BEC_y = expParams['omega_BEC_y'] / T_exp2th; omega_BEC_z = expParams['omega_BEC_z'] / T_exp2th
    omega_Imp_x = expParams['omega_Imp_x'] / T_exp2th
    trapParams = {'n0_TF_BEC': n0_TF, 'RTF_BEC_X': RTF_BEC_X, 'RTF_BEC_Y': RTF_BEC_Y, 'RTF_BEC_Z': RTF_BEC_Z, 'n0_thermal_BEC': n0_thermal, 'RG_BEC_X': RG_BEC_X, 'RG_BEC_Y': RG_BEC_Y, 'RG_BEC_Z': RG_BEC_Z, 'omega_Imp_x': omega_Imp_x}
    mu_div_hbar = expParams['mu_div_hbar'] / T_exp2th
    mu = 1 * mu_div_hbar  # hbar in theory units is just 1
    # mu_th = 4 * np.pi * (hbar**2) * expParams['aBB'] * expParams['n0_TF'] / expParams['mB']
    # print(mu_th / hbar)

    cBEC = pfs.nu(mB, n0, gBB)
    cBEC_exp = cBEC * T_exp2th / L_exp2th
    xi = (8 * np.pi * n0 * aBB)**(-1 / 2)
    xi_exp = xi / L_exp2th
    tscale_exp = xi_exp / cBEC_exp
    print('c_BEC (um/ms): {:.2f}, xi (um): {:.2f}, xi/c_BEC (ms): {:.2f}'.format(cBEC_exp * 1e3, xi_exp * 1e6, tscale_exp * 1e3))

    # RTF (Calculating from trap frequencies doesn't match explicit measurements)

    n0_TF_pred = n0_TF
    # RTF_BEC_X_pred = np.sqrt(2 * gBB * n0_TF_pred / (mB * (omega_BEC_x**2))); RTF_BEC_Y_pred = np.sqrt(2 * gBB * n0_TF_pred / (mB * (omega_BEC_y**2))); RTF_BEC_Z_pred = np.sqrt(2 * gBB * n0_TF_pred / (mB * (omega_BEC_z**2)))
    RTF_BEC_X_pred = np.sqrt(2 * mu / (mB * (omega_BEC_x**2))); RTF_BEC_Y_pred = np.sqrt(2 * mu / (mB * (omega_BEC_y**2))); RTF_BEC_Z_pred = np.sqrt(2 * mu / (mB * (omega_BEC_z**2)))
    RTF_BEC_X_pred_exp = RTF_BEC_X_pred / L_exp2th; RTF_BEC_Y_pred_exp = RTF_BEC_Y_pred / L_exp2th; RTF_BEC_Z_pred_exp = RTF_BEC_Z_pred / L_exp2th

    print(expParams['RTF_BEC_X'] * 1e6, RTF_BEC_X_pred_exp * 1e6)
    print(expParams['RTF_BEC_Y'] * 1e6, RTF_BEC_Y_pred_exp * 1e6)
    print(expParams['RTF_BEC_Z'] * 1e6, RTF_BEC_Z_pred_exp * 1e6)

    print(gBB * n0_TF_pred / mu)
    n0_BEC = expParams['n0_BEC'] / (L_exp2th**3)
    print(gBB * n0_BEC / E_exp2th / hbar)

    # RTF_X = np.sqrt(expParams['aBB'] * 8 * np.pi * (hbar**2) * expParams['n0_TF'] / ((expParams['mB']**2) * (expParams['omega_BEC_x']**2)))
    # RTF_X = np.sqrt(2 * hbar * expParams['mu_div_hbar'] / (expParams['mB'] * (expParams['omega_BEC_x']**2)))
    # print(RTF_X * 1e6)

    # # MF IMPURITY POTENTIAL

    # gIB_Vals = gIB_Vals_Born
    # # gIB_Vals = gIB_Vals_LS

    # X_Vals = np.linspace(-1 * trapParams['RTF_BEC_X'] * 0.99, trapParams['RTF_BEC_X'] * 0.99, 1e3)
    # X_Vals_m = X_Vals / L_exp2th

    # def V_Imp_trap(X, omega_Imp_x, mI):
    #     return 0.5 * mI * (omega_Imp_x**2) * (X**2)

    # n_BEC_Vals = pf_dynamic_sph.n_BEC(X_Vals, 0, 0, n0_TF, n0_thermal, RTF_BEC_X, RTF_BEC_Y, RTF_BEC_Z, RG_BEC_X, RG_BEC_Y, RG_BEC_Z)
    # V_Imp_Vals = V_Imp_trap(X_Vals, trapParams['omega_Imp_x'], mI)

    # Xlim_fit = 0.9 * trapParams['RTF_BEC_X']
    # Xfit_mask = np.abs(X_Vals) <= Xlim_fit
    # X_Vals_fit = X_Vals[Xfit_mask]
    # omega_Imp_eff = np.zeros(gIB_Vals.size)
    # VMF_Mat = np.zeros((gIB_Vals.size, X_Vals.size))
    # VMF_HarmApprox_Mat = np.zeros((gIB_Vals.size, X_Vals.size))

    # for indg, gIB in enumerate(gIB_Vals):
    #     VMF_Mat[indg, :] = V_Imp_Vals + gIB * n_BEC_Vals
    #     [p2, p1, p0] = np.polyfit(X_Vals_fit, VMF_Mat[indg][Xfit_mask], deg=2)
    #     omega_Imp_eff[indg] = np.sqrt(2 * p2 / mI)
    #     VMF_HarmApprox_Mat[indg, :] = V_Imp_trap(X_Vals, omega_Imp_eff[indg], mI) + p0

    # f_Imp_eff = (omega_Imp_eff / (2 * np.pi)) * T_exp2th

    # MF_freqIncreasePercentage = 100 * (omega_Imp_eff - omega_Imp_x) / omega_Imp_x

    # V_Imp_Hz = (2 * np.pi * hbar)**(-1) * V_Imp_Vals / E_exp2th
    # VMF_Mat_Hz = (2 * np.pi * hbar)**(-1) * VMF_Mat / E_exp2th
    # VMF_HarmApprox_Mat_Hz = (2 * np.pi * hbar)**(-1) * VMF_HarmApprox_Mat / E_exp2th

    # shiftPot = False
    # # indf = 20
    # indf = -1

    # if shiftPot is True:
    #     shift = -1 * np.min(VMF_Mat_Hz[indf])
    #     shiftLabel = ' (Shifted)'
    # else:
    #     shift = 0
    #     shiftLabel = ''

    # fig = plt.figure(figsize=plt.figaspect(0.25))
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    # ax1.plot(X_Vals_m * 1e6, V_Imp_Hz * 1e-3, 'b-', label=r'$V_{imp}(X)$' + ' ({:.2f} Hz)'.format(expParams['omega_Imp_x'] / (2 * np.pi)))
    # ax1.plot(X_Vals_m * 1e6, (VMF_Mat_Hz[indf] + shift) * 1e-3, 'g-', label=r'$V_{eff,MF}(X)=V_{imp}(X)+g_{IB}n_{BEC}(X)$' + shiftLabel)
    # ax1.plot(X_Vals_m * 1e6, (VMF_HarmApprox_Mat_Hz[indf] + shift) * 1e-3, 'r--', label=r'$V_{eff,MF}$' + ' Harmonic Fit ({:.2f} Hz)'.format(f_Imp_eff[indf]) + shiftLabel)
    # ax1.legend()
    # ax1.set_xlabel('X ($\mu$m)')
    # ax1.set_ylabel('Frequency (kHz)')
    # ax1.set_title('MF Potential for Impurity ' + r'($a_{IB}=$' + '{:.0f}'.format(aIBexp_Vals[indf] / a0_exp) + r'$a_{0}$)')
    # # ax1.set_xlim([-50, 50])
    # # ax1.set_ylim([-100, 3000])

    # # ax2.plot(aIBexp_Vals / a0_exp, MF_freqIncreasePercentage, 'b-')
    # ax2.plot(aIBexp_Vals / a0_exp, f_Imp_eff, 'r-', label='Effective Trap Frequency')
    # ax2.plot(aIBexp_Vals / a0_exp, (T_exp2th * omega_Imp_x / (2 * np.pi)) * np.ones(aIBexp_Vals.size), 'b-', label='Bare Trap Frequency')
    # ax2.set_xlabel(r'$a_{IB}$ [$a_{0}$]')
    # # ax2.set_ylabel('Frequency Increase from Bare Impurity Trap (%)')
    # ax2.set_ylabel('Effective Impurity Trap Frequency (Hz)')
    # ax2.set_title('Impurity Frequency Shift from MF Potential')
    # # ax2.set_xlim([-1000, 0]); ax2.set_ylim([100, 300])
    # ax2.legend()

    # plt.show()

    # # BEC DENSITY PROFILE (TF + THERMAL CLOUD)

    # X_Vals = np.linspace(-1 * trapParams['RTF_BEC_X'] * 2, trapParams['RTF_BEC_X'] * 2, 1e3)
    # X_Vals_m = X_Vals / L_exp2th

    # n_BEC_Vals = pf_dynamic_sph.n_BEC(X_Vals, 0, 0, n0_TF, n0_thermal, RTF_BEC_X, RTF_BEC_Y, RTF_BEC_Z, RG_BEC_X, RG_BEC_Y, RG_BEC_Z)
    # n_BEC_TF = pf_dynamic_sph.n_thomasFermi(X_Vals, 0, 0, n0_TF, RTF_BEC_X, RTF_BEC_Y, RTF_BEC_Z)
    # n_BEC_thermal = pf_dynamic_sph.n_thermal(X_Vals, 0, 0, n0_thermal, RG_BEC_X, RG_BEC_Y, RG_BEC_Z)
    # fig3, ax3 = plt.subplots()
    # ax3.plot(X_Vals_m * 1e6, n_BEC_Vals * (L_exp2th**3) * 1e-6, 'b-', label='Total')
    # ax3.plot(X_Vals_m * 1e6, n_BEC_TF * (L_exp2th**3) * 1e-6, 'g-', label='TF')
    # ax3.plot(X_Vals_m * 1e6, n_BEC_thermal * (L_exp2th**3) * 1e-6, 'r-', label='Thermal')
    # ax3.legend()
    # ax3.set_xlabel('$X$ ($\mu$m)')
    # ax3.set_ylabel('$n(X)$ ($cm^{-3}$)')
    # ax3.set_title('BEC Density Profile')

    # plt.show()

    # MF ENERGY POTENTIAL (CURRENT CODE JUST EXTENDS THE MF POTENTAL FROM BEC DENSITY PAST THE TF RADIUS)

    cParams = {}; cParams['aIBi'] = aIBi_Vals[100]
    # print(aIBexp_Vals[100] / a0_exp)
    X_Vals = np.linspace(-1 * trapParams['RTF_BEC_X'] * 0.99, trapParams['RTF_BEC_X'] * 0.99, 100)
    E_Pol_tck = pf_dynamic_sph.V_Pol_interp(kgrid, X_Vals, cParams, sParams, trapParams)

    def F_pol_func(X): return pf_dynamic_sph.F_pol(X, E_Pol_tck)

    fig, ax = plt.subplots()
    ax.plot(1e6 * X_Vals / L_exp2th, 1 * interpolate.splev(X_Vals, E_Pol_tck, der=0), 'b-')
    X_Vals_extend = np.linspace(-1 * trapParams['RTF_BEC_X'] * 2, trapParams['RTF_BEC_X'] * 2, 400)
    ax.plot(1e6 * X_Vals_extend / L_exp2th, 1 * interpolate.splev(X_Vals_extend, E_Pol_tck, der=0), 'g--')

    n0_TF = trapParams['n0_TF_BEC']; RTF_X = trapParams['RTF_BEC_X']; RTF_Y = trapParams['RTF_BEC_Y']; RTF_Z = trapParams['RTF_BEC_Z']
    n0_thermal = trapParams['n0_thermal_BEC']; RG_X = trapParams['RG_BEC_X']; RG_Y = trapParams['RG_BEC_Y']; RG_Z = trapParams['RG_BEC_Z']

    # Find a_star^{-1} values for different densities
    aSiVals = np.zeros(X_Vals.size)
    for ind, X in enumerate(X_Vals):
        n = pf_dynamic_sph.n_BEC(X, 0, 0, n0_TF, n0_thermal, RTF_X, RTF_Y, RTF_Z, RG_X, RG_Y, RG_Z)
        aSiVals[ind] = pfs.aSi_grid(kgrid, 0, mI, mB, n, gBB)
    aSVals = 1 / (aSiVals * L_exp2th) / a0_exp

    fig2, ax2 = plt.subplots()
    # ax2.plot(1e6 * X_Vals / L_exp2th, aSVals)
    ax2.plot(1e6 * X_Vals / L_exp2th, -1 / aSVals)

    plt.show()
