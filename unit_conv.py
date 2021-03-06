import numpy as np
import pf_dynamic_sph as pfs

if __name__ == "__main__":

    # Constants (SI units)

    a0 = 5.29e-11  # Bohr radius (m)
    hbar = 1.0555e-34  # reduced Planck's constant (J*s/rad)
    u = 1.661e-27  # atomic mass unit (kg)

    # Experimental parameters (SI units)

    aIB_exp = -2600 * a0
    aBB_exp = 52 * a0
    n0_exp = 2e14 * 1e6  # BEC peak density
    nI_exp = 1.4e11 * 1e6  # impurity peak density
    (omega_x_exp, omega_y_exp, omega_z_exp) = (2 * np.pi * 13, 2 * np.pi * 41, 2 * np.pi * 101)  # BEC trapping frequencies
    mI_exp = 39.96 * u
    mB_exp = 22.99 * u
    EF_exp_Hz = 10e3  # (impurity) Fermi energy

    # Derived quantities (SI units)
    kn_exp = (6 * np.pi**2 * n0_exp)**(1 / 3)
    En_exp = hbar**2 * kn_exp**2 / (2 * mB_exp)
    En_exp_Hz = En_exp / (2 * np.pi * hbar)
    inv_kn_aIB_exp = (kn_exp * aIB_exp)**(-1)
    gBB_exp = (4 * np.pi * hbar**2 / mB_exp) * aBB_exp
    nu_exp = np.sqrt(n0_exp * gBB_exp / mB_exp)
    xi_exp = (8 * np.pi * n0_exp * aBB_exp)**(-1 / 2)
    tscale_exp = xi_exp / nu_exp
    Fscale_exp = 2 * np.pi * hbar * nu_exp / (xi_exp**2)
    EF_exp = 2 * np.pi * hbar * EF_exp_Hz
    # kF_exp = np.sqrt(2 * mI_exp * EF_exp) / hbar  # (impurity) Fermi momentum
    EF_nu_ratio_exp = np.sqrt(2 * mI_exp * 2 * np.pi * hbar * EF_exp_Hz) / (mI_exp * nu_exp)
    EF_nu_rat_exp = (2 * np.pi * hbar * nI_exp**(1 / 3)) / (mI_exp * nu_exp)
    RTF_x_exp = np.sqrt(2 * gBB_exp * n0_exp / (mB_exp * omega_x_exp**2))
    RTF_y_exp = np.sqrt(2 * gBB_exp * n0_exp / (mB_exp * omega_y_exp**2))
    RTF_z_exp = np.sqrt(2 * gBB_exp * n0_exp / (mB_exp * omega_z_exp**2))

    # Theory parameters (n0^(-1/3)=1, mB = 1, hbar = 1 scale)
    mB_th = 1
    n0_th = 1
    hbar_th = 1

    L_th_exp = n0_th**(-1 / 3) / n0_exp**(-1 / 3)
    M_th_exp = mB_th / mB_exp
    T_th_exp = (mB_th * (n0_th)**(-2 / 3) / (2 * np.pi * hbar_th)) / (mB_exp * (n0_exp)**(-2 / 3) / (2 * np.pi * hbar))

    mI_th = mI_exp * M_th_exp
    aBB_th = aBB_exp * L_th_exp
    # aBB_th = 0.062
    aIB_th = aIB_exp * L_th_exp
    (omega_x_th, omega_y_th, omega_z_th) = (omega_x_exp / T_th_exp, omega_y_exp / T_th_exp, omega_z_exp / T_th_exp)

    # Derived quantities (theory units)
    kn_th = (6 * np.pi**2 * n0_th)**(1 / 3)
    En_th = hbar_th**2 * kn_th**2 / (2 * mB_th)
    En_th_Hz = En_th / (2 * np.pi * hbar_th)
    inv_kn_aIB_th = (kn_th * aIB_th)**(-1)
    gBB_th = (4 * np.pi * hbar_th**2 / mB_th) * aBB_th
    nu_th = np.sqrt(n0_th * gBB_th / mB_th)
    xi_th = (8 * np.pi * n0_th * aBB_th)**(-1 / 2)
    tscale_th = xi_th / nu_th
    Fscale_th = 2 * np.pi * hbar_th * nu_th / (xi_th**2)
    TF_th = 0.5 * mI_th * nu_th / Fscale_th

    # Comparing impurity trap and external force
    X_th = np.linspace(0, 10, 20)
    X_exp = X_th / L_th_exp
    omega_impTrap_shallow_th = 2 * np.pi * 40 / T_th_exp
    omega_impTrap_deep_th = 2 * np.pi * 143 / T_th_exp
    impTrap_Force_th = -mI_th * omega_impTrap_deep_th**2 * X_th

    # Other
    print(Fscale_exp)
    print(8 / L_th_exp)
    print(1 / (kn_exp * aIB_exp), kn_exp * aBB_exp)
    print(1 / aIB_th, aBB_th)
    print(RTF_x_exp * 1e6, RTF_y_exp * 1e6, RTF_z_exp * 1e6)
    # print(omega_impTrap_deep_th)
    # print(omega_impTrap_deep_th / (2 * np.pi * nu_th / xi_th))
    # print(impTrap_Force_th / Fscale_th)
    # print(xi_th / L_th_exp, nu_th * T_th_exp / L_th_exp, tscale_th / T_th_exp, Fscale_th * T_th_exp**2 / (M_th_exp * L_th_exp), (Fscale_th * T_th_exp**2 / (M_th_exp * L_th_exp)) / (2 * np.pi))

    # density

    nTF_peak_exp = 6e13 * 1e6
    nG_peak_exp = 0.9e13 * 1e6
    RTF_BEC_X = 103e-6; RTF_BEC_Y = 32e-6; RTF_BEC_Z = 13e-6
    RG_BEC_X = 95e-6; RG_BEC_Y = 29e-6; RG_BEC_Z = 12e-6
    nBEC_peak_exp = pfs.n_BEC(0, 0, 0, nTF_peak_exp, nG_peak_exp, RTF_BEC_X, RTF_BEC_Y, RTF_BEC_Z, RG_BEC_X, RG_BEC_Y, RG_BEC_Z)

    NTF = nTF_peak_exp * (15 / (8 * np.pi * RTF_BEC_X * RTF_BEC_Y * RTF_BEC_Z)) ** (-1)
    NG = nG_peak_exp * (1 / (RG_BEC_X * RG_BEC_Y * RG_BEC_Z * np.pi**(1.5)))**(-1)
    print('{:E}, {:E}, {:E}'.format(NTF, NG, NTF + NG))
