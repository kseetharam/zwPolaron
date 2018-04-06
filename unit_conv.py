import numpy as np


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
    (omega_x_exp, omega_y_exp, omega_z_exp) = (2 * np.pi * 13, 2 * np.pi * 41, 2 * np.pi * 47)  # impurity trapping frequencies
    mI_exp = 39.96 * u
    mB_exp = 22.99 * u

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
    impTrap_Force_th = -1 * mI_th * (omega_y_th**2) * X_th
    impTrap_Force_exp = -1 * mI_exp * (omega_y_exp**2) * X_exp
    print(impTrap_Force_exp / Fscale_exp)
    print(impTrap_Force_th / Fscale_th)

    # Other
    # print(aBB_th, (aIB_th)**(-1))
    print(omega_x_th, omega_y_th, omega_z_th)
    print(10 * T_th_exp * 1e-3)
    print(En_exp_Hz, En_th_Hz * T_th_exp)
    print(TF_th, TF_th / T_th_exp)
    # print(xi_th / L_th_exp, nu_th * T_th_exp / L_th_exp, tscale_th / T_th_exp, Fscale_th * T_th_exp**2 / (M_th_exp * L_th_exp), (Fscale_th * T_th_exp**2 / (M_th_exp * L_th_exp)) / (2 * np.pi))
