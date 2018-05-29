import numpy as np
import pf_dynamic_sph as pfs
import pf_dynamic_cart as pfc


class LDA_PolaronHamiltonian:
        # """ This is a class that stores information about the Hamiltonian"""

    def __init__(self, coherent_state, Params, LDA_funcs, fParams, trapParams, toggleDict):

        # Params = [aIBi, mI, mB, n0, gBB]
        self.Params = Params

        self.LDA_funcs = LDA_funcs
        self.fParams = fParams
        self.trapParams = trapParams

        self.grid = coherent_state.kgrid
        self.coordinate_system = coherent_state.coordinate_system
        self.kz = coherent_state.kzg_flat
        self.k0mask = coherent_state.k0mask

        self.k2 = coherent_state.k2_flat

        self.dynamicsType = toggleDict['Dynamics']
        self.couplingType = toggleDict['Coupling']
        self.BEC_density_var = toggleDict['BEC_density']
        self.BEC_density_osc = toggleDict['BEC_density_osc']

        if self.couplingType == 'frohlich':
            [aIBi, mI, mB, n0, gBB] = self.Params
            self.gnum = (2 * np.pi / pfs.ur(mI, mB)) * (1 / aIBi)

        if(self.coordinate_system == "SPHERICAL_2D"):
            self.gnum = pfs.g(self.grid, *Params[0:])
            self.Omega0_grid = pfs.Omega(self.grid, 0, *Params[1:])
            self.Wk_grid = pfs.Wk(self.grid, *Params[2:])
            # self.Wk_grid[self.k0mask] = 1  # k0mask should be all False in the Spherical case so the second line shouldn't do anything
            self.Wki_grid = 1 / self.Wk_grid

        if(self.coordinate_system == "CARTESIAN_3D"):
            self.kxg, self.kyg, self.kzg = coherent_state.kxg, coherent_state.kyg, coherent_state.kzg
            self.gnum = pfc.g(self.kxg, self.kyg, self.kzg, coherent_state.dVk[0], *Params[0:])
            self.Omega0_grid = pfc.Omega(self.kxg, self.kyg, self.kzg, 0, *Params[1:]).flatten()
            self.Wk_grid = pfc.Wk(self.kxg, self.kyg, self.kzg, *Params[2:]).flatten(); self.Wk_grid[self.k0mask] = 1  # this is where |k| = 0 -> changing this value to 1 arbitrarily shouldn't affect the actual calculation as we are setting Beta_k = 0 here too
            self.Wki_grid = 1 / self.Wk_grid

    # @profile
    def update(self, t, system_vars, coherent_state):

        amplitude = system_vars[0:-3]
        # phase = system_vars[-3].real.astype(float)
        P = system_vars[-2].real.astype(float)
        X = system_vars[-1].real.astype(float)

        [aIBi, mI, mB, n0, gBB] = self.Params
        F_ext_func = self.LDA_funcs['F_ext']; F_pol_func = self.LDA_funcs['F_pol']
        dP = self.fParams['dP_ext']; F = self.fParams['Fext_mag']

        RTF_X = self.trapParams['RTF_BEC_X']; RTF_Y = self.trapParams['RTF_BEC_Y']; RTF_Z = self.trapParams['RTF_BEC_Z']; RG_X = self.trapParams['RG_BEC_X']; RG_Y = self.trapParams['RG_BEC_Y']; RG_Z = self.trapParams['RG_BEC_Z']
        n0_TF = self.trapParams['n0_TF_BEC']; n0_thermal = self.trapParams['n0_thermal_BEC']
        omega_BEC_osc = self.trapParams['omega_BEC_osc']

        # Update BEC density dependent quantities

        if self.BEC_density_var == 'on':
            n = pfs.n_BEC(X, 0, 0, n0_TF, n0_thermal, RTF_X, RTF_Y, RTF_Z, RG_X, RG_Y, RG_Z)  # ASSUMING PARTICLE IS IN CENTER OF TRAP IN Y AND Z DIRECTIONS
            if self.BEC_density_osc == 'on':
                n = n * pfs.n_BEC_osc(t, omega_BEC_osc)
            if(self.coordinate_system == "SPHERICAL_2D"):
                self.Omega0_grid = pfs.Omega(self.grid, 0, mI, mB, n, gBB)
                self.Wk_grid = pfs.Wk(self.grid, mB, n, gBB)
                self.Wki_grid = 1 / self.Wk_grid
            if(self.coordinate_system == "CARTESIAN_3D"):
                self.Omega0_grid = pfc.Omega(self.kxg, self.kyg, self.kzg, 0, mI, mB, n, gBB).flatten()
                self.Wk_grid = pfc.Wk(self.kxg, self.kyg, self.kzg, mB, n, gBB).flatten(); self.Wk_grid[self.k0mask] = 1
                self.Wki_grid = 1 / self.Wk_grid
        else:
            n = n0

        # Calculate updates

        amplitude[self.k0mask] = 0  # set Beta_k = 0 where |k| = 0 to avoid numerical issues (this is an unphysical point)
        system_vars_new = np.zeros(system_vars.size, dtype=complex)

        dVk = coherent_state.dVk

        betaSum = amplitude + np.conjugate(amplitude)
        xp = 0.5 * np.dot(self.Wk_grid, betaSum * dVk)

        betaDiff = amplitude - np.conjugate(amplitude)
        xm = 0.5 * np.dot(self.Wki_grid, betaDiff * dVk)

        PB = np.dot(self.kz * np.abs(amplitude)**2, dVk)

        # FOR FROLICH:
        if self.couplingType == 'frohlich':
            xp = 0
            xm = 0

        if self.dynamicsType == 'real':
            # FOR REAL TIME DYNAMICS:
            amplitude_new_temp = -1j * (self.gnum * np.sqrt(n) * self.Wk_grid +
                                        amplitude * (self.Omega0_grid - self.kz * (P - PB) / mI) +
                                        self.gnum * (self.Wk_grid * xp + self.Wki_grid * xm))
            phase_new_temp = self.gnum * n + self.gnum * np.sqrt(n) * xp + (P**2 - PB**2) / (2 * mI)
            P_new_temp = F_ext_func(t, F, dP) + F_pol_func(X)
            X_new_temp = (P - PB) / mI

        elif self.dynamicsType == 'imaginary':
            # FOR IMAGINARY TIME DYNAMICS
            amplitude_new_temp = -1 * (self.gnum * np.sqrt(n) * self.Wk_grid +
                                       amplitude * (self.Omega0_grid - self.kz * (P - PB) / mI) +
                                       self.gnum * (self.Wk_grid * xp + self.Wki_grid * xm))
            phase_new_temp = -1j * (self.gnum * n + self.gnum * np.sqrt(n) * xp + (P**2 - PB**2) / (2 * mI))
            P_new_temp = -1j * (F_ext_func(t, F, dP) + F_pol_func(X))
            X_new_temp = -1j * (P - PB) / mI

        amplitude_new_temp[self.k0mask] = 0  # ensure Beta_k remains equal to 0 where |k| = 0 to avoid numerical issues (this is an unphysical point)

        # Assign updates

        system_vars_new[0:-3] = amplitude_new_temp
        system_vars_new[-3] = phase_new_temp
        system_vars_new[-2] = P_new_temp
        system_vars_new[-1] = X_new_temp

        return system_vars_new
