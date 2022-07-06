import numpy as np
import pf_dynamic_sph as pfs
import pf_dynamic_cart as pfc
from scipy import interpolate


class zw2021_PolaronHamiltonian_2D:
        # """ This is a class that stores information about the Hamiltonian"""

    def __init__(self, coherent_state, Params, LDA_funcs, trapParams, toggleDict):

        # Params = [aIBi, mI, mB, n0, gBB]
        self.Params = Params

        self.LDA_funcs = LDA_funcs
        self.trapParams = trapParams

        self.grid = coherent_state.kgrid
        self.coordinate_system = coherent_state.coordinate_system
        self.kz = coherent_state.kzg_flat
        self.k0mask = coherent_state.k0mask

        self.k2 = coherent_state.k2_flat

        self.dynamicsType = toggleDict['Dynamics']
        self.couplingType = toggleDict['Coupling']
        self.BEC_density_var = toggleDict['BEC_density']
        self.Pol_Potential = toggleDict['Polaron_Potential']
        self.PP_Type = toggleDict['PP_Type']
        self.CS_Dyn = toggleDict['CS_Dyn']

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

        amplitude = system_vars[0:-5]
        phase = system_vars[-5].real.astype(float)
        X = system_vars[-4].real.astype(float)
        PX = system_vars[-3].real.astype(float)
        Y = system_vars[-2].real.astype(float)
        PY = system_vars[-1].real.astype(float)
        YLab = Y + pfs.x_BEC_osc_zw2021(t, self.trapParams['omega_BEC_osc'], self.trapParams['gamma_BEC_osc'], self.trapParams['phi_BEC_osc'], self.trapParams['amp_BEC_osc'])

        [aIBi, mI, mB, n0, gBB] = self.Params
        RTF_BEC_X = self.trapParams['RTF_BEC_X']; RTF_BEC_Y = self.trapParams['RTF_BEC_Y']
        F_BEC_osc_func = self.LDA_funcs['F_BEC_osc']; F_Imp_trap_Y_func = self.LDA_funcs['F_Imp_trap_Y']
        densityFunc = self.trapParams['densityFunc']; densityGradFunc = self.trapParams['densityGradFunc']

        # Update BEC density dependent quantities

        if self.BEC_density_var == 'on':
            n = densityFunc(X, Y)
            nGrad = densityGradFunc(X, Y)
            dndx = nGrad[0]; dndy = nGrad[1]

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

        # FOR REAL TIME DYNAMICS:
        amplitude_new_temp = -1j * (self.gnum * np.sqrt(n) * self.Wk_grid +
                                    amplitude * (self.Omega0_grid - self.kz * (P - PB) / mI) +
                                    self.gnum * (self.Wk_grid * xp + self.Wki_grid * xm))
        phase_new_temp = self.gnum * n + self.gnum * np.sqrt(n) * xp + (PY**2 - PB**2) / (2 * mI)

        if self.Pol_Potential == 'off':
            P_new_temp = - F_BEC_osc_func(t) + F_Imp_trap_Y_func(YLab)
        else:
            amp_re = np.real(amplitude); amp_im = np.imag(amplitude)
            Wk2_grid = self.Wk_grid**2; Wk3_grid = self.Wk_grid**3; omegak_grid = pfs.omegak_grid(self.grid, mB, n, gBB)
            eta1 = np.dot(Wk2_grid * np.abs(amplitude)**2, dVk); eta2 = np.dot((Wk3_grid / omegak_grid) * amp_re, dVk); eta3 = np.dot((self.Wk_grid / omegak_grid) * amp_im, dVk)
            xp_re = 0.5 * np.dot(self.Wk_grid * amp_re, dVk); xm_im = 0.5 * np.dot(self.Wki_grid * amp_im, dVk)
            A_PP = self.gnum * (1 + 2 * xp_re / n) + gBB * eta1 - self.gnum * gBB * ((np.sqrt(n) + 2 * xp_re) * eta2 + 2 * xm_im * eta3)
            F_pol = -1 * A_PP * dndy

            PY_new_temp = F_pol - F_BEC_osc_func(t) + F_Imp_trap_Y_func(X, YLab)
            PX_new_temp = -1 * self.gnum * dndx + F_Imp_trap_X_func(X, YLab)
        

        if self.BEC_density_var == 'on':
            if np.isclose(np.heaviside(1 - X ** 2 / RTF_BEC_Y ** 2 - Y ** 2 / RTF_BEC_Y ** 2, 1 / 2), 0):
                amplitude_new_temp = 0 * amplitude_new_temp
                phase_new_temp = 0 * phase_new_temp
                PY_new_temp = - F_BEC_osc_func(t) + F_Imp_trap_Y_func(X, YLab)
                PX_new_temp = F_Imp_trap_X_func(X, YLab)

        X_new_temp = PX / mI
        Y_new_temp = (PY - PB) / mI

        amplitude_new_temp[self.k0mask] = 0  # ensure Beta_k remains equal to 0 where |k| = 0 to avoid numerical issues (this is an unphysical point)

        if self.CS_Dyn == 'off':
            amplitude_new_temp = 0 * amplitude_new_temp
            phase_new_temp = 0 * phase_new_temp

        # Assign updates

        system_vars_new[0:-5] = amplitude_new_temp
        system_vars_new[-5] = phase_new_temp
        system_vars_new[-4] = X_new_temp
        system_vars_new[-3] = PX_new_temp
        system_vars_new[-2] = Y_new_temp
        system_vars_new[-1] = PY_new_temp

        return system_vars_new
