import numpy as np
import pf_dynamic_sph as pfs
import pf_dynamic_cart as pfc


class PolaronHamiltonian:
        # """ This is a class that stores information about the Hamiltonian"""

    def __init__(self, coherent_state, Params):

        # Params = [P, aIBi, mI, mB, n0, gBB]

        self.Params = Params

        self.grid = coherent_state.kgrid
        self.coordinate_system = coherent_state.coordinate_system
        self.kz = coherent_state.kzg_flat
        self.k0mask = coherent_state.k0mask

        if(self.coordinate_system == "SPHERICAL_2D"):
            self.gnum = pfs.g(self.grid, *Params[1:])
            self.Omega0_grid = pfs.Omega(self.grid, 0, *Params[2:])
            # self.Wk_grid = pf.Wk(self.grid, *Params)
            self.Wk_grid = pfs.Wk(self.grid, *Params[3:])
            # ; self.Wk_grid[self.k0mask] = 1  # k0mask should be all False in the Spherical case so the second line shouldn't do anything
            self.Wki_grid = 1 / self.Wk_grid

            # dV = coherent_state.dVk; dV[self.k0mask] = 0
            # print(np.dot(np.abs(self.Wk_grid), dV))
            # print(np.dot(np.abs(self.Omega0_grid), dV))

        if(self.coordinate_system == "CARTESIAN_3D"):
            self.kxg, self.kyg, self.kzg = coherent_state.kxg, coherent_state.kyg, coherent_state.kzg
            self.gnum = pfc.g(self.kxg, self.kyg, self.kzg, *Params[1:])
            self.Omega0_grid = pfc.Omega(self.kxg, self.kyg, self.kzg, 0, *Params[2:]).flatten()
            self.Wk_grid = pfc.Wk(self.kxg, self.kyg, self.kzg, *Params[3:]).flatten(); self.Wk_grid[self.k0mask] = 1  # this is where |k| = 0 -> changing this value to 1 arbitrarily shouldn't affect the actual calculation as we are setting Beta_k = 0 here too
            self.Wki_grid = 1 / self.Wk_grid

            # dV = coherent_state.dVk; dV[self.k0mask] = 0
            # print(np.pi * np.dot(np.abs(self.Wk_grid), dV))
            # print(np.pi * np.dot(np.abs(self.Omega0_grid), dV))

    def update(self, t, amplitude_phase, coherent_state):
        # here on can write any method induding Runge-Kutta 4
        amplitude = amplitude_phase[0:-1]
        amplitude[self.k0mask] = 0  # set Beta_k = 0 where |k| = 0 to avoid numerical issues (this is an unphysical point)
        amplitude_phase_new = np.zeros(amplitude_phase.size, dtype=complex)

        [P, aIBi, mI, mB, n0, gBB] = self.Params

        dVk = coherent_state.dVk

        betaSum = amplitude + np.conjugate(amplitude)
        xp = 0.5 * np.dot(self.Wk_grid, betaSum * dVk)

        # betaDiff = amplitude - np.conjugate(amplitude)
        # xm = 0.5 * np.dot(self.Wki_grid, betaDiff * dVk)

        PB = np.dot(self.kz * np.abs(amplitude)**2, dVk)

        # print(xp, xm, PB)

        # xp = 0
        xm = 0

        # amplitude_new_temp = -1j * (self.gnum * np.sqrt(n0) * self.Wk_grid +
        #                             amplitude * (self.Omega0_grid - self.kz * (P - PB) / mI) +
        #                             self.gnum * (self.Wk_grid * xp + self.Wki_grid * xm))

        amplitude_new_temp = -1 * (self.gnum * np.sqrt(n0) * self.Wk_grid +
                                   amplitude * (self.Omega0_grid - self.kz * (P - PB) / mI) +
                                   self.gnum * (self.Wk_grid * xp + self.Wki_grid * xm))

        amplitude_new_temp[self.k0mask] = 0  # ensure Beta_k remains equal to 0 where |k| = 0 to avoid numerical issues (this is an unphysical point)
        amplitude_phase_new[0:-1] = amplitude_new_temp
        amplitude_phase_new[-1] = self.gnum * n0 + self.gnum * np.sqrt(n0) * xp + (P**2 - PB**2) / (2 * mI)

        # PB = np.dot(self.kz, amplitude * np.conjugate(amplitude) * dVk)

        # amplitude_phase_new[0:-1] = -1j * (self.gnum * np.sqrt(n0) * self.Wk_grid +
        #                                    amplitude * (self.Omega0_grid - self.kz * (P - PB) / mI) +
        #                                    self.gnum * (self.Wk_grid * xp + self.Wki_grid * xm))
        # amplitude_phase_new[-1] = self.gnum * n0 + self.gnum * np.sqrt(n0) * xp + (P**2 - PB**2) / (2 * mI)

        # # Frohlich model (without two phonon contribution)
        # # gf = (2 * np.pi / pfs.ur(mI, mB)) * (1 / aIBi)
        # # amplitude_phase_new[0:-1] = -1j * (gf * np.sqrt(n0) * self.Wk_grid +
        # #                                    amplitude * (self.Omega0_grid - self.kz * (P - PB) / mI))

        # # amplitude_phase_new[-1] = gf * n0 + gf * np.sqrt(n0) * xp + (P**2 - PB**2) / (2 * mI)

        return amplitude_phase_new
