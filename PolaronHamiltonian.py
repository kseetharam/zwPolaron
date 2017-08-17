import numpy as np
import polaron_functions as pf


class PolaronHamiltonian:
        # """ This is a class that stores information about the Hamiltonian"""

    def __init__(self, coherent_state, Params):

        # Params = [P, aIBi, mI, mB, n0, gBB]
        self.Params = Params

        self.grid = coherent_state.kgrid

        self.gnum = pf.g(self.grid, *Params)
        self.Omega0_grid = pf.omega0(self.grid, *Params)
        self.Wk_grid = pf.Wk(self.grid, *Params)
        self.Wki_grid = 1 / self.Wk_grid
        self.kcos = pf.kcos_func(self.grid)

        # print(self.Omega0_grid.shape)

    # def phase_update(self, t, phase, coherent_state):

    #     [P, aIBi, mI, mB, n0, gBB] = self.Params

    #     dv = coherent_state.dV

    #     amplitude_t = coherent_state.amplitude
    #     PB_t = coherent_state.get_PhononMomentum()

    #     betaSum = amplitude_t + np.conjugate(amplitude_t)

    #     xp_t = 0.5 * np.dot(self.Wk_grid, betaSum * dv)

    #     return self.gnum * n0 + self.gnum * np.sqrt(n0) * xp_t + (P**2 - PB_t**2) / (2 * mI)

    # def amplitude_update(self, t, amplitude, coherent_state):
    #     # here on can write any method induding Runge-Kutta 4

    #     [P, aIBi, mI, mB, n0, gBB] = self.Params

    #     dV = coherent_state.dV

    #     PB_t = np.dot(self.kcos, amplitude * np.conjugate(amplitude) * dV)

    #     betaSum = amplitude + np.conjugate(amplitude)
    #     xp_t = 0.5 * np.dot(self.Wk_grid, betaSum * dV)

    #     betaDiff = amplitude - np.conjugate(amplitude)
    #     xm_t = 0.5 * np.dot(self.Wki_grid, betaDiff * dV)

    #     return -1j * (self.gnum * np.sqrt(n0) * self.Wk_grid +
    #                   amplitude * (self.Omega0_grid - self.kcos * (P - PB_t) / mI) +
    #                   self.gnum * (self.Wk_grid * xp_t + self.Wki_grid * xm_t))

    def update(self, t, amplitude_phase, coherent_state):
        # here on can write any method induding Runge-Kutta 4
        amplitude = amplitude_phase[0:-1]
        amplitude_phase_new = np.zeros(amplitude_phase.size, dtype=complex)

        [P, aIBi, mI, mB, n0, gBB] = self.Params

        dV = coherent_state.dV

        PB = np.dot(self.kcos, amplitude * np.conjugate(amplitude) * dV)

        betaSum = amplitude + np.conjugate(amplitude)
        xp = 0.5 * np.dot(self.Wk_grid, betaSum * dV)

        betaDiff = amplitude - np.conjugate(amplitude)
        xm = 0.5 * np.dot(self.Wki_grid, betaDiff * dV)

        amplitude_phase_new[0:-1] = -1j * (self.gnum * np.sqrt(n0) * self.Wk_grid +
                                           amplitude * (self.Omega0_grid - self.kcos * (P - PB) / mI) +
                                           self.gnum * (self.Wk_grid * xp + self.Wki_grid * xm))

        amplitude_phase_new[-1] = self.gnum * n0 + self.gnum * np.sqrt(n0) * xp + (P**2 - PB**2) / (2 * mI)
        return amplitude_phase_new
