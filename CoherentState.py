import numpy as np
from polaron_functions import kcos_func
from scipy.integrate import ode


class CoherentState:
    # """ This is a class that stores information about coherent state """

    def __init__(self, grid_space):

        size = grid_space.size()
        self.amplitude = np.zeros(size, dtype=complex)
        self.phase = 0 + 0j
        self.time = 0
        self.grid = grid_space

        self.dV = grid_space.dV()
        self.kcos = kcos_func(self.grid)

        self.abs_error = 1.0e-8
        self.rel_error = 1.0e-6

    # EVOLUTION

    # def evolve(self, dt, hamiltonian):

    #     self.phase = self.phase + dt * hamiltonian.phi_update(self)
    #     self.amplitude = self.amplitude + dt * hamiltonian.amplitude_update(self)

    def evolve(self, dt, hamiltonian):

        amp_solver = ode(hamiltonian.amplitude_update).set_integrator('zvode', method='bdf', atol=self.abs_error, rtol=self.rel_error)
        amp_solver.set_initial_value(self.amplitude, self.time).set_f_params(self)
        self.amplitude = amp_solver.integrate(amp_solver.t + dt)

        ph_solver = ode(hamiltonian.phase_update).set_integrator('zvode', method='bdf')
        ph_solver.set_initial_value(self.phase, self.time).set_f_params(self)
        self.phase = ph_solver.integrate(ph_solver.t + dt)

        self.time = self.time + dt

        # self.phase = self.phase + dt * hamiltonian.phase_update(0, 0, self)
        # self.amplitude = self.amplitude + dt * hamiltonian.amplitude_update(0, 0, self)

    # OBSERVABLES
    def get_PhononNumber(self):
        return np.dot(self.amplitude * np.conjugate(self.amplitude), self.dV)

    def get_PhononMomentum(self):
        return np.dot(self.kcos, self.amplitude * np.conjugate(self.amplitude) * self.dV)

    def get_DynOverlap(self):
        # dynamical overlap/Ramsey interferometry signal
        NB_vec = self.get_PhononNumber()
        exparg = -1j * self.phase - (1 / 2) * NB_vec
        return np.exp(exparg)
