import numpy as np
from polaron_functions import kcos_func
from scipy.integrate import ode
from copy import copy


class CoherentState:
    # """ This is a class that stores information about coherent state """

    def __init__(self, grid_space):

        size = grid_space.size()
        self.amplitude_phase = np.zeros(size + 1, dtype=complex)
        self.time = 0
        self.grid = grid_space

        self.dV = grid_space.dV()
        self.kcos = kcos_func(self.grid)

        self.abs_error = 1.0e-8
        self.rel_error = 1.0e-6

        self.amplitude = np.zeros(size, dtype=complex)
        self.phase = 0 + 0j

    # EVOLUTION

    # def Aevolve(self, dt, hamiltonian):

    #     amp0 = copy(self.amplitude)
    #     t0 = copy(self.time)
    #     amp_solver = ode(hamiltonian.amplitude_update).set_integrator('zvode', method='bdf', atol=self.abs_error, rtol=self.rel_error)
    #     amp_solver.set_initial_value(amp0, t0).set_f_params(self)
    #     self.amplitude = amp_solver.integrate(amp_solver.t + dt)

    #     # ph_solver = ode(hamiltonian.phase_update).set_integrator('zvode', method='bdf')
    #     # ph_solver.set_initial_value(self.phase, self.time).set_f_params(self)
    #     # self.phase = ph_solver.integrate(ph_solver.t + dt)

    #     self.time = self.time + dt

    #     # self.phase = self.phase + dt * hamiltonian.phase_update(0, self.phase, self)
    #     # self.amplitude = self.amplitude + dt * hamiltonian.amplitude_update(0, self.amplitude, self)

    # # OBSERVABLES
    # def Aget_PhononNumber(self):
    #     return np.dot(self.amplitude * np.conjugate(self.amplitude), self.dV)

    # def Aget_PhononMomentum(self):
    #     return np.dot(self.kcos, self.amplitude * np.conjugate(self.amplitude) * self.dV)

    # def Aget_DynOverlap(self):
    #     # dynamical overlap/Ramsey interferometry signal
    #     NB_vec = self.get_PhononNumber()
    #     exparg = -1j * self.phase - (1 / 2) * NB_vec
    #     return np.exp(exparg)

    # EVOLUTION

    def evolve(self, dt, hamiltonian):

        amp_phase0 = copy(self.amplitude_phase)
        t0 = copy(self.time)
        amp_solver = ode(hamiltonian.update).set_integrator('zvode', method='bdf', atol=self.abs_error, rtol=self.rel_error)
        amp_solver.set_initial_value(amp_phase0, t0).set_f_params(self)
        self.amplitude_phase = amp_solver.integrate(amp_solver.t + dt)
        self.time = self.time + dt

    # OBSERVABLES

    def get_PhononNumber(self):
        amplitude = self.amplitude_phase[0:-1]
        return np.dot(amplitude * np.conjugate(amplitude), self.dV)

    def get_PhononMomentum(self):
        amplitude = self.amplitude_phase[0:-1]
        return np.dot(self.kcos, amplitude * np.conjugate(amplitude) * self.dV)

    def get_DynOverlap(self):
        # dynamical overlap/Ramsey interferometry signal
        NB_vec = self.get_PhononNumber()
        phase = self.amplitude_phase[-1]
        exparg = -1j * phase - (1 / 2) * NB_vec
        return np.exp(exparg)
