import numpy as np


class Coherent:
    # """ This is a class that stores information about coherent state """

    def __init__(self, grid_space):

        size = grid_space.size()
        self.amplitude = np.zeros(size, dtype=complex)
        self.phase = 0 + 0j

    # EVOLUTION
    def evolve(self, dt):

        dt = grid_time.dV()
        self.phase = self.phase + dt * self.phi_update()
        self.amplitude = self.amplitude + dt * self.amplitude_update()

    def phi_update(self, Hamiltonian):
        # here on can write any method induding Runge-Kutta 4
        return

    def amplitude_update(self, Hamiltonian):
        # here on can write any method induding Runge-Kutta 4

        return

    # OBSERVABLES
    def Number_of_phonons(self, grid_space)

        coherent_amplitude = self.amplitude
        dv = grid_space.dV()
        return np.dot(coherent_amplitude * np.conjugate(coherent_amplitude), dv)


class Hamiltonian:
        # """ This is a class that stores information about the Hamiltonian"""

    def __init__(self, grid_space):

        self.update
