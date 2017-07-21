import numpy as np
from polaron_functions import kcos_func, ksin_func, kpow2_func, FTkernal_func
from scipy.integrate import ode
from copy import copy


class CoherentState:
    # """ This is a class that stores information about coherent state """

    def __init__(self, kgrid, xgrid):

        size = kgrid.size()
        # last element of self.amplitude_phase is the phase, the rest is the amplitude
        self.amplitude_phase = np.zeros(size + 1, dtype=complex)

        self.time = 0
        self.grid = kgrid

        self.dV = self.grid.dV()
        self.kcos = kcos_func(self.grid)
        self.ksin = ksin_func(self.grid)
        self.kpow2 = kpow2_func(self.grid)
        self.FTkernal = FTkernal_func(self.kcos, self.ksin, xgrid)

        self.abs_error = 1.0e-8
        self.rel_error = 1.0e-6

    # EVOLUTION

    def evolve(self, dt, hamiltonian):

        amp_phase0 = copy(self.amplitude_phase)
        t0 = copy(self.time)
        amp_solver = ode(hamiltonian.update).set_integrator('zvode', method='bdf', atol=self.abs_error, rtol=self.rel_error, nsteps=100000)
        amp_solver.set_initial_value(amp_phase0, t0).set_f_params(self)
        self.amplitude_phase = amp_solver.integrate(amp_solver.t + dt)
        self.time = self.time + dt

    # CHARACTERISTICS

    def get_Amplitude(self):
        return self.amplitude_phase[0:-1]

    def get_Phase(self):
        return self.amplitude_phase[-1].real.astype(float)

    # OBSERVABLES

    def get_PhononNumber(self):
        amplitude = self.amplitude_phase[0:-1]
        return np.dot(amplitude * np.conjugate(amplitude), self.dV).real.astype(float)

    def get_PhononMomentum(self):
        amplitude = self.amplitude_phase[0:-1]
        return np.dot(self.kcos, amplitude * np.conjugate(amplitude) * self.dV).real.astype(float)

    def get_DynOverlap(self):
        # dynamical overlap/Ramsey interferometry signal
        NB = self.get_PhononNumber()
        phase = self.amplitude_phase[-1]
        exparg = -1j * phase - (1 / 2) * NB
        return np.exp(exparg)

    def get_MomentumDispersion(self):
        amplitude = self.amplitude_phase[0:-1]
        return np.dot(self.kpow2 * amplitude * np.conjugate(amplitude), self.dV).real.astype(float)

    def get_PositionDistribution(self):
        amplitude = self.amplitude_phase[0:-1]
        return np.dot(self.dV * amplitude, self.FTkernal)
