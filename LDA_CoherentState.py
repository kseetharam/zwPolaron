import numpy as np
from pf_dynamic_sph import kcos_func, kpow2_func
from scipy.integrate import ode
from copy import copy


class LDA_CoherentState:
    # """ This is a class that stores information about coherent state with local density approximation to include confining potentials, nonuniform BEC density, and force"""

    def __init__(self, kgrid, xgrid):

        self.system_vars = np.zeros(kgrid.size() + 3, dtype=complex)  # indices [0:-3] is the coherent state amplitude, index [-3] is the coherent state phase, index [-2] is total system momentum P, and index [-1] is average impurity position X=(P-Pph)/mI
        self.time = 0

        self.kgrid = kgrid
        self.xgrid = xgrid

        self.k0mask = np.zeros(kgrid.size(), dtype=bool)  # this is where |k| = 0 in the provided kgrid -> never true for Spherical grid

        self.coordinate_system = kgrid.coordinate_system
        if(kgrid.coordinate_system != xgrid.coordinate_system):
            print('GRID COORDINATE SYSTEM MISTMATCH - DISTRIBUTION INFO UNAVAILABLE')

        # precompute quantities to make observable calculations computationally cheaper
        self.dVk = self.kgrid.dV()

        if(self.coordinate_system == "SPHERICAL_2D"):
            self.kzg_flat = kcos_func(kgrid)
            self.k2_flat = kpow2_func(kgrid)
        if(self.coordinate_system == "CARTESIAN_3D"):
            self.xg, self.yg, self.zg = np.meshgrid(self.xgrid.getArray('x'), self.xgrid.getArray('y'), self.xgrid.getArray('z'), indexing='ij')
            self.kxg, self.kyg, self.kzg = np.meshgrid(self.kgrid.getArray('kx'), self.kgrid.getArray('ky'), self.kgrid.getArray('kz'), indexing='ij')
            self.Nx, self.Ny, self.Nz = len(self.xgrid.getArray('x')), len(self.xgrid.getArray('y')), len(self.xgrid.getArray('z'))
            self.kzg_flat = self.kzg.reshape(self.kzg.size)
            self.dVx_const = ((2 * np.pi)**(3)) * self.xgrid.dV()[0]
            self.k0mask[(self.Nx * self.Ny * self.Nz) // 2] = True  # this is where |k| = sqrt(kx^2 + ky^2 + kz^2) = 0 in the Cartesian grid

            self.k2 = self.kxg**2 + self.kyg**2 + self.kzg**2
            self.k2_flat = self.k2.reshape(self.k2.size)

        # error for ODE solver
        self.abs_error = 1.0e-8
        self.rel_error = 1.0e-6

    # SET INITIAL CONDITION

    def set_initState(self, amplitude, phase, P, X):
        self.system_vars[0:-3] = amplitude
        self.system_vars[-3] = phase
        self.system_vars[-2] = P
        self.system_vars[-1] = X
        return

    # EVOLUTION

    # @profile
    def evolve(self, dt, hamiltonian):

        system_vars0 = copy(self.system_vars)
        t0 = copy(self.time)

        sys_solver = ode(hamiltonian.update).set_integrator('zvode', method='bdf', atol=self.abs_error, rtol=self.rel_error, nsteps=100000)
        sys_solver.set_initial_value(system_vars0, t0).set_f_params(self)

        self.system_vars = sys_solver.integrate(sys_solver.t + dt)
        self.time = self.time + dt

    # CHARACTERISTICS

    def get_Amplitude(self):
        return self.system_vars[0:-3]

    def get_Phase(self):
        return self.system_vars[-3].real.astype(float)

    def get_totMom(self):
        return self.system_vars[-2].real.astype(float)

    def get_impPos(self):
        return self.system_vars[-1].real.astype(float)

    # MOMENTUM SPACE DEPENDENT OBSERVABLES

    def get_PhononNumber(self):
        amplitude = self.get_Amplitude()
        return np.dot(np.abs(amplitude)**2, self.dVk).real.astype(float)

    def get_PhononMomentum(self):
        amplitude = self.get_Amplitude()
        return np.dot(self.kzg_flat * np.abs(amplitude)**2, self.dVk).real.astype(float)

    def get_DynOverlap(self):
        # dynamical overlap/Ramsey interferometry signal -> S(t) = <psi_0|psi(t)>
        NB = self.get_PhononNumber()
        phase = self.get_Phase()
        exparg = -1j * phase - (1 / 2) * NB
        return np.exp(exparg)

    # DISTRIBUTION

    def get_PhononDistributions(self):
        amplitude = self.get_Amplitude()  # this is flattened and stored w.r.t. kgrid

        if self.coordinate_system != "CARTESIAN_3D":
            print('INVALID COORDINATE SYSTEM')
            return -1

        # generation

        beta_kxkykz = np.fft.ifftshift(amplitude.reshape((self.Nx, self.Ny, self.Nz)))  # unflatten Beta_k, FFT shift it to prepare for Fourier transform
        beta2_kxkykz = np.abs(beta_kxkykz)**2

        decay_length = 5
        decay_xyz = np.exp(-1 * (self.xg**2 + self.yg**2 + self.zg**2) / (2 * decay_length**2))

        # Calculate Nph
        Nph = self.get_PhononNumber()
        # Nph = np.sum(beta2_kxkykz) * self.dVk[0]

        # Fourier transform
        amp_beta_xyz_preshift = np.fft.ifftn(beta_kxkykz) / self.dVx_const
        amp_beta_xyz = np.fft.fftshift(amp_beta_xyz_preshift)
        nxyz = np.abs(amp_beta_xyz)**2  # this is the unnormalized phonon position distribution in 3D Cartesian coordinates
        nxyz_norm = nxyz / Nph  # this is the normalized phonon position distribution in 3D Cartesian coordinates

        # Fourier transform
        beta2_xyz_preshift = np.fft.ifftn(beta2_kxkykz) / self.dVx_const
        beta2_xyz = np.fft.fftshift(beta2_xyz_preshift)

        # Exponentiate
        fexp = (np.exp(beta2_xyz - Nph) - np.exp(-Nph)) * decay_xyz

        # Inverse Fourier transform
        nPB_preshift = np.fft.fftn(fexp) * self.dVx_const
        nPB_complex = np.fft.fftshift(nPB_preshift) / ((2 * np.pi)**3)  # this is the phonon momentum distribution in 3D Cartesian coordinates
        nPB = np.abs(nPB_complex)
        nPB_deltaK0 = np.exp(-Nph)

        phonon_pos_dist = nxyz_norm  # this is a 3D matrix in terms of x,y,z
        phonon_mom_dist = nPB  # this is a 3D matrix in terms of kx, ky, kz -> more accurately PB_x, PB_y, PB_z
        phonon_mom_k0deltapeak = nPB_deltaK0  # this is the weight of the delta peak in nPB at kx=ky=kz=0

        return phonon_pos_dist, phonon_mom_dist, phonon_mom_k0deltapeak
