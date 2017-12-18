import numpy as np
from pf_spherical import kcos_func, ksin_func, kpow2_func
from scipy.integrate import ode
from copy import copy


class CoherentState:
    # """ This is a class that stores information about coherent state """

    def __init__(self, kgrid, xgrid):

        # last element of self.amplitude_phase is the phase, the rest is the amplitude
        self.amplitude_phase = np.zeros(kgrid.size() + 1, dtype=complex)
        self.time = 0

        self.kgrid = kgrid
        self.xgrid = xgrid
        self.coordinate_system = kgrid.coordinate_system
        if(kgrid.coordinate_system != xgrid.coordinate_system):
            print('ERROR: GRID COORDINATE SYSTEM MISTMATCH')

        self.dVk = self.kgrid.dV()

        if(self.coordinate_system == "SPHERICAL_2D"):
            self.kcos = kcos_func(kgrid)
            self.ksin = ksin_func(kgrid)
            self.kpow2 = kpow2_func(kgrid)

        self.abs_error = 1.0e-8
        self.rel_error = 1.0e-6

#  # THIS WAS FOR DISTRIBUTION FUNCTION ATTEMPT IN SPHERICAL COORDINATES -- DEPRACATED
        # self.th = kgrid.function_prod(list(kgrid.arrays.keys()), [lambda k: 0 * k + 1, lambda th: th])
        # self.xgrid = xgrid
        # self.xmagVals = xgrid.function_prod(list(xgrid.arrays.keys()), [lambda x: x, lambda th: 0 * th + 1])
        # self.xthetaVals = xgrid.function_prod(list(xgrid.arrays.keys()), [lambda x: 0 * x + 1, lambda th: th])
        # self.dV_x = (2 * np.pi)**3 * self.xgrid.dV()

        # self.PBgrid = PBgrid
        # self.PBmagVals = PBgrid.function_prod(list(PBgrid.arrays.keys()), [lambda PB: PB, lambda th: 0 * th + 1])
        # self.PBthetaVals = PBgrid.function_prod(list(PBgrid.arrays.keys()), [lambda PB: 0 * PB + 1, lambda th: th])
        # self.dV_PB = (2 * np.pi)**3 * self.PBgrid.dV()

        # self.FTkernel_kx = FTkernel_func(kgrid, xgrid, True)
        # # self.FTkernel_xPB = FTkernel_func(xgrid, PBgrid, True)

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

    # PURELY MOMENTUM SPACE DEPENDENT OBSERVABLES

    def get_PhononNumber(self):
        amplitude = self.amplitude_phase[0:-1]
        return np.dot(amplitude * np.conjugate(amplitude), self.dVk).real.astype(float)  # should this be changed to np.abs() for Cartesian?

    def get_PhononMomentum(self):
        amplitude = self.amplitude_phase[0:-1]
        if self.coordinate_system == "SPHERICAL_2D":
            return np.dot(self.kcos, amplitude * np.conjugate(amplitude) * self.dVk).real.astype(float)
        elif self.coordinate_system == "CARTESIAN_3D":
            np.dot(np.abs(beta2_kz), kzF * dkz)
            return
            # !!!!!
        else:
            print('INVALID COORDINATE SYSTEM')
            return

    def get_DynOverlap(self):
        # dynamical overlap/Ramsey interferometry signal
        NB = self.get_PhononNumber()
        phase = self.amplitude_phase[-1]
        exparg = -1j * phase - (1 / 2) * NB
        return np.exp(exparg)

    # def get_MomentumDispersion(self):
    #     amplitude = self.amplitude_phase[0:-1]
    #     return np.dot(self.kpow2 * amplitude * np.conjugate(amplitude), self.dVk).real.astype(float)

    # POSITION SPACE DEPENDENT OBSERVABLES

    def get_Distributions(self):
        amplitude = self.amplitude_phase[0:-1]

        if self.coordinate_system != "CARTESIAN_3D":
            print('INVALID COORDINATE SYSTEM')
            return -1

        # unpack grid args

        x = self.xgrid.getArray('x'); y = self.xgrid.getArray('y'); z = self.xgrid.getArray('z')
        (Nx, Ny, Nz) = (len(x), len(y), len(z))
        dx = self.xgrid.arrays_diff['x']; dy = self.xgrid.arrays_diff['y']; dz = self.xgrid.arrays_diff['z']

        kxF = kFgrid.getArray('kx'); kyF = kFgrid.getArray('ky'); kzF = kFgrid.getArray('kz')

        kx = kgrid.getArray('kx'); ky = kgrid.getArray('ky'); kz = kgrid.getArray('kz')
        dkx = kgrid.arrays_diff['kx']; dky = kgrid.arrays_diff['ky']; dkz = kgrid.arrays_diff['kz']
        dVk = dkx * dky * dkz

        xg, yg, zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
        kxg, kyg, kzg = np.meshgrid(kx, ky, kz, indexing='ij', sparse=True)
        kxFg, kyFg, kzFg = np.meshgrid(kxF, kyF, kzF, indexing='ij', sparse=True)

        # generation

        beta_kxkykz = (2 * np.pi)**(-3 / 2) * amplitude.reshape(())
        beta2_kxkykz = np.abs(beta_kxkykz)**2
        decay_length = 5
        decay_xyz = np.exp(-1 * (xg**2 + yg**2 + zg**2) / (2 * decay_length**2))

        # Fourier transform
        amp_beta_xyz_0 = np.fft.fftn(np.sqrt(beta2_kxkykz))
        amp_beta_xyz = np.fft.fftshift(amp_beta_xyz_0) * dkx * dky * dkz  # this is the position distribution

        # Calculate Nph
        Nph = self.get_PhononNumber()
        # Nph = np.real(np.sum(beta2_kxkykz) * dkx * dky * dkz)

        # Fourier transform
        beta2_xyz_preshift = np.fft.fftn(beta2_kxkykz)
        beta2_xyz = np.fft.fftshift(beta2_xyz_preshift) * dkx * dky * dkz

        # Exponentiate
        fexp = (np.exp(beta2_xyz - Nph) - np.exp(-Nph)) * decay_xyz

        # Inverse Fourier transform
        nPB_preshift = np.fft.ifftn(fexp) * 1 / (dkx * dky * dkz)
        nPB = np.fft.fftshift(nPB_preshift)
        nPB_deltaK0 = np.exp(-Nph)

        # Flipping domain for P_I instead of P_B so now nPB(PI) -> nPI

        PI_x = -1 * kx
        PI_y = -1 * ky
        PI_z = P - kz

        # Calculate magnitude distribution nPB(P) and nPI(P) where P_IorB = sqrt(Px^2 + Py^2 + Pz^2) - calculate CDF from this

        PB = np.sqrt(kxg**2 + kyg**2 + kzg**2)
        PI = np.sqrt((-kxg)**2 + (-kyg)**2 + (P - kzg)**2)
        PB_flat = PB.reshape(PB.size)
        PI_flat = PI.reshape(PI.size)
        nPB_flat = np.abs(nPB.reshape(nPB.size))

        PB_series = pd.Series(nPB_flat, index=PB_flat)
        PI_series = pd.Series(nPB_flat, index=PI_flat)

        nPBm_unique = PB_series.groupby(PB_series.index).sum() * dkx * dky * dkz
        nPIm_unique = PI_series.groupby(PI_series.index).sum() * dkx * dky * dkz

        PB_unique = nPBm_unique.keys().values
        PI_unique = nPIm_unique.keys().values

        nPBm_cum = nPBm_unique.cumsum()
        nPIm_cum = nPIm_unique.cumsum()

        # CDF and PDF pre-processing

        PBm_Vec, dPBm = np.linspace(0, np.max(PB_unique), 200, retstep=True)
        PIm_Vec, dPIm = np.linspace(0, np.max(PI_unique), 200, retstep=True)

        nPBm_cum_smooth = nPBm_cum.groupby(pd.cut(x=nPBm_cum.index, bins=PBm_Vec, right=True, include_lowest=True)).mean()
        nPIm_cum_smooth = nPIm_cum.groupby(pd.cut(x=nPIm_cum.index, bins=PIm_Vec, right=True, include_lowest=True)).mean()

        # one less bin than bin edge so consider each bin average to correspond to left bin edge and throw out last (rightmost) edge
        PBm_Vec = PBm_Vec[0:-1]
        PIm_Vec = PIm_Vec[0:-1]

        # smooth data has NaNs in it from bins that don't contain any points - forward fill these holes
        PBmapping = pd.Series(PBm_Vec, index=nPBm_cum_smooth.keys())
        PImapping = pd.Series(PIm_Vec, index=nPIm_cum_smooth.keys())
        nPBm_cum_smooth = nPBm_cum_smooth.rename(PBmapping).fillna(method='ffill')
        nPIm_cum_smooth = nPIm_cum_smooth.rename(PImapping).fillna(method='ffill')

        nPBm_Vec = np.gradient(nPBm_cum_smooth, dPBm)
        nPIm_Vec = np.gradient(nPIm_cum_smooth, dPIm)

        pos_dist = amp_beta_xyz
        mom_dist = nPB
        mag_dist_List = [PBm_Vec, nPBm_Vec, PIm_Vec, nPIm_Vec]

        return pos_dist, mom_dist, mag_dist_List


#  # THIS WAS FOR DISTRIBUTION FUNCTION ATTEMPT IN SPHERICAL COORDINATES -- DEPRACATED
    # def get_PositionDistribution(self):
    #     # outputs a vector of values corresponding to x, thetap pairs
    #     amplitude = self.amplitude_phase[0:-1]
    #     return (np.abs(np.dot(self.dV * amplitude, self.FTkernel_kx))**2).real.astype(float)

    # def get_MomentumDistribution(self, PBgrid):
    #     amplitude = self.amplitude_phase[0:-1]
    #     Nph = self.get_PhononNumber()
    #     FTkernel_xPB = FTkernel_func(self.xgrid, PBgrid, False)
    #     # G = np.exp(np.dot(self.dV * amplitude * np.conjugate(amplitude), self.FTkernel_kx) - Nph)
    #     # Ntheta = self.xgrid.arrays['th'].size
    #     # G0 = G[0:Ntheta - 1]
    #     MD = np.dot(self.dV_x * np.exp(np.dot(self.dV * amplitude * np.conjugate(amplitude), self.FTkernel_kx) - Nph), FTkernel_xPB)
    #     return MD.real.astype(float)
