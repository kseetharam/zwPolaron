# SPLIT SPLINE INTERPOLATION

# PB_mask = PB_unique < nuV
# PI_mask = PI_unique < nuV

# PB_unique_S = PB_unique[PB_mask]
# PB_unique_L = PB_unique[np.logical_not(PB_mask)]
# nPBm_cum_S = nPBm_cum[PB_mask]
# nPBm_cum_L = nPBm_cum[np.logical_not(PB_mask)]

# PI_unique_S = PI_unique[PI_mask]
# PI_unique_L = PI_unique[np.logical_not(PI_mask)]
# nPIm_cum_S = nPIm_cum[PI_mask]
# nPIm_cum_L = nPIm_cum[np.logical_not(PI_mask)]

# nPBm_tck_S = interpolate.splrep(PB_unique_S, nPBm_cum_S, k=3, s=1)
# nPBm_tck_L = interpolate.splrep(PB_unique_L, nPBm_cum_L, k=3, s=1)

# nPIm_tck_S = interpolate.splrep(PI_unique_S, nPIm_cum_S, k=3, s=1)
# nPIm_tck_L = interpolate.splrep(PI_unique_L, nPIm_cum_L, k=3, s=1)

# PBm_Vec_S = np.linspace(0, nuV, 50, endpoint=False)
# PBm_Vec_L = np.linspace(nuV, np.max(PB_unique), 50)
# PIm_Vec_S = np.linspace(0, nuV, 50, endpoint=False)
# PIm_Vec_L = np.linspace(nuV, np.max(PI_unique), 50)

# nPBm_cum_Vec_S = interpolate.splev(PBm_Vec_S, nPBm_tck_S, der=0)
# nPBm_cum_Vec_L = interpolate.splev(PBm_Vec_L, nPBm_tck_L, der=0)
# nPIm_cum_Vec_S = interpolate.splev(PIm_Vec_S, nPIm_tck_S, der=0)
# nPIm_cum_Vec_L = interpolate.splev(PIm_Vec_L, nPIm_tck_L, der=0)

# nPBm_Vec_S = interpolate.splev(PBm_Vec_S, nPBm_tck_S, der=1)
# nPBm_Vec_L = interpolate.splev(PBm_Vec_L, nPBm_tck_L, der=1)
# nPIm_Vec_S = interpolate.splev(PIm_Vec_S, nPIm_tck_S, der=1)
# nPIm_Vec_L = interpolate.splev(PIm_Vec_L, nPIm_tck_L, der=1)

# PBm_Vec = np.concatenate((PBm_Vec_S, PBm_Vec_L))
# PIm_Vec = np.concatenate((PIm_Vec_S, PIm_Vec_L))
# nPBm_cum_Vec = np.concatenate((nPBm_cum_Vec_S, nPBm_cum_Vec_L))
# nPIm_cum_Vec = np.concatenate((nPIm_cum_Vec_S, nPIm_cum_Vec_L))
# nPBm_Vec = np.concatenate((nPBm_Vec_S, nPBm_Vec_L))
# nPIm_Vec = np.concatenate((nPIm_Vec_S, nPIm_Vec_L))

# dPBm_L = PBm_Vec[-1] - PBm_Vec[-2]
# dPIm_L = PIm_Vec[-1] - PIm_Vec[-2]
# nPBm_Tot = np.dot(nPBm_Vec, np.ediff1d(nPBm_Vec, to_end=dPBm_L)) + nPB_deltaK0
# nPIm_Tot = np.dot(nPIm_Vec, np.ediff1d(nPIm_Vec, to_end=dPIm_L)) + nPB_deltaK0

# PBm_max = PBm_Vec[np.argmax(nPBm_Vec)]
# PIm_max = PIm_Vec[np.argmax(nPIm_Vec)]

####

# ax[2, 0].plot(PB_unique, nPBm_unique, 'k*')
# # ax[2, 0].plot(np.zeros(PB_unique.size), np.linspace(0, nPB_deltaK0, PB_unique.size))
# ax[2, 0].set_title(r'$n_{\vec{P_B}}$')
# ax[2, 0].set_xlabel(r'$|P_{B}|$')

# ax[2, 1].plot(PI_unique, nPIm_unique, 'k*')
# # ax[2, 1].plot(P * np.ones(PI_unique.size), np.linspace(0, nPB_deltaK0, PI_unique.size))
# ax[2, 1].set_title(r'$n_{\vec{P_I}}$')
# ax[2, 1].set_xlabel(r'$|P_{I}|$')

# ax[2, 2].plot(PBm_Vec, nPBm_Vec)
# ax[2, 2].set_title(r'$n_{\vec{P_B}}$')
# ax[2, 2].set_xlabel(r'$|P_{B}|$')
# ax[2, 2].plot(np.zeros(PB_unique.size), np.linspace(0, nPB_deltaK0, PB_unique.size))

# ax[3, 2].plot(PIm_Vec, nPIm_Vec)
# ax[3, 2].set_title(r'$n_{\vec{P_I}}$')
# ax[3, 2].set_xlabel(r'$|P_{I}|$')
# ax[3, 2].plot(P * np.ones(PI_unique.size), np.linspace(0, nPB_deltaK0, PI_unique.size))

# ax[3, 0].plot(PB_unique, nPBm_cum, 'k*')
# ax[3, 0].set_title('Cumulative Distribution Function')
# ax[3, 0].set_xlabel(r'$|P_{B}|$')
# ax[3, 0].plot(PBm_Vec, nPBm_cum_Vec, 'r-')

# ax[3, 1].plot(PI_unique, nPIm_cum, 'k*')
# ax[3, 1].set_title('Cumulative Distribution Function')
# ax[3, 1].set_xlabel(r'$|P_{I}|$')
# ax[3, 1].plot(PIm_Vec, nPIm_cum_Vec, 'r-')

# # # dist funct

# def staticDistCalc(gridargs, params, datapath):
#     [xgrid, kgrid, kFgrid] = gridargs
#     [P, aIBi, aSi, DP, mI, mB, n0, gBB] = params
#     bparams = [aIBi, aSi, DP, mI, mB, n0, gBB]

#     # unpack grid args
#     x = xgrid.getArray('x'); y = xgrid.getArray('y'); z = xgrid.getArray('z')
#     (Nx, Ny, Nz) = (len(x), len(y), len(z))
#     dx = xgrid.arrays_diff['x']; dy = xgrid.arrays_diff['y']; dz = xgrid.arrays_diff['z']

#     kxF = kFgrid.getArray('kx'); kyF = kFgrid.getArray('ky'); kzF = kFgrid.getArray('kz')

#     kx = kgrid.getArray('kx'); ky = kgrid.getArray('ky'); kz = kgrid.getArray('kz')
#     dkx = kgrid.arrays_diff['kx']; dky = kgrid.arrays_diff['ky']; dkz = kgrid.arrays_diff['kz']

#     # generation
#     xg, yg, zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
#     kxg, kyg, kzg = np.meshgrid(kx, ky, kz, indexing='ij', sparse=True)
#     kxFg, kyFg, kzFg = np.meshgrid(kxF, kyF, kzF, indexing='ij', sparse=True)

#     beta2_kxkykz = np.abs(BetaK(kxFg, kyFg, kzFg, *bparams))**2
#     mask = np.isnan(beta2_kxkykz); beta2_kxkykz[mask] = 0

#     decay_length = 5
#     decay_xyz = np.exp(-1 * (xg**2 + yg**2 + zg**2) / (2 * decay_length**2))

#     # Fourier transform
#     amp_beta_xyz_0 = np.fft.fftn(np.sqrt(beta2_kxkykz))
#     amp_beta_xyz = np.fft.fftshift(amp_beta_xyz_0) * dkx * dky * dkz

#     # Calculate Nph
#     Nph = np.real(np.sum(beta2_kxkykz) * dkx * dky * dkz)
#     Nph_x = np.real(np.sum(np.abs(amp_beta_xyz)**2) * dx * dy * dz * (2 * np.pi)**(-3))

#     # Fourier transform
#     beta2_xyz_preshift = np.fft.fftn(beta2_kxkykz)
#     beta2_xyz = np.fft.fftshift(beta2_xyz_preshift) * dkx * dky * dkz

#     # Exponentiate
#     fexp = (np.exp(beta2_xyz - Nph) - np.exp(-Nph)) * decay_xyz

#     # Inverse Fourier transform
#     nPB_preshift = np.fft.ifftn(fexp) * 1 / (dkx * dky * dkz)
#     nPB = np.fft.fftshift(nPB_preshift)
#     nPB_deltaK0 = np.exp(-Nph)

#     # Integrating out y and z

#     beta2_kz = np.sum(np.abs(beta2_kxkykz), axis=(0, 1)) * dkx * dky
#     nPB_kx = np.sum(np.abs(nPB), axis=(1, 2)) * dky * dkz
#     nPB_ky = np.sum(np.abs(nPB), axis=(0, 2)) * dkx * dkz
#     nPB_kz = np.sum(np.abs(nPB), axis=(0, 1)) * dkx * dky
#     nx_x = np.sum(np.abs(amp_beta_xyz)**2, axis=(1, 2)) * dy * dz
#     nx_y = np.sum(np.abs(amp_beta_xyz)**2, axis=(0, 2)) * dx * dz
#     nx_z = np.sum(np.abs(amp_beta_xyz)**2, axis=(0, 1)) * dx * dy
#     nx_x_norm = np.real(nx_x / Nph_x); nx_y_norm = np.real(nx_y / Nph_x); nx_z_norm = np.real(nx_z / Nph_x)

#     nPB_Tot = np.sum(np.abs(nPB) * dkx * dky * dkz) + nPB_deltaK0
#     nPB_Mom1 = np.dot(np.abs(nPB_kz), kz * dkz)
#     beta2_kz_Mom1 = np.dot(np.abs(beta2_kz), kzF * dkz)

#     # Flipping domain for P_I instead of P_B so now nPB(PI) -> nPI

#     PI_x = -1 * kx
#     PI_y = -1 * ky
#     PI_z = P - kz

#     # Calculate FWHM

#     PI_z_ord = np.flip(PI_z, 0)
#     nPI_z = np.flip(np.real(nPB_kz), 0)

#     if np.abs(np.max(nPI_z) - np.min(nPI_z)) < 1e-2:
#         FWHM = 0
#     else:
#         D = nPI_z - np.max(nPI_z) / 2
#         indices = np.where(D > 0)[0]
#         FWHM = PI_z_ord[indices[-1]] - PI_z_ord[indices[0]]

#     # Calculate magnitude distribution nPB(p) and nPI(p) where p_IorB = sqrt(px^2 + py^2 + pz^2)

#     PB = np.sqrt(kxg**2 + kyg**2 + kzg**2)
#     PI = np.sqrt((-kxg)**2 + (-kyg)**2 + (P - kzg)**2)
#     PB_flat = PB.reshape(PB.size)
#     PI_flat = PI.reshape(PI.size)
#     nPB_flat = nPB.reshape(nPB.size)

#     PB_unique, PB_uind, PB_ucounts = np.unique(PB_flat, return_inverse=True, return_counts=True)
#     PI_unique, PI_uind, PI_ucounts = np.unique(PI_flat, return_inverse=True, return_counts=True)
#     nPBm_unique = np.zeros(PB_unique.size)
#     nPIm_unique = np.zeros(PI_unique.size)

#     for ind, val in enumerate(np.abs(nPB_flat) * dkx * dky * dkz):
#         PB_index = PB_uind[ind]
#         PI_index = PI_uind[ind]
#         nPBm_unique[PB_index] += val
#         nPIm_unique[PI_index] += val

#     # Calculate CDF

#     nPBm_cum = np.cumsum(nPBm_unique)
#     nPIm_cum = np.cumsum(nPIm_unique)

#     # CDF and PDF smoothing/averaging

#     PBm_Vec = np.linspace(0, np.max(PB_unique), 200)
#     PIm_Vec = np.linspace(0, np.max(PI_unique), 200)
#     dPBm = PBm_Vec[1] - PBm_Vec[0]
#     dPIm = PIm_Vec[1] - PIm_Vec[0]

#     nPBm_cum_smooth, bin_edgesB, binnumberB = binned_statistic(x=PB_unique, values=nPBm_cum, bins=PBm_Vec, statistic='mean')
#     nPIm_cum_smooth, bin_edgesI, binnumberI = binned_statistic(x=PI_unique, values=nPIm_cum, bins=PIm_Vec, statistic='mean')

#     nPBm_dat = np.gradient(nPBm_cum_smooth, dPBm)
#     nPIm_dat = np.gradient(nPIm_cum_smooth, dPIm)

#     PB_mask = np.isnan(nPBm_dat)
#     PI_mask = np.isnan(nPIm_dat)

#     if(any(PB_mask) or any(PI_mask)):
#         print('Zeros in nP*m_dat')
#     nPBm_dat[PB_mask] = 0
#     nPIm_dat[PI_mask] = 0

#     PBm_Vec = PBm_Vec[0:-1]
#     PIm_Vec = PIm_Vec[0:-1]

#     nPBm_Tot = np.sum(nPBm_dat * dPBm) + nPB_deltaK0
#     nPIm_Tot = np.sum(nPIm_dat * dPIm) + nPB_deltaK0

#     # Metrics/consistency checks

#     print("FWHM = {0}, Var = {1}".format(FWHM, (FWHM / 2.355)**2))
#     print("Nph = \sum b^2 = %f" % (Nph))
#     print("Nph_x = %f " % (Nph_x))
#     print("\int np dp = %f" % (nPB_Tot))
#     print("\int p np dp = %f" % (nPB_Mom1))
#     print("\int k beta^2 dk = %f" % (beta2_kz_Mom1))
#     print("Exp[-Nph] = %f" % (nPB_deltaK0))
#     print("\int n(PB_mag) dPB_mag = %f" % (nPBm_Tot))
#     print("\int n(PI_mag) dPI_mag = %f" % (nPIm_Tot))

#     # Save data

#     Dist_data = np.concatenate((DP * np.ones(Nz)[:, np.newaxis], Nph * np.ones(Nz)[:, np.newaxis], Nph_x * np.ones(Nz)[:, np.newaxis], nPB_Tot * np.ones(Nz)[:, np.newaxis], nPB_Mom1 * np.ones(Nz)[:, np.newaxis], beta2_kz_Mom1 * np.ones(Nz)[:, np.newaxis], FWHM * np.ones(Nz)[:, np.newaxis], x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis], nx_x_norm[:, np.newaxis], nx_y_norm[:, np.newaxis], nx_z_norm[:, np.newaxis], kx[:, np.newaxis], ky[:, np.newaxis], kz[:, np.newaxis], np.real(nPB_kx)[:, np.newaxis], np.real(nPB_ky)[:, np.newaxis], np.real(nPB_kz)[:, np.newaxis], PI_z_ord[:, np.newaxis], np.real(nPI_z)[:, np.newaxis]), axis=1)
#     np.savetxt(datapath + '/3Ddist_aIBi_{:.2f}_P_{:.2f}.dat'.format(aIBi, P), Dist_data)
