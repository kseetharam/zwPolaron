import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os


matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})


def epsilon(kx, ky, kz, mB):
    return (kx**2 + ky**2 + kz**2) / (2 * mB)


def omegak(kx, ky, kz, mB):
    ep = epsilon(kx, ky, kz, mB)
    gBB = (4 * np.pi / mB) * 0.05
    n = 1
    return np.sqrt(ep * (ep + 2 * gBB * n))


def Omega(kx, ky, kz, DP, mI, mB):
    return omegak(kx, ky, kz, mB) + epsilon(kx, ky, kz, mB) / mI - kx * DP / mI


def Wk(kx, ky, kz, mB):
    return np.sqrt(epsilon(kx, ky, kz, mB) / omegak(kx, ky, kz, mB))


def Bkt(kx, ky, kz, aIBi, aSi, DP, mI, mB):
    ur = mI * mB / (mI + mB)
    n = 1
    Bk = -2 * np.pi * np.sqrt(n) * Wk(kx, ky, kz, mB) / (ur * Omega(kx, ky, kz, DP, mI, mB) * (aIBi - aSi))
    prefactor = (2 * np.pi)**(-3 / 2)
    return prefactor * Bk


def staticDistCalc(gridargs, params, datapath):
    [Nx, Ny, Nz, x, y, z, dx, dy, dz, kx, ky, kz, dkx, dky, dkz, kxfft, kyfft, kzfft] = gridargs
    [aIBi, aSi, DP, mI, mB] = params

    # generation
    beta2_kxkykz = np.zeros((Nx, Ny, Nz)).astype('complex')
    decay_xyz = np.zeros((Nx, Ny, Nz)).astype('complex')

    for indx in np.arange(Nx):
        for indy in np.arange(Ny):
            for indz in np.arange(Nz):
                kxF = kxfft[indx]
                kyF = kyfft[indy]
                kzF = kzfft[indz]

                x_ = x[indx]
                y_ = y[indx]
                z_ = z[indx]

                if(kxF == 0 and kyF == 0 and kzF == 0):
                    beta2_kxkykz[indx, indy, indz] = 0
                else:
                    # decay_momentum = 100
                    # decay = np.exp(-1 * epsilon(kxF, kyF, kzF, mB) / (decay_momentum**2))
                    # beta_kxkykz[indx, indy, indz] = np.abs(Bkt(kxF, kyF, kzF, aIBi, aSi, DP, mI, mB))**2 * decay
                    beta2_kxkykz[indx, indy, indz] = np.abs(Bkt(kxF, kyF, kzF, *params))**2

                    decay_length = 8
                    decay_xyz[indx, indy, indz] = np.exp(-1 * (x_**2 + y_**2 + z_**2) / (2 * decay_length**2))

    # Fourier transform and slice
    amp_beta_xyz_0 = np.fft.fftn(np.sqrt(beta2_kxkykz))
    amp_beta_xyz = np.fft.fftshift(amp_beta_xyz_0) * dkx * dky * dkz

    # Calculate Nph
    Nph = np.real(np.sum(beta2_kxkykz) * dkx * dky * dkz)
    Nph_x = np.real(np.sum(np.abs(amp_beta_xyz)**2) * dx * dy * dz * (2 * np.pi)**(-3))

    # Fourier transform and slice
    beta2_xyz_preshift = np.fft.fftn(beta2_kxkykz)
    beta2_xyz = np.fft.fftshift(beta2_xyz_preshift) * dkx * dky * dkz
    beta2_x = beta2_xyz[:, Ny // 2 + 1, Nz // 2 + 1]

    # Exponentiate, slice
    fexp = (np.exp(beta2_xyz - Nph) - np.exp(-Nph)) * decay_xyz
    # fexp = (np.exp(beta_xyz - Nph) - np.exp(-Nph))
    fexp_x = fexp[:, Ny // 2 + 1, Nz // 2 + 1]

    # Inverse Fourier transform
    nPB_preshift = np.fft.ifftn(fexp) * 1 / (dkx * dky * dkz)
    nPB = np.fft.fftshift(nPB_preshift)
    nPB_deltaK0 = np.exp(-Nph)

    # Integrating out y and z
    beta2_kx = np.zeros(Nx).astype('complex')

    nPB_kx = np.zeros(Nx).astype('complex')
    nPB_ky = np.zeros(Ny).astype('complex')
    nPB_kz = np.zeros(Nz).astype('complex')

    nx_x = np.zeros(Nx).astype('complex')
    nx_y = np.zeros(Ny).astype('complex')
    nx_z = np.zeros(Nz).astype('complex')

    for indx in np.arange(Nx):
        for indy in np.arange(Ny):
            for indz in np.arange(Nz):
                beta2_kx[indx] = beta2_kx[indx] + np.abs(beta2_kxkykz[indx, indy, indz]) * dky * dkz

                nPB_kx[indx] = nPB_kx[indx] + np.abs(nPB[indx, indy, indz]) * dky * dkz
                nPB_ky[indy] = nPB_ky[indy] + np.abs(nPB[indx, indy, indz]) * dkx * dkz
                nPB_kz[indz] = nPB_kz[indz] + np.abs(nPB[indx, indy, indz]) * dkx * dky

                nx_x[indx] = nx_x[indx] + np.abs(amp_beta_xyz[indx, indy, indz])**2 * dy * dz
                nx_y[indy] = nx_y[indy] + np.abs(amp_beta_xyz[indx, indy, indz])**2 * dx * dz
                nx_z[indz] = nx_z[indz] + np.abs(amp_beta_xyz[indx, indy, indz])**2 * dx * dy

    nx_x_norm = np.real(nx_x / Nph_x)
    nx_y_norm = np.real(nx_y / Nph_x)
    nx_z_norm = np.real(nx_z / Nph_x)

    nPB_Tot = np.sum(np.abs(nPB) * dkx * dky * dkz) + nPB_deltaK0
    # nPB_Totx = np.sum(np.abs(nPB_kx) * dkx) + nPB_deltaK0
    nPB_Mom1 = np.dot(np.abs(nPB_kx), kx * dkx)
    beta2_kx_Mom1 = np.dot(np.abs(beta2_kx), kxfft * dkx)

    print("Nph = \sum b^2 = %f" % (Nph))
    print("Nph_x = %f " % (Nph_x))
    print("\int np dp = %f" % (nPB_Tot))
    print("\int p np dp = %f" % (nPB_Mom1))
    print("\int k beta^2 dk = %f" % (beta2_kx_Mom1))
    print("Exp[-Nph] = %f" % (nPB_deltaK0))

    # Save data
    Dist_data = np.concatenate((Nph * np.ones(Nx)[:, np.newaxis], Nph_x * np.ones(Nx)[:, np.newaxis], nPB_Tot * np.ones(Nx)[:, np.newaxis], nPB_Mom1 * np.ones(Nx)[:, np.newaxis], beta2_kx_Mom1 * np.ones(Nx)[:, np.newaxis], x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis], nx_x_norm[:, np.newaxis], nx_y_norm[:, np.newaxis], nx_z_norm[:, np.newaxis], kx[:, np.newaxis], ky[:, np.newaxis], kz[:, np.newaxis], np.real(nPB_kx)[:, np.newaxis], np.real(nPB_ky)[:, np.newaxis], np.real(nPB_kz)[:, np.newaxis]), axis=1)
    np.savetxt(datapath + '/3Ddist_aIBi_%.2f_DP_%.2f.dat' % (aIBi, DP), Dist_data)

    # Plot
    # fig, ax = plt.subplots(nrows=1, ncols=5)

    # ax[0].plot(kxfft, np.abs(beta2_kx))
    # ax[0].set_title(r'$|\beta_{\vec{k}}|^2$')
    # ax[0].set_xlabel(r'$k_{x}$')

    # ax[3].plot(kx, np.real(nPB_kx))
    # ax[3].plot(np.zeros(Nx), np.linspace(0, nPB_deltaK0, Nx))
    # ax[3].set_title(r'$n_{\vec{P_B}}$')
    # ax[3].set_xlabel(r'$P_{B,x}$')

    # ax[4].plot(x, np.real(nx_x_norm))
    # ax[4].set_title(r'$\frac{n(\vec{x})}{N_{ph}}$')
    # ax[4].set_xlabel(r'$x$')

    # ax[1].plot(x, np.real(beta2_x))
    # ax[2].plot(x, np.abs(fexp_x))

    # # ax[1].plot(ky, np.real(nPB_ky))
    # # ax[1].plot(np.zeros(Ny), np.linspace(0, nPB_deltaK0, Ny))
    # # ax[2].plot(kz, np.real(nPB_kz))
    # # ax[2].plot(np.zeros(Nz), np.linspace(0, nPB_deltaK0, Nz))

    # fig.tight_layout()
    # plt.show()


# Create grids
(Lx, Ly, Lz) = (15, 15, 15)
(dx, dy, dz) = (1e-01, 1e-01, 1e-01)
x = np.arange(- Lx, Lx + dx, dx)
y = np.arange(- Ly, Ly + dy, dy)
z = np.arange(- Lz, Lz + dz, dz)

(Lkx, Lky, Lkz) = (np.pi / dy, np.pi / dy, np.pi / dz)
(dkx, dky, dkz) = (np.pi / Lx, np.pi / Ly, np.pi / Lz)

# FFT prep - this is the same as previous one
(Nx, Ny, Nz) = (len(x), len(y), len(z))

kxfft = np.fft.fftfreq(Nx) * 2 * np.pi / dx
kyfft = np.fft.fftfreq(Nx) * 2 * np.pi / dy
kzfft = np.fft.fftfreq(Nx) * 2 * np.pi / dz

kx = np.fft.fftshift(kxfft)
ky = np.fft.fftshift(kyfft)
kz = np.fft.fftshift(kzfft)

gridargs = [Nx, Ny, Nz, x, y, z, dx, dy, dz, kx, ky, kz, dkx, dky, dkz, kxfft, kyfft, kzfft]

# Set parameters
aIBi = 2
aSi = 0
DP = 0.75
mI = 1
mB = 1
params = [aIBi, aSi, DP, mI, mB]

aIBi_Vals = [-10, -5, -2, 2, 5, 10]
DP_Vals = [0.0, 0.25, 0.5, 0.75]
paramsList_aIBi = [[aIBi, aSi, 0.5, mI, mB] for aIBi in aIBi_Vals]
paramsList_DP = [[-2, aSi, DP, mI, mB] for DP in DP_Vals]
paramsList = paramsList_aIBi + paramsList_DP

# Call function
datapath = os.path.dirname(os.path.realpath(__file__)) + '/fftdata'
for p in paramsList:
    staticDistCalc(gridargs, p, datapath)
