import numpy as np
import Grid
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import polaron_functions_cart as pfc


def epsilon(kx, ky, kz):
    mB = 1
    return (kx**2 + ky**2 + kz**2) / (2 * mB)


def omegak(kx, ky, kz):
    ep = epsilon(kx, ky, kz)
    mB = 1
    gBB = (4 * np.pi / mB) * 0.05
    n = 1
    return np.sqrt(ep * (ep + 2 * gBB * n))


def Omega(kx, ky, kz):
    mI = 1
    DP = 0.2
    return omegak(kx, ky, kz) + epsilon(kx, ky, kz) / mI - kx * DP / mI


def Bkt(kx, ky, kz):
    aIBi = -5
    aSi = 0
    mI = 1
    mB = 1
    ur = mI * mB / (mI + mB)
    Bk = 2 * np.pi * np.sqrt(epsilon(kx, ky, kz) / omegak(kx, ky, kz)) / (ur * Omega(kx, ky, kz) * (aIBi - aSi))
    prefactor = (2 * np.pi)**(-3 / 2)
    return prefactor * Bk


# Create grids
(Lx, Ly, Lz) = (10, 10, 10)
(dx, dy, dz) = (2.5e-01, 2.5e-01, 2.5e-01)
x = np.arange(- Lx, Lx + dx, dx)
y = np.arange(- Ly, Ly + dy, dy)
z = np.arange(- Lz, Lz + dz, dz)

(Lkx, Lky, Lkz) = (np.pi / dy, np.pi / dy, np.pi / dz)
(dkx, dky, dkz) = (np.pi / Lx, np.pi / Ly, np.pi / Lz)
kx = dkx * np.arange(- Lx / dx, Lx / dx + 1, 1)
ky = dky * np.arange(- Ly / dy, Ly / dy + 1, 1)
kz = dkz * np.arange(- Lz / dz, Lz / dz + 1, 1)

# FFT prep - this is the same as previous one
(Nx, Ny, Nz) = (len(x), len(y), len(z))

kxfft = np.fft.fftfreq(Nx) * 2 * np.pi * Nx / Lx / 2
kyfft = np.fft.fftfreq(Nx) * 2 * np.pi * Ny / Ly / 2
kzfft = np.fft.fftfreq(Nx) * 2 * np.pi * Nz / Lz / 2

# print(kx)
# print(kxfft)

# generation

# beta_xyz = np.zeros((Nx, Ny, Nz)).astype('complex')
beta_kxkykz = np.zeros((Nx, Ny, Nz)).astype('complex')
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
                beta_kxkykz[indx, indy, indz] = 0
            else:
                decay_momentum = 100
                decay = np.exp(-1 * epsilon(kxF, kyF, kzF) / (decay_momentum**2))
                beta_kxkykz[indx, indy, indz] = np.abs(Bkt(kxF, kyF, kzF))**2 * decay
                decay_length = 10
                decay_xyz[indx, indy, indz] = np.exp(-1 * (x_**2 + y_**2 + z_**2) / (2 * decay_length**2))


dVk = np.ones((Nx, Ny, Nz)).astype('complex')
Nph = np.real(np.sum(beta_kxkykz * dVk) * dkx * dky * dkz)
print("Nph = \sum b^2 = %f" % (Nph))

# Slice x,y=0,z=0
beta_kx = np.zeros(Nx).astype('complex')

# Fourier transform and slice
amp_beta_xyz_0 = np.fft.fftn(np.sqrt(beta_kxkykz))
amp_beta_xyz = np.fft.fftshift(amp_beta_xyz_0) * dkx * dky * dkz
nx_x = np.zeros(Nx).astype('complex')


# Fourier transform and slice
beta_xyz_0 = np.fft.fftn(beta_kxkykz)
beta_xyz = np.fft.fftshift(beta_xyz_0) * dkx * dky * dkz
beta_x = beta_xyz[:, Ny // 2 + 1, Nz // 2 + 1]

# Exponenciate, slice
fexp = (np.exp(beta_xyz - Nph) - np.exp(-Nph)) * decay_xyz
fexp_x = fexp[:, Ny // 2 + 1, Nz // 2 + 1]

# Inverse Fourier transform
Gexp0 = np.fft.ifftn(fexp) * 1 / (dkx * dky * dkz)
Gexp = np.fft.fftshift(Gexp0)
Gexp_kx = np.zeros(Nx).astype('complex')

for indx in np.arange(Nx):
    for indy in np.arange(Ny):
        for indz in np.arange(Nz):
            beta_kx[indx] = beta_kx[indx] + np.abs(beta_kxkykz[indx, indy, indz]) * dky * dkz
            Gexp_kx[indx] = Gexp_kx[indx] + np.abs(Gexp[indx, indy, indz]) * dky * dkz
            nx_x[indx] = nx_x[indx] + np.abs(amp_beta_xyz[indx, indy, indz])**2 * dy * dz


dxVec = dx * np.ones(Nx) * (2 * np.pi)**(-3)
dkxVec = dkx * np.ones(Nx)

print("(\int beta_x dx)^2 = %f " % (np.dot(np.abs(nx_x), dxVec)))
print("\int np dp = %f" % (np.dot(np.abs(Gexp_kx), dkxVec) + np.exp(-Nph)))
print("\int p np dp = %f" % (np.dot(np.abs(Gexp_kx), kx * dkxVec)))
print("\int k beta^2 dk = %f" % (np.dot(np.abs(beta_kx), kxfft * dkxVec)))
print("Exp[-Nph] = %f" % (np.exp(-Nph)))

fig, ax = plt.subplots(nrows=1, ncols=4)

ax[0].plot(kx, np.abs(beta_kx))
ax[1].plot(x, np.real(beta_x))
ax[2].plot(x, np.abs(fexp_x))
ax[3].plot(kx, np.real(Gexp_kx))


fig.tight_layout()
plt.show()
