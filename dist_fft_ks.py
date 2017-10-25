import numpy as np
import Grid
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import polaron_functions_cart as pfc


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


def Bk(kx, ky, kz, aIBi, aSi, DP, mI, mB):
    ur = mI * mB / (mI + mB)
    n = 1
    return -2 * np.pi * np.sqrt(n) * Wk(kx, ky, kz, mB) / (ur * Omega(kx, ky, kz, DP, mI, mB) * (aIBi - aSi))


# set parameters

aIBi = -5
aSi = 0
DP = 0
mI = 1
mB = 1

# Create grids
(Lx, Ly, Lz) = (10, 10, 10)
(dx, dy, dz) = (5e-01, 5e-01, 5e-01)
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

kxfft = np.fft.fftfreq(Nx) * 2 * np.pi / dx
kyfft = np.fft.fftfreq(Nx) * 2 * np.pi / dy
kzfft = np.fft.fftfreq(Nx) * 2 * np.pi / dz


# print(kx)
# print(kxfft)

# generation

# beta_xyz = np.zeros((Nx, Ny, Nz)).astype('complex')
beta_kxkykz = np.zeros((Nx, Ny, Nz)).astype('complex')
decay_kxkykz = np.zeros((Nx, Ny, Nz)).astype('complex')
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
                decay_momentum = 3.5
                decay_kxkykz[indx, indy, indz] = np.exp(-1 * epsilon(kxF, kyF, kzF, mB) / (2 * decay_momentum**2))
                beta_kxkykz[indx, indy, indz] = Bk(kxF, kyF, kzF, aIBi, aSi, DP, mI, mB)
                decay_length = 3.5
                decay_xyz[indx, indy, indz] = np.exp(-1 * (x_**2 + y_**2 + z_**2) / (2 * decay_length**2))


# Slice x,y=0,z=0

beta2_kxkykz = np.abs(beta_kxkykz * decay_kxkykz)**2

# Fourier transform and slice
beta_xyz_0 = np.fft.fftn(beta_kxkykz * decay_kxkykz)
beta_xyz = np.fft.fftshift(beta_xyz_0) * dkx * dky * dkz
nX = np.abs(beta_xyz * decay_xyz)**2

beta_x = beta_xyz[:, Ny // 2 + 1, Nz // 2 + 1]

# Fourier transform and slice
beta2_xyz_0 = np.fft.fftn(beta2_kxkykz)
beta2_xyz = np.fft.fftshift(beta2_xyz_0) * dkx * dky * dkz
beta2_x = beta2_xyz[:, Ny // 2 + 1, Nz // 2 + 1]

# Phonon number
dVk = np.ones((Nx, Ny, Nz)).astype('complex')
Nph = np.real(np.sum(beta2_kxkykz * dVk) * dkx * dky * dkz)
print("Nph = \sum b^2 = %f" % (Nph))

Nph_x = np.real(np.sum(nX) * dx * dy * dz) * (2 * np.pi)**(-3)
print("Nph_x = \sum n(x) = %f" % (Nph_x))

# Exponenciate, slice
fexp = np.exp(beta2_xyz - Nph) * decay_xyz
# fexp = np.exp(beta_xyz - Nph) - np.exp(-Nph)


fexp_x = fexp[:, Ny // 2 + 1, Nz // 2 + 1]

# Inverse Fourier transform
nPB_0 = np.fft.ifftn(fexp) * 1 / (dkx * dky * dkz)
nPB = np.fft.fftshift(nPB_0)

# delta = np.fft.ifftn(np.ones((Nx, Ny, Nz))) * 1 / (dkx * dky * dkz)
# delta_sum = np.sum(delta * dkx * dky * dkz)
# print(delta_sum)
# nPB_k0 = np.exp(-1*Nph) * np.abs(delta_sum)

beta2_kx = np.zeros(Nx).astype('complex')
nPB_kx = np.zeros(Nx).astype('complex')
nX_x = np.zeros(Nx).astype('complex')

for indx in np.arange(Nx):
    for indy in np.arange(Ny):
        for indz in np.arange(Nz):
            beta2_kx[indx] = beta2_kx[indx] + np.abs(beta2_kxkykz[indx, indy, indz]) * dky * dkz
            nPB_kx[indx] = nPB_kx[indx] + np.abs(nPB[indx, indy, indz]) * dky * dkz
            nX_x[indx] = nX_x[indx] + np.abs(nX[indx, indy, indz]) * dx * dy * (2 * np.pi)**(-2)

dxVec = dx * np.ones(Nx)
dkxVec = dkx * np.ones(Nx)

int_np = np.dot(np.abs(nPB_kx), dkxVec)
# int_np = np.dot(np.abs(nPB_kx), dkxVec) + nPB_k0
print("\int np dp = %f" % (int_np))

# int_nx = np.dot(np.abs(nX_x), dxVec * (2 * np.pi)**(-1))
# print("\int nx dx = %f" % (int_nx))


# moment test
kxVec = np.fft.fftshift(kxfft)
np_M1_kx = np.dot(np.abs(kxVec * nPB_kx), dkxVec)
beta2_M1_kx = np.dot(np.abs(kxVec * beta2_kx), dkxVec)

print("1st Moment (nPB, Beta^2) = (%f,%f)" % (np_M1_kx, beta2_M1_kx))

# plotting

fig, ax = plt.subplots(nrows=1, ncols=5)

ax[0].plot(kxVec, np.abs(beta2_kx))
ax[1].plot(x, np.real(beta2_x))
ax[2].plot(x, np.abs(fexp_x))
ax[3].plot(kxVec, np.real(nPB_kx))
ax[4].plot(x, np.real(nX_x) / Nph_x)
# ax[3].plot(np.zeros(Nx), np.linspace(0, Gexp_k0, Nx))


fig.tight_layout()
plt.show()
