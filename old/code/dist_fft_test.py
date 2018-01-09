import numpy as np
import Grid
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import polaron_functions_cart as pfc


# Grid

# xGrid = Grid.Grid("CARTESIAN_3D")
# L = 5
# dx = 0.5
# xGrid.initArray('x', -L, L, dx)
# xGrid.initArray('y', -L, L, dx)
# xGrid.initArray('z', -L, L, dx)

# kGrid = Grid.Grid("CARTESIAN_3D")
# dk = np.pi / L
# Lk = np.pi / dx
# kGrid.initArray('kx', -Lk, Lk, dk)
# kGrid.initArray('ky', -Lk, Lk, dk)
# kGrid.initArray('kz', -Lk, Lk, dk)


# Create grids
Lx = 10; dx = 1e-01; x = np.arange(-Lx, Lx + dx, dx)
Ly = 10; dy = 1e-01; y = np.arange(-Ly, Ly + dy, dy)
Lz = 10; dz = 1e-01; z = np.arange(-Lz, Lz + dz, dz)

Lkx = np.pi / dx; dkx = np.pi / Lx; kx = np.arange(-Lkx, Lkx + dkx, dkx)
Lky = np.pi / dy; dky = np.pi / Ly; ky = np.arange(-Lky, Lky + dky, dky)
Lkz = np.pi / dz; dkz = np.pi / Lz; kz = np.arange(-Lkz, Lkz + dkz, dkz)

xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
kxv, kyv, kzv = np.meshgrid(kx, ky, kz, indexing='ij')

# FFT prep
Nx = len(x); Ny = len(y); Nz = len(z)
Nyx = Nx // 2 + 1; Nyy = Ny // 2 + 1; Nzz = Nz // 2 + 1

kny_x = (2 * np.pi / dx) / 2; kxfft = np.linspace(-kny_x, kny_x, Nx)
kny_y = (2 * np.pi / dy) / 2; kyfft = np.linspace(-kny_y, kny_y, Ny)
kny_z = (2 * np.pi / dz) / 2; kzfft = np.linspace(-kny_z, kny_z, Nz)
norm = dx * dy * dz
knorm = dkx * dky * dkz

# Calculate analytic function and its FT
# mx = 0; varx = 1
# Gx = 1 / np.sqrt(2 * np.pi * varx**2) * np.exp(-(x - mx)**2 / (2 * varx**2))

# my = 0; vary = 1
# Gy = 1 / np.sqrt(2 * np.pi * vary**2) * np.exp(-(y - my)**2 / (2 * vary**2))

# mz = 0; varz = 1
# Gz = 1 / np.sqrt(2 * np.pi * varz**2) * np.exp(-(z - mz)**2 / (2 * varz**2))

# # Gxyz = np.outer(Gx, Gy, Gz)

# Gkx = np.exp(-1j * mx * kx) * np.exp(-(kx**2 * varx**2) / 2)
# Gky = np.exp(-1j * my * ky) * np.exp(-(ky**2 * vary**2) / 2)
# Gkz = np.exp(-1j * mx * kz) * np.exp(-(kz**2 * varz**2) / 2)

# # Gkxkykz = np.outer(Gkx, Gky, Gkz)


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
    DP = 0.75
    return omegak(kx, ky, kz) + epsilon(kx, ky, kz) / mI - kx * DP / mI


def Bk(kx, ky, kz):
    aIBi = -1
    aSi = 0
    return np.sqrt(epsilon(kx, ky, kz) / omegak(kx, ky, kz)) / (Omega(kx, ky, kz) * (aIBi - aSi))

# generation


Gxyz = np.zeros((Nx, Ny, Nz)).astype('complex')
Gkxkykz = np.zeros((Nx, Ny, Nz)).astype('complex')
for indx in np.arange(Nx):
    for indy in np.arange(Ny):
        for indz in np.arange(Nz):
            kxF = kxfft[indx]; kyF = kyfft[indy]; kzF = kzfft[indz]
            if(kxF == 0 and kyF == 0 and kzF == 0):
                Gkxkykz[indx, indy, indz] = 0
                # Gkxkykz[indx, indy, indz] = Bk(kxF, kyF, kzF)
            else:
                decay = np.exp(-1*epsilon(kxF,kyF,kzF))
                Gkxkykz[indx, indy, indz] = decay*np.abs(Bk(kxF, kyF, kzF))**2

#

# Calculate FFT, post-process FFT (switching freqs), and iFFT
F = np.fft.fftn(Gkxkykz)
F1 = np.concatenate((F[Nyx:, :, :], F[0:Nyx, :, :]), axis=0)
F2 = np.concatenate((F1[:, Nyy:, :], F1[:, 0:Nyy, :]), axis=1)
F3 = np.concatenate((F2[:, :, Nzz:], F2[:, :, 0:Nzz]), axis=2)
Gxyz_fft = knorm * F3
Gxyz_ifft = knorm*np.fft.ifftn(Gkxkykz)
# kxv_fft, kyv_fft, kzv_fft = np.meshgrid(kxfft, kyfft, kzfft, indexing='ij')

# Gkxkykz_ifft = (1 / knorm) * np.fft.ifftn(Gxyz_fft)
dVk = knorm*np.ones((Nx, Ny, Nz)).astype('complex')
Nph = np.sum(Gkxkykz*dVk)

fexp = np.exp(Gxyz_fft)*np.exp(-1*Nph)
Gexp = (1 / knorm) * np.fft.ifftn(fexp)

# plotting
# Gx = np.zeros(Nx).astype('complex')
Gkx = np.zeros(Nx).astype('complex')
# Gkx_ifft = np.zeros(Nx).astype('complex')
fexp_x = np.zeros(Nx).astype('complex')
Gexp_kx = np.zeros(Nx).astype('complex')

Gft_x = np.zeros(Nx).astype('complex')
Gift_x = np.zeros(Nx).astype('complex')
for indx in np.arange(Nx):
    for indy in np.arange(Ny):
        for indz in np.arange(Nz):
            # Gx[indx] = Gx[indx] + Gxyz_fft[indx, indy, indz] * dy * dz
            Gkx[indx] = Gkx[indx] + np.abs(Gkxkykz[indx, indy, indz]) * dky * dkz
            # Gkx_ifft[indx] = Gkx_ifft[indx] + np.abs(Gkxkykz_ifft[indx, indy, indz]) * dky * dkz
            fexp_x[indx] = fexp_x[indx] + np.abs(fexp[indx, indy, indz]) * dy * dz
            Gexp_kx[indx] = Gexp_kx[indx] + np.abs(Gexp[indx, indy, indz]) * dky * dkz

            Gft_x[indx] = Gft_x[indx] + np.abs(Gxyz_fft[indx, indy, indz]) * dy * dz
            Gift_x[indx] = Gift_x[indx] + np.abs(Gxyz_ifft[indx, indy, indz]) * dy * dz


# for indx in np.arange(len(kxfft)):
#     GA = Gkxkykz_fft[indx, :, :]
#     dkyVec = dky * np.ones(len(kyfft))
#     dkzVec = dkz * np.ones(len(kzfft))
#     dA = np.outer(dkyVec, dkzVec)
#     Gkx[indx] = np.sum(GA * dA)

dxVec = dx * np.ones(Nx)
dkxVec = dkx * np.ones(Nx)
# print(np.dot(np.abs(Gx)**2, dxVec))
print(Nph, np.dot(np.abs(Gkx), dkxVec))
# print(np.dot(np.abs(Gkx_ifft)**2, dkxVec))

fig, ax = plt.subplots(nrows=1, ncols=4)
# ax[0].plot(x, np.abs(Gx))
ax[0].plot(kx, np.abs(Gkx))
# plt.plot(kxfft, np.abs(Gkx_fft))
# ax[1].plot(kx, np.abs(Gkx))
# ax[1].plot(kx, np.abs(Gkx_ifft))
ax[1].plot(x,fexp_x)
ax[2].plot(kx, np.abs(Gexp_kx))
ax[3].plot(x,np.abs(Gft_x))
ax[3].plot(x,np.abs(Gift_x))
fig.tight_layout()
plt.show()
