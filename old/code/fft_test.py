import numpy as np
import Grid
from scipy.special import jv
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


xGrid = Grid.Grid("CARTESIAN_3D")
L = 5
dx = 0.5
xGrid.initArray('x', -L, L, dx)
xGrid.initArray('y', -L, L, dx)
xGrid.initArray('z', -L, L, dx)

kGrid = Grid.Grid("CARTESIAN_3D")
dk = np.pi / L
Lk = np.pi / dx
kGrid.initArray('kx', -Lk, Lk, dk)
kGrid.initArray('ky', -Lk, Lk, dk)
kGrid.initArray('kz', -Lk, Lk, dk)


def gauss1d(k, sigma, mean):
    return np.exp(-1 * (k - mean)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)


sigma = 1
Nex = 3.1
names = list(kGrid.arrays.keys())
kgauss = kGrid.function_prod(names, [lambda kx: gauss1d(kx, sigma, 0), lambda ky: gauss1d(ky, sigma, 0), lambda kz: gauss1d(kz, sigma, 0)])
Nkgauss = Nex * kgauss

kx_array = kGrid.getArray('kx')
ky_array = kGrid.getArray('ky')
kz_array = kGrid.getArray('kz')
fk_mat = Nkgauss.reshape((len(kx_array), len(ky_array), len(kz_array)))


# CHECK TO INTEGRATE FUNCTION kx*ky*kz
# names = list(kGrid.arrays.keys())
# kxyz = kGrid.function_prod(names, [lambda kx: kx, lambda ky: ky, lambda kz: kz])
# intk = np.dot(kxyz, kGrid.dV())
# print(intk)
# print((kmax**2 / (4 * np.pi))**3)


# def ft(dk, Nk, funcVec):
#     # spectral function (Fourier Transform of dynamical overlap)
#     # tstep = t_Vec[1] - t_Vec[0]
#     # N = t_Vec.size
#     # tdecay = 3
#     # decayFactor = np.exp(-1 * t_Vec / tdecay)
#     decayFactor = 1
#     ft = 2 * np.real(np.fft.ifftn(funcVec * decayFactor))
#     # omega = 2 * np.pi * np.fft.fftfreq(Nk, d=tstep)
#     # return omega, sf
#     return ft


ft = np.fft.ifftn(fk_mat)

x_array = xGrid.getArray('x')
y_array = xGrid.getArray('y')
z_array = xGrid.getArray('z')

# xnames = list(xGrid.arrays.keys())
# print(ft)


# SINE SIGNAL TEST

# t = np.linspace(0, 3, 1000)
# omega = (2 * np.pi) * 3
# A = 100
# sig = A * np.sin(omega * t)

# # plt.plot(t, sig)
# hann = np.hanning(len(sig))
# Y = np.fft.fft(hann * sig)

# N = int(len(Y) / 2 + 1)
# dt = t[1] - t[0]
# fny = (1 / dt) / 2
# spectrum_Vals = Y[0:N]
# f_Vals = np.linspace(0, fny, N)
# norm = 2 / N


# plt.plot(f_Vals, norm * np.abs(spectrum_Vals))


# # 1D GAUSSIAN TEST

# mx = 0
# vx = 1
# vk = 1 / vx

# L = 10
# dx = 1e-03
# x = np.arange(-L, L + dx, dx)
# Gx = 1 / np.sqrt(2 * np.pi * vx**2) * np.exp(-(x - mx)**2 / (2 * vx**2))

# N = len(x)
# Lk = np.pi / dx
# dk = np.pi / L
# k = np.arange(-Lk, Lk + dk, dk)
# Gk = np.exp(-1j * mx * k) * np.exp(-k**2 / (2 * vk**2))


# # hann = np.hanning(len(Gx))
# hann = 1
# Y = np.fft.fft(hann * Gx)
# Ny = N // 2 + 1  # add the 1 just for indexing - the Nyquist frequency occurs at bin len(Y)/2
# kny = (2 * np.pi / dx) / 2
# kfft = np.linspace(-kny, kny, N)
# # norm = L / Ny  # why this factor of L? should be Ny-1?
# norm = dx
# Gkfft = norm * np.concatenate((Y[Ny:], Y[0:Ny]), axis=0)
# Gxifft = (1 / norm) * np.fft.ifft(Gkfft)

# # FT magnitude

# fig, ax = plt.subplots(nrows=1, ncols=3)

# ax[0].plot(k, np.abs(Gk), label='Analytical')
# ax[0].plot(kfft, np.abs(Gkfft), label='FFT')
# ax[0].legend()
# ax[0].set_xlim([-10, 10])
# ax[0].set_title('FT (Momentum Space) Magnitude')

# # FT magnitude

# ax[1].plot(k, np.angle(Gk), label='Analytical')
# ax[1].plot(kfft, np.angle(Gkfft), label='FFT')
# ax[1].legend()
# ax[1].set_xlim([-10, 10])
# ax[1].set_title('FT (Momentum Space) Phase')

# # inverse FT magnitude

# ax[2].plot(x, Gx, label='Analytical')
# ax[2].plot(x, np.abs(Gxifft), label='FFT')
# ax[2].legend()
# ax[2].set_title('Inverse FT (Original/Real Space)')

# diff = np.abs(np.angle(Gk) - np.angle(Gkfft))
# for d in diff:
#     print(d)
# # print(np.allclose(np.abs(Gk), np.abs(Gkfft), atol=1e-03))
# # fig.tight_layout()
# plt.show()


# # 2D GAUSSIAN TEST

# var = 3

# Lx = 10
# dx = 1
# x = np.arange(-Lx, Lx + dx, dx)
# mx = 4
# Gx = 1 / np.sqrt(2 * np.pi * var**2) * np.exp(-(x - mx)**2 / (2 * var**2))

# Ly = 10
# dy = 1
# y = np.arange(-Ly, Ly + dy, dy)
# Gy = 1 / np.sqrt(2 * np.pi * var**2) * np.exp(-y**2 / (2 * var**2))

# Lkx = np.pi / dx
# dkx = np.pi / Lx
# kx = np.arange(-Lkx, Lkx + dkx, dkx)
# Gkx = np.exp(-1j * mx * kx) * np.exp(-(kx**2 * var**2) / 2)

# Lky = np.pi / dy
# dky = np.pi / Ly
# ky = np.arange(-Lky, Lky + dky, dky)
# Gky = np.exp(-(ky**2 * var**2) / 2)

# xv, yv = np.meshgrid(x, y, indexing='ij')
# Gxy = np.outer(Gx, Gy)
# kxv, kyv = np.meshgrid(kx, ky, indexing='ij')
# Gkxky = np.outer(Gkx, Gky)
# Nx = len(x)
# Ny = len(y)

# F = np.fft.fftn(Gxy)
# Nyx = Nx // 2 + 1
# Nyy = Ny // 2 + 1

# kny_x = (2 * np.pi / dx) / 2
# kxfft = np.linspace(-kny_x, kny_x, Nx)
# kny_y = (2 * np.pi / dy) / 2
# kyfft = np.linspace(-kny_y, kny_y, Ny)
# # norm_x = Lx / Nyx
# # norm_y = Ly / Nyy
# norm = dx * dy

# F1 = np.concatenate((F[Nyx:, :], F[0:Nyx, :]), axis=0)
# F2 = np.concatenate((F1[:, Nyy:], F1[:, 0:Nyy]), axis=1)
# Gkxky_fft = norm * F2
# kxv_fft, kyv_fft = np.meshgrid(kxfft, kyfft, indexing='ij')
# # kxtest = np.outer(kxfft, np.ones(len(kxfft)))

# Gxy_ifft = (1 / norm) * np.fft.ifftn(Gkxky_fft)
# # print(Gkxky - Gkxky_fft)

# fig = plt.figure()


# ax = fig.add_subplot(2, 2, 1, projection='3d')
# ax.plot_surface(kxv, kyv, np.abs(Gkxky), rstride=2, cstride=2, linewidth=0)

# ax = fig.add_subplot(2, 2, 2, projection='3d')
# ax.plot_surface(kxv_fft, kyv_fft, np.abs(Gkxky_fft), rstride=2, cstride=2, linewidth=0)

# ax = fig.add_subplot(2, 2, 3, projection='3d')
# ax.plot_surface(xv, yv, np.abs(Gxy), rstride=2, cstride=2, linewidth=0)

# ax = fig.add_subplot(2, 2, 4, projection='3d')
# ax.plot_surface(xv, yv, np.abs(Gxy_ifft), rstride=2, cstride=2, linewidth=0)

# plt.show()


# # 2D NON-PRODUCT STATE TEST (CIRCLE - BESSEL)


# # Create grids
# Lx = 10
# dx = 1e-01
# x = np.arange(-Lx, Lx + dx, dx)

# Ly = 10
# dy = 1e-01
# y = np.arange(-Ly, Ly + dy, dy)

# Lkx = np.pi / dx
# dkx = np.pi / Lx
# kx = np.arange(-Lkx, Lkx + dkx, dkx)

# Lky = np.pi / dy
# dky = np.pi / Ly
# ky = np.arange(-Lky, Lky + dky, dky)

# xv, yv = np.meshgrid(x, y, indexing='ij')
# kxv, kyv = np.meshgrid(kx, ky, indexing='ij')

# # FFT prep
# Nx = len(x)
# Ny = len(y)
# Nyx = Nx // 2 + 1
# Nyy = Ny // 2 + 1

# kny_x = (2 * np.pi / dx) / 2
# kxfft = np.linspace(-kny_x, kny_x, Nx)
# kny_y = (2 * np.pi / dy) / 2
# kyfft = np.linspace(-kny_y, kny_y, Ny)
# norm = dx * dy

# # Calculate analytic function and its FT
# a = 5
# mask = np.sqrt(xv**2 + yv**2) < a
# Gxy = np.zeros((len(x), len(y)))
# Gxy[mask] = 1

# rho = np.sqrt(kxv**2 + kyv**2)
# Gkxky = a * jv(2 * np.pi * a * rho, 1) / rho

# # Calculate FFT, post-process FFT (switching freqs), and iFFT
# F = np.fft.fftn(Gxy)
# F1 = np.concatenate((F[Nyx:, :], F[0:Nyx, :]), axis=0)
# F2 = np.concatenate((F1[:, Nyy:], F1[:, 0:Nyy]), axis=1)
# Gkxky_fft = norm * F2
# kxv_fft, kyv_fft = np.meshgrid(kxfft, kyfft, indexing='ij')

# Gxy_ifft = (1 / norm) * np.fft.ifftn(Gkxky_fft)

# # plotting
# fig = plt.figure()

# ax = fig.add_subplot(2, 2, 1, projection='3d')
# ax.plot_surface(xv, yv, np.abs(Gxy), rstride=4, cstride=4, linewidth=0)
# ax.set_title('Original Function (Real Space)')

# ax = fig.add_subplot(2, 2, 2, projection='3d')
# ax.plot_surface(xv, yv, np.abs(Gxy_ifft), rstride=4, cstride=4, linewidth=0)
# ax.set_title('Inverse FFT (Real Space)')

# ax = fig.add_subplot(2, 2, 3, projection='3d')
# ax.plot_surface(kxv, kyv, np.abs(Gkxky), rstride=2, cstride=2, linewidth=0)
# ax.set_title('Analytical FT (Momentum Space)')
# ax.set_xlim([-3, 3])
# ax.set_ylim([-3, 3])

# ax = fig.add_subplot(2, 2, 4, projection='3d')
# ax.plot_surface(kxv_fft, kyv_fft, np.abs(Gkxky_fft), rstride=2, cstride=2, linewidth=0)
# ax.set_title('FFT (Momentum Space)')
# ax.set_xlim([-3, 3])
# ax.set_ylim([-3, 3])

# fig.tight_layout()
# plt.show()

# # # contour plotting
# # fig = plt.figure()

# # plt.subplot(2, 2, 1)
# # plt.contourf(xv, yv, np.abs(Gxy))
# # plt.title('Original Function (Real Space)')

# # plt.subplot(2, 2, 2)
# # plt.contourf(xv, yv, np.abs(Gxy_ifft))
# # plt.title('Inverse FFT (Real Space)')

# # plt.subplot(2, 2, 3)
# # plt.contourf(kxv, kyv, np.abs(Gkxky))
# # plt.title('Analytical FT (Momentum Space)')

# # plt.subplot(2, 2, 4)
# # plt.contourf(kxv_fft, kyv_fft, np.abs(Gkxky_fft))
# # plt.title('FFT (Momentum Space)')

# # fig.tight_layout()
# # plt.show()


# # 3D TEST


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

# Calculate analytic function and its FT
mx = 0; varx = 1
Gx = 1 / np.sqrt(2 * np.pi * varx**2) * np.exp(-(x - mx)**2 / (2 * varx**2))

my = 0; vary = 1
Gy = 1 / np.sqrt(2 * np.pi * vary**2) * np.exp(-(y - my)**2 / (2 * vary**2))

mz = 0; varz = 1
Gz = 1 / np.sqrt(2 * np.pi * varz**2) * np.exp(-(z - mz)**2 / (2 * varz**2))

# Gxyz = np.outer(Gx, Gy, Gz)

Gkx = np.exp(-1j * mx * kx) * np.exp(-(kx**2 * varx**2) / 2)
Gky = np.exp(-1j * my * ky) * np.exp(-(ky**2 * vary**2) / 2)
Gkz = np.exp(-1j * mx * kz) * np.exp(-(kz**2 * varz**2) / 2)

# Gkxkykz = np.outer(Gkx, Gky, Gkz)


# generation

Gxyz = np.zeros((Nx, Ny, Nz)).astype('complex')
Gkxkykz = np.zeros((Nx, Ny, Nz)).astype('complex')
for indx in np.arange(Nx):
    for indy in np.arange(Ny):
        for indz in np.arange(Nz):
            Gxyz[indx, indy, indz] = Gx[indx] * Gy[indy] * Gz[indz]
            Gkxkykz[indx, indy, indz] = Gkx[indx] * Gky[indy] * Gkz[indz]

#

# Calculate FFT, post-process FFT (switching freqs), and iFFT
F = np.fft.fftn(Gxyz)
F1 = np.concatenate((F[Nyx:, :, :], F[0:Nyx, :, :]), axis=0)
F2 = np.concatenate((F1[:, Nyy:, :], F1[:, 0:Nyy, :]), axis=1)
F3 = np.concatenate((F2[:, :, Nzz:], F2[:, :, 0:Nzz]), axis=2)
Gkxkykz_fft = norm * F3
kxv_fft, kyv_fft, kzv_fft = np.meshgrid(kxfft, kyfft, kzfft, indexing='ij')

Gxyz_ifft = (1 / norm) * np.fft.ifftn(Gkxkykz_fft)

# plotting
Gx = np.zeros(Nx).astype('complex')
Gkx = np.zeros(Nx).astype('complex')
Gkx_fft = np.zeros(Nx).astype('complex')
Gx_ifft = np.zeros(Nx).astype('complex')
for indx in np.arange(Nx):
    for indy in np.arange(Ny):
        for indz in np.arange(Nz):
            Gx[indx] = Gx[indx] + Gxyz[indx, indy, indz] * dy * dz
            Gkx[indx] = Gkx[indx] + np.abs(Gkxkykz[indx, indy, indz]) * dky * dkz
            Gkx_fft[indx] = Gkx_fft[indx] + np.abs(Gkxkykz_fft[indx, indy, indz]) * dky * dkz
            Gx_ifft[indx] = Gx_ifft[indx] + np.abs(Gxyz_ifft[indx, indy, indz]) * dy * dz

# for indx in np.arange(len(kxfft)):
#     GA = Gkxkykz_fft[indx, :, :]
#     dkyVec = dky * np.ones(len(kyfft))
#     dkzVec = dkz * np.ones(len(kzfft))
#     dA = np.outer(dkyVec, dkzVec)
#     Gkx[indx] = np.sum(GA * dA)

# plt.plot(x, np.abs(Gx))
# plt.plot(kx, np.abs(Gkx))
# plt.plot(kxfft, np.abs(Gkx_fft))
plt.plot(x, np.abs(Gx_ifft))
plt.show()
