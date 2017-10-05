import numpy as np
import Grid
import matplotlib.pyplot as plt
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


# 1D GAUSSIAN TEST

# fig, ax = plt.subplots()

# mx = 0
# vx = 1
# vk = 1 / vx

# L = 10
# dx = 0.01
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

# ax.plot(k, Gk)
# ax.plot(kfft, np.abs(Gkfft))
# ax.set_xlim([-10, 10])

# # inverse

# # Gxifft = (1 / norm) * np.fft.ifft(Gkfft)

# # ax.plot(x, Gx)
# # ax.plot(x, np.abs(Gxifft))

# plt.show()


# 2D GAUSSIAN TEST

var = 1

Lx = 10
dx = 1
x = np.arange(-Lx, Lx + dx, dx)
Gx = 1 / np.sqrt(2 * np.pi * var**2) * np.exp(-x**2 / (2 * var**2))

Ly = 10
dy = 1
y = np.arange(-Ly, Ly + dy, dy)
Gy = 1 / np.sqrt(2 * np.pi * var**2) * np.exp(-y**2 / (2 * var**2))

Lkx = np.pi / dx
dkx = np.pi / Lx
kx = np.arange(-Lkx, Lkx + dkx, dkx)
Gkx = np.exp(-(kx**2 * var**2) / 2)

Lky = np.pi / dy
dky = np.pi / Ly
ky = np.arange(-Lky, Lky + dky, dky)
Gky = np.exp(-(ky**2 * var**2) / 2)

xv, yv = np.meshgrid(x, y, indexing='ij')
Gxy = np.outer(Gx, Gy)
kxv, kyv = np.meshgrid(kx, ky, indexing='ij')
Gkxky = np.outer(Gkx, Gky)
Nx = len(x)
Ny = len(y)

F = np.fft.fft2(Gxy)
Nyx = Nx // 2 + 1
Nyy = Ny // 2 + 1

kny_x = (2 * np.pi / dx) / 2
kxfft = np.linspace(-kny_x, kny_x, Nx)
kny_y = (2 * np.pi / dy) / 2
kyfft = np.linspace(-kny_y, kny_y, Ny)
# norm_x = Lx / Nyx
# norm_y = Ly / Nyy
norm = dx * dy

F1 = np.concatenate((F[Nyx:, :], F[0:Nyx, :]), axis=0)
F2 = np.concatenate((F1[:, Nyy:], F1[:, 0:Nyy]), axis=1)
Gkxky_fft = norm * F2
kxv_fft, kyv_fft = np.meshgrid(kxfft, kyfft, indexing='ij')
# kxtest = np.outer(kxfft, np.ones(len(kxfft)))

# print(Gkxky - Gkxky_fft)

fig = plt.figure()


ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(kxv, kyv, Gkxky, rstride=4, cstride=4, linewidth=0)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(kxv_fft, kyv_fft, np.abs(Gkxky_fft), rstride=4, cstride=4, linewidth=0)

plt.show()
