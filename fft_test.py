import numpy as np
import Grid


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


print(ft)
