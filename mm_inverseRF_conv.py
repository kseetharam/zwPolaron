import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import os
import itertools
import Grid


if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    # gParams

    (Lx, Ly, Lz) = (20, 20, 20)
    (dx, dy, dz) = (0.2, 0.2, 0.2)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    xgrid = Grid.Grid('CARTESIAN_3D')
    xgrid.initArray('x', -Lx, Lx, dx); xgrid.initArray('y', -Ly, Ly, dy); xgrid.initArray('z', -Lz, Lz, dz)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    NGridPoints_desired = (1 + 2 * Lx / dx) * (1 + 2 * Lz / dz)
    Ntheta = 50
    Nk = np.ceil(NGridPoints_desired / Ntheta)

    theta_max = np.pi
    thetaArray, dtheta = np.linspace(0, theta_max, Ntheta, retstep=True)

    # k_max = np.sqrt((np.pi / dx)**2 + (np.pi / dy)**2 + (np.pi / dz)**2)
    k_max = ((2 * np.pi / dx)**3 / (4 * np.pi / 3))**(1 / 3)

    k_min = 1e-5
    kArray, dk = np.linspace(k_min, k_max, Nk, retstep=True)
    if dk < k_min:
        print('k ARRAY GENERATION ERROR')

    kgrid = Grid.Grid("SPHERICAL_2D")
    kgrid.initArray_premade('k', kArray)
    kgrid.initArray_premade('th', thetaArray)
    dVk = kgrid.dV()

    # Basic parameters

    mI = 1.7
    mB = 1
    n0 = 1
    aBB = 0.075
    gBB = (4 * np.pi / mB) * aBB

    # datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    # datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    datapath = '/n/regal/demler_lab/kis/ZwierleinExp_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)

    innerdatapath = datapath + '/redyn_spherical'
    outputdatapath = datapath + '/mm_invRF'

    # # Individual Datasets

    for ind, filename in enumerate(os.listdir(innerdatapath)):
        if filename == 'quench_Dataset_sph.nc':
            continue
        with xr.open_dataset(innerdatapath + '/' + filename) as ds:
            # ds = xr.open_dataset(innerdatapath + '/' + filename)
            aIBi = ds.attrs['aIBi']
            P = ds.attrs['P']
            tgrid = ds.coords['t'].values
            aIBiVec = aIBi * np.ones(tgrid.size)
            PVec = P * np.ones(tgrid.size)
            data = np.concatenate((PVec[:, np.newaxis], aIBiVec[:, np.newaxis], (PVec**2 / (2 * mI))[:, np.newaxis], tgrid[:, np.newaxis], ds['Real_DynOv'].values[:, np.newaxis], ds['Imag_DynOv'].values[:, np.newaxis]), axis=1)
            np.savetxt(outputdatapath + '/quench_P_{:.3f}_aIBi_{:.2f}.dat'.format(P, aIBi), data)

    # for ind, filename in enumerate(os.listdir(outputdatapath)):
    #     PVec, aIBiVec, P2mVec, tGrid, ReSt, ImSt = np.loadtxt(outputdatapath + '/' + filename, unpack=True)
    #     P = PVec[0]; aIBi = aIBiVec[0]; P2m = P2mVec[0]
    #     print(filename, P, aIBi)
    #     St = np.exp(1j * P2m) * (ReSt + 1j * ImSt)
    #     fig, ax = plt.subplots()
    #     ax.plot(tGrid, np.imag(St))
    #     plt.show()
