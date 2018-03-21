import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import os
import itertools
import Grid
import pf_dynamic_sph as pfs
import pf_static_sph

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

    names = list(kgrid.arrays.keys())  # ***need to have arrays added as k, th when kgrid is created
    if names[0] != 'k':
        print('CREATED kgrid IN WRONG ORDER')
    functions_wk = [lambda k: pfs.omegak(k, mB, n0, gBB), lambda th: 0 * th + 1]
    wk = kgrid.function_prod(names, functions_wk)

    # datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    # datapath = '/n/regal/demler_lab/kis/ZwierleinExp_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)

    # innerdatapath = datapath + '/imdyn_spherical'
    innerdatapath = datapath + '/redyn_nonint'
    outputdatapath = datapath + '/mm'

    def dirRF(dataset, kgrid):
        CSAmp = dataset['Real_CSAmp'] + 1j * dataset['Imag_CSAmp']
        Phase = dataset['Phase']
        dVk = kgrid.dV()
        tgrid = CSAmp.coords['t'].values
        CSA0 = CSAmp.isel(t=0).values; CSA0 = CSA0.reshape(CSA0.size)
        Phase0 = Phase.isel(t=0).values
        DynOv_Vec = np.zeros(tgrid.size, dtype=complex)
        for tind, t in enumerate(tgrid):
            CSAt = CSAmp.sel(t=t).values; CSAt = CSAt.reshape(CSAt.size)
            Phaset = Phase.sel(t=t).values
            exparg = np.dot(np.abs(CSAt)**2 + np.abs(CSA0)**2 - 2 * CSA0.conjugate() * CSAt, dVk)
            DynOv_Vec[tind] = np.exp(-1j * (Phaset - Phase0)) * np.exp((-1 / 2) * exparg)

        ReDynOv_da = xr.DataArray(np.real(DynOv_Vec), coords=[tgrid], dims=['t'])
        ImDynOv_da = xr.DataArray(np.imag(DynOv_Vec), coords=[tgrid], dims=['t'])
        dirRF_ds = xr.Dataset({'Real_DynOv': ReDynOv_da, 'Imag_DynOv': ImDynOv_da}, coords={'t': tgrid}, attrs=dataset.attrs)
        return dirRF_ds

    # # # Individual Datasets

    # Nsteps = 1e2
    # # pf_static_sph.createSpline_grid(Nsteps, kgrid, mI, mB, n0, gBB)
    # aSi_tck = np.load('aSi_spline_sph.npy')
    # PBint_tck = np.load('PBint_spline_sph.npy')

    # dVk = kgrid.dV()
    # kzg_flat = pfs.kcos_func(kgrid)

    # for ind, filename in enumerate(os.listdir(innerdatapath)):
    #     if filename == 'quench_Dataset_sph.nc':
    #         continue
    #     with xr.open_dataset(innerdatapath + '/' + filename) as ds:
    #         # ds = xr.open_dataset(innerdatapath + '/' + filename)
    #         aIBi = ds.attrs['aIBi']
    #         P = ds.attrs['P']
    #         tgrid = ds.coords['t'].values
    #         dirRF_ds = dirRF(ds, kgrid)

    #     # calculate energy explictly from imdyn polaron state
    #     gIB = pfs.g(kgrid, aIBi, mI, mB, n0, gBB)
    #     Amp_ds = xr.open_dataset(datapath + '/imdyn_spherical/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))
    #     CSAmp = (Amp_ds['Real_CSAmp'] + 1j * Amp_ds['Imag_CSAmp']).values; CSAmp = CSAmp.reshape(CSAmp.size)
    #     PB_id = np.dot(kzg_flat * np.abs(CSAmp)**2, dVk).real.astype(float)
    #     DP_id = P - PB_id
    #     Energy_id = (P**2 - PB_id**2) / (2 * mI) + np.dot(pfs.Omega(kgrid, DP_id, mI, mB, n0, gBB) * np.abs(CSAmp)**2, dVk) + gIB * (np.dot(pfs.Wk(kgrid, mB, n0, gBB) * CSAmp, dVk) + np.sqrt(n0))**2
    #     Energy_id = Energy_id.real.astype(float)

    #     # calculate energy from steady state formula
    #     DP = pf_static_sph.DP_interp(0, P, aIBi, aSi_tck, PBint_tck)
    #     aSi = pf_static_sph.aSi_interp(DP, aSi_tck)
    #     PB_Val = pf_static_sph.PB_interp(DP, aIBi, aSi_tck, PBint_tck)
    #     Energy = pf_static_sph.Energy(P, PB_Val, aIBi, aSi, mI, mB, n0)
    #     eMass = pf_static_sph.effMass(P, PB_Val, mI)

    #     # print(np.abs((Energy_id - Energy) / Energy))

    #     aIBiVec = aIBi * np.ones(tgrid.size)
    #     PVec = P * np.ones(tgrid.size)
    #     EVec = Energy_id * np.ones(tgrid.size)
    #     data = np.concatenate((PVec[:, np.newaxis], aIBiVec[:, np.newaxis], EVec[:, np.newaxis], tgrid[:, np.newaxis], dirRF_ds['Real_DynOv'].values[:, np.newaxis], dirRF_ds['Imag_DynOv'].values[:, np.newaxis]), axis=1)
    #     np.savetxt(outputdatapath + '/quench_P_{:.3f}_aIBi_{:.2f}.dat'.format(P, aIBi), data)

    #     # St = np.exp(1j * Energy_id) * (dirRF_ds['Real_DynOv'] + 1j * dirRF_ds['Imag_DynOv'])
    #     # fig, ax = plt.subplots()
    #     # ax.plot(tgrid, np.abs(St.values), 'k-')
    #     # plt.show()

    for ind, filename in enumerate(os.listdir(outputdatapath)):
        PVec, aIBiVec, EVec, tGrid, ReSt, ImSt = np.loadtxt(outputdatapath + '/' + filename, unpack=True)
        P = PVec[0]; aIBi = aIBiVec[0]; Energy = EVec[0]
        print(filename, P, aIBi)
        St = np.exp(1j * Energy) * (ReSt + 1j * ImSt)
        fig, ax = plt.subplots()
        ax.plot(tGrid, np.abs(St))
        plt.show()
