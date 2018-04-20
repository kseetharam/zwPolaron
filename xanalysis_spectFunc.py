import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import pf_dynamic_sph as pfs
import Grid
import itertools
import os

if __name__ == "__main__":

    # # Initialization

    matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    # gParams

    (Lx, Ly, Lz) = (20, 20, 20)
    (dx, dy, dz) = (0.2, 0.2, 0.2)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    aBB = 0.011
    # aBB = 0.062

    expParams = pfs.Zw_expParams()
    L_exp2th, M_exp2th, T_exp2th = pfs.unitConv_exp2th(expParams['n0_BEC'], expParams['mB'])

    # Toggle parameters

    toggleDict = {'Location': 'home', 'RF': 'inverse'}

    # kgrid

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

    # ---- SET OUTPUT DATA FOLDER ----

    if toggleDict['Location'] == 'home':
        datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}'.format(aBB, NGridPoints_cart)
    elif toggleDict['Location'] == 'work':
        datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}'.format(aBB, NGridPoints_cart)
    elif toggleDict['Location'] == 'cluster':
        datapath = '/n/regal/demler_lab/kis/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}'.format(aBB, NGridPoints_cart)

    # if toggleDict['RF'] == 'direct':
    #     innerdatapath = datapath + '/dirRF'
    # elif toggleDict['RF'] == 'inverse':
    #     innerdatapath = datapath + '/invRF'

    if toggleDict['RF'] == 'direct':
        innerdatapath = datapath + '/redyn_spherical_nonint'
        plotTitle = 'Direct RF Spectral Function'
        prefac = 1
    elif toggleDict['RF'] == 'inverse':
        innerdatapath = datapath + '/redyn_spherical'
        plotTitle = 'Inverse RF Spectral Function'
        prefac = -1

    ds_name = toggleDict['RF'] + 'RF_Dataset_sph.nc'
    ds_path = innerdatapath + '/' + ds_name

    # # # Concatenate Individual Datasets

    # ds_list = []; P_list = []; aIBi_list = []; mI_list = []
    # for ind, filename in enumerate(os.listdir(innerdatapath)):
    #     if filename == ds_name:
    #         continue
    #     print(filename)

    #     ds = xr.open_dataset(innerdatapath + '/' + filename)
    #     ds_list.append(ds)
    #     P_list.append(ds.attrs['P'])
    #     aIBi_list.append(ds.attrs['aIBi'])
    #     mI_list.append(ds.attrs['mI'])

    # s = sorted(zip(aIBi_list, P_list, ds_list))
    # g = itertools.groupby(s, key=lambda x: x[0])

    # aIBi_keys = []; aIBi_groups = []; aIBi_ds_list = []
    # for key, group in g:
    #     aIBi_keys.append(key)
    #     aIBi_groups.append(list(group))

    # for ind, group in enumerate(aIBi_groups):
    #     aIBi = aIBi_keys[ind]
    #     _, P_list_temp, ds_list_temp = zip(*group)
    #     ds_temp = xr.concat(ds_list_temp, pd.Index(P_list_temp, name='P'))
    #     aIBi_ds_list.append(ds_temp)

    # ds_tot = xr.concat(aIBi_ds_list, pd.Index(aIBi_keys, name='aIBi'))
    # del(ds_tot.attrs['P']); del(ds_tot.attrs['aIBi']); del(ds_tot.attrs['gIB'])
    # ds_tot.to_netcdf(ds_path)

    # # # Add Spectral Function

    # with xr.open_dataset(ds_path) as qds:
    #     if 'SpectFunc' in qds.data_vars:
    #         qds = qds.drop('SpectFunc')
    #         qds = qds.drop('omega')
    #     aIBi_Vals = qds.coords['aIBi'].values
    #     P_Vals = qds.coords['P'].values
    #     tgrid = qds.coords['t'].values
    #     spectFunc_da = xr.DataArray(np.full((aIBi_Vals.size, P_Vals.size, tgrid.size), np.nan, dtype=float), coords=[aIBi_Vals, P_Vals, np.arange(tgrid.size)], dims=['aIBi', 'P', 'omega'])
    #     tdecay = 3
    #     for Aind, aIBi in enumerate(aIBi_Vals):
    #         for Pind, P in enumerate(P_Vals):
    #             qPA = qds.sel(aIBi=aIBi, P=P)
    #             St = qPA['Real_DynOv'].values + 1j * qPA['Imag_DynOv'].values

    #             if toggleDict['RF'] == 'direct':  # why do we need this?
    #                 prefac = -1
    #             elif toggleDict['RF'] == 'inverse':
    #                 prefac = 1
    #             omega, sf = pfs.spectFunc(tgrid, St, tdecay)
    #             spectFunc_da.coords['omega'] = omega
    #             spectFunc_da.sel(aIBi=aIBi, P=P)[:] = prefac * sf

    #             spectFunc_da.sel(aIBi=aIBi, P=P).plot()
    #             plt.show()
    #     qds['SpectFunc'] = spectFunc_da
    #     RF_ds = qds.copy(deep=True)

    # RF_ds.to_netcdf(ds_path)

    # # Analysis of Total Dataset

    # qds = xr.open_dataset(ds_path)
    # aIBi = -1.17
    # qds['SpectFunc'].isel(P=1).sel(aIBi=aIBi).plot()
    # # qds['SpectFunc'].sel(aIBi=aIBi).plot()
    # # qds['SpectFunc'].sel(aIBi=aIBi).mean(dim='P').plot()
    # plt.show()

    def dirRF(dataset, kgrid, cParams):
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

        # # calculate polaron energy (energy of initial state CSA0)
        # [P, aIBi] = cParams
        # [mI, mB, n0, gBB] = sParams
        # dVk = kgrid.dV()
        # kzg_flat = kcos_func(kgrid)
        # gIB = g(kgrid, aIBi, mI, mB, n0, gBB)
        # PB0 = np.dot(kzg_flat * np.abs(CSA0)**2, dVk).real.astype(float)
        # DP0 = P - PB0
        # Energy0 = (P**2 - PB0**2) / (2 * mI) + np.dot(Omega(kgrid, DP0, mI, mB, n0, gBB) * np.abs(CSA0)**2, dVk) + gIB * (np.dot(Wk(kgrid, mB, n0, gBB) * CSA0, dVk) + np.sqrt(n0))**2

        # calculate full dynamical overlap
        # DynOv_Vec = np.exp(1j * Energy0) * DynOv_Vec
        ReDynOv_da = xr.DataArray(np.real(DynOv_Vec), coords=[tgrid], dims=['t'])
        ImDynOv_da = xr.DataArray(np.imag(DynOv_Vec), coords=[tgrid], dims=['t'])
        DynOv_ds = xr.Dataset({'Real_DynOv': ReDynOv_da, 'Imag_DynOv': ImDynOv_da}, coords={'t': tgrid}, attrs=dataset.attrs)
        # DynOv_ds = dataset[['Real_CSAmp', 'Imag_CSAmp', 'Phase']]; DynOv_ds['Real_DynOv'] = ReDynOv_da; DynOv_ds['Imag_DynOv'] = ImDynOv_da; DynOv_ds.attrs = dataset.attrs
        return DynOv_ds

    P_sub = 0.1
    P_ss = 1.073
    aIBi = -1.77
    qds_sub = xr.open_dataset(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P_sub, aIBi))
    qds_ss = xr.open_dataset(innerdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P_ss, aIBi))

    if toggleDict['RF'] == 'inverse':
        DynOv_da_sub = qds_sub['Real_DynOv'] + 1j * qds_sub['Imag_DynOv']
        DynOv_da_ss = qds_ss['Real_DynOv'] + 1j * qds_ss['Imag_DynOv']
    elif toggleDict['RF'] == 'direct':
        dirRF_ds_sub = dirRF(qds_sub, kgrid, [P_sub, aIBi])
        DynOv_da_sub = dirRF_ds_sub['Real_DynOv'] + 1j * dirRF_ds_sub['Imag_DynOv']
        dirRF_ds_ss = dirRF(qds_ss, kgrid, [P_ss, aIBi])
        DynOv_da_ss = dirRF_ds_ss['Real_DynOv'] + 1j * dirRF_ds_ss['Imag_DynOv']

    tdecay = 3
    omega_sub, sf_sub = pfs.spectFunc(DynOv_da_sub.coords['t'].values, DynOv_da_sub.values, tdecay)
    omega_ss, sf_ss = pfs.spectFunc(DynOv_da_ss.coords['t'].values, DynOv_da_ss.values, tdecay)

    fig, ax = plt.subplots()
    ax.plot(omega_sub * T_exp2th * (prefac * 1e-3 / (2 * np.pi)), sf_sub, 'b-', label='Subsonic ($P=0.1$)')
    ax.plot(omega_ss * T_exp2th * (prefac * 1e-3 / (2 * np.pi)), sf_ss, 'r-', label='Supersonic ($P=1.5m_{I}c$)')
    ax.legend()
    ax.set_xlabel('Frequency (kHz)')
    ax.set_title(plotTitle)
    plt.show()
