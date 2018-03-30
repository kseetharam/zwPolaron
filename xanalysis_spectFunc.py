import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import pf_dynamic_sph as pfs
import os

if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    # gParams

    (Lx, Ly, Lz) = (20, 20, 20)
    (dx, dy, dz) = (0.2, 0.2, 0.2)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    # Toggle parameters

    toggleDict = {'Location': 'work', 'RF': 'inverse'}

    # ---- SET OUTPUT DATA FOLDER ----

    if toggleDict['Location'] == 'home':
        datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    elif toggleDict['Location'] == 'work':
        datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    elif toggleDict['Location'] == 'cluster':
        datapath = '/n/regal/demler_lab/kis/ZwierleinExp_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)

    if toggleDict['RF'] == 'direct':
        innerdatapath = datapath + '/dirRF'
    elif toggleDict['RF'] == 'inverse':
        innerdatapath = datapath + '/invRF'

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

    # # Add Spectral Function

    qds = xr.open_dataset(ds_path)
    aIBi_Vals = qds.coords['aIBi'].values
    P_Vals = qds.coords['P'].values
    tgrid = qds.coords['t'].values

    for Aind, aIBi in enumerate(aIBi_Vals):
        for Pind, P in enumerate(P_Vals):
            qPA = qds.sel(aIBi=aIBi, P=P)
            St = qPA['Real_DynOv'].values + 1j * qPA['Imag_DynOv'].values
            omega, sf = pfs.spectFunc(tgrid, St)
            fig, ax = plt.subplots()
            ax.plot(omega, sf)
            plt.show()

    # # # Analysis of Total Dataset

    # qds = xr.open_dataset(ds_path)
