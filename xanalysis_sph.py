import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import itertools

if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    # gParams

    (Lx, Ly, Lz) = (21, 21, 21)
    (dx, dy, dz) = (0.375, 0.375, 0.375)

    # (Lx, Ly, Lz) = (21, 21, 21)
    # (dx, dy, dz) = (0.25, 0.25, 0.25)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    # datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    innerdatapath = datapath + '/spherical'
    figdatapath = datapath + '/figures'

    # # # Concatenate Individual Datasets

    # ds_list = []; P_list = []; aIBi_list = []; mI_list = []
    # for ind, filename in enumerate(os.listdir(innerdatapath)):
    #     if filename == 'quench_Dataset_sph.nc':
    #         continue
    #     print(filename)
    #     with xr.open_dataset(innerdatapath + '/' + filename) as dsf:
    #         ds = dsf.compute()
    #         ds_list.append(ds)
    #         P_list.append(ds.attrs['P'])
    #         aIBi_list.append(ds.attrs['aIBi'])
    #         mI_list.append(ds.attrs['mI'])

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
    # ds_tot.to_netcdf(innerdatapath + '/quench_Dataset_sph.nc')

    # # Analysis of Total Dataset

    qds = xr.open_dataset(innerdatapath + '/quench_Dataset_sph.nc')
    # print(qds.attrs)

    fig, ax = plt.subplots()
    qds_St = np.sqrt(qds['Real_DynOv']**2 + qds['Imag_DynOv']**2)
    qds_Pimp = qds.coords['P'] - qds['PB']

    # qds['NB'].isel(t=-1).sel(aIBi=-2).plot(ax=ax)
    # qds['NB'].isel(P=40).sel(aIBi=-2).isel(t=np.arange(-300, 0)).plot(ax=ax)
    # qds_St.isel(t=-1).sel(aIBi=-5).plot(ax=ax)
    # qds['PB'].isel(P=22).sel(aIBi=-10).rolling(t=5).mean().isel(t=np.arange(-495, 0)).plot(ax=ax)

    aIBi = -10
    ax.plot(qds.coords['t'].values, qds.attrs['mI'] * qds.attrs['nu'] * np.ones(qds.coords['t'].values.size), 'k--', label=r'$P=m_{I}\nu$')
    PindList = [4, 10, 14, 16, 22, 45, 90]
    for Pind in PindList:
        qds_Pimp.isel(P=Pind).sel(aIBi=aIBi).rolling(t=1).mean().plot(ax=ax, label='P={:.2f}'.format(qds_Pimp.coords['P'].values[Pind]))

    ax.set_xscale('linear'); ax.set_yscale('linear')
    ax.legend()
    ax.set_title('Impurity Momentum at Interaction aIBi={:.2f}'.format(aIBi))
    ax.set_ylabel(r'$P_{imp}$')
    ax.set_xlabel(r'$t$')
    plt.show()
