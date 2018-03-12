import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import itertools
import Grid

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
    # innerdatapath = datapath + '/redyn_spherical'
    # innerdatapath = datapath + '/imdyn_spherical'
    innerdatapath = datapath + '/imdyn_spherical_long'

    def CSAmp_dists(qds_ap):
        # takes in an xarray dataset selected for the specific P, aIBi you want to look at
        # outputs xarray datasets (dataarray) that represents the Beta_|k| distribution (probability of magnitude |k| momentum phonons in the coherent state) and its time-derivative

        qds_Bk = qds_ap['Real_CSAmp'] + 1j * qds_ap['Imag_CSAmp']
        qds_DBk = qds_ap['Real_Delta_CSAmp'] + 1j * qds_ap['Imag_Delta_CSAmp']
        qds_Bk2 = np.abs(qds_Bk)**2
        qds_DBk2 = (qds_DBk * qds_Bk.conjugate() + qds_DBk.conjugate() * qds_Bk)
        # tB = np.sqrt(qds_ap['Real_CSAmp']**2 + qds_ap['Imag_CSAmp']**2)
        # tB.sel(t=100).plot(ax=ax)

        k = qds_ap.coords['k'].values; th = qds_ap.coords['th'].values
        kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', k); kgrid.initArray_premade('th', th)
        tgrid = qds.coords['t'].values

        Bk2_k_da = xr.DataArray(np.full((tgrid.size, len(k)), np.nan, dtype=float), coords=[tgrid, k], dims=['t', 'k'])
        DBk2_k_da = xr.DataArray(np.full((tgrid.size, len(k)), np.nan, dtype=float), coords=[tgrid, k], dims=['t', 'k'])

        for ind, t in enumerate(tgrid):
            qBk2t = qds_Bk2.sel(t=t).values.real.astype(float)
            qDBk2t = qds_DBk2.sel(t=t).values.real.astype(float)
            Bk2_k_da.sel(t=t)[:] = kgrid.integrateFunc(qBk2t.reshape(qBk2t.size), 'th')
            DBk2_k_da.sel(t=t)[:] = kgrid.integrateFunc(qDBk2t.reshape(qDBk2t.size), 'th')

        return Bk2_k_da, DBk2_k_da

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
    qds_Pimp = qds.coords['P'] - qds['PB']
    aIBi = -5
    P = 5.0
    fig, ax = plt.subplots()

    qds_ap = qds.sel(P=P, aIBi=aIBi)
    qds_St = np.sqrt(qds['Real_DynOv']**2 + qds['Imag_DynOv']**2)

    # Bk2_k_da, DBk2_k_da = CSAmp_dists(qds_ap)
    # DBk2_k_da.isel(t=-1).plot(ax=ax)

    nu = 0.792665459521

    # qds['Nph'].isel(t=-1).sel(aIBi=-2).plot(ax=ax)
    # qds['Nph'].isel(P=40).sel(aIBi=-2).isel(t=np.arange(-300, 0)).plot(ax=ax)
    # qds_St.isel(t=-1).sel(aIBi=-5).plot(ax=ax)
    # qds['Pph'].isel(P=22).sel(aIBi=-10).rolling(t=5).mean().isel(t=np.arange(-495, 0)).plot(ax=ax)

    # ax.plot(qds.coords['t'].values, np.abs(qds.attrs['mI'] * qds.attrs['nu'] * np.ones(qds.coords['t'].values.size) - nu), 'k--', label=r'$P=m_{I}\nu$')
    # PindList = [4, 10, 14, 16, 22, 45, 90]
    # PindList = [0]

    # for Pind in PindList:
    #     dat = qds_Pimp.isel(P=Pind).sel(aIBi=aIBi).rolling(t=1).mean()
    #     datm = np.abs(dat - nu)
    #     datm.plot(ax=ax, label='P={:.2f}'.format(qds_Pimp.coords['P'].values[Pind]))

    # ax.set_xscale('log'); ax.set_yscale('log')
    # ax.legend()
    # ax.set_title('Impurity Momentum at Interaction aIBi={:.2f}'.format(aIBi))
    # ax.set_ylabel(r'$P_{imp}$')
    # ax.set_xlabel(r'$t$')
    # plt.show()

    qds['NB'].sel(P=P, aIBi=aIBi).plot(ax=ax)
    # qds_St.sel(P=P, aIBi=aIBi).plot(ax=ax)
    ax.set_xscale('log'); ax.set_yscale('log')

    plt.show()
