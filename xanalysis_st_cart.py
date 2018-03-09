import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import itertools
from scipy.interpolate import griddata

if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    # gParams

    (Lx, Ly, Lz) = (21, 21, 21)
    (dx, dy, dz) = (0.375, 0.375, 0.375)

    NGridPoints = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    # datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints)
    datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints)

    innerdatapath = datapath + '/steadystate_cart'

    # # # Concatenate Individual Datasets

    # ds_list = []; P_list = []; aIBi_list = []; mI_list = []
    # for ind, filename in enumerate(os.listdir(innerdatapath)):
    #     if filename == 'quench_Dataset_cart.nc':
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
    # ds_tot.to_netcdf(innerdatapath + '/quench_Dataset_cart.nc')

    # # Analysis of Total Dataset

    fig, axes = plt.subplots()
    qds = xr.open_dataset(innerdatapath + '/quench_Dataset_cart.nc')
    nu = 0.792665459521

    P = 0.8
    aIBi = -10
    PIm = qds.coords['PI_mag'].values

    qds['nPI_mag'].sel(P=P, aIBi=aIBi).dropna('PI_mag').plot(ax=axes, label='')
    axes.plot(P * np.ones(PIm.size), np.linspace(0, qds['mom_deltapeak'].sel(P=P, aIBi=-10).values, PIm.size), 'g--', label=r'$\delta$-peak')
    axes.plot(nu * np.ones(len(PIm)), np.linspace(0, 1, len(PIm)), 'k:', label=r'$m_{I}\nu$')

    axes.set_ylim([0, 1])
    axes.set_title('$P=${:.2f}'.format(P))
    axes.set_xlabel(r'$|P_{I}|$')
    axes.set_ylabel(r'$n_{|P_{I}|}$')
    axes.legend()
    plt.show()

    # qd_slice = qds['nPI_xz_slice'].sel(P=P, aIBi=aIBi).dropna('PI_z')
    # PI_x = qd_slice.coords['PI_x'].values
    # PI_z = qd_slice.coords['PI_z'].values
    # PI_zg, PI_xg = np.meshgrid(PI_z, PI_x)

    # PI_x_interp = np.linspace(np.min(PI_x), np.max(PI_x), 2 * PI_x.size)
    # PI_z_interp = np.linspace(np.min(PI_z), np.max(PI_z), 2 * PI_z.size)
    # PI_zg_interp, PI_xg_interp = np.meshgrid(PI_z_interp, PI_x_interp)

    # slice_interp = griddata((PI_z, PI_x), qd_slice.values, (PI_zg_interp, PI_xg_interp), method='nearest')
    # print(slice_interp.shape)
    # print(PI_z_interp.shape, PI_x_interp.shape)
    # print(qd_slice.values.shape)
    # # axes.pcolormesh(PI_zg, PI_xg, qd_slice.values)
    # axes.pcolormesh(PI_z_interp, PI_x_interp, slice_interp)

    # qds['nPI_xz_slice'].sel(P=P, aIBi=aIBi).dropna('PI_z').plot(ax=axes)

    # axes.set_title('Impurity Longitudinal Momentum Distribution ' + r'($a_{IB}^{-1}=$' + '{:.2f}'.format(aIBi) + '$P=${:.2f})'.format(P))
    # axes.set_ylabel(r'$P_{I,x}$')
    # axes.set_xlabel(r'$P_{I,z}$')
    # axes.set_xlim([-2, 2])
    # axes.set_ylim([-2, 2])
    # axes.grid(True, linewidth=0.5)
    # plt.show()
