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

    def xinterp2D(xdataset, coord1, coord2, mult):
        # xdataset is the desired xarray dataset with the desired plotting quantity already selected
        # coord1 and coord2 are the two coordinates making the 2d plot
        # mul is the multiplicative factor by which you wish to increase the resolution of the grid
        # e.g. xdataset = qds['nPI_xz_slice'].sel(P=P,aIBi=aIBi).dropna('PI_z'), coord1 = 'PI_x', coord2 = 'PI_z'
        # returns meshgrid values for C1_interp and C2_interp as well as the function value on this 2D grid -> these are ready to plot

        C1 = xdataset.coords[coord1].values
        C2 = xdataset.coords[coord2].values
        C1g, C2g = np.meshgrid(C1, C2, indexing='ij')

        C1_interp = np.linspace(np.min(C1), np.max(C1), mult * C1.size)
        C2_interp = np.linspace(np.min(C2), np.max(C2), mult * C2.size)
        C1g_interp, C2g_interp = np.meshgrid(C1_interp, C2_interp, indexing='ij')

        interp_vals = griddata((C1g.flatten(), C2g.flatten()), xdataset.values.flatten(), (C1g_interp, C2g_interp), method='cubic')

        return interp_vals, C1g_interp, C2g_interp

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
    aIBi = -5
    PIm = qds.coords['PI_mag'].values

    # qds['nPI_mag'].sel(P=P, aIBi=aIBi).dropna('PI_mag').plot(ax=axes, label='')
    # axes.plot(P * np.ones(PIm.size), np.linspace(0, qds['mom_deltapeak'].sel(P=P, aIBi=-10).values, PIm.size), 'g--', label=r'$\delta$-peak')
    # axes.plot(nu * np.ones(len(PIm)), np.linspace(0, 1, len(PIm)), 'k:', label=r'$m_{I}\nu$')

    # axes.set_ylim([0, 1])
    # axes.set_title('$P=${:.2f}'.format(P))
    # axes.set_xlabel(r'$|P_{I}|$')
    # axes.set_ylabel(r'$n_{|P_{I}|}$')
    # axes.legend()
    # plt.show()

    qd_slice = qds['nPI_xz_slice'].sel(P=P, aIBi=aIBi).dropna('PI_z')
    slice_interp, PI_xg_interp, PI_zg_interp = xinterp2D(qd_slice, 'PI_x', 'PI_z', 8)
    axes.pcolormesh(PI_zg_interp, PI_xg_interp, slice_interp)
    # axes.pcolormesh(PI_zg, PI_xg, qd_slice.values)
    # qds['nPI_xz_slice'].sel(P=P, aIBi=aIBi).dropna('PI_z').plot(ax=axes)

    axes.set_title('Impurity Longitudinal Momentum Distribution ' + r'($a_{IB}^{-1}=$' + '{:.2f}'.format(aIBi) + '$P=${:.2f})'.format(P))
    axes.set_ylabel(r'$P_{I,x}$')
    axes.set_xlabel(r'$P_{I,z}$')
    axes.set_xlim([-2, 2])
    axes.set_ylim([-2, 2])
    axes.grid(True, linewidth=0.5)
    plt.show()
