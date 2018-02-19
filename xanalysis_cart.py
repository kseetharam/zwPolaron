import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import os
import itertools

if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    # # gParams

    # (Lx, Ly, Lz) = (21, 21, 21)
    # (dx, dy, dz) = (0.375, 0.375, 0.375)

    # # (Lx, Ly, Lz) = (20, 20, 20)
    # # (dx, dy, dz) = (0.5, 0.5, 0.5)

    # NGridPoints = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    # datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints)
    # # datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints)

    # ds_list = []; P_list = []; aIBi_list = []; mI_list = []
    # for ind, filename in enumerate(os.listdir(datapath)):
    #     ds = xr.open_dataset(datapath + '/' + filename)
    #     ds_list.append(ds)
    #     P_list.append(ds.attrs['P'])
    #     aIBi_list.append(ds.attrs['aIBi'])
    #     mI_list.append(ds.attrs['mI'])

    # s = sorted(zip(aIBi_list, P_list, ds_list))
    # g = itertools.groupby(s, key=lambda x: x[0])

    # aIBi_keys = []
    # aIBi_groups = []
    # for key, group in g:
    #     aIBi_keys.append(key)
    #     aIBi_groups.append(list(group))

    # aIBi_ds_list = []
    # for ind, group in enumerate(aIBi_groups):
    #     aIBi = aIBi_keys[ind]
    #     _, P_list_temp, ds_list_temp = zip(*group)
    #     ds_temp = xr.concat(ds_list_temp, pd.Index(P_list_temp, name='P'))
    #     aIBi_ds_list.append(ds_temp)

    # ds_tot = xr.concat(aIBi_ds_list, pd.Index(aIBi_keys, name='aIBi'))

    # test_datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/test.nc'
    # ds_tot.to_netcdf(test_datapath)

    test_datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/test.nc'
    # datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/test.nc'
    dst = xr.open_dataset(test_datapath)
    del(dst.attrs['P']); del(dst.attrs['aIBi']); del(dst.attrs['nu']); del(dst.attrs['gIB'])
    # print(dst)

    # # ds_ind = -1
    # ds = ds_list[ds_ind]; P = P_list[ds_ind]; aIBi = aIBi_list[ds_ind]; mI = mI_list[ds_ind]

    # print('P: {0}'.format(P))
    # print('aIBi: {0}'.format(aIBi))
    # print('nu: {0}'.format(ds_list[0].attrs['nu']))

    # S_mag = np.sqrt(ds['Real_DynOv']**2 + ds['Imag_DynOv']**2)

    # ds['NB'].plot()
    # S_mag.plot()
    # ds['nPI_mag'].isel(t=-1).plot()
    dst['nPI_xz_slice'].isel(P=-1, aIBi=-1, t=-1).plot()
    plt.show()
