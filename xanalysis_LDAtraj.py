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

    (Lx, Ly, Lz) = (20, 20, 20)
    (dx, dy, dz) = (0.2, 0.2, 0.2)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    # Toggle parameters

    toggleDict = {'Location': 'work', 'Dynamics': 'real', 'Coupling': 'twophonon', 'Grid': 'spherical'}

    # ---- SET OUTPUT DATA FOLDER ----

    if toggleDict['Location'] == 'home':
        datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/NGridPoints_{:.2E}/LDA'.format(NGridPoints_cart)
    elif toggleDict['Location'] == 'work':
        datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/NGridPoints_{:.2E}/LDA'.format(NGridPoints_cart)
    elif toggleDict['Location'] == 'cluster':
        datapath = '/n/regal/demler_lab/kis/ZwierleinExp_data/NGridPoints_{:.2E}/LDA'.format(NGridPoints_cart)

    if toggleDict['Dynamics'] == 'real':
        innerdatapath = datapath + '/redyn'
    elif toggleDict['Dynamics'] == 'imaginary':
        innerdatapath = datapath + '/imdyn'

    if toggleDict['Grid'] == 'cartesian':
        innerdatapath = innerdatapath + '_cart'
    elif toggleDict['Grid'] == 'spherical':
        innerdatapath = innerdatapath + '_spherical'

    if toggleDict['Coupling'] == 'frohlich':
        innerdatapath = innerdatapath + '_froh'
    elif toggleDict['Coupling'] == 'twophonon':
        innerdatapath = innerdatapath

    # # # Concatenate Individual Datasets

    # ds_list = []; F_list = []; aIBi_list = []; mI_list = []
    # for ind, filename in enumerate(os.listdir(innerdatapath)):
    #     if filename == 'quench_Dataset_sph.nc':
    #         continue
    #     print(filename)
    #     # with xr.open_dataset(innerdatapath + '/' + filename) as dsf:
    #     #     ds = dsf.compute()
    #     #     ds_list.append(ds)
    #     #     F_list.append(ds.attrs['F'])
    #     #     aIBi_list.append(ds.attrs['aIBi'])
    #     #     mI_list.append(ds.attrs['mI'])

    #     ds = xr.open_dataset(innerdatapath + '/' + filename)
    #     # ds = ds.drop(['Real_Delta_CSAmp', 'Imag_Delta_CSAmp'])
    #     ds_list.append(ds)
    #     F_list.append(ds.attrs['F'])
    #     aIBi_list.append(ds.attrs['aIBi'])
    #     mI_list.append(ds.attrs['mI'])

    # s = sorted(zip(aIBi_list, F_list, ds_list))
    # g = itertools.groupby(s, key=lambda x: x[0])

    # aIBi_keys = []; aIBi_groups = []; aIBi_ds_list = []
    # for key, group in g:
    #     aIBi_keys.append(key)
    #     aIBi_groups.append(list(group))

    # for ind, group in enumerate(aIBi_groups):
    #     aIBi = aIBi_keys[ind]
    #     _, F_list_temp, ds_list_temp = zip(*group)
    #     ds_temp = xr.concat(ds_list_temp, pd.Index(F_list_temp, name='F'))
    #     aIBi_ds_list.append(ds_temp)

    # ds_tot = xr.concat(aIBi_ds_list, pd.Index(aIBi_keys, name='aIBi'))
    # del(ds_tot.attrs['F']); del(ds_tot.attrs['aIBi']); del(ds_tot.attrs['gIB'])
    # ds_tot.to_netcdf(innerdatapath + '/LDA_Dataset_sph.nc')

    # # # Analysis of Total Dataset

    # qds = xr.open_dataset(innerdatapath + '/LDA_Dataset_sph.nc')
    qds = xr.open_dataset(innerdatapath + '/F_0.057_aIBi_-1.17.nc')

    print(qds.attrs['TF'])

    fig, ax = plt.subplots()
    X_Vals = qds['X'].values
    t_Vals = qds['t'].values
    ax.plot(t_Vals, np.gradient(X_Vals))
    # qds['X'].plot(ax=ax)
    # ax.set_xscale('log'); ax.set_yscale('log')
    plt.show()
