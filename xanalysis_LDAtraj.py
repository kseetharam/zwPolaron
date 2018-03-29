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

    # # # # Concatenate Individual Datasets

    # ds_list = []; F_list = []; aIBi_list = []
    # for ind, filename in enumerate(os.listdir(innerdatapath)):
    #     if filename == 'LDA_Dataset_sph.nc':
    #         continue
    #     print(filename)
    #     ds = xr.open_dataset(innerdatapath + '/' + filename)
    #     ds_list.append(ds)
    #     F_list.append(ds.attrs['Fext_mag'])
    #     aIBi_list.append(ds.attrs['aIBi'])

    # dP = ds_list[0].attrs['Fext_mag'] * ds_list[0].attrs['TF']
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
    # del(ds_tot.attrs['Fext_mag']); del(ds_tot.attrs['aIBi']); del(ds_tot.attrs['gIB']); del(ds_tot.attrs['TF'])
    # ds_tot.attrs['dP'] = dP
    # ds_tot.to_netcdf(innerdatapath + '/LDA_Dataset_sph.nc')

    # # # Analysis of Total Dataset

    aIBi = -1.17
    qds = xr.open_dataset(innerdatapath + '/LDA_Dataset_sph.nc')
    attrs = qds.attrs
    dP = attrs['dP']
    mI = attrs['mI']
    Fscale = attrs['nu'] / attrs['xi']**2
    FVals = qds['F'].values
    tVals = qds['t'].values
    qds_aIBi = qds.sel(aIBi=aIBi)

    # v_ds = (qds_aIBi['X'].diff('t')).rename('v')
    # for Find, F in enumerate(FVals):
    #     fig, ax = plt.subplots()
    #     v_ds.sel(F=F).plot(ax=ax)
    #     ax.plot((dP / F) * np.ones(tVals.size), np.linspace(0, v_ds.sel(F=F).max('t'), tVals.size), 'g--')
    #     ax.set_xlim([0, 20])
    #     # ax.set_xscale('log'); ax.set_yscale('log')
    #     # print('TF: {0}'.format(dP / F))
    #     vf = v_ds.sel(F=F).isel(t=-1).values
    #     ms = dP / vf
    #     print('ms/mI: {0}'.format(ms / attrs['mI']))
    #     plt.show()

    x_ds = qds_aIBi['X']
    numPoints = 10
    vf_Vals = np.zeros(FVals.size)
    ms_Vals = np.zeros(FVals.size)
    for Find, F in enumerate(FVals):
        XTail = x_ds.sel(F=F).isel(t=np.arange(-1 * numPoints, 0))
        tTail = XTail.coords['t']
        [vf_Vals[Find], const] = np.polyfit(tTail.values, XTail.values, deg=1)
        ms_Vals[Find] = dP / vf_Vals[Find]

    vf_ave = np.average(vf_Vals)
    ms_ave = np.average(ms_Vals)

    # # Plotting
    # fig, ax = plt.subplots()

    # ax.plot(FVals / Fscale, vf_Vals, 'r-')
    # ax.set_ylim([0.975 * vf_ave, 1.025 * vf_ave])
    # ax.set_ylabel(r'$v_{f}=\frac{d<X>}{dx}|_{t=\infty}$')
    # ax.set_xlabel(r'$\frac{F}{\eta}$' + ' with ' + r'$\eta=\frac{c}{\xi^{2}}$')
    # ax.set_title('Final (avergae) impurity velocity')

    # ax.plot(FVals, ms_Vals / mI, 'b-')
    # ax.set_ylim([0.975 * ms_ave / mI, 1.025 * ms_ave / mI])
    # ax.set_ylabel(r'$\frac{m^{*}}{m_{I}}=\frac{1}{m_{I}} \cdot \frac{F \cdot T_{F}}{v_{f}}$')
    # ax.set_xlabel(r'$\frac{F}{\eta}$' + ' with ' + r'$\eta=\frac{c}{\xi^{2}}$')
    # ax.set_title('Impurity mass enhancement')

    # plt.show()

    aIBi_Vals = qds['aIBi'].values
    vf_AVals = np.zeros(aIBi_Vals.size)
    ms_AVals = np.zeros(aIBi_Vals.size)
    for aind, aIBi in enumerate(aIBi_Vals):
        x_ds = qds.sel(aIBi=aIBi)['X']
        numPoints = 10
        vf_Vals = np.zeros(FVals.size)
        ms_Vals = np.zeros(FVals.size)
        for Find, F in enumerate(FVals):
            XTail = x_ds.sel(F=F).isel(t=np.arange(-1 * numPoints, 0))
            tTail = XTail.coords['t']
            [vf_Vals[Find], const] = np.polyfit(tTail.values, XTail.values, deg=1)
            ms_Vals[Find] = dP / vf_Vals[Find]

        vf_AVals[aind] = np.average(vf_Vals)
        ms_AVals[aind] = np.average(ms_Vals)

    fig, ax = plt.subplots()
    ax.plot(aIBi_Vals, ms_AVals / mI, 'ro')
    ax.set_ylabel(r'$\frac{m^{*}}{m_{I}}$')
    ax.set_xlabel(r'$a_{IB}^{-1}$')
    ax.set_title('Impurity mass enhancement')
    plt.show()
