import numpy as np
import pandas as pd
import xarray as xr
import pf_static_sph as pfs
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import itertools
import Grid
import pf_dynamic_sph

if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    # gParams

    (Lx, Ly, Lz) = (20, 20, 20)
    (dx, dy, dz) = (0.2, 0.2, 0.2)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    aBB = 0.013
    # tfin = 100

    # Toggle parameters

    toggleDict = {'Location': 'work', 'Dynamics': 'real', 'Interaction': 'on', 'InitCS': 'steadystate', 'InitCS_datapath': '', 'Coupling': 'twophonon', 'Grid': 'spherical',
                  'F_ext': 'off', 'BEC_density': 'on', 'BEC_density_osc': 'on', 'Large_freq': 'true', 'P0': 0.6, 'a_osc': 0.75}

    # ---- SET OUTPUT DATA FOLDER ----

    if toggleDict['Location'] == 'home':
        datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/LDA'.format(aBB, NGridPoints_cart)
    elif toggleDict['Location'] == 'work':
        datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/LDA'.format(aBB, NGridPoints_cart)
    elif toggleDict['Location'] == 'cluster':
        datapath = '/n/regal/demler_lab/kis/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/LDA'.format(aBB, NGridPoints_cart)

    if toggleDict['Dynamics'] == 'real':
        innerdatapath = datapath + '/redyn'
    elif toggleDict['Dynamics'] == 'imaginary':
        innerdatapath = datapath + '/imdyn'

    if toggleDict['Grid'] == 'cartesian':
        innerdatapath = innerdatapath + '_cart'
    elif toggleDict['Grid'] == 'spherical':
        innerdatapath = innerdatapath + '_spherical'

    if toggleDict['F_ext'] == 'on':
        innerdatapath = innerdatapath + '_extForce'
    elif toggleDict['F_ext'] == 'off':
        innerdatapath = innerdatapath

    if toggleDict['BEC_density'] == 'on':
        innerdatapath = innerdatapath + '_BECden'
    elif toggleDict['BEC_density'] == 'off':
        innerdatapath = innerdatapath

    if toggleDict['BEC_density_osc'] == 'on':
        innerdatapath = innerdatapath + '_BECosc'
        if toggleDict['Large_freq'] == 'true':
            innerdatapath = innerdatapath + 'LF'
    elif toggleDict['BEC_density_osc'] == 'off':
        innerdatapath = innerdatapath

    if toggleDict['Coupling'] == 'frohlich':
        innerdatapath = innerdatapath + '_froh'
    elif toggleDict['Coupling'] == 'twophonon':
        innerdatapath = innerdatapath

    if toggleDict['InitCS'] == 'file':
        toggleDict['InitCS_datapath'] = datapath + '/P_0_PolStates'
        innerdatapath = innerdatapath + '_ImDynStart'
    elif toggleDict['InitCS'] == 'steadystate':
        toggleDict['InitCS_datapath'] = 'InitCS ERROR'
        innerdatapath = innerdatapath + '_SteadyStart_P_0.1'
    elif toggleDict['InitCS'] == 'default':
        toggleDict['InitCS_datapath'] = 'InitCS ERROR'

    if toggleDict['Interaction'] == 'off':
        innerdatapath = innerdatapath + '_nonint'
    elif toggleDict['Interaction'] == 'on':
        innerdatapath = innerdatapath

    # # # Concatenate Individual Datasets

    # ds_list = []; aIBi_list = []
    # for ind, filename in enumerate(os.listdir(innerdatapath)):
    #     if filename[0:3] == 'LDA':
    #         continue
    #     ds = xr.open_dataset(innerdatapath + '/' + filename)

    #     print(filename)
    #     # ds = ds.sel(t=slice(0, tfin))
    #     ds_list.append(ds)
    #     aIBi_list.append(ds.attrs['aIBi'])

    # s = sorted(zip(aIBi_list, ds_list))
    # g = itertools.groupby(s, key=lambda x: x[0])

    # aIBi_keys = []; aIBi_groups = []; aIBi_ds_list = []
    # for key, group in g:
    #     aIBi_keys.append(key)
    #     aIBi_groups.append(list(group))

    # for ind, group in enumerate(aIBi_groups):
    #     aIBi = aIBi_keys[ind]
    #     _, ds_temp = zip(*group)
    #     aIBi_ds_list.append(ds_temp[0])

    # ds_tot = xr.concat(aIBi_ds_list, pd.Index(aIBi_keys, name='aIBi'))
    # del(ds_tot.attrs['Fext_mag']); del(ds_tot.attrs['aIBi']); del(ds_tot.attrs['gIB']); del(ds_tot.attrs['TF']); del(ds_tot.attrs['Delta_P'])
    # ds_tot.to_netcdf(innerdatapath + '/LDA_Dataset.nc')

    # # # Analysis of Total Dataset

    filepath = innerdatapath + '/LDA_Dataset.nc'
    qds = xr.open_dataset(filepath)
    qds_nonosc = xr.open_dataset(datapath + '/redyn_spherical_BECden_SteadyStart_P_0.1/LDA_Dataset.nc')
    # if toggleDict['Large_freq'] == 'true':
    #     qds_nonosc = qds_nonosc.sel(t=slice(0, 25))
    expParams = pf_dynamic_sph.Zw_expParams()
    L_exp2th, M_exp2th, T_exp2th = pf_dynamic_sph.unitConv_exp2th(expParams['n0_BEC_scale'], expParams['mB'])

    attrs = qds.attrs
    mI = attrs['mI']
    nu = attrs['nu']
    xi = attrs['xi']
    tscale = xi / nu
    tVals = qds['t'].values
    aIBiVals = qds['aIBi'].values
    dt = tVals[1] - tVals[0]
    ts = tVals / tscale
    omega_BEC_osc = attrs['omega_BEC_osc']
    print(omega_BEC_osc, 2 * np.pi / omega_BEC_osc, qds_nonosc.attrs['omega_BEC_osc'])
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    print(1e6 * tscale / T_exp2th)

    # POSITION VS TIME

    x_ds = qds['X']
    x_ds_nonosc = qds_nonosc['X']
    fig, ax = plt.subplots()
    for ind, aIBi in enumerate(aIBiVals):
        ax.plot(ts, 1e6 * x_ds.sel(aIBi=aIBi).values / L_exp2th, color=colors[ind], linestyle='-', label=r'$aIB^{-1}=$' + '{:.2f}'.format(aIBi))
        ax.plot(x_ds_nonosc['t'].values / tscale, 1e6 * x_ds_nonosc.sel(aIBi=aIBi).values / L_exp2th, color=colors[ind], linestyle='--', label='')
    ax.plot(ts, np.sin(omega_BEC_osc * tVals), 'k:', label='BEC Peak Oscillation')
    ax.legend()
    ax.set_ylabel(r'$<X> (\mu m)$')
    ax.set_xlabel(r'$t$ [$\frac{\xi}{c}$]')
    ax.set_title('Impurity Trajectory')
    plt.show()

    # VELOCITY VS TIME

    v_ds = (qds['X'].diff('t') / dt).rename('v')
    ts = v_ds['t'].values / tscale
    v_ds_nonosc = (qds_nonosc['X'].diff('t') / dt).rename('v')

    fig, ax = plt.subplots()
    for ind, aIBi in enumerate(aIBiVals):
        ax.plot(ts, v_ds.sel(aIBi=aIBi).values, color=colors[ind], linestyle='-', label=r'$aIB^{-1}=$' + '{:.2f}'.format(aIBi))
        ax.plot(v_ds_nonosc['t'].values / tscale, v_ds_nonosc.sel(aIBi=aIBi).values, color=colors[ind], linestyle='--', label='')

    ax.legend()
    ax.set_ylabel(r'$v=\frac{d<X>}{dt}$')
    ax.set_xlabel(r'$t$ [$\frac{\xi}{c}$]')
    ax.set_title('Impurity Velocity')
    plt.show()
