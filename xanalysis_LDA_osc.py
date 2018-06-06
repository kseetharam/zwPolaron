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
                  'F_ext': 'off', 'BEC_density': 'on', 'BEC_density_osc': 'on', 'Large_freq': 'true'}
    trapParams_List = [{'X0': 0.0, 'P0': 0.1, 'a_osc': 0.5},
                       {'X0': 0.0, 'P0': 0.1, 'a_osc': 0.75},
                       {'X0': 0.0, 'P0': 0.6, 'a_osc': 0.75},
                       {'X0': 95.6, 'P0': 0.1, 'a_osc': 0.75},
                       {'X0': 95.6, 'P0': 0.6, 'a_osc': 0.75}]

    trapParams_noscList = [{'X0': 0.0, 'P0': 0.1, 'a_osc': 0.0},
                           {'X0': 0.0, 'P0': 0.6, 'a_osc': 0.0},
                           {'X0': 95.6, 'P0': 0.1, 'a_osc': 0.0},
                           {'X0': 95.6, 'P0': 0.6, 'a_osc': 0.0}]

    # trapParams_noscList = [{'X0': 0.0, 'P0': 0.1, 'a_osc': 0.0}]

    # ---- SET OUTPUT DATA FOLDER ----

    datapath_List = []
    for trapParams in trapParams_List:
        if toggleDict['Location'] == 'home':
            datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/LDA/X0={:.1f}_P0={:.1f}_aosc={:.2f}'.format(aBB, NGridPoints_cart, trapParams['X0'], trapParams['P0'], trapParams['a_osc'])
        elif toggleDict['Location'] == 'work':
            datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/LDA/X0={:.1f}_P0={:.1f}_aosc={:.2f}'.format(aBB, NGridPoints_cart, trapParams['X0'], trapParams['P0'], trapParams['a_osc'])
        innerdatapath = datapath + '/redyn_spherical_BECden'
        if toggleDict['BEC_density_osc'] == 'on':
            innerdatapath = innerdatapath + '_BECosc'
            if toggleDict['Large_freq'] == 'true':
                innerdatapath = innerdatapath + 'LF'
        elif toggleDict['BEC_density_osc'] == 'off':
            innerdatapath = innerdatapath
        innerdatapath = innerdatapath + '_SteadyStart_P_{:.1f}'.format(trapParams['P0'])
        datapath_List.append(innerdatapath)

    datapath_noscList = []
    for trapParams in trapParams_noscList:
        if toggleDict['Location'] == 'home':
            datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/LDA/X0={:.1f}_P0={:.1f}_aosc={:.2f}'.format(aBB, NGridPoints_cart, trapParams['X0'], trapParams['P0'], trapParams['a_osc'])
        elif toggleDict['Location'] == 'work':
            datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/LDA/X0={:.1f}_P0={:.1f}_aosc={:.2f}'.format(aBB, NGridPoints_cart, trapParams['X0'], trapParams['P0'], trapParams['a_osc'])
        innerdatapath = datapath + '/redyn_spherical_BECden_SteadyStart_P_{:.1f}'.format(trapParams['P0'])
        datapath_noscList.append(innerdatapath)

    # # # Concatenate Individual Datasets

    # for innerdatapath in datapath_noscList:
    #     ds_list = []; aIBi_list = []
    #     for ind, filename in enumerate(os.listdir(innerdatapath)):
    #         if filename[0:3] == 'LDA':
    #             continue
    #         ds = xr.open_dataset(innerdatapath + '/' + filename)

    #         print(filename)
    #         # ds = ds.sel(t=slice(0, tfin))
    #         ds_list.append(ds)
    #         aIBi_list.append(ds.attrs['aIBi'])

    #     s = sorted(zip(aIBi_list, ds_list))
    #     g = itertools.groupby(s, key=lambda x: x[0])

    #     aIBi_keys = []; aIBi_groups = []; aIBi_ds_list = []
    #     for key, group in g:
    #         aIBi_keys.append(key)
    #         aIBi_groups.append(list(group))

    #     for ind, group in enumerate(aIBi_groups):
    #         aIBi = aIBi_keys[ind]
    #         _, ds_temp = zip(*group)
    #         aIBi_ds_list.append(ds_temp[0])

    #     ds_tot = xr.concat(aIBi_ds_list, pd.Index(aIBi_keys, name='aIBi'))
    #     del(ds_tot.attrs['Fext_mag']); del(ds_tot.attrs['aIBi']); del(ds_tot.attrs['gIB']); del(ds_tot.attrs['TF']); del(ds_tot.attrs['Delta_P'])
    #     ds_tot.to_netcdf(innerdatapath + '/LDA_Dataset.nc')

    # # # Analysis of Total Dataset

    ds_Dict = {}
    for ind, innerdatapath in enumerate(datapath_List):
        trapParams = trapParams_List[ind]
        ds_Dict[(trapParams['X0'], trapParams['P0'], trapParams['a_osc'])] = xr.open_dataset(innerdatapath + '/LDA_Dataset.nc')
    ds_noscDict = {}
    for ind, innerdatapath in enumerate(datapath_noscList):
        trapParams = trapParams_noscList[ind]
        ds_noscDict[(trapParams['X0'], trapParams['P0'])] = xr.open_dataset(innerdatapath + '/LDA_Dataset.nc')
    # if toggleDict['Large_freq'] == 'true':
    #     qds_nosc = qds_nosc.sel(t=slice(0, 25))
    expParams = pf_dynamic_sph.Zw_expParams()
    L_exp2th, M_exp2th, T_exp2th = pf_dynamic_sph.unitConv_exp2th(expParams['n0_BEC_scale'], expParams['mB'])

    X0 = 95.6; P0 = 0.6; a_osc = 0.75
    qds = ds_Dict[(X0, P0, a_osc)]
    qds_nosc = ds_noscDict[(X0, P0)]

    attrs = qds.attrs
    mI = attrs['mI']
    nu = attrs['nu']
    xi = attrs['xi']
    tscale = xi / nu
    tVals = qds_nosc['t'].values
    aIBiVals = qds_nosc['aIBi'].values
    dt = tVals[1] - tVals[0]
    ts = tVals / tscale
    omega_BEC_osc = attrs['omega_BEC_osc']
    print(omega_BEC_osc, (2 * np.pi / omega_BEC_osc), (1e-3 * T_exp2th * omega_BEC_osc / (2 * np.pi)), qds_nosc.attrs['omega_BEC_osc'])
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    print('xi/c (exp in ms): {0}'.format(1e3 * tscale / T_exp2th))

    # POSITION VS TIME

    x_ds = qds['X']
    x_ds_nosc = qds_nosc['X']
    fig, ax = plt.subplots()
    for ind, aIBi in enumerate(aIBiVals):
        ax.plot(ts, 1e6 * x_ds.sel(aIBi=aIBi).values / L_exp2th, color=colors[ind], linestyle='-', label=r'$aIB^{-1}=$' + '{:.2f}'.format(aIBi))
        ax.plot(x_ds_nosc['t'].values / tscale, 1e6 * x_ds_nosc.sel(aIBi=aIBi).values / L_exp2th, color=colors[ind], linestyle='--', label='')
    ax.plot(ts, np.sin(omega_BEC_osc * tVals) + np.min(1e6 * x_ds.sel(aIBi=aIBi).values / L_exp2th), 'k:', label='BEC Peak Oscillation (arbit amp)')
    ax.legend()
    ax.set_ylabel(r'$<X> (\mu m)$')
    ax.set_xlabel(r'$t$ [$\frac{\xi}{c}$]')
    ax.set_title('Impurity Trajectory')

    # VELOCITY VS TIME

    v_ds = (qds['X'].diff('t') / dt).rename('v')
    ts = v_ds['t'].values / tscale
    v_ds_nosc = (qds_nosc['X'].diff('t') / dt).rename('v')

    fig2, ax2 = plt.subplots()
    for ind, aIBi in enumerate(aIBiVals):
        ax2.plot(ts, v_ds.sel(aIBi=aIBi).values * (1e3 * T_exp2th / L_exp2th), color=colors[ind], linestyle='-', label=r'$aIB^{-1}=$' + '{:.2f}'.format(aIBi))
        ax2.plot(v_ds_nosc['t'].values / tscale, v_ds_nosc.sel(aIBi=aIBi).values * (1e3 * T_exp2th / L_exp2th), color=colors[ind], linestyle='--', label='')
    ax2.plot(ts, np.sin(omega_BEC_osc * v_ds['t'].values), 'k:', label='BEC Peak Oscillation (arbit amp)')
    ax2.legend()
    ax2.set_ylabel(r'$v=\frac{d<X>}{dt} (\frac{\mu m}{ms})$')
    ax2.set_xlabel(r'$t$ [$\frac{\xi}{c}$]')
    ax2.set_title('Impurity Velocity')
    plt.show()
