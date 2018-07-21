import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import itertools
import Grid
import pf_dynamic_sph as pfs

if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    # gParams

    # (Lx, Ly, Lz) = (30, 30, 30)
    # (dx, dy, dz) = (0.2, 0.2, 0.2)

    (Lx, Ly, Lz) = (20, 20, 20)
    (dx, dy, dz) = (0.2, 0.2, 0.2)

    # (Lx, Ly, Lz) = (30, 30, 30)
    # (dx, dy, dz) = (0.25, 0.25, 0.25)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    aBB = 0.013
    # tfin = 100

    # Toggle parameters

    toggleDict = {'Location': 'home'}
    dParams_List = [{'f_BEC_osc': 500, 'f_Imp_x': 1000, 'a_osc': 0.5, 'X0': 0.0, 'P0': 0.6}]

    # ---- SET OUTPUT DATA FOLDER ----

    datapath_List = []
    datapath_noscList = []
    for dParams in dParams_List:
        if toggleDict['Location'] == 'home':
            datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/BEC_osc/fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(aBB, NGridPoints_cart, dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])
        elif toggleDict['Location'] == 'work':
            datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/BEC_osc/fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(aBB, NGridPoints_cart, dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])
        datapath_List.append(datapath)

    # # Concatenate Individual Datasets

    # for innerdatapath in datapath_List:
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
        dParams = dParams_List[ind]
        ds_Dict[(dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])] = xr.open_dataset(innerdatapath + '/LDA_Dataset.nc')
    # if toggleDict['Large_freq'] == 'true':
    #     qds_nosc = qds_nosc.sel(t=slice(0, 25))
    expParams = pfs.Zw_expParams()
    L_exp2th, M_exp2th, T_exp2th = pfs.unitConv_exp2th(expParams['n0_BEC_scale'], expParams['mB'])
    RTF_BEC_X = expParams['RTF_BEC_X'] * L_exp2th
    omega_Imp_x = expParams['omega_Imp_x'] / T_exp2th

    f_BEC_osc = 500; f_Imp_x = 1000; a_osc = 0.5; X0 = 0.0; P0 = 0.6
    qds = ds_Dict[(f_BEC_osc, f_Imp_x, a_osc, X0, P0)]
    # qds_nosc = ds_Dict[(X0, P0, 0.0)]

    attrs = qds.attrs
    mI = attrs['mI']
    nu = attrs['nu']
    xi = attrs['xi']
    tscale = xi / nu
    tVals = qds['t'].values
    aIBiVals = qds['aIBi'].values
    dt = tVals[1] - tVals[0]
    ts = tVals / tscale
    tscale_exp = tscale / T_exp2th
    omega_BEC_osc = attrs['omega_BEC_osc']
    # print(omega_BEC_osc, (2 * np.pi / omega_BEC_osc), (1e-3 * T_exp2th * omega_BEC_osc / (2 * np.pi)), qds_nosc.attrs['omega_BEC_osc'])
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    print('xi/c (exp in ms): {0}'.format(1e3 * tscale_exp))

    # aIBi_noPlotList = [-1000.0]
    aIBi_noPlotList = []

    # # POSITION VS TIME

    # x_ds = qds['XLab']
    # fig1, ax1 = plt.subplots()
    # for ind, aIBi in enumerate(aIBiVals):
    #     if aIBi in aIBi_noPlotList:
    #         continue
    #     x0 = x_ds.sel(aIBi=aIBi).isel(t=0).values
    #     x0 = 0
    #     ax1.plot(ts, 1e6 * x_ds.sel(aIBi=aIBi).values / L_exp2th, color=colors[ind], linestyle='-', label=r'$aIB^{-1}=$' + '{:.2f}'.format(aIBi))
    # xBEC = pfs.x_BEC_osc(tVals, omega_BEC_osc, RTF_BEC_X, a_osc)
    # ax1.plot(ts, xBEC, 'k:', label='BEC Peak Position')
    # ax1.plot(ts, xBEC[0] * np.cos(omega_Imp_x * tVals), color='orange', linestyle=':', marker='', label='Impurity Trap Frequency')

    # ax1.legend()
    # ax1.set_ylabel(r'$<X> (\mu m)$')
    # ax1.set_xlabel(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]'.format(1e3 * tscale_exp))
    # ax1.set_title('Impurity Trajectory (Lab Frame)')

    # # OSCILLATION FREQUENCY PLOT
    # maxph = 0
    # fig2, ax2 = plt.subplots()
    # for ind, aIBi in enumerate(aIBiVals):
    #     if aIBi in aIBi_noPlotList:
    #         continue
    #     xVals = x_ds.sel(aIBi=aIBi).values
    #     x0 = xVals[0]
    #     dt = tVals[1] - tVals[0]
    #     FTVals = np.fft.fftshift(dt * np.fft.fft(xVals))
    #     fVals = np.fft.fftshift(np.fft.fftfreq(xVals.size) / dt)
    #     ax2.plot(fVals * T_exp2th, np.abs(FTVals), color=colors[ind], linestyle='-', label=r'$aIB^{-1}=$' + '{:.2f}'.format(aIBi))
    #     if np.max(np.abs(FTVals)) > maxph:
    #         maxph = np.max(np.abs(FTVals))

    # ax2.plot(omega_BEC_osc * T_exp2th / (2 * np.pi) * np.ones(fVals.size), np.linspace(0, maxph, fVals.size), 'k:', label='BEC Oscillation Frequency')
    # ax2.plot(omega_Imp_x * T_exp2th / (2 * np.pi) * np.ones(fVals.size), np.linspace(0, maxph, fVals.size), color='orange', linestyle=':', marker='', label='Impurity Trap Frequency')
    # ax2.legend()
    # ax2.set_xlim([0, 1500])
    # ax2.set_xlabel('f (Hz)')
    # ax2.set_ylabel(r'$\mathscr{F}[<X>]$')
    # ax2.set_title('Impurity Trajectory Frequency Spectrum')

    # # VELOCITY VS TIME (LAB FRAME)

    # v_ds = (qds['XLab'].diff('t') / dt).rename('v')
    # ts = v_ds['t'].values / tscale
    # v_BEC_osc = np.diff(xBEC) / dt
    # cBEC = nu * np.ones(v_BEC_osc.size)
    # v_ImpTrap = -1 * xBEC[0] * omega_Imp_x * np.sin(omega_Imp_x * v_ds['t'].values)
    # fig3, ax3 = plt.subplots()
    # for ind, aIBi in enumerate(aIBiVals):
    #     if aIBi in aIBi_noPlotList:
    #         continue
    #     ax3.plot(ts, v_ds.sel(aIBi=aIBi).values * (1e3 * T_exp2th / L_exp2th), color=colors[ind], linestyle='-', label=r'$aIB^{-1}=$' + '{:.2f}'.format(aIBi))

    # ax3.plot(ts[::20], v_BEC_osc[::20] * (1e3 * T_exp2th / L_exp2th), 'ko', mfc='none', label='BEC Peak Velocity')
    # ax3.plot(ts[::20], v_ImpTrap[::20] * (1e3 * T_exp2th / L_exp2th), color='orange', linestyle='', marker='o', mfc='none', label='Impurity Trap Frequency')
    # # ax3.plot(ts[::10], v_BEC_osc[::10] * (1e3 * T_exp2th / L_exp2th), 'ko', mfc='none', label='BEC Peak Velocity')
    # # ax3.plot(ts[::2], v_ImpTrap[::2] * (1e3 * T_exp2th / L_exp2th), color='orange', linestyle='', marker='o', mfc='none', label='Impurity Trap Frequency')
    # ax3.legend()
    # ax3.set_ylabel(r'$v=\frac{d<X>}{dt} (\frac{\mu m}{ms})$')
    # ax3.set_xlabel(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]'.format(1e3 * tscale_exp))
    # ax3.set_title('Impurity Velocity (Lab Frame)')

    # # VELOCITY VS TIME (BEC FRAME)

    # v_ds = (qds['X'].diff('t') / dt).rename('v')
    # ts = v_ds['t'].values / tscale
    # cBEC = nu * np.ones(ts.size)
    # fig4, ax4 = plt.subplots()
    # for ind, aIBi in enumerate(aIBiVals):
    #     if aIBi in aIBi_noPlotList:
    #         continue
    #     ax4.plot(ts, v_ds.sel(aIBi=aIBi).values * (1e3 * T_exp2th / L_exp2th), color=colors[ind], linestyle='-', label=r'$aIB^{-1}=$' + '{:.2f}'.format(aIBi))
    # ax4.fill_between(ts, -cBEC * (1e3 * T_exp2th / L_exp2th), cBEC * (1e3 * T_exp2th / L_exp2th), facecolor='yellow', alpha=0.5, label='Subsonic Region ($|v|<c_{BEC}$)')
    # ax4.legend()
    # ax4.set_ylabel(r'$v=\frac{d<X>}{dt} (\frac{\mu m}{ms})$')
    # ax4.set_xlabel(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]'.format(1e3 * tscale_exp))
    # ax4.set_title('Impurity Velocity (BEC Frame)')

    # ENERGY VS TIME

    E_ds = qds['Energy']
    fig5, ax5 = plt.subplots()
    for ind, aIBi in enumerate(aIBiVals):
        if aIBi in aIBi_noPlotList:
            continue
        E0 = E_ds.sel(aIBi=aIBi).isel(t=0).values
        E0 = 0
        ax5.plot(ts, E_ds.sel(aIBi=aIBi).values, color=colors[ind], linestyle='-', label=r'$aIB^{-1}=$' + '{:.2f}'.format(aIBi))
    xBEC = pfs.x_BEC_osc(tVals, omega_BEC_osc, RTF_BEC_X, a_osc)
    ax5.plot(ts, xBEC, 'k:', label='BEC Peak Position')
    # ax5.plot(ts, xBEC[0] * np.cos(omega_Imp_x * tVals), color='orange', linestyle=':', marker='', label='Impurity Trap Frequency')

    ax5.legend()
    ax5.set_ylabel(r'$Energy$')
    ax5.set_xlabel(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]'.format(1e3 * tscale_exp))
    ax5.set_title('Total System Energy')

    plt.show()
