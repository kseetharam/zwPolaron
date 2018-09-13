import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy import interpolate
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
from matplotlib.animation import writers
import os
import itertools
import Grid
import pf_dynamic_sph as pfs

if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})
    mpegWriter = writers['ffmpeg'](fps=20, bitrate=1800)

    # gParams

    (Lx, Ly, Lz) = (20, 20, 20)
    (dx, dy, dz) = (0.2, 0.2, 0.2)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    aBB = 0.013
    # tfin = 100

    # Toggle parameters

    toggleDict = {'Location': 'work', 'CS_Dyn': 'on', 'PosScat': 'off', 'ObsONLY': 'false', 'tFin': 'true'}
    dParams_List = [{'f_BEC_osc': 500, 'f_Imp_x': 1000, 'a_osc': 0.5, 'X0': 0.0, 'P0': 0.6}]
    # dParams_List = [{'f_BEC_osc': 500, 'f_Imp_x': 1000, 'a_osc': 0.5, 'X0': 358.6, 'P0': 0.6}]

    # ---- SET OUTPUT DATA FOLDER ----

    datapath_List = []
    datapath_noscList = []
    for dParams in dParams_List:
        if toggleDict['Location'] == 'home':
            datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}'.format(aBB, NGridPoints_cart)
            animpath = '/home/kis/Dropbox/ZwierleinExp/figures/aBB={:.3f}/BEC_osc'.format(aBB)
        elif toggleDict['Location'] == 'work':
            datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}'.format(aBB, NGridPoints_cart)
            animpath = '/media/kis/Storage/Dropbox/ZwierleinExp/figures/aBB={:.3f}/BEC_osc'.format(aBB)
        if toggleDict['PosScat'] == 'on':
            innerdatapath = datapath + '/BEC_osc/PosScat'
        else:
            innerdatapath = datapath + '/BEC_osc'
        if toggleDict['CS_Dyn'] == 'off':
            innerdatapath = innerdatapath + '/NoCSdyn_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])
        else:
            innerdatapath = innerdatapath + '/fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])
        if toggleDict['ObsONLY'] == 'true':
            innerdatapath = innerdatapath + '_ObsONLY'
        if toggleDict['tFin'] == 'true':
            innerdatapath = innerdatapath + '_tF'

        datapath_List.append(innerdatapath)

    if toggleDict['PosScat'] == 'on':
        animpath = animpath + '/PosScat'

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

    #     if toggleDict['PosScat'] == 'on':
    #         aIBi_keys = aIBi_keys[::-1]
    #         aIBi_ds_list = aIBi_ds_list[::-1]

    #     ds_tot = xr.concat(aIBi_ds_list, pd.Index(aIBi_keys, name='aIBi'))
    #     del(ds_tot.attrs['Fext_mag']); del(ds_tot.attrs['aIBi']); del(ds_tot.attrs['gIB']); del(ds_tot.attrs['TF']); del(ds_tot.attrs['Delta_P'])
    #     ds_tot.to_netcdf(innerdatapath + '/LDA_Dataset.nc')

    # # # # Analysis of Total Dataset (FOR SPECIFIC INTERACTION)

    # aIBi = -0.25
    # ds_Dict = {}
    # for ind, innerdatapath in enumerate(datapath_List):
    #     dParams = dParams_List[ind]
    #     ds_Dict[(dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])] = xr.open_dataset(innerdatapath + '/aIBi_{:.2f}.nc'.format(aIBi))
    # expParams = pfs.Zw_expParams()
    # L_exp2th, M_exp2th, T_exp2th = pfs.unitConv_exp2th(expParams['n0_BEC_scale'], expParams['mB'])
    # RTF_BEC_X = expParams['RTF_BEC_X'] * L_exp2th
    # omega_Imp_x = expParams['omega_Imp_x'] / T_exp2th

    # f_BEC_osc = 500; f_Imp_x = 1000; a_osc = 0.5; X0 = 0.0; P0 = 0.6
    # qds = ds_Dict[(f_BEC_osc, f_Imp_x, a_osc, X0, P0)]

    # attrs = qds.attrs
    # mI = attrs['mI']
    # mB = attrs['mB']
    # nu = attrs['nu']
    # xi = attrs['xi']
    # tscale = xi / nu
    # tVals = qds['t'].values
    # dt = tVals[1] - tVals[0]
    # ts = tVals / tscale
    # tscale_exp = tscale / T_exp2th
    # omega_BEC_osc = attrs['omega_BEC_osc']
    # xBEC = pfs.x_BEC_osc(tVals, omega_BEC_osc, RTF_BEC_X, a_osc)
    # xB0 = xBEC[0]
    # x_ds = qds['XLab']
    # print('xi/c (exp in ms): {0}'.format(1e3 * tscale_exp))
    # print('mI*c: {0}'.format(mI * nu))

    # # INDIVIDUAL PHONON MOMENTUM DISTRIBUTION PLOT (FOR SPECIFIC INTERACTION)

    # tind = -1
    # CSAmp_ds = (qds['Real_CSAmp'] + 1j * qds['Imag_CSAmp']).isel(t=tind)
    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', CSAmp_ds.coords['k'].values); kgrid.initArray_premade('th', CSAmp_ds.coords['th'].values)
    # kVec = kgrid.getArray('k')
    # thVec = kgrid.getArray('th')
    # print('dk: {0}'.format(kVec[1] - kVec[0]))
    # CSAmp_Vals = CSAmp_ds.values
    # Nph = qds.isel(t=tind)['Nph'].values
    # Bk_2D_vals = CSAmp_Vals.reshape((len(kVec), len(thVec)))
    # kg, thg = np.meshgrid(kVec, thVec, indexing='ij')
    # kxg = kg * np.sin(thg)
    # kzg = kg * np.cos(thg)
    # PhDen = ((1 / Nph) * np.abs(Bk_2D_vals)**2).real.astype(float)
    # # Normalization of the original data array - this checks out
    # dk = kVec[1] - kVec[0]
    # dth = thVec[1] - thVec[0]
    # Bk_norm = np.sum(dk * dth * (2 * np.pi)**(-2) * kg**2 * np.sin(thg) * PhDen)
    # print('Original (1/Nph)|Bk|^2 normalization (Spherical 2D): {0}'.format(Bk_norm))

    # # # Interpolate 2D slice of position distribution
    # # posmult = 5
    # # k_interp = np.linspace(np.min(kVec), np.max(kVec), posmult * kVec.size); th_interp = np.linspace(np.min(thVec), np.max(thVec), posmult * thVec.size)
    # # kg_interp, thg_interp = np.meshgrid(k_interp, th_interp, indexing='ij')
    # # PhDen_interp = interpolate.griddata((kg.flatten(), thg.flatten()), PhDen.flatten(), (kg_interp, thg_interp), method='linear')

    # # # Normalization of the original data array - this checks out
    # # dk_interp = k_interp[1] - k_interp[0]
    # # dth_interp = th_interp[1] - th_interp[0]
    # # Bk_norm_interp = np.sum(dk_interp * dth_interp * (2 * np.pi)**(-2) * kg_interp**2 * np.sin(thg_interp) * PhDen_interp)
    # # print('Interp (1/Nph)|Bk|^2 normalization (Spherical 2D): {0}'.format(Bk_norm_interp))

    # # kxg_interp = kg_interp * np.sin(thg_interp)
    # # kzg_interp = kg_interp * np.cos(thg_interp)
    # # print(np.min(PhDen_interp))

    # fig1, ax1 = plt.subplots()
    # # quad1 = ax1.pcolormesh(kzg_interp, kxg_interp, PhDen_interp, norm=colors.LogNorm(vmin=1e-3, vmax=1e9), cmap='inferno')
    # # quad1m = ax1.pcolormesh(kzg_interp, -1 * kxg_interp, PhDen_interp, norm=colors.LogNorm(vmin=1e-3, vmax=1e9), cmap='inferno')
    # quad1 = ax1.pcolormesh(kzg, kxg, PhDen, norm=colors.LogNorm(vmin=1e0, vmax=1e9), cmap='inferno')
    # quad1m = ax1.pcolormesh(kzg, -1 * kxg, PhDen, norm=colors.LogNorm(vmin=1e0, vmax=1e9), cmap='inferno')
    # ax1.set_xlim([-1 * 0.3, 0.3])
    # ax1.set_ylim([-1 * 0.3, 0.3])
    # ax1.set_xlabel('kz (Impurity Propagation Direction)')
    # ax1.set_ylabel('kx')
    # ax1.set_title('Individual Phonon Momentum Distribution')
    # fig1.colorbar(quad1, ax=ax1, extend='both')

    # plt.show()

    # # INDIVIDUAL PHONON MOMENTUM DISTRIBUTION ANIMATION - TIME (FOR SPECIFIC INTERACTION)

    # tVals = tVals[::100]
    # ts = ts[::100]
    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', qds.coords['k'].values); kgrid.initArray_premade('th', qds.coords['th'].values)
    # kVec = kgrid.getArray('k')
    # thVec = kgrid.getArray('th')
    # kg, thg = np.meshgrid(kVec, thVec, indexing='ij')
    # kxg = kg * np.sin(thg)
    # kzg = kg * np.cos(thg)

    # PhDen_array = np.empty(tVals.size, dtype=np.object)
    # for tind, t in enumerate(tVals):
    #     CSAmp_Vals = (qds['Real_CSAmp'] + 1j * qds['Imag_CSAmp']).isel(t=tind).values
    #     Nph = qds.isel(t=tind)['Nph'].values
    #     Bk_2D_vals = CSAmp_Vals.reshape((len(kVec), len(thVec)))
    #     PhDen_array[tind] = ((1 / Nph) * np.abs(Bk_2D_vals)**2).real.astype(float)

    # fig1, ax1 = plt.subplots()
    # quad1 = ax1.pcolormesh(kzg, kxg, PhDen_array[0][:-1, :-1], norm=colors.LogNorm(vmin=1e0, vmax=1e9), cmap='inferno')
    # quad1m = ax1.pcolormesh(kzg, -1 * kxg, PhDen_array[0][:-1, :-1], norm=colors.LogNorm(vmin=1e0, vmax=1e9), cmap='inferno')
    # t_text = ax1.text(0.01, 0.9, r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]='.format(1e3 * tscale_exp) + '{:.2f}'.format(ts[0]), transform=ax1.transAxes, color='r')

    # ax1.set_xlim([-1 * 0.3, 0.3])
    # ax1.set_ylim([-1 * 0.3, 0.3])
    # ax1.set_xlabel('kz (Impurity Propagation Direction)')
    # ax1.set_ylabel('kx')
    # ax1.set_title('Individual Phonon Momentum Distribution' + ' (aIBi={:.2f})'.format(aIBi))
    # fig1.colorbar(quad1, ax=ax1, extend='both')

    # def animate_PhDen(i):
    #     if i >= tVals.size:
    #         return
    #     quad1.set_array(PhDen_array[i][:-1, :-1].ravel())
    #     quad1m.set_array(PhDen_array[i][:-1, :-1].ravel())
    #     t_text.set_text(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]='.format(1e3 * tscale_exp) + '{:.2f}'.format(ts[i]))

    # anim_PhDen = FuncAnimation(fig1, animate_PhDen, interval=200, frames=range(tVals.size))
    # anim_PhDen_filename = '/PhDenAnimTime_aIBi={:.2f}_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.mp4'.format(aIBi, f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # # anim_PhDen.save(animpath + anim_PhDen_filename, writer=mpegWriter)
    # plt.show()

    # # # # Analysis of Total Dataset (FOR FINAL TIME)

    # ds_Dict = {}
    # for ind, innerdatapath in enumerate(datapath_List):
    #     dParams = dParams_List[ind]
    #     ds_Dict[(dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])] = xr.open_dataset(innerdatapath + '/LDA_Dataset.nc')
    # expParams = pfs.Zw_expParams()
    # L_exp2th, M_exp2th, T_exp2th = pfs.unitConv_exp2th(expParams['n0_BEC_scale'], expParams['mB'])
    # RTF_BEC_X = expParams['RTF_BEC_X'] * L_exp2th
    # omega_Imp_x = expParams['omega_Imp_x'] / T_exp2th

    # f_BEC_osc = 500; f_Imp_x = 1000; a_osc = 0.5; X0 = 0.0; P0 = 0.6
    # qds = ds_Dict[(f_BEC_osc, f_Imp_x, a_osc, X0, P0)]

    # attrs = qds.attrs
    # mI = attrs['mI']
    # mB = attrs['mB']
    # nu = attrs['nu']
    # xi = attrs['xi']
    # aIBiVals = qds['aIBi'].values
    # print('mI*c: {0}'.format(mI * nu))

    # # INDIVIDUAL PHONON MOMENTUM DISTRIBUTION ANIMATION - INTERACTION STRENGTH (FOR FINAL TIME)

    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', qds.coords['k'].values); kgrid.initArray_premade('th', qds.coords['th'].values)
    # kVec = kgrid.getArray('k')
    # thVec = kgrid.getArray('th')
    # kg, thg = np.meshgrid(kVec, thVec, indexing='ij')
    # kxg = kg * np.sin(thg)
    # kzg = kg * np.cos(thg)

    # PhDen_array = np.empty(aIBiVals.size, dtype=np.object)
    # for aind, t in enumerate(aIBiVals):
    #     CSAmp_Vals = (qds['Real_CSAmp'] + 1j * qds['Imag_CSAmp']).isel(aIBi=aind).values
    #     Nph = qds.isel(aIBi=aind)['Nph'].values
    #     Bk_2D_vals = CSAmp_Vals.reshape((len(kVec), len(thVec)))
    #     PhDen_array[aind] = ((1 / Nph) * np.abs(Bk_2D_vals)**2).real.astype(float)

    # fig1, ax1 = plt.subplots()
    # quad1 = ax1.pcolormesh(kzg, kxg, PhDen_array[0][:-1, :-1], norm=colors.LogNorm(vmin=1e0, vmax=1e9), cmap='inferno')
    # quad1m = ax1.pcolormesh(kzg, -1 * kxg, PhDen_array[0][:-1, :-1], norm=colors.LogNorm(vmin=1e0, vmax=1e9), cmap='inferno')
    # aIBi_text = ax1.text(0.01, 0.9, r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[0]), transform=ax1.transAxes, color='r')

    # ax1.set_xlim([-1 * 0.2, 0.2])
    # ax1.set_ylim([-1 * 0.2, 0.2])
    # ax1.set_xlabel('kz (Impurity Propagation Direction)')
    # ax1.set_ylabel('kx')
    # ax1.set_title('Individual Phonon Momentum Distribution (Final Time)')
    # fig1.colorbar(quad1, ax=ax1, extend='both')

    # def animate_PhDen(i):
    #     if i >= aIBiVals.size:
    #         return
    #     quad1.set_array(PhDen_array[i][:-1, :-1].ravel())
    #     quad1m.set_array(PhDen_array[i][:-1, :-1].ravel())
    #     aIBi_text.set_text(r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[i]))

    # anim_PhDen = FuncAnimation(fig1, animate_PhDen, interval=100, frames=range(aIBiVals.size))
    # anim_PhDen_filename = '/PhDenAnimInteraction_tF_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.mp4'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # # anim_PhDen.save(animpath + anim_PhDen_filename, writer=mpegWriter)
    # plt.show()
