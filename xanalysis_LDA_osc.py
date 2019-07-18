import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy import interpolate
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import writers
import os
import itertools
import Grid
import pf_dynamic_sph as pfs

if __name__ == "__main__":

    # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})
    # mpegWriter = writers['ffmpeg'](fps=1, bitrate=1800)
    mpegWriter = writers['ffmpeg'](bitrate=1800)

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

    toggleDict = {'Location': 'work', 'CS_Dyn': 'on', 'PosScat': 'off', 'ObsONLY': 'false'}
    dParams_List = [{'f_BEC_osc': 80, 'f_Imp_x': 150, 'a_osc': 0.7, 'X0': 0.0, 'P0': 0.4}]
    # dParams_List = [{'f_BEC_osc': 500, 'f_Imp_x': 1000, 'a_osc': 0.5, 'X0': 0.0, 'P0': 0.6}]
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

        datapath_List.append(innerdatapath)

    if toggleDict['PosScat'] == 'on':
        animpath = animpath + '/PosScat'

    # # # Concatenate Individual Datasets

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

    # # # Analysis of Total Dataset

    ds_Dict = {}
    for ind, innerdatapath in enumerate(datapath_List):
        dParams = dParams_List[ind]
        print(innerdatapath)
        ds_Dict[(dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])] = xr.open_dataset(innerdatapath + '/LDA_Dataset.nc')
    # if toggleDict['Large_freq'] == 'true':
    #     qds_nosc = qds_nosc.sel(t=slice(0, 25))
    expParams = pfs.Zw_expParams()
    L_exp2th, M_exp2th, T_exp2th = pfs.unitConv_exp2th(expParams['n0_BEC_scale'], expParams['mB'])
    RTF_BEC_X = expParams['RTF_BEC_X'] * L_exp2th
    omega_Imp_x = expParams['omega_Imp_x'] / T_exp2th
    a0_exp = 5.29e-11; a0_th = a0_exp * L_exp2th

    f_BEC_osc = 80; f_Imp_x = 150; a_osc = 0.7; X0 = 0.0; P0 = 0.4
    # f_BEC_osc = 500; f_Imp_x = 1000; a_osc = 0.5; X0 = 0.0; P0 = 0.6
    # f_BEC_osc = 500; f_Imp_x = 1000; a_osc = 0.5; X0 = 358.6; P0 = 0.6
    qds = ds_Dict[(f_BEC_osc, f_Imp_x, a_osc, X0, P0)]
    # qds_nosc = ds_Dict[(X0, P0, 0.0)]

    attrs = qds.attrs
    mI = attrs['mI']
    mB = attrs['mB']
    nu = attrs['nu']
    xi = attrs['xi']
    tscale = xi / nu
    tVals = qds['t'].values
    aIBiVals = qds['aIBi'].values
    dt = tVals[1] - tVals[0]
    ts = tVals / tscale
    tscale_exp = tscale / T_exp2th
    omega_BEC_osc = attrs['omega_BEC_osc']
    xBEC = pfs.x_BEC_osc(tVals, omega_BEC_osc, RTF_BEC_X, a_osc)
    xB0 = xBEC[0]
    x_ds = qds['XLab']
    # print(omega_BEC_osc, (2 * np.pi / omega_BEC_osc), (1e-3 * T_exp2th * omega_BEC_osc / (2 * np.pi)), qds_nosc.attrs['omega_BEC_osc'])
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    print('xi/c (exp in ms): {0}'.format(1e3 * tscale_exp))
    print('mI*c: {0}'.format(mI * nu))
    print(omega_Imp_x)

    # aIBi_noPlotList = [-1000.0]
    aIBi_noPlotList = []

    # # POSITION VS TIME

    # aIBi_des = -0.3
    # xds = qds['XLab'].sel(aIBi=aIBi_des, method='nearest')
    # xDat = xds.values
    # aIBi = xds['aIBi'].values

    # fig1, ax1 = plt.subplots()
    # ax1.plot(ts, 1e6 * xDat / L_exp2th, color='g', linestyle='-', label=r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBi))
    # ax1.plot(ts, xBEC, 'k:', label='BEC Peak Position')
    # ax1.plot(ts, xBEC[0] * np.cos(omega_Imp_x * tVals), color='orange', linestyle=':', marker='', label='Impurity Trap Frequency')
    # ax1.legend(loc=2)
    # ax1.set_ylabel(r'$<X> (\mu m)$')
    # ax1.set_xlabel(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]'.format(1e3 * tscale_exp))
    # ax1.set_title('Impurity Trajectory (Lab Frame)')

    # # POSITION VS TIME ANIMATION

    # inverseScat = False

    # x_ds = qds['XLab']
    # xI_array = np.empty(aIBiVals.size, dtype=np.object)
    # for ind, aIBi in enumerate(aIBiVals):
    #     xI_array[ind] = x_ds.isel(aIBi=ind).values

    # fig6, ax6 = plt.subplots()
    # ax6.plot(ts, xBEC, 'k:', label='BEC Peak Position')
    # ax6.plot(ts, xBEC[0] * np.cos(omega_Imp_x * tVals), color='orange', linestyle=':', marker='', label='Impurity Trap Frequency')
    # curve = ax6.plot(ts, 1e6 * xI_array[0] / L_exp2th, color='g', lw=2, label='')[0]
    # if inverseScat is True:
    #     aIBi_text = ax6.text(0.8, 0.9, r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[0]), transform=ax6.transAxes, color='r')
    # else:
    #     aIB_text = ax6.text(0.75, 0.9, r'$a_{IB}=$' + '{:d}'.format(((1 / aIBiVals[0]) / a0_th).astype(int)) + r' [$a_{0}$]', transform=ax6.transAxes, color='r')

    # ax6.legend(loc=2)
    # ax6.set_ylabel(r'$<X> (\mu m)$')
    # ax6.set_xlabel(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]'.format(1e3 * tscale_exp))
    # ax6.set_title('Impurity Trajectory (Lab Frame)')

    # def animate_pos(i):
    #     # if i >= aIBiVals.size:
    #     #     return
    #     curve.set_ydata(1e6 * xI_array[i] / L_exp2th)
    #     if inverseScat is True:
    #         aIBi_text.set_text(r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[i]))
    #     else:
    #         aIB_text.set_text(r'$a_{IB}=$' + '{:d}'.format(((1 / aIBiVals[i]) / a0_th).astype(int)) + r' [$a_{0}$]')
    # anim_p = FuncAnimation(fig6, animate_pos, interval=50, frames=range(aIBiVals.size), repeat=False)

    # # anim_p_filename = '/TrajAnim_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.gif'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # # anim_p.save(animpath + anim_p_filename, writer='imagemagick')

    # anim_p_filename = '/TrajAnim_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.mp4'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # if toggleDict['CS_Dyn'] == 'off':
    #     anim_p_filename = '/NoCSdyn_' + anim_p_filename[1:]
    # # anim_p.save(animpath + anim_p_filename, writer=mpegWriter)

    # # POSITION 2D PLOT

    # aIBiVals = aIBiVals[5:]
    # x_ds = qds['XLab'].sel(aIBi=aIBiVals)
    # x_ds = 1e6 * x_ds / L_exp2th
    # x_interp, t_interp, aIBi_interp = pfs.xinterp2D(x_ds, 't', 'aIBi', 10)
    # fig6, ax6 = plt.subplots()
    # quadx = ax6.pcolormesh(t_interp / tscale, aIBi_interp, x_interp, vmin=-70, vmax=70)
    # if toggleDict['PosScat'] != 'on':
    #     ax6.set_ylim([aIBiVals[-1], aIBiVals[0]])
    # ax6.set_xlabel(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]'.format(1e3 * tscale_exp))
    # ax6.set_ylabel(r'$a_{IB}^{-1}$')
    # ax6.set_title('Impurity Trajectory (Lab Frame) in ' + r'$\mu m$')
    # fig6.colorbar(quadx, ax=ax6, extend='max')

    # plt.show()

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
    #     ax2.plot(fVals * T_exp2th, np.abs(FTVals), color=colors[ind % 7], linestyle='-', label=r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBi))
    #     if np.max(np.abs(FTVals)) > maxph:
    #         maxph = np.max(np.abs(FTVals))

    # ax2.plot(omega_BEC_osc * T_exp2th / (2 * np.pi) * np.ones(fVals.size), np.linspace(0, maxph, fVals.size), 'k:', label='BEC Oscillation Frequency')
    # ax2.plot(omega_Imp_x * T_exp2th / (2 * np.pi) * np.ones(fVals.size), np.linspace(0, maxph, fVals.size), color='orange', linestyle=':', marker='', label='Impurity Trap Frequency')
    # ax2.legend()
    # ax2.set_xlim([0, 1500])
    # ax2.set_xlabel('f (Hz)')
    # ax2.set_ylabel(r'$\mathscr{F}[<X>]$')
    # ax2.set_title('Impurity Trajectory Frequency Spectrum')

    # # OSCILLATION FREQUENCY ANIMATION

    # freq_array = np.empty(aIBiVals.size, dtype=np.object)
    # maxph = 0
    # for ind, aIBi in enumerate(aIBiVals):
    #     if aIBi in aIBi_noPlotList:
    #         continue
    #     xVals = x_ds.sel(aIBi=aIBi).values
    #     x0 = xVals[0]
    #     dt = tVals[1] - tVals[0]
    #     FTVals = np.fft.fftshift(dt * np.fft.fft(xVals))
    #     fVals = np.fft.fftshift(np.fft.fftfreq(xVals.size) / dt)
    #     freq_array[ind] = np.abs(FTVals)
    #     if np.max(np.abs(FTVals)) > maxph:
    #         maxph = np.max(np.abs(FTVals))

    # fig7, ax7 = plt.subplots()
    # curve = ax7.plot(fVals * T_exp2th, freq_array[0], color='g', lw=2, label='')[0]
    # aIBi_text = ax7.text(0.8, 0.9, r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[0]), transform=ax7.transAxes, color='r')

    # ax7.plot(omega_BEC_osc * T_exp2th / (2 * np.pi) * np.ones(fVals.size), np.linspace(0, maxph, fVals.size), 'k:', label='BEC Oscillation Frequency')
    # ax7.plot(omega_Imp_x * T_exp2th / (2 * np.pi) * np.ones(fVals.size), np.linspace(0, maxph, fVals.size), color='orange', linestyle=':', marker='', label='Impurity Trap Frequency')
    # ax7.legend(loc=2)
    # ax7.set_xlim([0, 1500])
    # ax7.set_xlabel('f (Hz)')
    # ax7.set_ylabel(r'$|\mathscr{F}[<X>]|$')
    # ax7.set_title('Impurity Trajectory Frequency Spectrum')

    # def animate_freq(i):
    #     if i >= aIBiVals.size:
    #         return
    #     curve.set_ydata(freq_array[i])
    #     aIBi_text.set_text(r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[i]))

    # anim_freq = FuncAnimation(fig7, animate_freq, interval=50, frames=range(fVals.size))
    # # anim_freq_filename = '/FreqAnim_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.gif'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # # anim_freq.save(animpath + anim_freq_filename, writer='imagemagick')
    # anim_freq_filename = '/FreqAnim_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.mp4'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # if toggleDict['CS_Dyn'] == 'off':
    #     anim_freq_filename = '/NoCSdyn_' + anim_freq_filename[1:]
    # # anim_freq.save(animpath + anim_freq_filename, writer=mpegWriter)

    # # OSCILLATION FREQUENCY 2D PLOT

    # inverseScat = False
    # a0ylim = 1000

    # dt = tVals[1] - tVals[0]
    # fVals = np.fft.fftshift(np.fft.fftfreq(tVals.size) / dt)
    # # aIBiVals = aIBiVals[2:]
    # aIBVals = (1 / aIBiVals) / a0_th
    # freq_da = xr.DataArray(np.full((fVals.size, len(aIBiVals)), np.nan, dtype=float), coords=[fVals, aIBiVals], dims=['f', 'aIBi'])
    # maxph = 0
    # for ind, aIBi in enumerate(aIBiVals):
    #     if aIBi in aIBi_noPlotList:
    #         continue
    #     xVals = x_ds.sel(aIBi=aIBi).values
    #     x0 = xVals[0]
    #     dt = tVals[1] - tVals[0]
    #     # FTVals = np.fft.fftshift(dt * np.fft.fft(xVals))
    #     FTVals = np.fft.fftshift(dt * np.fft.fft(np.fft.fftshift(xVals)))
    #     fVals = np.fft.fftshift(np.fft.fftfreq(xVals.size) / dt)
    #     absFTVals = np.abs(FTVals)
    #     freq_da.sel(aIBi=aIBi)[:] = absFTVals
    #     if (inverseScat is True) and (np.abs(1 / aIBi / a0_th) > a0ylim):
    #         continue
    #     if np.max(absFTVals) > maxph:
    #         maxph = np.max(absFTVals)

    # print(maxph)
    # # vmax = 60000
    # vmax = maxph

    # absFT_interp, f_interp, aIBi_interp = pfs.xinterp2D(freq_da, 'f', 'aIBi', 5)

    # # absFT_interp = freq_da.values
    # # f_interp, aIBi_interp = np.meshgrid(freq_da['f'].values, freq_da['aIBi'].values, indexing='ij')

    # fig7, ax7 = plt.subplots()
    # if inverseScat is True:
    #     aIBi_interp = a0_th * aIBi_interp
    #     aIBiVals = a0_th * aIBiVals

    #     quadF = ax7.pcolormesh(f_interp * T_exp2th, aIBi_interp, absFT_interp, vmin=0, vmax=vmax)
    #     ax7.set_ylabel(r'$(\frac{a_{IB}}{a_{0}})^{-1}$')
    #     ax7.plot(omega_BEC_osc * T_exp2th / (2 * np.pi) * np.ones(aIBiVals.size), aIBiVals, 'k:', label='BEC Oscillation Frequency')
    #     ax7.plot(omega_Imp_x * T_exp2th / (2 * np.pi) * np.ones(aIBiVals.size), aIBiVals, color='orange', linestyle=':', marker='', label='Impurity Trap Frequency')
    #     if toggleDict['PosScat'] != 'on':
    #         ax7.set_ylim([aIBiVals[-1], aIBiVals[0]])
    #     ax7.legend(loc=2)

    # else:
    #     quadF = ax7.pcolormesh(f_interp * T_exp2th, (1 / aIBi_interp) / a0_th, absFT_interp, vmin=0, vmax=vmax)
    #     ax7.set_ylabel(r'$a_{IB}$ [$a_{0}$]')
    #     ax7.plot(omega_BEC_osc * T_exp2th / (2 * np.pi) * np.ones(aIBVals.size), aIBVals, 'k:', label='BEC Oscillation Frequency')
    #     ax7.plot(omega_Imp_x * T_exp2th / (2 * np.pi) * np.ones(aIBVals.size), aIBVals, color='orange', linestyle=':', marker='', label='Impurity Trap Frequency')
    #     if toggleDict['PosScat'] != 'on':
    #         # ax7.set_ylim([aIBVals[-7], aIBVals[0]])
    #         ax7.set_ylim([-1 * a0ylim, np.max(aIBVals)])
    #     ax7.legend(loc=4)

    # ax7.set_xlabel('f (Hz)')
    # # ax7.set_xlim([0, 1250])
    # ax7.set_xlim([0, 300])
    # ax7.set_title('Impurity Trajectory Frequency Spectrum')
    # fig7.colorbar(quadF, ax=ax7, extend='max')
    # plt.show()

    # # VELOCITY VS TIME (LAB FRAME)

    # # v_ds = (qds['XLab'].diff('t') / dt).rename('v')
    # # ts = v_ds['t'].values / tscale
    # # v_BEC_osc = np.diff(xBEC) / dt
    # # v_ImpTrap = -1 * xBEC[0] * omega_Imp_x * np.sin(omega_Imp_x * v_ds['t'].values)

    # v_BEC_osc = np.gradient(xBEC, tVals)
    # v_ImpTrap = -1 * xBEC[0] * omega_Imp_x * np.sin(omega_Imp_x * tVals)

    # cBEC = nu * np.ones(v_BEC_osc.size)
    # fig3, ax3 = plt.subplots()
    # for ind, aIBi in enumerate(aIBiVals):
    #     if aIBi in aIBi_noPlotList:
    #         continue

    #     # qds_aIBi = qds.sel(aIBi=aIBi)
    #     # P = 0.5 * (qds_aIBi['P'].isel(t=1).values + qds_aIBi['P'].isel(t=0).values)
    #     # Pph = 0.5 * (qds_aIBi['Pph'].isel(t=1).values + qds_aIBi['Pph'].isel(t=0).values)
    #     # PI = P - Pph
    #     # ms = mI * P / (P - Pph)
    #     # # print(P, P - Pph)
    #     # v0 = v_ds.sel(aIBi=aIBi).values[0]
    #     # print(np.abs(v0 - PI / mI) / v0)

    #     xVals = x_ds.sel(aIBi=aIBi).values
    #     vVals = np.gradient(xVals, tVals)
    #     ax3.plot(ts, vVals * (1e3 * T_exp2th / L_exp2th), color=colors[ind % 7], linestyle='-', label=r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBi))
    #     # ax3.plot(ts, v_ds.sel(aIBi=aIBi).values * (1e3 * T_exp2th / L_exp2th), color=colors[ind % 7], linestyle='-', label=r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBi))

    # ax3.plot(ts[::20], v_BEC_osc[::20] * (1e3 * T_exp2th / L_exp2th), 'ko', mfc='none', label='BEC Peak Velocity')
    # ax3.plot(ts[::20], v_ImpTrap[::20] * (1e3 * T_exp2th / L_exp2th), color='orange', linestyle='', marker='o', mfc='none', label='Impurity Trap Frequency')
    # # ax3.plot(ts[::10], v_BEC_osc[::10] * (1e3 * T_exp2th / L_exp2th), 'ko', mfc='none', label='BEC Peak Velocity')
    # # ax3.plot(ts[::2], v_ImpTrap[::2] * (1e3 * T_exp2th / L_exp2th), color='orange', linestyle='', marker='o', mfc='none', label='Impurity Trap Frequency')
    # ax3.legend()
    # ax3.set_ylabel(r'$v=\frac{d<X>}{dt} (\frac{\mu m}{ms})$')
    # ax3.set_xlabel(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]'.format(1e3 * tscale_exp))
    # ax3.set_title('Impurity Velocity (Lab Frame)')

    # # VELOCITY VS TIME (BEC FRAME)

    # aIBi = -0.25
    # # v_ds = (qds['X'].diff('t') / dt).rename('v')
    # # ts = v_ds['t'].values / tscale
    # cBEC = nu * np.ones(ts.size)
    # fig4, ax4 = plt.subplots()
    # ax4.plot(ts, np.gradient(qds['X'].sel(aIBi=aIBi).values, tVals) * (1e3 * T_exp2th / L_exp2th), color=colors[ind % 7], linestyle='-', label=r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBi))
    # ax4.fill_between(ts, -cBEC * (1e3 * T_exp2th / L_exp2th), cBEC * (1e3 * T_exp2th / L_exp2th), facecolor='yellow', alpha=0.5, label='Subsonic Region ($|v|<c_{BEC}$)')
    # ax4.legend(loc=2)
    # ax4.set_ylim([-450, 450])
    # ax4.set_ylabel(r'$v=\frac{d<X>}{dt} (\frac{\mu m}{ms})$')
    # ax4.set_xlabel(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]'.format(1e3 * tscale_exp))
    # ax4.set_title('Impurity Velocity (BEC Frame)')

    # # VELOCITY VS TIME ANIMATION (BEC FRAME)

    # inverseScat = False

    # cBEC = nu * np.ones(ts.size)
    # vI_array = np.empty(aIBiVals.size, dtype=np.object)
    # for ind, aIBi in enumerate(aIBiVals):
    #     vI_array[ind] = np.gradient(qds['X'].sel(aIBi=aIBi).values, tVals)

    # fig6, ax6 = plt.subplots()
    # curve = ax6.plot(ts, vI_array[0] * (1e3 * T_exp2th / L_exp2th), color='b', linestyle='-', lw=2, label='')[0]
    # if inverseScat is True:
    #     aIBi_text = ax6.text(0.8, 0.9, r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[0]), transform=ax6.transAxes, color='r')
    # else:
    #     aIB_text = ax6.text(0.75, 0.9, r'$a_{IB}=$' + '{:d}'.format(((1 / aIBiVals[0]) / a0_th).astype(int)) + r' [$a_{0}$]', transform=ax6.transAxes, color='r')
    # ax6.fill_between(ts, -cBEC * (1e3 * T_exp2th / L_exp2th), cBEC * (1e3 * T_exp2th / L_exp2th), facecolor='yellow', alpha=0.5, label='Subsonic Region ($|v|<c_{BEC}$)')

    # ax6.legend(loc=2)
    # ax6.set_ylabel(r'$v=\frac{d<X>}{dt} (\frac{\mu m}{ms})$')
    # ax6.set_xlabel(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]'.format(1e3 * tscale_exp))
    # ax6.set_title('Impurity Velocity (BEC Frame)')
    # # ax6.set_ylim([-450, 450])
    # ax6.set_ylim([-25, 25])

    # def animate_pos(i):
    #     if i >= aIBiVals.size:
    #         return
    #     curve.set_ydata(vI_array[i] * (1e3 * T_exp2th / L_exp2th))
    #     if inverseScat is True:
    #         aIBi_text.set_text(r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[i]))
    #     else:
    #         aIB_text.set_text(r'$a_{IB}=$' + '{:d}'.format(((1 / aIBiVals[i]) / a0_th).astype(int)) + r' [$a_{0}$]')

    # anim_v = FuncAnimation(fig6, animate_pos, interval=50, frames=range(aIBiVals.size), repeat=False)
    # # anim_v_filename = '/VelAnim_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.gif'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # # anim_v.save(animpath + anim_v_filename, writer='imagemagick')
    # anim_v_filename = '/VelAnim_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.mp4'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # if toggleDict['CS_Dyn'] == 'off':
    #     anim_v_filename = '/NoCSdyn_' + anim_v_filename[1:]
    # # anim_v.save(animpath + anim_v_filename, writer=mpegWriter)

    # # ENERGY VS TIME

    # E_ds = qds['Energy']
    # fig5, ax5 = plt.subplots()
    # for ind, aIBi in enumerate(aIBiVals):
    #     if aIBi in aIBi_noPlotList:
    #         continue
    #     E0 = E_ds.sel(aIBi=aIBi).isel(t=0).values
    #     E0 = 0
    #     ax5.plot(ts, E_ds.sel(aIBi=aIBi).values, color=colors[ind], linestyle='-', label=r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBi))
    # xBEC = pfs.x_BEC_osc(tVals, omega_BEC_osc, RTF_BEC_X, a_osc)
    # ax5.plot(ts, xBEC, 'k:', label='BEC Peak Position')
    # # ax5.plot(ts, xBEC[0] * np.cos(omega_Imp_x * tVals), color='orange', linestyle=':', marker='', label='Impurity Trap Frequency')

    # ax5.legend()
    # ax5.set_ylabel(r'$Energy$')
    # ax5.set_xlabel(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]'.format(1e3 * tscale_exp))
    # ax5.set_title('Total System Energy')

    # # ENERGY VS TIME ANIMATION

    # Energy_array = np.empty(aIBiVals.size, dtype=np.object)
    # for ind, aIBi in enumerate(aIBiVals):
    #     Energy_array[ind] = qds['Energy'].isel(aIBi=ind).values

    # fig6, ax6 = plt.subplots()
    # ax6.plot(ts, 10 * xBEC, 'k:', label='BEC Frequency')
    # ax6.plot(ts, 10 * xBEC[0] * np.cos(omega_Imp_x * tVals), color='orange', linestyle=':', marker='', label='Impurity Trap Frequency')
    # curve = ax6.plot(ts, Energy_array[0], color='g', lw=2, label='')[0]
    # aIBi_text = ax6.text(0.8, 0.9, r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[0]), transform=ax6.transAxes, color='r')

    # ax6.legend(loc=2)
    # ax6.set_ylabel(r'$<H>$')
    # ax6.set_xlabel(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]'.format(1e3 * tscale_exp))
    # ax6.set_title('Total System Energy')
    # ax6.set_ylim([-3000, 6000])

    # def animate_en(i):
    #     if i >= aIBiVals.size:
    #         return
    #     curve.set_ydata(Energy_array[i])
    #     aIBi_text.set_text(r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[i]))

    # anim_e = FuncAnimation(fig6, animate_en, interval=50, frames=range(ts.size))
    # # anim_e_filename = '/TrajAnim_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.gif'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # # anim_e.save(animpath + anim_e_filename, writer='imagemagick')
    # anim_e_filename = '/EnergyAnim_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.mp4'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # # anim_e.save(animpath + anim_e_filename, writer=mpegWriter)

    ###############################################################################################################################
    # # FIT IN THE BEC FRAME
    ###############################################################################################################################

    # # ODE FIT TO POSITION (TRAJECTORY) (***USE THIS)

    # def EqMotion(y, t, gamma, beta):
    #     # y1 = x, y2 = dx/dt
    #     y1, y2 = y
    #     dy1dt = y2
    #     dy2dt = -2 * gamma * y2 - (omega_Imp_x**2 + beta) * y1 + (omega_BEC_osc**2 - omega_Imp_x**2) * xB0 * np.cos(omega_BEC_osc * t)
    #     return [dy1dt, dy2dt]

    # def yint(t, gamma, beta, y0):
    #     y = odeint(EqMotion, y0, t, args=(gamma, beta))
    #     return y.ravel(order='F')

    # def gphiVals(gamma, beta, omega_Imp_x, omega_BEC_osc, xB0):
    #     kappa = omega_Imp_x**2 + beta
    #     zeta = omega_BEC_osc**2 - omega_Imp_x**2
    #     d = zeta * xB0 / np.sqrt((kappa - omega_BEC_osc**2)**2 + 4 * gamma**2 * omega_BEC_osc**2)
    #     delta = np.arctan(2 * gamma * omega_BEC_osc / (omega_BEC_osc**2 - kappa))
    #     g = np.sqrt(d**2 + xB0**2 + d * xB0 * np.cos(delta))
    #     phi = np.arctan(np.sin(delta) / (np.cos(delta) + xB0 / d))
    #     return g, phi

    # x_ds = qds['X']
    # xI_DatArray_LAB = np.empty(aIBiVals.size, dtype=np.object)
    # xI_FitArray_LAB = np.empty(aIBiVals.size, dtype=np.object)

    # xI_DatArray = np.empty(aIBiVals.size, dtype=np.object)
    # vI_DatArray = np.empty(aIBiVals.size, dtype=np.object)
    # xI_FitArray = np.empty(aIBiVals.size, dtype=np.object)
    # vI_FitArray = np.empty(aIBiVals.size, dtype=np.object)
    # R2_Array = np.empty(aIBiVals.size, dtype=np.object)
    # MSErr_Array = np.empty(aIBiVals.size, dtype=np.object)

    # y0Vals = np.empty(aIBiVals.size, dtype=np.object)
    # gammaVals = np.empty(aIBiVals.size)
    # betaVals = np.empty(aIBiVals.size)
    # gVals = np.empty(aIBiVals.size)
    # phiVals = np.empty(aIBiVals.size)
    # msVals = np.empty(aIBiVals.size)
    # for ind, aIBi in enumerate(aIBiVals):
    #     # if ind != 10:
    #     #     continue
    #     xVals = x_ds.sel(aIBi=aIBi).values
    #     vVals = np.gradient(xVals, tVals)
    #     x0 = xVals[0]
    #     v0 = vVals[0]
    #     # v0 = (qds['P'].sel(aIBi=aIBi).isel(t=0).values - qds['Pph'].sel(aIBi=aIBi).isel(t=0).values) / mI
    #     y0 = [x0, v0]
    #     data = np.concatenate((xVals, vVals))
    #     if ind == 0:
    #         p0 = [1e-3, 1e-3]
    #         lowerbound = [0, 0]
    #         upperbound = [np.inf, np.inf]

    #     else:
    #         p0 = [gammaVals[ind - 1], betaVals[ind - 1]]
    #         # lowerbound = [gammaVals[ind - 1], 0]
    #         lowerbound = [0, 0]
    #         upperbound = [np.inf, np.inf]
    #     popt, cov = curve_fit(lambda t, gamma, beta: yint(t, gamma, beta, y0), tVals, data, p0=p0, bounds=(lowerbound, upperbound))
    #     gopt, bopt = popt
    #     y0Vals[ind] = y0; gammaVals[ind] = gopt; betaVals[ind] = bopt
    #     gVals[ind], phiVals[ind] = gphiVals(gopt, bopt, omega_Imp_x, omega_BEC_osc, xB0)

    #     fitvals = yint(tVals, gammaVals[ind], betaVals[ind], y0Vals[ind])
    #     xfit = fitvals[0:tVals.size]
    #     vfit = fitvals[tVals.size:]
    #     xI_DatArray[ind] = xVals
    #     vI_DatArray[ind] = vVals
    #     xI_FitArray[ind] = xfit
    #     vI_FitArray[ind] = vfit
    #     R2_Array[ind] = r2_score(xVals, xfit)
    #     MSErr_Array[ind] = mean_squared_error(xVals, xfit)

    #     xI_DatArray_LAB[ind] = qds['XLab'].sel(aIBi=aIBi).values
    #     xI_FitArray_LAB[ind] = xfit + xBEC
    #     P = qds['P'].sel(aIBi=aIBi).isel(t=0).values
    #     Pph = qds['Pph'].sel(aIBi=aIBi).isel(t=0).values
    #     msVals[ind] = mI * P / (P - Pph)

    # # ANALYTICAL SOLUTION FIT TO POSITION (TRAJECTORY)

    # def traj(t, gamma, beta, y0, omega_Imp_x, omega_BEC_osc, xB0):
    #     [x0, v0] = y0
    #     kappa = omega_Imp_x**2 + beta
    #     zeta = omega_BEC_osc**2 - omega_Imp_x**2
    #     # print(gamma**2 - kappa)
    #     omega = np.sqrt(kappa - gamma**2)

    #     d = zeta * xB0 / np.sqrt((kappa - omega_BEC_osc**2)**2 + 4 * gamma**2 * omega_BEC_osc**2)
    #     delta = np.arctan(2 * gamma * omega_BEC_osc / (omega_BEC_osc**2 - kappa))
    #     h = np.sqrt((x0 - d * np.cos(delta))**2 + omega**(-2) * (v0 + omega_BEC_osc * d * np.sin(delta))**2)
    #     eta = -1 * np.arctan(omega**(-1) * (v0 + omega_BEC_osc * d * np.sin(delta)) / (x0 - d * np.cos(delta)))
    #     xt = h * np.exp(-1 * gamma * t) * np.cos(omega * t + eta) + d * np.cos(omega_BEC_osc * t + delta)
    #     return xt

    # aIBiVals = aIBiVals[(aIBiVals < -8)]
    # x_ds = qds['X']
    # xI_DatArray = np.empty(aIBiVals.size, dtype=np.object)
    # xI_FitArray = np.empty(aIBiVals.size, dtype=np.object)

    # y0Vals = np.empty(aIBiVals.size, dtype=np.object)
    # gammaVals = np.empty(aIBiVals.size)
    # betaVals = np.empty(aIBiVals.size)
    # deltaVals = np.empty(aIBiVals.size)
    # msVals = np.empty(aIBiVals.size)
    # for ind, aIBi in enumerate(aIBiVals):
    #     # if ind != 10:
    #     #     continue
    #     xVals = x_ds.sel(aIBi=aIBi).values
    #     vVals = np.gradient(xVals, tVals)
    #     x0 = xVals[0]
    #     v0 = vVals[0]
    #     y0 = [x0, v0]
    #     if ind == 0:
    #         p0 = [1e-3, 1e-3]
    #         bounds = ([0, 0], [np.inf, np.inf])

    #     else:
    #         p0 = [gammaVals[ind - 1], betaVals[ind - 1]]
    #         bounds = ([0, 0], [np.inf, np.inf])
    #     # print(aIBi)
    #     popt, cov = curve_fit(lambda t, gamma, beta: traj(t, gamma, beta, y0, omega_Imp_x, omega_BEC_osc, xB0), tVals, xVals, p0=p0, bounds=bounds)
    #     gopt, bopt = popt
    #     y0Vals[ind] = y0; gammaVals[ind] = gopt; betaVals[ind] = bopt
    #     deltaVals[ind] = np.arctan(2 * gopt * omega_BEC_osc / (omega_BEC_osc**2 - omega_Imp_x**2 - bopt))

    #     xI_DatArray[ind] = xVals
    #     xI_FitArray[ind] = traj(tVals, gammaVals[ind], betaVals[ind], y0Vals[ind], omega_Imp_x, omega_BEC_osc, xB0)

    #     P = qds['P'].sel(aIBi=aIBi).isel(t=0).values
    #     Pph = qds['Pph'].sel(aIBi=aIBi).isel(t=0).values
    #     msVals[ind] = mI * P / (P - Pph)

    # # POSITION (BEC) ANIMATION

    # fig, ax = plt.subplots()
    # curve_Dat = ax.plot(tVals[::20], xI_DatArray[0][::20], color='k', linestyle='', marker='o', label='')[0]
    # curve_Fit = ax.plot(tVals, xI_FitArray[0], color='orange', lw=2, label='')[0]
    # aIBi_text = ax.text(0.8, 0.9, r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[0]), transform=ax.transAxes, color='r')
    # Gamma_text = ax.text(0.8, 0.85, r'$\gamma=$' + '{:.2E}'.format(gammaVals[0]), transform=ax.transAxes, color='g')
    # Beta_text = ax.text(0.8, 0.8, r'$\beta=$' + '{:.2E}'.format(betaVals[0]), transform=ax.transAxes, color='b')

    # ax.set_xlabel('t')
    # ax.set_ylabel('<X>')
    # ax.set_title('Impurity Trajectory (BEC Frame)')

    # def animate_fit(i):
    #     if i >= aIBiVals.size:
    #         return
    #     curve_Dat.set_ydata(xI_DatArray[i][::20])
    #     curve_Fit.set_ydata(xI_FitArray[i])
    #     aIBi_text.set_text(r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[i]))
    #     Gamma_text.set_text(r'$\gamma=$' + '{:.2E}'.format(gammaVals[i]))
    #     Beta_text.set_text(r'$\beta=$' + '{:.2E}'.format(betaVals[i]))

    # anim_fit = FuncAnimation(fig, animate_fit, interval=75, frames=range(tVals.size))
    # anim_fit_filename = '/TrajFitBECAnim_BECFit_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.mp4'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # if toggleDict['CS_Dyn'] == 'off':
    #     anim_fit_filename = '/NoCSdyn_' + anim_fit_filename[1:]
    # # anim_fit.save(animpath + anim_fit_filename, writer=mpegWriter)

    # # POSITION (LAB) ANIMATION

    # inverseScat = False

    # fig, ax = plt.subplots()
    # curve_Dat = ax.plot(ts[::20], xI_DatArray_LAB[0][::20] * 1e6 / L_exp2th, color='k', linestyle='', marker='o', label='Simulation Data')[0]
    # curve_Fit = ax.plot(ts, xI_FitArray_LAB[0] * 1e6 / L_exp2th, color='orange', lw=2, label='ODE Fit')[0]
    # Gamma_text = ax.text(0.8, 0.85, r'$\gamma=$' + '{:.2E}'.format(gammaVals[0]), transform=ax.transAxes, color='g')
    # Beta_text = ax.text(0.8, 0.8, r'$\beta=$' + '{:.2E}'.format(betaVals[0]), transform=ax.transAxes, color='b')
    # rhoVals = gammaVals / np.sqrt(betaVals + omega_Imp_x**2)
    # Rho_text = ax.text(0.74, 0.74, r'$\frac{\gamma}{\sqrt{\omega_{0}^{2}+\beta}}=$' + '{:.2E}'.format(rhoVals[0]), transform=ax.transAxes, color='m')

    # if inverseScat is True:
    #     aIBi_text = ax.text(0.8, 0.9, r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[0]), transform=ax.transAxes, color='r')
    # else:
    #     aIB_text = ax.text(0.75, 0.9, r'$a_{IB}=$' + '{:d}'.format(((1 / aIBiVals[0]) / a0_th).astype(int)) + r' [$a_{0}$]', transform=ax.transAxes, color='r')

    # ax.legend(loc=2)
    # ax.set_ylabel(r'$<X> (\mu m)$')
    # ax.set_xlabel(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]'.format(1e3 * tscale_exp))
    # ax.set_title('Impurity Trajectory (Lab Frame)')

    # def animate_fit(i):
    #     if i >= aIBiVals.size:
    #         return
    #     curve_Dat.set_ydata(xI_DatArray_LAB[i][::20] * 1e6 / L_exp2th)
    #     curve_Fit.set_ydata(xI_FitArray_LAB[i] * 1e6 / L_exp2th)
    #     Gamma_text.set_text(r'$\gamma=$' + '{:.2E}'.format(gammaVals[i]))
    #     Beta_text.set_text(r'$\beta=$' + '{:.2E}'.format(betaVals[i]))
    #     Rho_text.set_text(r'$\frac{\gamma}{\sqrt{\omega_{0}^{2}+\beta}}=$' + '{:.2E}'.format(rhoVals[i]))
    #     if inverseScat is True:
    #         aIBi_text.set_text(r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[i]))
    #     else:
    #         aIB_text.set_text(r'$a_{IB}=$' + '{:d}'.format(((1 / aIBiVals[i]) / a0_th).astype(int)) + r' [$a_{0}$]')

    # anim_fit = FuncAnimation(fig, animate_fit, interval=50, frames=range(aIBiVals.size), repeat=False)
    # # # # anim_fit_filename = '/TrajFitAnim_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.gif'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # # # # anim_fit.save(animpath + anim_fit_filename, writer='imagemagick')
    # anim_fit_filename = '/TrajFitAnim_BECFit_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.mp4'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # if toggleDict['CS_Dyn'] == 'off':
    #     anim_fit_filename = '/NoCSdyn_' + anim_fit_filename[1:]
    # anim_fit.save(animpath + anim_fit_filename, writer=mpegWriter)

    # # PARAMETER CURVES (& ESTIMATE alpha = m*Beta) (***PLOT THIS)

    # NGridPoints_desired = (1 + 2 * Lx / dx) * (1 + 2 * Lz / dz)
    # Ntheta = 50
    # Nk = np.ceil(NGridPoints_desired / Ntheta)
    # theta_max = np.pi
    # thetaArray, dtheta = np.linspace(0, theta_max, Ntheta, retstep=True)
    # k_max = ((2 * np.pi / dx)**3 / (4 * np.pi / 3))**(1 / 3)
    # k_min = 1e-5
    # kArray, dk = np.linspace(k_min, k_max, Nk, retstep=True)
    # kgrid = Grid.Grid("SPHERICAL_2D")
    # kgrid.initArray_premade('k', kArray)
    # kgrid.initArray_premade('th', thetaArray)
    # n0_TF = expParams['n0_TF'] / (L_exp2th**3)
    # n0_thermal = expParams['n0_thermal'] / (L_exp2th**3)
    # RTF_BEC_X = expParams['RTF_BEC_X'] * L_exp2th; RTF_BEC_Y = expParams['RTF_BEC_Y'] * L_exp2th; RTF_BEC_Z = expParams['RTF_BEC_Z'] * L_exp2th
    # RG_BEC_X = expParams['RG_BEC_X'] * L_exp2th; RG_BEC_Y = expParams['RG_BEC_Y'] * L_exp2th; RG_BEC_Z = expParams['RG_BEC_Z'] * L_exp2th
    # trapParams = {'n0_TF_BEC': n0_TF, 'RTF_BEC_X': RTF_BEC_X, 'RTF_BEC_Y': RTF_BEC_Y, 'RTF_BEC_Z': RTF_BEC_Z, 'n0_thermal_BEC': n0_thermal, 'RG_BEC_X': RG_BEC_X, 'RG_BEC_Y': RG_BEC_Y, 'RG_BEC_Z': RG_BEC_Z,
    #               'omega_Imp_x': omega_Imp_x, 'omega_BEC_osc': omega_BEC_osc, 'X0': X0, 'P0': P0, 'a_osc': a_osc}
    # n0 = expParams['n0_BEC'] / (L_exp2th**3)  # should ~ 1
    # mB = expParams['mB'] * M_exp2th  # should = 1
    # mI = expParams['mI'] * M_exp2th
    # aBB = expParams['aBB'] * L_exp2th
    # gBB = (4 * np.pi / mB) * aBB
    # sParams = [mI, mB, n0, gBB]

    # X_Vals = np.linspace(-1 * RTF_BEC_X * 0.99, RTF_BEC_X * 0.99, 100)
    # # aIBiVals = aIBiVals[::10]
    # aVals_Est = np.empty(aIBiVals.size)
    # for ind, aIBi in enumerate(aIBiVals):
    #     cParams = {'aIBi': aIBi}
    #     E_Pol_tck = pfs.V_Pol_interp(kgrid, X_Vals, cParams, sParams, trapParams)
    #     aVals_Est[ind] = interpolate.splev(0, E_Pol_tck, der=2)

    # # PLOT PARAMETERS (DEPRECATED)
    # rhoVals = gammaVals**2 - betaVals - omega_Imp_x**2
    # critdamp_ind = np.argwhere(np.sign(rhoVals) >= 0)[0][0]
    # fig2, ax2 = plt.subplots()
    # ax2.plot(aIBiVals, gammaVals, 'g-', label=r'$\gamma$')
    # ax2.plot(aIBiVals, msVals * gammaVals, 'g:', label=r'$\xi=m^{*}\gamma$')
    # # ax2.plot(aIBiVals, rhoVals, 'm--', label=r'$\gamma^{2}-\beta-\omega_{0}^{2}$')
    # ax2.plot(aIBiVals, betaVals, color='b', linestyle='', marker='o', markerfacecolor='none', label=r'$\beta$')
    # ax2.plot(aIBiVals, msVals * betaVals, color='b', linestyle='', marker='x', label=r'$\alpha=m^{*}\beta$')
    # # ax2.plot(aIBiVals, aVals_Est, 'r-', label=r'$\alpha_{est}=\frac{d^{2}E_{pol}}{dx^{2}}|_{x_{peak}}$')
    # ax2.plot(aIBiVals, phiVals, color='orange', linestyle=':', label=r'$\varphi$')
    # ax2.plot(aIBiVals[critdamp_ind] * np.ones(aIBiVals.size), np.linspace(0, np.max(msVals * gammaVals), aIBiVals.size), 'y--', label='Critical Damping')
    # # ax2.plot(aIBiVals, msVals, 'y-', label=r'$m^{*}$')
    # ax2.set_xlabel(r'$a_{IB}^{-1}$')
    # ax2.set_title('Oscillation Fit Parameters')
    # ax2.set_ylim([-2, 4])
    # if toggleDict['PosScat'] == 'on':
    #     ax2.legend(loc=1)
    #     ax2.set_xlim([0, 40])
    # else:
    #     ax2.legend(loc=2)
    #     ax2.set_xlim([-40, 0])

    # # PLOT ERROR OF FIT
    # fig1, ax1 = plt.subplots()
    # ax1.plot(aIBiVals, R2_Array, color='r', linestyle='', marker='x', label=r'$R^{2}$')
    # # ax1.plot(aIBiVals, MSErr_Array, color='k', linestyle='', marker='x', label='Mean Squared Error')
    # # ax1.legend()
    # ax1.set_xlabel(r'$a_{IB}^{-1}$')
    # ax1.set_title(r'$R^{2}$' + ' Error')

    # # PLOT PARAMETERS FIXED (***PLOT THIS)
    # inverseScat = False
    # a0xlim = 1000

    # if inverseScat is True:
    #     rhoVals = gammaVals**2 - betaVals - omega_Imp_x**2
    #     # critdamp_ind = np.argwhere(np.sign(rhoVals) >= 0)[0][0]
    #     fig2, ax2 = plt.subplots()
    #     ax2.plot(aIBiVals, mI * gammaVals, 'g-', label=r'$m_{I} \gamma$')
    #     ax2.plot(aIBiVals, msVals * gammaVals, 'g:', label=r'$\xi=m^{*}\gamma$')
    #     # ax2.plot(aIBiVals, rhoVals, 'm--', label=r'$\gamma^{2}-\beta-\omega_{0}^{2}$')
    #     ax2.plot(aIBiVals, mI * betaVals * omega_Imp_x, color='b', linestyle='', marker='o', markerfacecolor='none', label=r'$m_{I} \beta \omega_{0}$')
    #     ax2.plot(aIBiVals, msVals * betaVals * omega_Imp_x, color='b', linestyle='', marker='x', label=r'$\alpha=m^{*}\beta \omega_{0}$')
    #     # ax2.plot(aIBiVals, aVals_Est, 'r-', label=r'$\alpha_{est}=\frac{d^{2}E_{pol}}{dx^{2}}|_{x_{peak}}$')
    #     ax2.plot(aIBiVals, phiVals, color='orange', linestyle=':', label=r'$\varphi$')
    #     # ax2.plot(aIBiVals[critdamp_ind] * np.ones(aIBiVals.size), np.linspace(0, np.max(msVals * gammaVals), aIBiVals.size), 'y--', label='Critical Damping')
    #     # ax2.plot(aIBiVals, msVals, 'y-', label=r'$m^{*}$')
    #     ax2.set_xlabel(r'$a_{IB}^{-1}$')
    #     ax2.set_title('Oscillation Fit Parameters')
    #     # ax2.set_ylim([-2, 4])
    #     if toggleDict['PosScat'] == 'on':
    #         ax2.legend(loc=1)
    #         ax2.set_xlim([0, 30])
    #     else:
    #         ax2.legend(loc=2)
    #         ax2.set_xlim([-30, 0])

    # else:
    #     aIBVals = 1 / aIBiVals / a0_th
    #     # rhoVals = gammaVals**2 - betaVals - omega_Imp_x**2
    #     rhoVals = gammaVals / np.sqrt(betaVals + omega_Imp_x**2)
    #     # critdamp_ind = np.argwhere(np.sign(rhoVals) >= 0)[0][0]
    #     fig2, ax2 = plt.subplots()
    #     ax2.plot(aIBVals, mI * gammaVals, 'g-', label=r'$m_{I} \gamma$')
    #     ax2.plot(aIBVals, msVals * gammaVals, 'g:', label=r'$\xi=m^{*}\gamma$')
    #     # ax2.plot(aIBVals, mI * betaVals * omega_Imp_x, color='b', linestyle='', marker='o', markerfacecolor='none', label=r'$m_{I} \beta \omega_{0}$')
    #     # ax2.plot(aIBVals, msVals * betaVals * omega_Imp_x, color='b', linestyle='', marker='x', label=r'$\alpha=m^{*}\beta \omega_{0}$')
    #     ax2.plot(aIBVals, mI * betaVals, color='b', linestyle='', marker='o', markerfacecolor='none', label=r'$m_{I} \beta$')
    #     ax2.plot(aIBVals, msVals * betaVals, color='b', linestyle='', marker='x', label=r'$\alpha=m^{*}\beta$')

    #     # ax2.plot(aIBVals, aVals_Est, 'r-', label=r'$\alpha_{est}=\frac{d^{2}E_{pol}}{dx^{2}}|_{x_{peak}}$')
    #     # ax2.plot(aIBVals, phiVals, color='orange', linestyle=':', label=r'$\varphi$')
    #     # ax2.plot(aIBVals[critdamp_ind] * np.ones(aIBVals.size), np.linspace(0, np.max(msVals * gammaVals), aIBVals.size), 'y--', label='Critical Damping')
    #     # ax2.plot(aIBVals, msVals, 'y-', label=r'$m^{*}$')
    #     ax2.set_xlabel(r'$a_{IB}$ [$a_{0}$]')
    #     ax2.set_title('Oscillation Fit Parameters')
    #     # ax2.set_ylim([-2, 4])
    #     if toggleDict['PosScat'] == 'on':
    #         ax2.legend(loc=1)
    #         # ax2.set_xlim([0, 30])
    #     else:
    #         ax2.legend(loc=1)
    #         # ax2.set_xlim([-1 * a0xlim, np.max(aIBVals)])
    #         # xmask = aIBVals > -1 * a0xlim
    #         # ymax = np.max(np.array([np.max(gammaVals[xmask]), np.max(betaVals[xmask])]))
    #         # ymin = np.min(np.array([np.min(gammaVals[xmask]), np.min(betaVals[xmask])]))
    #         # ax2.set_ylim([ymin, ymax])

    #     fig3, ax3 = plt.subplots()
    #     ax3.plot(aIBVals, rhoVals, 'm--')
    #     ax3.set_title('Oscillation Fit Damping Ratio')
    #     ax3.set_ylabel(r'$\frac{\gamma}{\sqrt{\omega_{0}^{2}+\beta}}$')
    #     ax3.set_xlabel(r'$a_{IB}$ [$a_{0}$]')
    #     # ax3.set_xlim([-1 * a0xlim, np.max(aIBVals)])

    #     print(aIBVals)
    #     print(gammaVals)
    #     print(betaVals)
    #     print(rhoVals)
    #     print(omega_Imp_x, omega_BEC_osc)

    # # PLOT ERROR OF FIT
    # fig1, ax1 = plt.subplots()
    # ax1.plot(aIBiVals, R2_Array, color='r', linestyle='', marker='x', label=r'$R^{2}$')
    # # ax1.plot(aIBiVals, MSErr_Array, color='k', linestyle='', marker='x', label='Mean Squared Error')
    # # ax1.legend()
    # ax1.set_xlabel(r'$a_{IB}^{-1}$')
    # ax1.set_title(r'$R^{2}$' + ' Error')

    # # DISSIPATION CONSTANT SCALING

    # aIBVals = 1 / aIBiVals
    # # weakMask = aIBiVals <= -25
    # weakMask = np.abs((aIBVals / a0_th)) <= 110
    # aIBiValsWeak = aIBiVals[weakMask]
    # gammaValsWeak = gammaVals[weakMask]
    # aIBValsWeak = aIBVals[weakMask]

    # daIB = aIBValsWeak[1] - aIBValsWeak[0]
    # pguess = np.diff(gammaValsWeak, n=2)[0] / daIB

    # def p2Fit(aIB, param):
    #     return param * aIB**2

    # popt, cov = curve_fit(p2Fit, aIBValsWeak, gammaValsWeak, p0=pguess)
    # p2param = popt[0]
    # print(pguess, p2param)

    # p2FitVals = p2Fit(aIBVals, p2param)

    # fig2, ax2 = plt.subplots()
    # ax2.plot(-1 * aIBVals / a0_th, gammaVals, 'g-', label=r'$\gamma$')
    # ax2.plot(-1 * aIBVals / a0_th, p2FitVals, 'b--', label=r'$\gamma_{fit}=($' + '{:.2f}'.format(p2param) + r'$)a_{IB}^{2}$')
    # ax2.plot(-1 * aIBValsWeak[0] * np.ones(aIBVals.size) / a0_th, np.linspace(np.min(p2FitVals), np.max(p2FitVals), p2FitVals.size), 'y:', label=r'$a_{IB}=$' + '{:.1f}'.format(1 / aIBiValsWeak[0] / a0_th) + r' [$a_{0}$]')
    # ax2.plot(-1 * aIBValsWeak[-1] * np.ones(aIBVals.size) / a0_th, np.linspace(np.min(p2FitVals), np.max(p2FitVals), p2FitVals.size), 'm:', label=r'$a_{IB}=$' + '{:.1f}'.format(1 / aIBiValsWeak[-1] / a0_th) + r' [$a_{0}$]')

    # ax2.legend(loc=1)
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    # ax2.set_xlabel(r'$(-1)\cdot a_{IB}$ [$a_{0}$]')
    # ax2.set_title('Quadratic Fit to Dissipation Constant (Attractive Interactions)')
    # plt.show()

    # # AVERAGE ENERGY, FREQUENCY WINDOW + FIT PARAMETRS

    # x_ds = qds['XLab']
    # FTDiff_array = np.empty(aIBiVals.size)
    # AveEnergy_array = np.empty(aIBiVals.size)
    # AvePhKinEn_array = np.empty(aIBiVals.size)
    # AveImpKinEn_array = np.empty(aIBiVals.size)
    # for ind, aIBi in enumerate(aIBiVals):
    #     En = qds['Energy'].isel(aIBi=ind).values
    #     Pph = qds['Pph'].isel(aIBi=ind).values
    #     Ptot = qds['P'].isel(aIBi=ind).values
    #     PImp = Ptot - Pph
    #     AveEnergy_array[ind] = np.average(En)
    #     AvePhKinEn_array[ind] = np.average((Pph**2) / (2 * mB))
    #     AveImpKinEn_array[ind] = np.average((PImp**2) / (2 * mI))
    #     xVals = x_ds.sel(aIBi=aIBi).values
    #     x0 = xVals[0]
    #     dt = tVals[1] - tVals[0]
    #     FTVals = np.fft.fftshift(dt * np.fft.fft(xVals))
    #     FTAmp_Vals = np.abs(FTVals)
    #     fVals = np.fft.fftshift(np.fft.fftfreq(xVals.size) / dt)
    #     ind_fBEC = (np.abs(2 * np.pi * fVals - omega_BEC_osc)).argmin()
    #     ind_fImpTrap = (np.abs(2 * np.pi * fVals - omega_Imp_x)).argmin()
    #     FTAmp_BEC = FTAmp_Vals[ind_fBEC]
    #     FTAmp_ImpTrap = FTAmp_Vals[ind_fImpTrap]
    #     FTDiff_array[ind] = np.abs(FTAmp_BEC - FTAmp_ImpTrap)
    #     # print(fVals[ind_fBEC] * T_exp2th, fVals[ind_fImpTrap] * T_exp2th)
    #     # print(FTAmp_BEC, FTAmp_ImpTrap)

    # fig7, ax7 = plt.subplots()
    # ax7.plot(aIBiVals, FTDiff_array / np.max(FTDiff_array), color='orange', linestyle='-', label='Spectral Max Difference (Normalized)')

    # # ax7.plot(aIBiVals, AveEnergy_array / np.max(AveEnergy_array), label='Time Averaged Energy (' + r'$<H>=\frac{1}{T}\sum_{t=0}^{T}<\psi(t)|H|\psi(t)>\Delta t$' + ') Normalized to ' + r'$max(<H>)$')
    # # ax7.plot(aIBiVals, AvePhKinEn_array / np.max(AveEnergy_array), label='Time Averaged BEC Frame Phonon Kinetic Energy (' + r'$\frac{<P_{ph}>^{2}}{2m_{B}}$' + ') Normalized to ' + r'$max(<H>)$')
    # # ax7.plot(aIBiVals, AveImpKinEn_array / np.max(AveEnergy_array), label='Averaged BEC Frame Impurity Kinetic Energy (' + r'$\frac{<P_{I}>^{2}}{2m_{I}}$' + ') Normalized to ' + r'$max(<H>)$')
    # # ax7.plot(aIBiVals, (AveEnergy_array - AvePhKinEn_array - AveImpKinEn_array) / np.max(AveEnergy_array), label='Time Averaged Potential Energy (' + r'$<H>-\frac{<P_{ph}>^{2}}{2m_{B}}-\frac{<P_{I}>^{2}}{2m_{I}}$' + ') Normalized to ' + r'$max(<H>)$')
    # ax7.plot(aIBiVals, AvePhKinEn_array / np.max(AveEnergy_array), 'm-', label='Phonon Kinetic Energy (Normalized, Time-Averaged)')
    # ax7.plot(aIBiVals, AveImpKinEn_array / np.max(AveEnergy_array), 'y-', label='Impurity Kinetic Energy (Normalized, Time-Averaged)')

    # xiVals = msVals * gammaVals
    # rhoVals = gammaVals**2 - betaVals - omega_Imp_x**2
    # critdamp_ind = np.argwhere(np.sign(rhoVals) >= 0)[0][0]
    # ax7.plot(aIBiVals, xiVals, 'g:', label='Decay Constant ' + r'$\xi$')
    # ax7.plot(aIBiVals, gammaVals, 'g-', label='Mass Renormalized Decay Constant ' + r'$\gamma$')
    # # ax7.plot(aIBiVals[critdamp_ind] * np.ones(aIBiVals.size), np.linspace(0, np.max(msVals * xiVals), aIBiVals.size), 'y--', label='Oscillator Fit Critical Damping Threshold')

    # if toggleDict['PosScat'] == 'on':
    #     ax7.legend(loc=1)
    # else:
    #     ax7.legend(loc=2)
    # # ax7.legend()
    # ax7.set_xlabel(r'$a_{IB}^{-1}$')
    # # ax7.set_title('Dissipation Characterization')
    # ax7.set_title('Average Kinetic Energy Characterization')
    # ax7.set_ylim([0, 1.05])

    plt.show()
