import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy import interpolate
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation
from matplotlib.animation import writers
import os
import itertools
import Grid
import pf_dynamic_sph as pfs

if __name__ == "__main__":

    # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})
    # # mpegWriter = writers['ffmpeg'](fps=1, bitrate=1800)
    mpegWriter = writers['ffmpeg'](bitrate=1800)
    matplotlib.rcParams.update({'font.size': 12})
    labelsize = 13
    legendsize = 12

    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    (Lx, Ly, Lz) = (20, 20, 20)
    (dx, dy, dz) = (0.2, 0.2, 0.2)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    aBB = 0.013
    # tfin = 100

    # Toggle parameters

    toggleDict = {'PosScat': 'off', 'CS_Dyn': 'off', 'Polaron_Potential': 'off'}
    dParams = {'f_BEC_osc': 80, 'f_Imp_x': 150, 'a_osc': 0.7, 'X0': 0.0, 'P0': 0.4}

    # ---- SET OUTPUT DATA FOLDER ----

    datapath_Neg_NoPP_NoCS = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/BEC_osc/NegScat/NoPolPot_NoCSDyn/fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(aBB, NGridPoints_cart, dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])
    datapath_Neg_NoPP_CS = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/BEC_osc/NegScat/NoPolPot_CSDyn/fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(aBB, NGridPoints_cart, dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])
    datapath_Neg_PP_NoCS = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/BEC_osc/NegScat/PolPot_NoCSDyn/fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(aBB, NGridPoints_cart, dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])
    datapath_Neg_PP_CS = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/BEC_osc/NegScat/PolPot_CSDyn/fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(aBB, NGridPoints_cart, dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])
    datapath_Pos_NoPP_NoCS = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/BEC_osc/PosScat/NoPolPot_NoCSDyn/fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(aBB, NGridPoints_cart, dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])
    datapath_Pos_NoPP_CS = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/BEC_osc/PosScat/NoPolPot_CSDyn/fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(aBB, NGridPoints_cart, dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])
    datapath_Pos_PP_NoCS = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/BEC_osc/PosScat/PolPot_NoCSDyn/fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(aBB, NGridPoints_cart, dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])
    datapath_Pos_PP_CS = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/BEC_osc/PosScat/PolPot_CSDyn/fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(aBB, NGridPoints_cart, dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])

    figdatapath = '/Users/kis/Dropbox/Apps/Overleaf/Cherenkov Polaron Paper pt2/figures/figdump'

    # # # Analysis of Total Dataset

    expParams = pfs.Zw_expParams()
    L_exp2th, M_exp2th, T_exp2th = pfs.unitConv_exp2th(expParams['n0_BEC_scale'], expParams['mB'])
    RTF_BEC_X = expParams['RTF_BEC_X'] * L_exp2th
    omega_Imp_x = expParams['omega_Imp_x'] / T_exp2th
    a0_exp = 5.29e-11
    a0_th = a0_exp * L_exp2th

    f_BEC_osc = dParams['f_BEC_osc']
    f_Imp_x = dParams['f_Imp_x']
    a_osc = dParams['a_osc']
    X0 = dParams['X0']
    P0 = dParams['P0']

    qds_Neg_NoPP_NoCS = xr.open_dataset(datapath_Neg_NoPP_NoCS + '/LDA_Dataset.nc')
    qds_Neg_NoPP_CS = xr.open_dataset(datapath_Neg_NoPP_CS + '/LDA_Dataset.nc')
    qds_Neg_PP_NoCS = xr.open_dataset(datapath_Neg_PP_NoCS + '/LDA_Dataset.nc')
    qds_Neg_PP_CS = xr.open_dataset(datapath_Neg_PP_CS + '/LDA_Dataset.nc')
    qds_Pos_NoPP_NoCS = xr.open_dataset(datapath_Pos_NoPP_NoCS + '/LDA_Dataset.nc')
    qds_Pos_NoPP_CS = xr.open_dataset(datapath_Pos_NoPP_CS + '/LDA_Dataset.nc')
    qds_Pos_PP_NoCS = xr.open_dataset(datapath_Pos_PP_NoCS + '/LDA_Dataset.nc')
    qds_Pos_PP_CS = xr.open_dataset(datapath_Pos_PP_CS + '/LDA_Dataset.nc')

    oscParams = {'X0': X0, 'P0': P0, 'a_osc': a_osc}

    attrs = qds_Neg_PP_CS.attrs
    mI = attrs['mI']
    mB = attrs['mB']
    nu = attrs['nu']
    xi = attrs['xi']
    gBB = attrs['gBB']
    tscale = xi / nu
    tVals = qds_Neg_PP_CS['t'].values
    dt = tVals[1] - tVals[0]
    ts = tVals / tscale
    tscale_exp = tscale / T_exp2th
    omega_BEC_osc = attrs['omega_BEC_osc']
    xBEC = pfs.x_BEC_osc(tVals, omega_BEC_osc, RTF_BEC_X, a_osc)
    xB0 = xBEC[0]
    # print(omega_BEC_osc, (2 * np.pi / omega_BEC_osc), (1e-3 * T_exp2th * omega_BEC_osc / (2 * np.pi)), qds_nosc.attrs['omega_BEC_osc'])
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    print('xi/c (exp in ms): {0}'.format(1e3 * tscale_exp))
    print('mI*c: {0}'.format(mI * nu))
    print(omega_Imp_x)
    print(tVals)

    xL_bareImp = (xBEC[0] + X0) * np.cos(omega_Imp_x * tVals) + (P0 / (omega_Imp_x * mI)) * np.sin(omega_Imp_x * tVals)  # gives the lab frame trajectory time trace of a bare impurity (only subject to the impurity trap) that starts at the same position w.r.t. the BEC as the polaron and has the same initial total momentum
    vL_bareImp = np.gradient(xL_bareImp, tVals)
    aL_bareImp = np.gradient(np.gradient(xL_bareImp, tVals), tVals)

    # print(1 / aIBiVals / a0_th)

    # # # # FIG 5 - 2D OSCILLATION FREQUENCY (HOMOGENEOUS BEC)

    # a0ylim = 6000

    # qds_List = [qds_Neg_NoPP_NoCS, qds_Pos_NoPP_NoCS, qds_Neg_NoPP_CS, qds_Pos_NoPP_CS]; signList = ['Neg', 'Pos', 'Neg', 'Pos']
    # freqda_List = []
    # aIBVals_List = []

    # for qds in qds_List:
    #     aIBiVals = qds['aIBi'].values
    #     x_ds = qds['XLab']
    #     tVals = qds['t'].values
    #     dt = tVals[1] - tVals[0]
    #     fVals = np.fft.fftshift(np.fft.fftfreq(tVals.size) / dt)
    #     Nf = fVals.size
    #     print('df: {0}'.format((fVals[1] - fVals[0]) * T_exp2th))
    #     # aIBiVals = aIBiVals[2:]
    #     aIBVals = (1 / aIBiVals) / a0_th
    #     freq_da = xr.DataArray(np.full((fVals.size, len(aIBiVals)), np.nan, dtype=float), coords=[fVals, aIBiVals], dims=['f', 'aIBi'])
    #     maxph = 0
    #     for ind, aIBi in enumerate(aIBiVals):
    #         xVals = x_ds.sel(aIBi=aIBi).values
    #         x0 = xVals[0]
    #         dt = tVals[1] - tVals[0]
    #         # FTVals = np.fft.fftshift(dt * np.fft.fft(xVals))
    #         FTVals = np.fft.fftshift(dt * np.fft.fft(np.fft.fftshift(xVals)))
    #         fVals = np.fft.fftshift(np.fft.fftfreq(xVals.size) / dt)
    #         absFTVals = np.abs(FTVals)

    #         freq_da.sel(aIBi=aIBi)[:] = absFTVals
    #         if np.max(absFTVals) > maxph:
    #             maxph = np.max(absFTVals)
    #     print(maxph)
    #     freqda_List.append(freq_da)
    #     aIBVals_List.append(aIBVals)

    # # vmax = maxph
    # vmax = 100000
    # # vmax = 200000

    # fig5, axes = plt.subplots(nrows=2, ncols=2)
    # axList = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    # for ind, freq_da in enumerate(freqda_List):
    #     ax = axList[ind]
    #     absFT_interp, f_interp, aIBi_interp = pfs.xinterp2D(freq_da, 'f', 'aIBi', 5)
    #     # absFT_interp = freq_da.values; f_interp = freq_da.coords['f'].values; aIBi_interp = freq_da.coords['aIBi'].values
    #     aIBVals = aIBVals_List[ind]
    #     quadF = ax.pcolormesh(f_interp * T_exp2th, (1 / aIBi_interp) / a0_th, absFT_interp, vmin=0, vmax=vmax)
    #     ax.set_ylabel(r'$a_{IB}$ [$a_{0}$]', fontsize=labelsize)
    #     ax.plot(omega_BEC_osc * T_exp2th / (2 * np.pi) * np.ones(aIBVals.size), aIBVals, 'k:', lw=3, label='BEC Oscillation Frequency')
    #     ax.plot(omega_Imp_x * T_exp2th / (2 * np.pi) * np.ones(aIBVals.size), aIBVals, color='orange', linestyle=':', marker='', lw=3, label='Impurity Trap Frequency')
    #     if signList[ind] == 'Neg':
    #         ax.set_ylim([-1 * a0ylim, np.max(aIBVals)])
    #     elif signList[ind] == 'Pos':
    #         ax.set_ylim([a0ylim, np.min(aIBVals)])
    #     ax.set_xlabel('f (Hz)', fontsize=labelsize)
    #     ax.set_xlim([0, 400])
    #     ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # cbar_ax = fig5.add_axes([0.9, 0.2, 0.02, 0.7])
    # fig5.colorbar(quadF, cax=cbar_ax, extend='max')
    # cbar_ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # # fig5.colorbar(quadF, cax=cbar_ax, extend='max', format=FormatStrFormatter('%.f'))
    # # fig5.colorbar(quadF, cax=cbar_ax, extend='max', format='%.0e')
    # handles, labels = axList[0].get_legend_handles_labels()
    # fig5.legend(handles, labels, ncol=2, loc='lower center', fontsize=legendsize)
    # fig5.subplots_adjust(bottom=0.17, top=0.95, right=0.85, hspace=0.45, wspace=0.45)
    # fig5.set_size_inches(7.8, 6.0)

    # fig5.text(0.05, 0.96, '(a)', fontsize=labelsize)
    # fig5.text(0.05, 0.51, '(b)', fontsize=labelsize)
    # fig5.text(0.47, 0.96, '(c)', fontsize=labelsize)
    # fig5.text(0.47, 0.51, '(d)', fontsize=labelsize)

    # # fig5.savefig(figdatapath + '/Fig5.pdf')
    # fig5.savefig(figdatapath + '/Fig5.jpg', quality=20)

    # # # # FIG 6 - 2D OSCILLATION FREQUENCY (INHOMOGENEOUS BEC)

    # a0ylim = 6000

    # qds_List = [qds_Neg_PP_NoCS, qds_Pos_PP_NoCS, qds_Neg_PP_CS, qds_Pos_PP_CS]; signList = ['Neg', 'Pos', 'Neg', 'Pos']
    # freqda_List = []
    # aIBVals_List = []

    # for qds in qds_List:
    #     aIBiVals = qds['aIBi'].values
    #     x_ds = qds['XLab']
    #     tVals = qds['t'].values
    #     dt = tVals[1] - tVals[0]
    #     fVals = np.fft.fftshift(np.fft.fftfreq(tVals.size) / dt)
    #     Nf = fVals.size
    #     print('df: {0}'.format((fVals[1] - fVals[0]) * T_exp2th))
    #     # aIBiVals = aIBiVals[2:]
    #     aIBVals = (1 / aIBiVals) / a0_th
    #     freq_da = xr.DataArray(np.full((fVals.size, len(aIBiVals)), np.nan, dtype=float), coords=[fVals, aIBiVals], dims=['f', 'aIBi'])
    #     maxph = 0
    #     for ind, aIBi in enumerate(aIBiVals):
    #         xVals = x_ds.sel(aIBi=aIBi).values
    #         x0 = xVals[0]
    #         dt = tVals[1] - tVals[0]
    #         # FTVals = np.fft.fftshift(dt * np.fft.fft(xVals))
    #         FTVals = np.fft.fftshift(dt * np.fft.fft(np.fft.fftshift(xVals)))
    #         fVals = np.fft.fftshift(np.fft.fftfreq(xVals.size) / dt)
    #         absFTVals = np.abs(FTVals)

    #         freq_da.sel(aIBi=aIBi)[:] = absFTVals
    #         if np.max(absFTVals) > maxph:
    #             maxph = np.max(absFTVals)
    #     print(maxph)
    #     freqda_List.append(freq_da)
    #     aIBVals_List.append(aIBVals)

    # # vmax = maxph
    # vmax = 100000
    # # vmax = 200000

    # fig6, axes = plt.subplots(nrows=2, ncols=2)
    # axList = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    # for ind, freq_da in enumerate(freqda_List):
    #     ax = axList[ind]
    #     absFT_interp, f_interp, aIBi_interp = pfs.xinterp2D(freq_da, 'f', 'aIBi', 5)
    #     # absFT_interp = freq_da.values; f_interp = freq_da.coords['f'].values; aIBi_interp = freq_da.coords['aIBi'].values
    #     aIBVals = aIBVals_List[ind]
    #     quadF = ax.pcolormesh(f_interp * T_exp2th, (1 / aIBi_interp) / a0_th, absFT_interp, vmin=0, vmax=vmax)
    #     ax.set_ylabel(r'$a_{IB}$ [$a_{0}$]', fontsize=labelsize)
    #     ax.plot(omega_BEC_osc * T_exp2th / (2 * np.pi) * np.ones(aIBVals.size), aIBVals, 'k:', lw=3, label='BEC Oscillation Frequency')
    #     ax.plot(omega_Imp_x * T_exp2th / (2 * np.pi) * np.ones(aIBVals.size), aIBVals, color='orange', linestyle=':', marker='', lw=3, label='Impurity Trap Frequency')
    #     if signList[ind] == 'Neg':
    #         ax.set_ylim([-1 * a0ylim, np.max(aIBVals)])
    #     elif signList[ind] == 'Pos':
    #         ax.set_ylim([a0ylim, np.min(aIBVals)])
    #     ax.set_xlabel('f (Hz)', fontsize=labelsize)
    #     ax.set_xlim([0, 400])
    #     ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # cbar_ax = fig6.add_axes([0.9, 0.2, 0.02, 0.7])
    # fig6.colorbar(quadF, cax=cbar_ax, extend='max')
    # cbar_ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # # fig6.colorbar(quadF, cax=cbar_ax, extend='max', format=FormatStrFormatter('%.f'))
    # # fig6.colorbar(quadF, cax=cbar_ax, extend='max', format='%.0e')
    # handles, labels = axList[0].get_legend_handles_labels()
    # fig6.legend(handles, labels, ncol=2, loc='lower center', fontsize=legendsize)
    # fig6.subplots_adjust(bottom=0.17, top=0.95, right=0.85, hspace=0.45, wspace=0.45)
    # fig6.set_size_inches(7.8, 6.0)

    # fig6.text(0.05, 0.96, '(a)', fontsize=labelsize)
    # fig6.text(0.05, 0.51, '(b)', fontsize=labelsize)
    # fig6.text(0.47, 0.96, '(c)', fontsize=labelsize)
    # fig6.text(0.47, 0.51, '(d)', fontsize=labelsize)

    # # fig6.savefig(figdatapath + '/Fig6.pdf')
    # fig6.savefig(figdatapath + '/Fig6.jpg', quality=20)

    # # POSITION VS TIME (LAB FRAME)

    # aIBdiva0_des = 400
    # aIBi_des = 1 / (aIBdiva0_des * a0_exp * L_exp2th)
    # xds = qds['XLab'].sel(aIBi=aIBi_des, method='nearest')
    # xDat = xds.values
    # aIBi = xds['aIBi'].values

    # fig1, ax1 = plt.subplots()
    # # ax1.plot(ts, 1e6 * xDat / L_exp2th, color='g', linestyle='-', label=r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBi))
    # ax1.plot(ts, 1e6 * xDat / L_exp2th, color='g', linestyle='-', label=r'$a_{IB}=$' + '{:.2f}'.format(1 / aIBi / a0_th) + r'[$a_{0}$]')
    # ax1.plot(ts, 1e6 * xBEC / L_exp2th, 'k:', label='BEC Peak Position')
    # ax1.plot(ts, 1e6 * xBEC[0] * np.cos(omega_Imp_x * tVals) / L_exp2th, color='orange', linestyle=':', marker='', label='Impurity Trap Frequency')
    # ax1.legend(loc=2)
    # ax1.set_ylabel(r'$<X> (\mu m)$')
    # ax1.set_xlabel(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]'.format(1e3 * tscale_exp))
    # ax1.set_title('Impurity Trajectory (Lab Frame)')

    # # # VELOCITY VS TIME (LAB FRAME)

    # v_BEC_osc = np.diff(xBEC) / dt
    # v_ImpTrap = -1 * xBEC[0] * omega_Imp_x * np.sin(omega_Imp_x * v_ds['t'].values)

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

    # # POSITION VS TIME ANIMATION (LAB FRAME)

    # inverseScat = False

    # x_ds = qds['XLab']
    # xI_array = np.empty(aIBiVals.size, dtype=np.object)
    # xmin = 1e20
    # xmax = -1e20
    # for ind, aIBi in enumerate(aIBiVals):
    #     xI_array[ind] = x_ds.isel(aIBi=ind).values
    #     xImax = np.max(xI_array[ind])
    #     xImin = np.min(xI_array[ind])
    #     if xmax < xImax:
    #         xmax = xImax
    #     if xmin > xImin:
    #         xmin = xImin

    # fig6, ax6 = plt.subplots()
    # if toggleDict['PosScat'] == 'on':
    #     # ax6.set_ylim([xmin, xmax])
    #     ax6.set_ylim([-50, 50])
    # else:
    #     edge = 1.5 * 1e6 * np.max(xBEC) / L_exp2th
    #     ax6.set_ylim([-1 * edge, edge])
    # ax6.plot(ts, 1e6 * xBEC / L_exp2th, 'k:', label='BEC Peak Position')
    # # ax6.plot(ts, 1e6 * xBEC[0] * np.cos(omega_Imp_x * tVals) / L_exp2th, color='orange', linestyle=':', marker='', label='Impurity Trap Frequency')
    # ax6.plot(ts, 1e6 * xL_bareImp / L_exp2th, color='orange', linestyle=':', marker='', label='Bare Impurity')

    # # xL_bareImp_mod = (xBEC[0] + X0) * np.cos(omega_Imp_x * tVals) + (P0 / (omega_Imp_x * mI)) * np.sin(omega_Imp_x * tVals)
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

    # # POSITION VS TIME ANIMATION (BEC FRAME)

    # inverseScat = False

    # x_ds = qds['X']
    # xI_array = np.empty(aIBiVals.size, dtype=np.object)
    # xmin = 1e20
    # xmax = -1e20
    # for ind, aIBi in enumerate(aIBiVals):
    #     xI_array[ind] = x_ds.isel(aIBi=ind).values
    #     xImax = np.max(xI_array[ind])
    #     xImin = np.min(xI_array[ind])
    #     if xmax < xImax:
    #         xmax = xImax
    #     if xmin > xImin:
    #         xmin = xImin

    # fig6, ax6 = plt.subplots()
    # if toggleDict['PosScat'] == 'on':
    #     # ax6.set_ylim([xmin, xmax])
    #     ax6.set_ylim([-50, 50])
    # else:
    #     edge = 2.5 * 1e6 * np.max(xBEC) / L_exp2th
    #     ax6.set_ylim([-1 * edge, edge])
    # ax6.plot(ts, 1e6 * RTF_BEC_X * np.ones(ts.size) / L_exp2th, 'k:', label='BEC TF Radius')
    # ax6.plot(ts, -1 * 1e6 * RTF_BEC_X * np.ones(ts.size) / L_exp2th, 'k:')
    # # ax6.plot(ts, 1e6 * (xBEC[0] * np.cos(omega_Imp_x * tVals) - xBEC) / L_exp2th, color='orange', linestyle=':', marker='', label='Impurity Trap Frequency')
    # ax6.plot(ts, 1e6 * (xL_bareImp - xBEC) / L_exp2th, color='orange', linestyle=':', marker='', label='Bare Impurity')
    # curve = ax6.plot(ts, 1e6 * xI_array[0] / L_exp2th, color='g', lw=2, label='')[0]
    # if inverseScat is True:
    #     aIBi_text = ax6.text(0.8, 0.9, r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[0]), transform=ax6.transAxes, color='r')
    # else:
    #     aIB_text = ax6.text(0.75, 0.9, r'$a_{IB}=$' + '{:d}'.format(((1 / aIBiVals[0]) / a0_th).astype(int)) + r' [$a_{0}$]', transform=ax6.transAxes, color='r')

    # ax6.legend(loc=2)
    # ax6.set_ylabel(r'$<X> (\mu m)$')
    # ax6.set_xlabel(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]'.format(1e3 * tscale_exp))
    # ax6.set_title('Impurity Trajectory (BEC Frame)')

    # def animate_pos(i):
    #     curve.set_ydata(1e6 * xI_array[i] / L_exp2th)
    #     if inverseScat is True:
    #         aIBi_text.set_text(r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[i]))
    #     else:
    #         aIB_text.set_text(r'$a_{IB}=$' + '{:d}'.format(((1 / aIBiVals[i]) / a0_th).astype(int)) + r' [$a_{0}$]')
    # anim_pB = FuncAnimation(fig6, animate_pos, interval=50, frames=range(aIBiVals.size), repeat=False)

    # anim_pB_filename = '/TrajAnim_BECframe_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.mp4'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # if toggleDict['CS_Dyn'] == 'off':
    #     anim_pB_filename = '/NoCSdyn_' + anim_pB_filename[1:]
    # # anim_pB.save(animpath + anim_pB_filename, writer=mpegWriter)

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

    # ax6.plot(ts, (vL_bareImp - np.gradient(xBEC, tVals)) * (1e3 * T_exp2th / L_exp2th), color='orange', linestyle=':', marker='', label='Bare Impurity')
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
    # anim_v_filename = '/VelAnim_BECframe_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.mp4'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # if toggleDict['CS_Dyn'] == 'off':
    #     anim_v_filename = '/NoCSdyn_' + anim_v_filename[1:]
    # # anim_v.save(animpath + anim_v_filename, writer=mpegWriter)

    # # ACCELERATION VS TIME ANIMATION (LAB FRAME)

    # inverseScat = False

    # aI_array = np.empty(aIBiVals.size, dtype=np.object)
    # xI_array = np.empty(aIBiVals.size, dtype=np.object)
    # Pph0_array = np.zeros(aIBiVals.size)

    # for ind, aIBi in enumerate(aIBiVals):
    #     aI_array[ind] = np.gradient(np.gradient(qds['XLab'].sel(aIBi=aIBi).values, tVals), tVals)
    #     xI_array[ind] = qds['XLab'].sel(aIBi=aIBi).values
    #     Pph0_array[ind] = qds['Pph'].sel(aIBi=aIBi).values[0]

    # fig6, ax6 = plt.subplots()
    # curve = ax6.plot(ts, aI_array[0] * (T_exp2th * T_exp2th / L_exp2th), color='b', linestyle='-', lw=2, label='')[0]
    # if inverseScat is True:
    #     aIBi_text = ax6.text(0.8, 0.9, r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[0]), transform=ax6.transAxes, color='r')
    # else:
    #     aIB_text = ax6.text(0.75, 0.9, r'$a_{IB}=$' + '{:d}'.format(((1 / aIBiVals[0]) / a0_th).astype(int)) + r' [$a_{0}$]', transform=ax6.transAxes, color='r')

    # ax6.plot(ts, aL_bareImp * (T_exp2th * T_exp2th / L_exp2th), color='orange', linestyle=':', marker='', label='Bare Impurity')
    # FIT_curve = ax6.plot(ts, (1 / mI) * pfs.F_Imp_trap(xI_array[0], omega_Imp_x, mI) * (T_exp2th * T_exp2th / L_exp2th), color='magenta', linestyle='dashed', marker='', label='Impurity Trap Force')[0]
    # # xL_bareImp = (xBEC[0] + X0) * np.cos(omega_Imp_x * tVals) + ((P0 - Pph0_array[0]) / (omega_Imp_x * mI)) * np.sin(omega_Imp_x * tVals)  # gives the lab frame trajectory time trace of a bare impurity (only subject to the impurity trap) that starts at the same position w.r.t. the BEC as the polaron and has the same initial total momentum
    # # bImod = ax6.plot(ts, (-(omega_Imp_x**2) * (xL_bareImp - (1 / mI) * Pph0_array[0] * tVals)**2) * (T_exp2th * T_exp2th / L_exp2th), color='orange', linestyle=':', marker='', label='BI Mod')[0]

    # ax6.legend(loc=2)
    # ax6.set_ylabel(r'$a=\frac{d^{2}<X>}{dt^{2}} (\frac{\mu m}{ms^{2}})$')
    # ax6.set_xlabel(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]'.format(1e3 * tscale_exp))
    # ax6.set_title('Impurity Acceleration (Lab Frame)')
    # # ax6.set_ylim([-450, 450])
    # ax6.set_ylim([-25, 25])

    # def animate_pos(i):
    #     if i >= aIBiVals.size:
    #         return
    #     curve.set_ydata(aI_array[i] * (T_exp2th * T_exp2th / L_exp2th))
    #     FIT_curve.set_ydata((1 / mI) * pfs.F_Imp_trap(xI_array[i], omega_Imp_x, mI) * (T_exp2th * T_exp2th / L_exp2th))
    #     # xL_bareImp = (xBEC[0] + X0) * np.cos(omega_Imp_x * tVals) + ((P0 - Pph0_array[i]) / (omega_Imp_x * mI)) * np.sin(omega_Imp_x * tVals)  # gives the lab frame trajectory time trace of a bare impurity (only subject to the impurity trap) that starts at the same position w.r.t. the BEC as the polaron and has the same initial total momentum
    #     # bImod.set_ydata((-(omega_Imp_x**2) * (xL_bareImp - (1 / mI) * Pph0_array[i] * tVals)**2) * (T_exp2th * T_exp2th / L_exp2th))

    #     if inverseScat is True:
    #         aIBi_text.set_text(r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[i]))
    #     else:
    #         aIB_text.set_text(r'$a_{IB}=$' + '{:d}'.format(((1 / aIBiVals[i]) / a0_th).astype(int)) + r' [$a_{0}$]')

    # anim_a = FuncAnimation(fig6, animate_pos, interval=50, frames=range(aIBiVals.size), repeat=False)
    # anim_a_filename = '/AccelAnim_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.mp4'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # if toggleDict['CS_Dyn'] == 'off':
    #     anim_a_filename = '/NoCSdyn_' + anim_a_filename[1:]
    # # anim_a.save(animpath + anim_a_filename, writer=mpegWriter)

    ##############################################################################################################################
    # FIT IN THE BEC FRAME
    ##############################################################################################################################

    # # ODE FIT TO POSITION (TRAJECTORY) - FIT Gamma & Beta

    # GammaFix = False

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
    # x0Vals = np.empty(aIBiVals.size)
    # v0Vals = np.empty(aIBiVals.size)
    # for ind, aIBi in enumerate(aIBiVals):
    #     # if ind != 10:
    #     #     continue
    #     xVals = x_ds.sel(aIBi=aIBi).values
    #     vVals = np.gradient(xVals, tVals)
    #     x0 = xVals[0]
    #     v0 = vVals[0]
    #     x0Vals[ind] = x0; v0Vals[ind] = v0
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

    # # ODE FIT TO POSITION (TRAJECTORY) - FIT Gamma & FIX Beta

    # GammaFix = True

    # NGridPoints_desired = (1 + 2 * Lx / dx) * (1 + 2 * Lz / dz); Ntheta = 50
    # Nk = np.ceil(NGridPoints_desired / Ntheta)
    # theta_max = np.pi; thetaArray, dtheta = np.linspace(0, theta_max, Ntheta, retstep=True)
    # k_max = ((2 * np.pi / dx)**3 / (4 * np.pi / 3))**(1 / 3); k_min = 1e-5
    # kArray, dk = np.linspace(k_min, k_max, Nk, retstep=True)
    # kgrid = Grid.Grid("SPHERICAL_2D"); kgrid.initArray_premade('k', kArray); kgrid.initArray_premade('th', thetaArray)

    # n0_TF = expParams['n0_TF'] / (L_exp2th**3)
    # n0_thermal = expParams['n0_thermal'] / (L_exp2th**3)
    # RTF_BEC_X = expParams['RTF_BEC_X'] * L_exp2th; RTF_BEC_Y = expParams['RTF_BEC_Y'] * L_exp2th; RTF_BEC_Z = expParams['RTF_BEC_Z'] * L_exp2th
    # RG_BEC_X = expParams['RG_BEC_X'] * L_exp2th; RG_BEC_Y = expParams['RG_BEC_Y'] * L_exp2th; RG_BEC_Z = expParams['RG_BEC_Z'] * L_exp2th
    # omega_BEC_osc = expParams['omega_BEC_osc'] / T_exp2th
    # omega_Imp_x = expParams['omega_Imp_x'] / T_exp2th
    # trapParams = {'n0_TF_BEC': n0_TF, 'RTF_BEC_X': RTF_BEC_X, 'RTF_BEC_Y': RTF_BEC_Y, 'RTF_BEC_Z': RTF_BEC_Z, 'n0_thermal_BEC': n0_thermal, 'RG_BEC_X': RG_BEC_X, 'RG_BEC_Y': RG_BEC_Y, 'RG_BEC_Z': RG_BEC_Z,
    #               'omega_Imp_x': omega_Imp_x, 'omega_BEC_osc': omega_BEC_osc, 'X0': oscParams['X0'], 'P0': oscParams['P0'], 'a_osc': oscParams['a_osc']}

    # n0 = expParams['n0_BEC'] / (L_exp2th**3)  # should ~ 1
    # sParams = [mI, mB, n0, gBB]

    # X_Vals = np.linspace(-1 * trapParams['RTF_BEC_X'] * 0.99, trapParams['RTF_BEC_X'] * 0.99, 100)

    # aVals_Fixed = np.empty(aIBiVals.size)
    # for ind, aIBi in enumerate(aIBiVals):
    #     cParams = {'aIBi': aIBi}
    #     E_Pol_tck = pfs.V_Pol_interp(kgrid, X_Vals, cParams, sParams, trapParams)
    #     aVals_Fixed[ind] = interpolate.splev(0, E_Pol_tck, der=2)

    # betaVals_Fixed = aVals_Fixed / mI

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
    # x0Vals = np.empty(aIBiVals.size)
    # v0Vals = np.empty(aIBiVals.size)
    # for ind, aIBi in enumerate(aIBiVals):
    #     # if ind != 10:
    #     #     continue
    #     xVals = x_ds.sel(aIBi=aIBi).values
    #     vVals = np.gradient(xVals, tVals)
    #     x0 = xVals[0]
    #     v0 = vVals[0]
    #     x0Vals[ind] = x0; v0Vals[ind] = v0
    #     # v0 = (qds['P'].sel(aIBi=aIBi).isel(t=0).values - qds['Pph'].sel(aIBi=aIBi).isel(t=0).values) / mI
    #     y0 = [x0, v0]
    #     data = np.concatenate((xVals, vVals))
    #     if ind == 0:
    #         p0 = [1e-3]
    #         lowerbound = [0]
    #         upperbound = [np.inf]

    #     else:
    #         p0 = [gammaVals[ind - 1]]
    #         # lowerbound = [gammaVals[ind - 1], 0]
    #         lowerbound = [0]
    #         upperbound = [np.inf]
    #     popt, cov = curve_fit(lambda t, gamma: yint(t, gamma, betaVals_Fixed[ind], y0), tVals, data, p0=p0, bounds=(lowerbound, upperbound))
    #     gopt = popt[0]
    #     y0Vals[ind] = y0; gammaVals[ind] = gopt; betaVals[ind] = betaVals_Fixed[ind]
    #     gVals[ind], phiVals[ind] = gphiVals(gopt, betaVals[ind], omega_Imp_x, omega_BEC_osc, xB0)

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

    # # ANALYTICAL SOLUTION FIT TO POSITION (TRAJECTORY) ***(OLD)

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
    # if GammaFix is True:
    #     anim_fit_filename = '/GammaOnly_' + anim_fit_filename[1::]
    # if toggleDict['CS_Dyn'] == 'off':
    #     anim_fit_filename = '/NoCSdyn_' + anim_fit_filename[1:]
    # # anim_fit.save(animpath + anim_fit_filename, writer=mpegWriter)
    # # print(anim_fit_filename)

    # # PLOT PARAMETERS FIXED

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

    #     w2_Un = omega_Imp_x**2 + betaVals - gammaVals**2
    #     freq_beta_Hz = (np.sqrt(betaVals) / (2 * np.pi)) * T_exp2th
    #     freq_MF_Hz = (np.sqrt(omega_Imp_x**2 + betaVals) / (2 * np.pi)) * T_exp2th
    #     freq_wUn_Hz = (np.sqrt(w2_Un) / (2 * np.pi)) * T_exp2th

    #     dVals = (omega_BEC_osc**2 - omega_Imp_x**2) * xB0 / np.sqrt(4 * gammaVals**2 * omega_BEC_osc**2 + (omega_Imp_x**2 + betaVals - omega_BEC_osc**2)**2)
    #     deltaVals = np.arctan2((2 * gammaVals * omega_BEC_osc), (omega_BEC_osc**2 - omega_Imp_x**2 - betaVals))
    #     gVals = np.sqrt(dVals**2 + xB0**2 + 2 * dVals * xB0 * np.cos(deltaVals))
    #     c1Un = np.sqrt((x0Vals - dVals * np.cos(deltaVals))**2 + ((v0Vals + omega_BEC_osc * dVals * np.sin(deltaVals))**2) / w2_Un)
    #     c2Un = -1 * np.arctan2((v0Vals + omega_BEC_osc * dVals * np.sin(deltaVals)), (x0Vals - dVals * np.cos(deltaVals) * np.sqrt(w2_Un)))

    #     fig3, ax3 = plt.subplots()
    #     ax3.plot(aIBVals, rhoVals, 'm--')
    #     ax3.set_title('Oscillation Fit Damping Ratio')
    #     ax3.set_ylabel(r'$\frac{\gamma}{\sqrt{\omega_{0}^{2}+\beta}}$')
    #     ax3.set_xlabel(r'$a_{IB}$ [$a_{0}$]')
    #     # ax3.set_xlim([-1 * a0xlim, np.max(aIBVals)])

    #     fig4, ax4 = plt.subplots()
    #     ax4.plot(aIBVals, freq_MF_Hz, color='m', linestyle='', marker='x', label=r'($\sqrt{\omega_{0}^2+\beta}$)')
    #     ax4.plot(aIBVals, freq_wUn_Hz, color='g', linestyle='', marker='x', label=r'($\sqrt{\omega_{0}^2+\beta-\gamma^2}$)')
    #     # ax4.plot(aIBVals, c2Un, color='b', linestyle='', marker='x', label='Phase ' + r'$(c_{1,Un}$)')
    #     # ax4.plot(aIBVals, freq_beta_Hz, color='b', linestyle='', marker='o', markerfacecolor='none', label=r'$\sqrt(\beta)$')
    #     ax4.legend()
    #     ax4.set_xlabel(r'$a_{IB}$ [$a_{0}$]')
    #     ax4.set_ylabel('Frequency (Hz)')
    #     ax4.set_title('Underdamped Frequency Parameters')

    #     fig5, ax5 = plt.subplots()
    #     ax5.plot(aIBVals, c1Un * 1e6 / L_exp2th, color='m', linestyle='', marker='x', label=r'$c_{1,Un}$')
    #     ax5.plot(aIBVals, gVals * 1e6 / L_exp2th, color='g', linestyle='', marker='x', label=r'$g$')
    #     ax5.legend()
    #     ax5.set_xlabel(r'$a_{IB}$ [$a_{0}$]')
    #     ax5.set_ylabel('Amplitude ' + r'($\mu$m)')
    #     ax5.set_title('Underdamped Amplitude Parameters')

    #     fig7, ax7 = plt.subplots()
    #     ax7.plot(aIBVals, dVals * 1e6 / L_exp2th, color='m', linestyle='', marker='x', label=r'$d$')
    #     ax7.plot(aIBVals, np.ones(dVals.size) * xB0 * 1e6 / L_exp2th, 'g--', label=r'$B_{0}$')
    #     ax7.legend()
    #     ax7.set_xlabel(r'$a_{IB}$ [$a_{0}$]')
    #     ax7.set_ylabel(r'($\mu$m)')
    #     ax7.set_title('Underdamped Amplitude-related Parameters')

    #     fig6, ax6 = plt.subplots()
    #     ax6.plot(aIBVals, gammaVals**2, color='g', linestyle='', marker='x', label=r'$\gamma^{2}$')
    #     ax6.plot(aIBVals, betaVals, color='m', linestyle='', marker='x', label=r'$\beta$')
    #     ax6.legend()
    #     ax6.set_xlabel(r'$a_{IB}$ [$a_{0}$]')
    #     ax6.set_title('Oscillation Fit Parameters')

    #     print(xB0, a_osc * RTF_BEC_X)
    #     print(aIBVals)
    #     print(gammaVals)
    #     print(betaVals)
    #     print(rhoVals)
    #     print(omega_Imp_x, omega_BEC_osc)
    #     print(freq_MF_Hz)

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
    # # weakMask = np.abs((aIBVals / a0_th)) <= 110
    # weakMask = np.abs((aIBVals / a0_th)) <= 50
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

    # plt.show()
