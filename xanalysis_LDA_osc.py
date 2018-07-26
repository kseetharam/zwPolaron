import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import writers
import os
import itertools
import Grid
import pf_dynamic_sph as pfs

if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})
    # mpegWriter = writers['ffmpeg'](fps=20, bitrate=1800)

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
            animpath = '/home/kis/Dropbox/ZwierleinExp/figures/aBB={:.3f}/BEC_osc'.format(aBB)
        elif toggleDict['Location'] == 'work':
            datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/BEC_osc/fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(aBB, NGridPoints_cart, dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])
            animpath = '/media/kis/Storage/Dropbox/ZwierleinExp/figures/aBB={:.3f}/BEC_osc'.format(aBB)
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
    xBEC = pfs.x_BEC_osc(tVals, omega_BEC_osc, RTF_BEC_X, a_osc)
    xB0 = xBEC[0]
    x_ds = qds['XLab']
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
    #     ax1.plot(ts, 1e6 * x_ds.sel(aIBi=aIBi).values / L_exp2th, color=colors[ind % 7], linestyle='-', label=r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBi))
    # ax1.plot(ts, xBEC, 'k:', label='BEC Peak Position')
    # ax1.plot(ts, xBEC[0] * np.cos(omega_Imp_x * tVals), color='orange', linestyle=':', marker='', label='Impurity Trap Frequency')
    # ax1.legend()
    # ax1.set_ylabel(r'$<X> (\mu m)$')
    # ax1.set_xlabel(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]'.format(1e3 * tscale_exp))
    # ax1.set_title('Impurity Trajectory (Lab Frame)')

    # # POSITION VS TIME ANIMATION

    # x_ds = qds['XLab']
    # xI_array = np.empty(aIBiVals.size, dtype=np.object)
    # for ind, aIBi in enumerate(aIBiVals):
    #     xI_array[ind] = x_ds.isel(aIBi=ind).values

    # fig6, ax6 = plt.subplots()
    # ax6.plot(ts, xBEC, 'k:', label='BEC Peak Position')
    # ax6.plot(ts, xBEC[0] * np.cos(omega_Imp_x * tVals), color='orange', linestyle=':', marker='', label='Impurity Trap Frequency')
    # curve = ax6.plot(ts, 1e6 * xI_array[0] / L_exp2th, color='g', lw=2, label='')[0]
    # aIBi_text = ax6.text(0.8, 0.9, r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[0]), transform=ax6.transAxes, color='r')

    # ax6.legend(loc=2)
    # ax6.set_ylabel(r'$<X> (\mu m)$')
    # ax6.set_xlabel(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]'.format(1e3 * tscale_exp))
    # ax6.set_title('Impurity Trajectory (Lab Frame)')

    # def animate_pos(i):
    #     if i >= aIBiVals.size:
    #         return
    #     curve.set_ydata(1e6 * xI_array[i] / L_exp2th)
    #     aIBi_text.set_text(r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[i]))

    # anim_p = FuncAnimation(fig6, animate_pos, interval=50, frames=range(ts.size))
    # # anim_p_filename = '/TrajAnim_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.gif'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # # anim_p.save(animpath + anim_p_filename, writer='imagemagick')
    # anim_p_filename = '/TrajAnim_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.mp4'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # anim_p.save(animpath + anim_p_filename, writer=mpegWriter)

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
    # anim_freq.save(animpath + anim_freq_filename, writer=mpegWriter)

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

    # # v_ds = (qds['X'].diff('t') / dt).rename('v')
    # # ts = v_ds['t'].values / tscale
    # cBEC = nu * np.ones(ts.size)
    # fig4, ax4 = plt.subplots()
    # for ind, aIBi in enumerate(aIBiVals):
    #     if aIBi in aIBi_noPlotList:
    #         continue
    #     ax4.plot(ts, np.gradient(qds['X'].sel(aIBi=aIBi).values, tVals) * (1e3 * T_exp2th / L_exp2th), color=colors[ind % 7], linestyle='-', label=r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBi))
    #     # ax4.plot(ts, v_ds.sel(aIBi=aIBi).values * (1e3 * T_exp2th / L_exp2th), color=colors[ind % 7], linestyle='-', label=r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBi))
    # ax4.fill_between(ts, -cBEC * (1e3 * T_exp2th / L_exp2th), cBEC * (1e3 * T_exp2th / L_exp2th), facecolor='yellow', alpha=0.5, label='Subsonic Region ($|v|<c_{BEC}$)')
    # ax4.legend()
    # ax4.set_ylabel(r'$v=\frac{d<X>}{dt} (\frac{\mu m}{ms})$')
    # ax4.set_xlabel(r'$t$ [$\frac{\xi}{c}=$' + '{:.2f} ms]'.format(1e3 * tscale_exp))
    # ax4.set_title('Impurity Velocity (BEC Frame)')

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

    # # ODE FIT TO POSITION (TRAJECTORY)

    # def EqMotion(y, t, gamma, beta):
    #     # y1 = x, y2 = dx/dt
    #     y1, y2 = y
    #     dy1dt = y2
    #     # dy2dt = -2 * gamma * y2 - (omega_Imp_x**2 - beta) * y1 + beta * xB0 * np.cos(omega_BEC_osc * t) + 2 * gamma * omega_BEC_osc * xB0 * np.sin(omega_BEC_osc * t)
    #     dy2dt = -2 * gamma * y2 - (omega_Imp_x**2 - beta) * y1 + beta * xB0 * np.cos(omega_BEC_osc * t)
    #     return [dy1dt, dy2dt]

    # def yint(t, gamma, beta, y0):
    #     y = odeint(EqMotion, y0, t, args=(gamma, beta))
    #     return y.ravel(order='F')

    # x_ds = qds['XLab']
    # xI_DatArray = np.empty(aIBiVals.size, dtype=np.object)
    # vI_DatArray = np.empty(aIBiVals.size, dtype=np.object)
    # xI_FitArray = np.empty(aIBiVals.size, dtype=np.object)
    # vI_FitArray = np.empty(aIBiVals.size, dtype=np.object)

    # y0Vals = np.empty(aIBiVals.size, dtype=np.object)
    # gVals = np.empty(aIBiVals.size)
    # bVals = np.empty(aIBiVals.size)
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
    #         p0 = [gVals[ind - 1], bVals[ind - 1]]
    #         # lowerbound = [gVals[ind - 1], 0]
    #         lowerbound = [0, 0]
    #         upperbound = [np.inf, np.inf]
    #     popt, cov = curve_fit(lambda t, gamma, beta: yint(t, gamma, beta, y0), tVals, data, p0=p0, bounds=(lowerbound, upperbound))
    #     gopt, bopt = popt
    #     y0Vals[ind] = y0; gVals[ind] = gopt; bVals[ind] = bopt

    #     fitvals = yint(tVals, gVals[ind], bVals[ind], y0Vals[ind])
    #     xfit = fitvals[0:tVals.size]
    #     vfit = fitvals[tVals.size:]
    #     xI_DatArray[ind] = xVals
    #     vI_DatArray[ind] = vVals
    #     xI_FitArray[ind] = xfit
    #     vI_FitArray[ind] = vfit

    #     P = qds['P'].sel(aIBi=aIBi).isel(t=0).values
    #     Pph = qds['Pph'].sel(aIBi=aIBi).isel(t=0).values
    #     msVals[ind] = mI * P / (P - Pph)

    # # ANALYTICAL SOLUTION FIT TO POSITION (TRAJECTORY)

    # def traj(t, gamma, beta, y0, omega_Imp_x, omega_BEC_osc, xB0):
    #     [x0, v0] = y0
    #     omega = np.sqrt(omega_Imp_x**2 - gamma**2 - beta)
    #     d = beta * xB0 / np.sqrt((omega_Imp_x**2 - omega_BEC_osc**2 - beta)**2 + 4 * gamma**2 * omega_BEC_osc**2)
    #     delta = np.arctan(2 * gamma * omega_BEC_osc / (omega_BEC_osc**2 - omega_Imp_x**2 + beta))
    #     h = np.sqrt((x0 - d * np.cos(delta))**2 + omega**(-2) * (v0 + omega_BEC_osc * d * np.sin(delta))**2)
    #     eta = -1 * np.arctan(omega**(-1) * (v0 + omega_BEC_osc * d * np.sin(delta)) / (x0 - d * np.cos(delta)))
    #     xt = h * np.exp(-1 * gamma * t) * np.cos(omega * t + eta) + d * np.cos(omega_BEC_osc * t + delta)
    #     return xt

    # x_ds = qds['XLab']
    # xI_DatArray = np.empty(aIBiVals.size, dtype=np.object)
    # xI_FitArray = np.empty(aIBiVals.size, dtype=np.object)

    # y0Vals = np.empty(aIBiVals.size, dtype=np.object)
    # gVals = np.empty(aIBiVals.size)
    # bVals = np.empty(aIBiVals.size)
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
    #         p0 = [1e-5, 1e-5]
    #         bounds = ([0, 0], [np.inf, np.inf])

    #     else:
    #         p0 = [gVals[ind - 1], bVals[ind - 1]]
    #         bounds = ([0, 0], [np.inf, np.inf])
    #     popt, cov = curve_fit(lambda t, gamma, beta: traj(t, gamma, beta, y0, omega_Imp_x, omega_BEC_osc, xB0), tVals, xVals, p0=p0, bounds=bounds)
    #     gopt, bopt = popt
    #     y0Vals[ind] = y0; gVals[ind] = gopt; bVals[ind] = bopt
    #     deltaVals[ind] = np.arctan(2 * gopt * omega_BEC_osc / (omega_BEC_osc**2 - omega_Imp_x**2 + bopt))

    #     xI_DatArray[ind] = xVals
    #     xI_FitArray[ind] = traj(tVals, gVals[ind], bVals[ind], y0Vals[ind], omega_Imp_x, omega_BEC_osc, xB0)

    #     P = qds['P'].sel(aIBi=aIBi).isel(t=0).values
    #     Pph = qds['Pph'].sel(aIBi=aIBi).isel(t=0).values
    #     msVals[ind] = mI * P / (P - Pph)

    # # POSITION (LAB) ANIMATION

    # fig, ax = plt.subplots()
    # curve_Dat = ax.plot(tVals[::20], xI_DatArray[0][::20], color='k', linestyle='', marker='o', label='')[0]
    # curve_Fit = ax.plot(tVals, xI_FitArray[0], color='orange', lw=2, label='')[0]
    # aIBi_text = ax.text(0.8, 0.9, r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[0]), transform=ax.transAxes, color='r')
    # Gamma_text = ax.text(0.8, 0.85, r'$\gamma=$' + '{:.2E}'.format(gVals[0]), transform=ax.transAxes, color='g')
    # Beta_text = ax.text(0.8, 0.8, r'$\beta=$' + '{:.2E}'.format(bVals[0]), transform=ax.transAxes, color='b')

    # ax.set_xlabel('t')
    # ax.set_ylabel('<X>')
    # ax.set_title('Impurity Trajectory (Lab Frame)')

    # def animate_fit(i):
    #     if i >= aIBiVals.size:
    #         return
    #     curve_Dat.set_ydata(xI_DatArray[i][::20])
    #     curve_Fit.set_ydata(xI_FitArray[i])
    #     aIBi_text.set_text(r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[i]))
    #     Gamma_text.set_text(r'$\gamma=$' + '{:.2E}'.format(gVals[i]))
    #     Beta_text.set_text(r'$\beta=$' + '{:.2E}'.format(bVals[i]))

    # anim_fit = FuncAnimation(fig, animate_fit, interval=75, frames=range(tVals.size))
    # # # anim_fit_filename = '/TrajFitAnim_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.gif'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # # # anim_fit.save(animpath + anim_fit_filename, writer='imagemagick')
    # # anim_fit_filename = '/TrajFitAnim_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.mp4'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # # anim_fit.save(animpath + anim_fit_filename, writer=mpegWriter)

    # # PARAMETER CURVES (& ESTIMATE alpha = m*Beta)

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
    #     aVals_Est[ind] = interpolate.splev(0.5 * RTF_BEC_X, E_Pol_tck, der=1)

    # # PLOT PARAMETERS

    # fig2, ax2 = plt.subplots()
    # ax2.plot(aIBiVals, gVals, 'g-', label=r'$\gamma$')
    # ax2.plot(aIBiVals, bVals, 'b-', label=r'$\beta$')
    # ax2.plot(aIBiVals, msVals * gVals, 'g:', label=r'$\xi=m^{*}\gamma$')
    # ax2.plot(aIBiVals, msVals * bVals, 'b:', label=r'$\alpha=m^{*}\beta$')
    # # ax2.plot(aIBiVals, deltaVals, color='orange', linestyle=':', label=r'$\delta$')
    # # ax2.plot(aIBiVals, bVals + gVals**2 - omega_Imp_x**2, 'm--', label=r'$\beta + \gamma^{2}-\omega_{0}^{2}$')
    # # ax2.plot(aIBiVals, aVals_Est, 'r-', label=r'$\frac{d^{2}E_{pol}}{dx^{2}}|_{0.5*X_{TF}}$')
    # # ax2.plot(aIBiVals, aVals_Est, 'r-', label=r'$\frac{dE_{pol}}{dx}|_{0.5*X_{TF}}$'))
    # # ax2.plot(aIBiVals, msVals, 'y-', label=r'$m^{*}$')
    # ax2.legend()
    # ax2.set_xlabel('aIBi')
    # ax2.set_title('Oscillation Fit Parameters')

    # plt.show()

    ###############################################################################################################################
    # # FIT IN THE BEC FRAME
    ###############################################################################################################################

    # ODE FIT TO POSITION (TRAJECTORY)

    def EqMotion(y, t, gamma, beta):
        # y1 = x, y2 = dx/dt
        y1, y2 = y
        dy1dt = y2
        dy2dt = -2 * gamma * y2 - (omega_Imp_x**2 + beta) * y1 + (omega_BEC_osc**2 - omega_Imp_x**2) * xB0 * np.cos(omega_BEC_osc * t)
        return [dy1dt, dy2dt]

    def yint(t, gamma, beta, y0):
        y = odeint(EqMotion, y0, t, args=(gamma, beta))
        return y.ravel(order='F')

    x_ds = qds['X']
    xI_DatArray_LAB = np.empty(aIBiVals.size, dtype=np.object)
    xI_FitArray_LAB = np.empty(aIBiVals.size, dtype=np.object)

    xI_DatArray = np.empty(aIBiVals.size, dtype=np.object)
    vI_DatArray = np.empty(aIBiVals.size, dtype=np.object)
    xI_FitArray = np.empty(aIBiVals.size, dtype=np.object)
    vI_FitArray = np.empty(aIBiVals.size, dtype=np.object)

    y0Vals = np.empty(aIBiVals.size, dtype=np.object)
    gVals = np.empty(aIBiVals.size)
    bVals = np.empty(aIBiVals.size)
    msVals = np.empty(aIBiVals.size)
    for ind, aIBi in enumerate(aIBiVals):
        # if ind != 10:
        #     continue
        xVals = x_ds.sel(aIBi=aIBi).values
        vVals = np.gradient(xVals, tVals)
        x0 = xVals[0]
        v0 = vVals[0]
        # v0 = (qds['P'].sel(aIBi=aIBi).isel(t=0).values - qds['Pph'].sel(aIBi=aIBi).isel(t=0).values) / mI
        y0 = [x0, v0]
        data = np.concatenate((xVals, vVals))
        if ind == 0:
            p0 = [1e-3, 1e-3]
            lowerbound = [0, 0]
            upperbound = [np.inf, np.inf]

        else:
            p0 = [gVals[ind - 1], bVals[ind - 1]]
            # lowerbound = [gVals[ind - 1], 0]
            lowerbound = [0, 0]
            upperbound = [np.inf, np.inf]
        popt, cov = curve_fit(lambda t, gamma, beta: yint(t, gamma, beta, y0), tVals, data, p0=p0, bounds=(lowerbound, upperbound))
        gopt, bopt = popt
        y0Vals[ind] = y0; gVals[ind] = gopt; bVals[ind] = bopt

        fitvals = yint(tVals, gVals[ind], bVals[ind], y0Vals[ind])
        xfit = fitvals[0:tVals.size]
        vfit = fitvals[tVals.size:]
        xI_DatArray[ind] = xVals
        vI_DatArray[ind] = vVals
        xI_FitArray[ind] = xfit
        vI_FitArray[ind] = vfit

        xI_DatArray_LAB[ind] = qds['XLab'].sel(aIBi=aIBi).values
        xI_FitArray_LAB[ind] = xfit + xBEC
        P = qds['P'].sel(aIBi=aIBi).isel(t=0).values
        Pph = qds['Pph'].sel(aIBi=aIBi).isel(t=0).values
        msVals[ind] = mI * P / (P - Pph)

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
    # gVals = np.empty(aIBiVals.size)
    # bVals = np.empty(aIBiVals.size)
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
    #         p0 = [gVals[ind - 1], bVals[ind - 1]]
    #         bounds = ([0, 0], [np.inf, np.inf])
    #     # print(aIBi)
    #     popt, cov = curve_fit(lambda t, gamma, beta: traj(t, gamma, beta, y0, omega_Imp_x, omega_BEC_osc, xB0), tVals, xVals, p0=p0, bounds=bounds)
    #     gopt, bopt = popt
    #     y0Vals[ind] = y0; gVals[ind] = gopt; bVals[ind] = bopt
    #     deltaVals[ind] = np.arctan(2 * gopt * omega_BEC_osc / (omega_BEC_osc**2 - omega_Imp_x**2 + bopt))

    #     xI_DatArray[ind] = xVals
    #     xI_FitArray[ind] = traj(tVals, gVals[ind], bVals[ind], y0Vals[ind], omega_Imp_x, omega_BEC_osc, xB0)

    #     P = qds['P'].sel(aIBi=aIBi).isel(t=0).values
    #     Pph = qds['Pph'].sel(aIBi=aIBi).isel(t=0).values
    #     msVals[ind] = mI * P / (P - Pph)

    # # POSITION (BEC) ANIMATION

    # fig, ax = plt.subplots()
    # curve_Dat = ax.plot(tVals[::20], xI_DatArray[0][::20], color='k', linestyle='', marker='o', label='')[0]
    # curve_Fit = ax.plot(tVals, xI_FitArray[0], color='orange', lw=2, label='')[0]
    # aIBi_text = ax.text(0.8, 0.9, r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[0]), transform=ax.transAxes, color='r')
    # Gamma_text = ax.text(0.8, 0.85, r'$\gamma=$' + '{:.2E}'.format(gVals[0]), transform=ax.transAxes, color='g')
    # Beta_text = ax.text(0.8, 0.8, r'$\beta=$' + '{:.2E}'.format(bVals[0]), transform=ax.transAxes, color='b')

    # ax.set_xlabel('t')
    # ax.set_ylabel('<X>')
    # ax.set_title('Impurity Trajectory (BEC Frame)')

    # def animate_fit(i):
    #     if i >= aIBiVals.size:
    #         return
    #     curve_Dat.set_ydata(xI_DatArray[i][::20])
    #     curve_Fit.set_ydata(xI_FitArray[i])
    #     aIBi_text.set_text(r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[i]))
    #     Gamma_text.set_text(r'$\gamma=$' + '{:.2E}'.format(gVals[i]))
    #     Beta_text.set_text(r'$\beta=$' + '{:.2E}'.format(bVals[i]))

    # anim_fit = FuncAnimation(fig, animate_fit, interval=75, frames=range(tVals.size))
    # # # # anim_fit_filename = '/TrajFitAnim_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.gif'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # # # # anim_fit.save(animpath + anim_fit_filename, writer='imagemagick')
    # # anim_fit_filename = '/TrajFitBECAnim_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.mp4'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # # anim_fit.save(animpath + anim_fit_filename, writer=mpegWriter)

    # # POSITION (LAB) ANIMATION

    # fig, ax = plt.subplots()
    # curve_Dat = ax.plot(tVals[::20], xI_DatArray_LAB[0][::20], color='k', linestyle='', marker='o', label='')[0]
    # curve_Fit = ax.plot(tVals, xI_FitArray_LAB[0], color='orange', lw=2, label='')[0]
    # aIBi_text = ax.text(0.8, 0.9, r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[0]), transform=ax.transAxes, color='r')
    # Gamma_text = ax.text(0.8, 0.85, r'$\gamma=$' + '{:.2E}'.format(gVals[0]), transform=ax.transAxes, color='g')
    # Beta_text = ax.text(0.8, 0.8, r'$\beta=$' + '{:.2E}'.format(bVals[0]), transform=ax.transAxes, color='b')

    # ax.set_xlabel('t')
    # ax.set_ylabel('<X>')
    # ax.set_title('Impurity Trajectory (BEC Frame)')

    # def animate_fit(i):
    #     if i >= aIBiVals.size:
    #         return
    #     curve_Dat.set_ydata(xI_DatArray_LAB[i][::20])
    #     curve_Fit.set_ydata(xI_FitArray_LAB[i])
    #     aIBi_text.set_text(r'$a_{IB}^{-1}=$' + '{:.2f}'.format(aIBiVals[i]))
    #     Gamma_text.set_text(r'$\gamma=$' + '{:.2E}'.format(gVals[i]))
    #     Beta_text.set_text(r'$\beta=$' + '{:.2E}'.format(bVals[i]))

    # anim_fit = FuncAnimation(fig, animate_fit, interval=75, frames=range(tVals.size))
    # # # # anim_fit_filename = '/TrajFitAnim_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.gif'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # # # # anim_fit.save(animpath + anim_fit_filename, writer='imagemagick')
    # # anim_fit_filename = '/TrajFitAnim_RevBEC_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.mp4'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # # anim_fit.save(animpath + anim_fit_filename, writer=mpegWriter)

    # PARAMETER CURVES (& ESTIMATE alpha = m*Beta)

    NGridPoints_desired = (1 + 2 * Lx / dx) * (1 + 2 * Lz / dz)
    Ntheta = 50
    Nk = np.ceil(NGridPoints_desired / Ntheta)
    theta_max = np.pi
    thetaArray, dtheta = np.linspace(0, theta_max, Ntheta, retstep=True)
    k_max = ((2 * np.pi / dx)**3 / (4 * np.pi / 3))**(1 / 3)
    k_min = 1e-5
    kArray, dk = np.linspace(k_min, k_max, Nk, retstep=True)
    kgrid = Grid.Grid("SPHERICAL_2D")
    kgrid.initArray_premade('k', kArray)
    kgrid.initArray_premade('th', thetaArray)
    n0_TF = expParams['n0_TF'] / (L_exp2th**3)
    n0_thermal = expParams['n0_thermal'] / (L_exp2th**3)
    RTF_BEC_X = expParams['RTF_BEC_X'] * L_exp2th; RTF_BEC_Y = expParams['RTF_BEC_Y'] * L_exp2th; RTF_BEC_Z = expParams['RTF_BEC_Z'] * L_exp2th
    RG_BEC_X = expParams['RG_BEC_X'] * L_exp2th; RG_BEC_Y = expParams['RG_BEC_Y'] * L_exp2th; RG_BEC_Z = expParams['RG_BEC_Z'] * L_exp2th
    trapParams = {'n0_TF_BEC': n0_TF, 'RTF_BEC_X': RTF_BEC_X, 'RTF_BEC_Y': RTF_BEC_Y, 'RTF_BEC_Z': RTF_BEC_Z, 'n0_thermal_BEC': n0_thermal, 'RG_BEC_X': RG_BEC_X, 'RG_BEC_Y': RG_BEC_Y, 'RG_BEC_Z': RG_BEC_Z,
                  'omega_Imp_x': omega_Imp_x, 'omega_BEC_osc': omega_BEC_osc, 'X0': X0, 'P0': P0, 'a_osc': a_osc}
    n0 = expParams['n0_BEC'] / (L_exp2th**3)  # should ~ 1
    mB = expParams['mB'] * M_exp2th  # should = 1
    mI = expParams['mI'] * M_exp2th
    aBB = expParams['aBB'] * L_exp2th
    gBB = (4 * np.pi / mB) * aBB
    sParams = [mI, mB, n0, gBB]

    X_Vals = np.linspace(-1 * RTF_BEC_X * 0.99, RTF_BEC_X * 0.99, 100)
    # aIBiVals = aIBiVals[::10]
    aVals_Est = np.empty(aIBiVals.size)
    for ind, aIBi in enumerate(aIBiVals):
        cParams = {'aIBi': aIBi}
        E_Pol_tck = pfs.V_Pol_interp(kgrid, X_Vals, cParams, sParams, trapParams)
        aVals_Est[ind] = interpolate.splev(0, E_Pol_tck, der=1)

    # PLOT PARAMETERS
    rhoVals = gVals**2 - bVals - omega_Imp_x**2
    critdamp_ind = np.argwhere(np.sign(rhoVals) >= 0)[0][0]
    fig2, ax2 = plt.subplots()
    ax2.plot(aIBiVals, gVals, 'g-', label=r'$\gamma$')
    ax2.plot(aIBiVals, bVals, 'b-', label=r'$\beta$')
    ax2.plot(aIBiVals, msVals * gVals, 'g:', label=r'$\xi=m^{*}\gamma$')
    ax2.plot(aIBiVals, msVals * bVals, 'b:', label=r'$\alpha=m^{*}\beta$')
    # ax2.plot(aIBiVals, deltaVals, color='orange', linestyle=':', label=r'$\delta$')
    ax2.plot(aIBiVals, gVals**2 - bVals - omega_Imp_x**2, 'm--', label=r'$\gamma^{2}-\beta-\omega_{0}^{2}$')
    ax2.plot(aIBiVals[critdamp_ind] * np.ones(aIBiVals.size), np.linspace(0, np.max(msVals * gVals), aIBiVals.size), 'y--', label='Critical Damping')
    ax2.plot(aIBiVals, aVals_Est, 'r-', label=r'$\alpha_{est}=\frac{d^{2}E_{pol}}{dx^{2}}|_{x_{peak}}$')
    # ax2.plot(aIBiVals, aVals_Est, 'r-', label=r'$\frac{dE_{pol}}{dx}|_{0.5*X_{TF}}$'))
    # ax2.plot(aIBiVals, msVals, 'y-', label=r'$m^{*}$')
    ax2.legend(loc=2)
    ax2.set_xlabel(r'$a_{IB}^{-1}$')
    ax2.set_title('Oscillation Fit Parameters')

    plt.show()
