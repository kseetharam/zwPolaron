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
import matplotlib.patches as patches
import os
import itertools
import Grid
import pf_dynamic_sph as pfs
import pf_static_sph as pfstat


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

    datapath_Neg_Homog_NoCS = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/BEC_osc/NegScat/HomogBEC/NoPolPot_NoCSDyn/fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(aBB, NGridPoints_cart, dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])
    datapath_Neg_Homog_CS = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/BEC_osc/NegScat/HomogBEC/NoPolPot_CSDyn/fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(aBB, NGridPoints_cart, dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])
    datapath_Pos_Homog_NoCS = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/BEC_osc/PosScat/HomogBEC/NoPolPot_NoCSDyn/fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(aBB, NGridPoints_cart, dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])
    datapath_Pos_Homog_CS = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}/BEC_osc/PosScat/HomogBEC/NoPolPot_CSDyn/fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(aBB, NGridPoints_cart, dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])

    figdatapath = '/Users/kis/Dropbox/Apps/Overleaf/Cherenkov Polaron Paper pt2/figures/figdump'
    datafilepath = 'tempData/'

    # # # Concatenate Individual Datasets

    # innerdatapath = datapath_Neg_Homog_NoCS
    # ds_list = []
    # aIBi_list = []
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

    # aIBi_keys = []
    # aIBi_groups = []
    # aIBi_ds_list = []
    # for key, group in g:
    #     aIBi_keys.append(key)
    #     aIBi_groups.append(list(group))

    # for ind, group in enumerate(aIBi_groups):
    #     aIBi = aIBi_keys[ind]
    #     _, ds_temp = zip(*group)
    #     aIBi_ds_list.append(ds_temp[0])

    # if toggleDict['PosScat'] == 'on':
    #     aIBi_keys = aIBi_keys[::-1]
    #     aIBi_ds_list = aIBi_ds_list[::-1]

    # ds_tot = xr.concat(aIBi_ds_list, pd.Index(aIBi_keys, name='aIBi'))
    # del(ds_tot.attrs['Fext_mag'])
    # del(ds_tot.attrs['aIBi'])
    # del(ds_tot.attrs['gIB'])
    # del(ds_tot.attrs['TF'])
    # del(ds_tot.attrs['Delta_P'])
    # ds_tot.to_netcdf(innerdatapath + '/LDA_Dataset.nc')

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

    qds_Neg_Homog_NoCS = xr.open_dataset(datapath_Neg_Homog_NoCS + '/LDA_Dataset.nc')
    qds_Neg_Homog_CS = xr.open_dataset(datapath_Neg_Homog_CS + '/LDA_Dataset.nc')
    qds_Pos_Homog_NoCS = xr.open_dataset(datapath_Pos_Homog_NoCS + '/LDA_Dataset.nc')
    qds_Pos_Homog_CS = xr.open_dataset(datapath_Pos_Homog_CS + '/LDA_Dataset.nc')

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

    xL_bareImp = (xBEC[0] + X0) * np.cos(omega_Imp_x * tVals) + (P0 / (omega_Imp_x * mI)) * np.sin(omega_Imp_x * tVals)  # gives the lab frame trajectory time trace of a bare impurity (only subject to the impurity trap) that starts at the same position w.r.t. the BEC as the polaron and has the same initial total momentum
    vL_bareImp = np.gradient(xL_bareImp, tVals)
    aL_bareImp = np.gradient(np.gradient(xL_bareImp, tVals), tVals)
    print(expParams['RTF_BEC_X'] * 1e6, expParams['RTF_BEC_Y'] * 1e6, expParams['RTF_BEC_Z'] * 1e6)

    # print(1 / aIBiVals / a0_th)

    # #############################################################################################################################
    # # ANIMATIONS
    # #############################################################################################################################

    # # POSITION VS TIME ANIMATION (LAB FRAME)

    # inverseScat = False

    # qds = qds_Pos_Homog_CS
    # aIBiVals = qds['aIBi'].values
    # aIBiVals = aIBiVals[::-1]
    # x_ds = qds['XLab']
    # xI_array = np.empty(aIBiVals.size, dtype=np.object)
    # xmin = 1e20
    # xmax = -1e20
    # for ind, aIBi in enumerate(aIBiVals):
    #     xI_array[ind] = x_ds.sel(aIBi=aIBi).values
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

    # anim_p_filename = '/PosHomogCS_TrajAnim_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.mp4'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # anim_p.save(figdatapath + anim_p_filename, writer=mpegWriter)

    # # POSITION VS TIME ANIMATION (BEC FRAME)

    # inverseScat = False

    # qds = qds_Pos_Homog_CS
    # aIBiVals = qds['aIBi'].values
    # aIBiVals = aIBiVals[::-1]
    # x_ds = qds['X']
    # xI_array = np.empty(aIBiVals.size, dtype=np.object)
    # xmin = 1e20
    # xmax = -1e20
    # for ind, aIBi in enumerate(aIBiVals):
    #     xI_array[ind] = x_ds.sel(aIBi=aIBi).values
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

    # anim_pB_filename = '/PosHomogCS_TrajAnim_BECframe_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.mp4'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # anim_pB.save(figdatapath + anim_pB_filename, writer=mpegWriter)

    # # VELOCITY VS TIME ANIMATION (BEC FRAME)

    # inverseScat = False

    # qds = qds_Pos_Homog_CS
    # aIBiVals = qds['aIBi'].values
    # aIBiVals = aIBiVals[::-1]

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

    # anim_v_filename = '/PosHomogCS_VelAnim_BECframe_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}.mp4'.format(f_BEC_osc, f_Imp_x, a_osc, X0, P0)
    # anim_v.save(figdatapath + anim_v_filename, writer=mpegWriter)

    # #############################################################################################################################
    # # PAPER FIGURES
    # #############################################################################################################################

    # # # # FIG 5 - 2D OSCILLATION FREQUENCY (HOMOGENEOUS BEC)

    # a0ylim = 4000

    # # # qds_List = [qds_Neg_NoPP_NoCS, qds_Pos_NoPP_NoCS, qds_Neg_NoPP_CS, qds_Pos_NoPP_CS]; signList = ['Neg', 'Pos', 'Neg', 'Pos']
    # # qds_List = [qds_Neg_Homog_NoCS, qds_Pos_Homog_NoCS, qds_Neg_Homog_CS, qds_Pos_Homog_CS]; signList = ['Neg', 'Pos', 'Neg', 'Pos']
    # qds_List = [qds_Neg_Homog_CS, qds_Pos_Homog_CS, qds_Neg_Homog_NoCS, qds_Pos_Homog_NoCS]; signList = ['Neg', 'Pos', 'Neg', 'Pos']

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
    #     ax.set_ylabel(r'$a_{\rm IB}/a_{0}$', fontsize=labelsize)
    #     ax.plot(omega_BEC_osc * T_exp2th / (2 * np.pi) * np.ones(aIBVals.size), aIBVals, 'k:', lw=3, label='BEC Oscillation Frequency')
    #     ax.plot(omega_Imp_x * T_exp2th / (2 * np.pi) * np.ones(aIBVals.size), aIBVals, color='orange', linestyle=':', marker='', lw=3, label='Impurity Trap Frequency')
    #     if signList[ind] == 'Neg':
    #         ax.set_ylim([-1 * a0ylim, np.max(aIBVals)])
    #     elif signList[ind] == 'Pos':
    #         ax.set_ylim([a0ylim, np.min(aIBVals)])
    #     ax.set_xlabel('f (Hz)', fontsize=labelsize)
    #     ax.set_xlim([0, 400])
    #     ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # rect = patches.Rectangle(xy=(0.03, 0.075), width=0.85, height=0.49, alpha=0.1, edgecolor='tab:gray', facecolor='tab:gray', zorder=1000, transform=fig5.transFigure, figure=fig5)
    # fig5.patches.extend([rect])

    # cbar_ax = fig5.add_axes([0.9, 0.2, 0.02, 0.7])
    # fig5.colorbar(quadF, cax=cbar_ax, extend='max')
    # cbar_ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # # fig5.colorbar(quadF, cax=cbar_ax, extend='max', format=FormatStrFormatter('%.f'))
    # # fig5.colorbar(quadF, cax=cbar_ax, extend='max', format='%.0e')
    # handles, labels = axList[0].get_legend_handles_labels()
    # fig5.legend(handles, labels, ncol=2, loc='lower center', fontsize=legendsize)
    # # fig5.subplots_adjust(bottom=0.17, top=0.95, right=0.85, hspace=0.45, wspace=0.45)
    # fig5.subplots_adjust(bottom=0.22, top=0.95, right=0.85, hspace=0.45, wspace=0.45)
    # fig5.set_size_inches(7.8, 6.0)

    # fig5.text(0.05, 0.96, '(a)', fontsize=labelsize)
    # fig5.text(0.05, 0.525, '(b)', fontsize=labelsize)
    # fig5.text(0.47, 0.96, '(c)', fontsize=labelsize)
    # fig5.text(0.47, 0.525, '(d)', fontsize=labelsize)

    # fig5.text(0.42, 0.11, 'No Phonons', color='tab:brown', fontsize=labelsize)

    # # fig5.savefig(figdatapath + '/Fig5.jpg', quality=100)

    # # # # FIG 8 - 2D OSCILLATION FREQUENCY (INHOMOGENEOUS BEC)

    # a0ylim = 6000

    # qds_List = [qds_Neg_PP_CS, qds_Pos_PP_CS, qds_Neg_PP_NoCS, qds_Pos_PP_NoCS]; signList = ['Neg', 'Pos', 'Neg', 'Pos']
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

    # fig8, axes = plt.subplots(nrows=2, ncols=2)
    # axList = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    # for ind, freq_da in enumerate(freqda_List):
    #     ax = axList[ind]
    #     absFT_interp, f_interp, aIBi_interp = pfs.xinterp2D(freq_da, 'f', 'aIBi', 5)
    #     # absFT_interp = freq_da.values; f_interp = freq_da.coords['f'].values; aIBi_interp = freq_da.coords['aIBi'].values
    #     aIBVals = aIBVals_List[ind]
    #     quadF = ax.pcolormesh(f_interp * T_exp2th, (1 / aIBi_interp) / a0_th, absFT_interp, vmin=0, vmax=vmax)
    #     ax.set_ylabel(r'$a_{\rm IB}/a_{0}$', fontsize=labelsize)
    #     ax.plot(omega_BEC_osc * T_exp2th / (2 * np.pi) * np.ones(aIBVals.size), aIBVals, 'k:', lw=3, label='BEC Oscillation Frequency')
    #     ax.plot(omega_Imp_x * T_exp2th / (2 * np.pi) * np.ones(aIBVals.size), aIBVals, color='orange', linestyle=':', marker='', lw=3, label='Impurity Trap Frequency')
    #     if signList[ind] == 'Neg':
    #         ax.set_ylim([-1 * a0ylim, np.max(aIBVals)])
    #     elif signList[ind] == 'Pos':
    #         ax.set_ylim([a0ylim, np.min(aIBVals)])
    #     ax.set_xlabel('f (Hz)', fontsize=labelsize)
    #     ax.set_xlim([0, 400])
    #     ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # rect = patches.Rectangle(xy=(0.03, 0.075), width=0.85, height=0.49, alpha=0.1, edgecolor='tab:gray', facecolor='tab:gray', zorder=1000, transform=fig8.transFigure, figure=fig8)
    # fig8.patches.extend([rect])

    # cbar_ax = fig8.add_axes([0.9, 0.2, 0.02, 0.7])
    # fig8.colorbar(quadF, cax=cbar_ax, extend='max')
    # cbar_ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # # fig8.colorbar(quadF, cax=cbar_ax, extend='max', format=FormatStrFormatter('%.f'))
    # # fig8.colorbar(quadF, cax=cbar_ax, extend='max', format='%.0e')
    # handles, labels = axList[0].get_legend_handles_labels()
    # fig8.legend(handles, labels, ncol=2, loc='lower center', fontsize=legendsize)
    # # fig8.subplots_adjust(bottom=0.17, top=0.95, right=0.85, hspace=0.45, wspace=0.45)
    # fig8.subplots_adjust(bottom=0.22, top=0.95, right=0.85, hspace=0.45, wspace=0.45)
    # fig8.set_size_inches(7.8, 6.0)

    # fig8.text(0.05, 0.96, '(a)', fontsize=labelsize)
    # fig8.text(0.05, 0.525, '(b)', fontsize=labelsize)
    # fig8.text(0.47, 0.96, '(c)', fontsize=labelsize)
    # fig8.text(0.47, 0.525, '(d)', fontsize=labelsize)

    # fig8.text(0.42, 0.11, 'No Phonons', color='tab:brown', fontsize=labelsize)

    # fig8.text(0.28, 0.72, r'$m^{*}$', color='w', fontsize=labelsize)
    # fig8.text(0.38, 0.72, r'$m$', color='w', fontsize=labelsize)
    # fig8.text(0.255, 0.28, r'$m^{*}=m$', color='w', fontsize=labelsize)

    # # Add MF potential (Vimp_bare + Epol) line

    # Ntheta = 50
    # NGridPoints_desired = (1 + 2 * Lx / dx) * (1 + 2 * Lz / dz)
    # Nk = np.ceil(NGridPoints_desired / Ntheta)

    # theta_max = np.pi
    # thetaArray, dtheta = np.linspace(0, theta_max, Ntheta, retstep=True)

    # k_max = ((2 * np.pi / dx)**3 / (4 * np.pi / 3))**(1 / 3)

    # k_min = 1e-5
    # kArray, dk = np.linspace(k_min, k_max, Nk, retstep=True)
    # if dk < k_min:
    #     print('k ARRAY GENERATION ERROR')

    # kgrid = Grid.Grid("SPHERICAL_2D")
    # kgrid.initArray_premade('k', kArray)
    # kgrid.initArray_premade('th', thetaArray)

    # aIBexp_Vals = np.concatenate((np.array([-12000, -8000, -7000, -6000, -5000]), np.linspace(-4000, -2000, 5, endpoint=False), np.linspace(-2000, -70, 5, endpoint=False), np.linspace(-70, -20, 3))) * a0_exp; aIBexp_Vals = np.concatenate((aIBexp_Vals, np.linspace(20, 650, 10,) * a0_exp))
    # aIBi_Vals = 1 / (aIBexp_Vals * L_exp2th)
    # aIBi = aIBi_Vals[2]
    # cParams = {'aIBi': aIBi}
    # n0 = expParams['n0_BEC'] / (L_exp2th**3)  # should = 1
    # sParams = [mI, mB, n0, gBB]

    # expParams = pfs.Zw_expParams()
    # L_exp2th, M_exp2th, T_exp2th = pfs.unitConv_exp2th(expParams['n0_BEC'], expParams['mB'])
    # E_exp2th = M_exp2th * L_exp2th**2 / T_exp2th**2
    # F_exp2th = M_exp2th * L_exp2th / T_exp2th**2

    # a0_exp = 5.29e-11  # Bohr radius (m)
    # hbar = 1.0555e-34  # reduced Planck's constant (J*s/rad)

    # n0_TF = expParams['n0_TF'] / (L_exp2th**3)
    # n0_thermal = expParams['n0_thermal'] / (L_exp2th**3)
    # RTF_BEC_X = expParams['RTF_BEC_X'] * L_exp2th; RTF_BEC_Y = expParams['RTF_BEC_Y'] * L_exp2th; RTF_BEC_Z = expParams['RTF_BEC_Z'] * L_exp2th
    # RG_BEC_X = expParams['RG_BEC_X'] * L_exp2th; RG_BEC_Y = expParams['RG_BEC_Y'] * L_exp2th; RG_BEC_Z = expParams['RG_BEC_Z'] * L_exp2th
    # omega_Imp_x = expParams['omega_Imp_x'] / T_exp2th
    # trapParams = {'n0_TF_BEC': n0_TF, 'RTF_BEC_X': RTF_BEC_X, 'RTF_BEC_Y': RTF_BEC_Y, 'RTF_BEC_Z': RTF_BEC_Z, 'n0_thermal_BEC': n0_thermal, 'RG_BEC_X': RG_BEC_X, 'RG_BEC_Y': RG_BEC_Y, 'RG_BEC_Z': RG_BEC_Z, 'omega_Imp_x': omega_Imp_x}

    # X_Vals = np.linspace(-1 * RTF_BEC_X * 0.99, RTF_BEC_X * 0.99, 1e3)
    # X_Vals_m = X_Vals / L_exp2th
    # E_Pol_tck = pfs.V_Pol_interp(kgrid, X_Vals, cParams, sParams, trapParams)
    # EpVals_interp = 1 * interpolate.splev(X_Vals, E_Pol_tck, der=0)
    # FpVals_interp = pfs.F_pol(X_Vals, E_Pol_tck)

    # X_Vals_poly = np.linspace(-1 * RTF_BEC_X * 0.5, RTF_BEC_X * 0.5, 50)
    # EpVals_poly = 1 * interpolate.splev(X_Vals_poly, E_Pol_tck, der=0)
    # [p2, p1, p0] = np.polyfit(X_Vals_poly, EpVals_poly, deg=2)
    # omegap = np.sqrt(2 * p2 / mI)
    # EpVals_harm = p2 * X_Vals**2 + p0

    # # EpVals_Hz = (2 * np.pi * hbar)**(-1) * EpVals / E_exp2th
    # EpVals_Hz = (2 * np.pi * hbar)**(-1) * EpVals_interp / E_exp2th
    # FpVals_N = FpVals_interp / F_exp2th
    # EpVals_harm_Hz = (2 * np.pi * hbar)**(-1) * EpVals_harm / E_exp2th
    # freq_p_Hz = (omegap / (2 * np.pi)) * T_exp2th

    # VtB = (gBB * n0 / RTF_BEC_X**2) * X_Vals**2
    # VtB_Hz = (2 * np.pi * hbar)**(-1) * VtB / E_exp2th
    # omega_tB = np.sqrt((2 / mB) * (gBB * n0 / RTF_BEC_X**2))
    # freq_tB_Hz = (omega_tB / (2 * np.pi)) * T_exp2th

    # n_BEC_Vals = pfs.n_BEC(X_Vals, 0, 0, n0_TF, n0_thermal, RTF_BEC_X, RTF_BEC_Y, RTF_BEC_Z, RG_BEC_X, RG_BEC_Y, RG_BEC_Z)

    # def V_Imp_trap(X, omega_Imp_x, mI):
    #     return 0.5 * mI * (omega_Imp_x**2) * (X**2)

    # mR = pfs.ur(mI, mB)
    # gIB_Vals_Born = (2 * np.pi / mR) * 1 / aIBi_Vals
    # X_Vals_poly = np.linspace(-1 * RTF_BEC_X * 0.5, RTF_BEC_X * 0.5, 50)
    # n_BEC_Vals = pfs.n_BEC(X_Vals_poly, 0, 0, n0_TF, n0_thermal, RTF_BEC_X, RTF_BEC_Y, RTF_BEC_Z, RG_BEC_X, RG_BEC_Y, RG_BEC_Z)

    # freqVals_Ep = np.zeros(aIBi_Vals.size)
    # freqVals_MF = np.zeros(aIBi_Vals.size)
    # for inda, aIBi in enumerate(aIBi_Vals):
    #     cParams = {'aIBi': aIBi}
    #     E_Pol_tck = pfs.V_Pol_interp(kgrid, X_Vals, cParams, sParams, trapParams)
    #     EpVals_interp = 1 * interpolate.splev(X_Vals, E_Pol_tck, der=0)

    #     EpVals_poly = 1 * interpolate.splev(X_Vals_poly, E_Pol_tck, der=0)

    #     # [p2, p1, p0] = np.polyfit(X_Vals_poly, EpVals_poly, deg=2)
    #     # omegap = np.sqrt(2 * p2 / mI)
    #     # freq_p_Hz = (omegap / (2 * np.pi)) * T_exp2th
    #     # freqVals_Ep[inda] = freq_p_Hz

    #     V_Imp_Vals = V_Imp_trap(X_Vals_poly, trapParams['omega_Imp_x'], mI)
    #     MF_pot = V_Imp_Vals + EpVals_poly
    #     [p2, p1, p0] = np.polyfit(X_Vals_poly, MF_pot, deg=2)
    #     omegap = np.sqrt(2 * p2 / mI)
    #     freq_p_Hz = (omegap / (2 * np.pi)) * T_exp2th
    #     freqVals_MF[inda] = freq_p_Hz

    # axes[1, 0].plot(freqVals_MF, aIBexp_Vals / a0_exp, 'w--')
    # axes[0, 0].plot(freqVals_MF, aIBexp_Vals / a0_exp, 'w--')

    # # add mass renormalized MF line

    # NGridPoints_desired = (1 + 2 * Lx / dx) * (1 + 2 * Lz / dz)
    # Ntheta = 50
    # Nk = np.ceil(NGridPoints_desired / Ntheta)

    # theta_max = np.pi
    # thetaArray, dtheta = np.linspace(0, theta_max, Ntheta, retstep=True)

    # k_max = ((2 * np.pi / dx)**3 / (4 * np.pi / 3))**(1 / 3)

    # k_min = 1e-5
    # kArray, dk = np.linspace(k_min, k_max, Nk, retstep=True)
    # if dk < k_min:
    #     print('k ARRAY GENERATION ERROR')

    # kgrid = Grid.Grid("SPHERICAL_2D")
    # kgrid.initArray_premade('k', kArray)
    # kgrid.initArray_premade('th', thetaArray)

    # Nsteps = 1e2
    # aSi_tck, PBint_tck = pfstat.createSpline_grid(Nsteps, kgrid, mI, mB, n0, gBB)

    # SS_ms_Avals = np.zeros(aIBi_Vals.size)

    # for Aind, aIBi in enumerate(aIBi_Vals):
    #     DP = pfstat.DP_interp(0, P0, aIBi, aSi_tck, PBint_tck)
    #     aSi = pfstat.aSi_interp(DP, aSi_tck)
    #     PB_Val = pfstat.PB_interp(DP, aIBi, aSi_tck, PBint_tck)
    #     SS_ms_Avals[Aind] = pfstat.effMass(P0, PB_Val, mI)

    # mE_steadystate = SS_ms_Avals / mI
    # mE_steadystate[mE_steadystate < 0] = np.nan

    # freqVals_MF_massRenorm = freqVals_MF * np.sqrt(1 / mE_steadystate)

    # axes[0, 0].plot(freqVals_MF_massRenorm, aIBexp_Vals / a0_exp, 'w:')

    # fig8.savefig(figdatapath + '/Fig8.jpg', quality=100)

    # # axes[0, 0].set_ylim([-300, -1])
    # # axes[1, 0].set_ylim([-300, -1])
    # # fig8.savefig(figdatapath + '/Fig8_zoom.jpg', quality=100)

    # # # # # FIG 6 - VELOCITY PLOTS (BEC FRAME) - NEGSCAT

    # fig6, axes = plt.subplots(nrows=2, ncols=2)
    # axList = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    # x_ds = qds_Neg_Homog_CS['X']
    # # x_ds = qds_Neg_PP_CS['X']
    # # x_ds = qds_Neg_PP_NoCS['X']

    # aIBiVals = x_ds['aIBi'].values
    # aIBVals = (1 / aIBiVals) / a0_th

    # aIB_des = np.array([-20, -150, -500, -1000])

    # aIBinds = np.zeros(aIB_des.size, dtype=int)
    # for ind, aIB in enumerate(aIB_des):
    #     aIBinds[ind] = np.abs(aIBVals - aIB).argmin().astype(int)

    # tVals = x_ds['t'].values
    # dt = tVals[1] - tVals[0]
    # ts = tVals / tscale

    # cBEC = nu * np.ones(ts.size)
    # vI_array = np.empty(aIBinds.size, dtype=np.object)
    # for ind, aIBind in enumerate(aIBinds):
    #     ax = axList[ind]
    #     vI_array[ind] = np.gradient(x_ds.isel(aIBi=aIBind).values, tVals)
    #     curve = ax.plot(ts, vI_array[ind] * (1e3 * T_exp2th / L_exp2th), color='b', linestyle='-', lw=2, label='Interacting Impurity')[0]
    #     ax.fill_between(ts, -cBEC * (1e3 * T_exp2th / L_exp2th), cBEC * (1e3 * T_exp2th / L_exp2th), facecolor='yellow', alpha=0.5, label=r'Subsonic Region ($|\langle v\rangle|<c$)')
    #     ax.plot(ts, (vL_bareImp - np.gradient(xBEC, tVals)) * (1e3 * T_exp2th / L_exp2th), color='orange', linestyle=':', marker='', lw=1.5, label='Free Impurity')
    #     ax.set_ylabel(r'$v_{\rm imp}$ ($\frac{\mu m}{ms}$)', fontsize=labelsize)
    #     ax.set_xlabel(r'$t/(\xi c^{-1})$', fontsize=labelsize)
    #     ax.set_title(r'$a_{\rm IB}=$' + '{:d}'.format((aIBVals[aIBind]).astype(int)) + r'$/a_{0}$')
    #     ax.set_ylim([-25, 25])

    # handles, labels = axList[0].get_legend_handles_labels()
    # fig6.legend(handles, labels, ncol=3, loc='lower center', fontsize=legendsize)
    # fig6.subplots_adjust(bottom=0.2, top=0.92, right=0.97, hspace=0.65, wspace=0.45)
    # fig6.set_size_inches(7.8, 6.0)

    # fig6.text(0.05, 0.96, '(a)', fontsize=labelsize)
    # fig6.text(0.05, 0.51, '(c)', fontsize=labelsize)
    # fig6.text(0.53, 0.96, '(b)', fontsize=labelsize)
    # fig6.text(0.53, 0.51, '(d)', fontsize=labelsize)

    # fig6.savefig(figdatapath + '/Fig6.pdf')
    # # fig6.savefig(figdatapath + '/Fig6_Inhomg.pdf')
    # # fig6.savefig(figdatapath + '/Fig6_Inhomg_noCS.pdf')

    # # # # FIG 9 - POSITION PLOTS (BEC FRAME) - POSSCAT

    # fig9, axes = plt.subplots(nrows=2, ncols=2)
    # axList = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    # x_ds = qds_Pos_PP_CS['X']
    # aIBiVals = x_ds['aIBi'].values
    # aIBVals = (1 / aIBiVals) / a0_th

    # aIB_des = np.array([20, 300, 960, 1500])

    # aIBinds = np.zeros(aIB_des.size, dtype=int)
    # for ind, aIB in enumerate(aIB_des):
    #     aIBinds[ind] = np.abs(aIBVals - aIB).argmin().astype(int)

    # tVals = x_ds['t'].values
    # dt = tVals[1] - tVals[0]
    # ts = tVals / tscale

    # cBEC = nu * np.ones(ts.size)
    # xI_array = np.empty(aIBinds.size, dtype=np.object)
    # for ind, aIBind in enumerate(aIBinds):
    #     ax = axList[ind]
    #     xI_array[ind] = x_ds.isel(aIBi=aIBind).values

    #     curve = ax.plot(ts, 1e6 * xI_array[ind] / L_exp2th, color='g', lw=2, label='Interacting Impurity')[0]
    #     ax.plot(ts, 1e6 * (xL_bareImp - xBEC) / L_exp2th, color='orange', linestyle=':', marker='', lw=1.5, label='Free Impurity')
    #     ax.plot(ts, 1e6 * RTF_BEC_X * np.ones(ts.size) / L_exp2th, 'k:', lw=1.5, label='BEC TF Radius')
    #     ax.plot(ts, -1 * 1e6 * RTF_BEC_X * np.ones(ts.size) / L_exp2th, 'k:', lw=1.5)
    #     ax.set_ylabel(r'$x_{\rm imp}$ ($\mu m$)', fontsize=labelsize)
    #     ax.set_xlabel(r'$t/(\xi c^{-1})$', fontsize=labelsize)
    #     ax.set_title(r'$a_{\rm IB}=$' + '{:d}'.format((aIBVals[aIBind]).astype(int)) + r'$/a_{0}$')
    #     ax.set_ylim([-25, 25])
    #     ax.set_ylim([-50, 50])  # For PosScat
    #     # edge = 2.5 * 1e6 * np.max(xBEC) / L_exp2th; ax.set_ylim([-1 * edge, edge])  # For NegScat

    # handles, labels = axList[0].get_legend_handles_labels()
    # fig9.legend(handles, labels, ncol=3, loc='lower center', fontsize=legendsize)
    # fig9.subplots_adjust(bottom=0.2, top=0.92, right=0.97, hspace=0.65, wspace=0.45)
    # fig9.set_size_inches(7.8, 6.0)

    # fig9.text(0.05, 0.96, '(a)', fontsize=labelsize)
    # fig9.text(0.05, 0.51, '(c)', fontsize=labelsize)
    # fig9.text(0.53, 0.96, '(b)', fontsize=labelsize)
    # fig9.text(0.53, 0.51, '(d)', fontsize=labelsize)

    # fig9.savefig(figdatapath + '/Fig9.pdf')

    # #############################################################################################################################
    # # FIT IN THE BEC FRAME
    # #############################################################################################################################

    # # ODE FIT TO POSITION (TRAJECTORY) - FIT Gamma & FIX Beta

    # GammaFix = True
    # BetaZero = True
    # aIBlim = 15000  # in units of a0
    # # aIBlim = 4000  # in units of a0

    # # qds_Neg = qds_Neg_NoPP_CS; qds_Pos = qds_Pos_NoPP_CS
    # qds_Neg = qds_Neg_Homog_CS; qds_Pos = qds_Pos_Homog_CS

    # x_ds_Neg = qds_Neg['X']
    # x_ds_Pos = qds_Pos['X']

    # aIBiVals_Neg = x_ds_Neg['aIBi'].values
    # aIBVals_Neg = (1 / aIBiVals_Neg) / a0_th
    # aIB_mask_Neg = np.abs(aIBVals_Neg) <= aIBlim; aIBiVals_Neg = aIBiVals_Neg[aIB_mask_Neg]

    # aIBiVals_Pos = x_ds_Pos['aIBi'].values
    # aIBVals_Pos = (1 / aIBiVals_Pos) / a0_th
    # aIB_mask_Pos = np.abs(aIBVals_Pos) <= aIBlim; aIBiVals_Pos = aIBiVals_Pos[aIB_mask_Pos]

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

    # if BetaZero:
    #     aVals_Fixed = np.zeros(aIBiVals_Neg.size)
    # else:
    #     aVals_Fixed = np.empty(aIBiVals_Neg.size)
    #     for ind, aIBi in enumerate(aIBiVals_Neg):
    #         cParams = {'aIBi': aIBi}
    #         E_Pol_tck = pfs.V_Pol_interp(kgrid, X_Vals, cParams, sParams, trapParams)
    #         aVals_Fixed[ind] = interpolate.splev(0, E_Pol_tck, der=2)

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

    # xI_DatArray_LAB_Neg = np.empty(aIBiVals_Neg.size, dtype=np.object)
    # xI_FitArray_LAB_Neg = np.empty(aIBiVals_Neg.size, dtype=np.object)

    # xI_DatArray_Neg = np.empty(aIBiVals_Neg.size, dtype=np.object)
    # vI_DatArray_Neg = np.empty(aIBiVals_Neg.size, dtype=np.object)
    # xI_FitArray_Neg = np.empty(aIBiVals_Neg.size, dtype=np.object)
    # vI_FitArray_Neg = np.empty(aIBiVals_Neg.size, dtype=np.object)
    # R2_Array_Neg = np.empty(aIBiVals_Neg.size, dtype=np.object)
    # MSErr_Array_Neg = np.empty(aIBiVals_Neg.size, dtype=np.object)

    # y0Vals_Neg = np.empty(aIBiVals_Neg.size, dtype=np.object)
    # gammaVals_Neg = np.empty(aIBiVals_Neg.size)
    # betaVals = np.empty(aIBiVals_Neg.size)
    # gVals_Neg = np.empty(aIBiVals_Neg.size)
    # phiVals_Neg = np.empty(aIBiVals_Neg.size)
    # msVals_Neg = np.empty(aIBiVals_Neg.size)
    # x0Vals_Neg = np.empty(aIBiVals_Neg.size)
    # v0Vals_Neg = np.empty(aIBiVals_Neg.size)
    # for ind, aIBi in enumerate(aIBiVals_Neg):
    #     # if ind != 10:
    #     #     continue
    #     xVals = x_ds_Neg.sel(aIBi=aIBi).values
    #     vVals = np.gradient(xVals, tVals)
    #     x0 = xVals[0]
    #     v0 = vVals[0]
    #     x0Vals_Neg[ind] = x0; v0Vals_Neg[ind] = v0
    #     # v0 = (qds['P'].sel(aIBi=aIBi).isel(t=0).values - qds['Pph'].sel(aIBi=aIBi).isel(t=0).values) / mI
    #     y0 = [x0, v0]
    #     data = np.concatenate((xVals, vVals))
    #     if ind == 0:
    #         p0 = [1e-3]
    #         lowerbound = [0]
    #         upperbound = [np.inf]

    #     else:
    #         p0 = [gammaVals_Neg[ind - 1]]
    #         # lowerbound = [gammaVals_Neg[ind - 1], 0]
    #         lowerbound = [0]
    #         upperbound = [np.inf]
    #     popt, cov = curve_fit(lambda t, gamma: yint(t, gamma, betaVals_Fixed[ind], y0), tVals, data, p0=p0, bounds=(lowerbound, upperbound))
    #     gopt = popt[0]
    #     y0Vals_Neg[ind] = y0; gammaVals_Neg[ind] = gopt; betaVals[ind] = betaVals_Fixed[ind]
    #     gVals_Neg[ind], phiVals_Neg[ind] = gphiVals(gopt, betaVals[ind], omega_Imp_x, omega_BEC_osc, xB0)

    #     fitvals = yint(tVals, gammaVals_Neg[ind], betaVals[ind], y0Vals_Neg[ind])
    #     xfit = fitvals[0:tVals.size]
    #     vfit = fitvals[tVals.size:]
    #     xI_DatArray_Neg[ind] = xVals
    #     vI_DatArray_Neg[ind] = vVals
    #     xI_FitArray_Neg[ind] = xfit
    #     vI_FitArray_Neg[ind] = vfit
    #     R2_Array_Neg[ind] = r2_score(xVals, xfit)
    #     MSErr_Array_Neg[ind] = mean_squared_error(xVals, xfit)

    #     xI_DatArray_LAB_Neg[ind] = qds_Neg['XLab'].sel(aIBi=aIBi).values
    #     xI_FitArray_LAB_Neg[ind] = xfit + xBEC
    #     P = qds_Neg['P'].sel(aIBi=aIBi).isel(t=0).values
    #     Pph = qds_Neg['Pph'].sel(aIBi=aIBi).isel(t=0).values
    #     msVals_Neg[ind] = mI * P / (P - Pph)

    # xI_DatArray_LAB_Pos = np.empty(aIBiVals_Pos.size, dtype=np.object)
    # xI_FitArray_LAB_Pos = np.empty(aIBiVals_Pos.size, dtype=np.object)

    # xI_DatArray_Pos = np.empty(aIBiVals_Pos.size, dtype=np.object)
    # vI_DatArray_Pos = np.empty(aIBiVals_Pos.size, dtype=np.object)
    # xI_FitArray_Pos = np.empty(aIBiVals_Pos.size, dtype=np.object)
    # vI_FitArray_Pos = np.empty(aIBiVals_Pos.size, dtype=np.object)
    # R2_Array_Pos = np.empty(aIBiVals_Pos.size, dtype=np.object)
    # MSErr_Array_Pos = np.empty(aIBiVals_Pos.size, dtype=np.object)

    # y0Vals_Pos = np.empty(aIBiVals_Pos.size, dtype=np.object)
    # gammaVals_Pos = np.empty(aIBiVals_Pos.size)
    # betaVals = np.empty(aIBiVals_Pos.size)
    # gVals_Pos = np.empty(aIBiVals_Pos.size)
    # phiVals_Pos = np.empty(aIBiVals_Pos.size)
    # msVals_Pos = np.empty(aIBiVals_Pos.size)
    # x0Vals_Pos = np.empty(aIBiVals_Pos.size)
    # v0Vals_Pos = np.empty(aIBiVals_Pos.size)
    # for ind, aIBi in enumerate(aIBiVals_Pos):
    #     # if ind != 10:
    #     #     continue
    #     xVals = x_ds_Pos.sel(aIBi=aIBi).values
    #     vVals = np.gradient(xVals, tVals)
    #     x0 = xVals[0]
    #     v0 = vVals[0]
    #     x0Vals_Pos[ind] = x0; v0Vals_Pos[ind] = v0
    #     # v0 = (qds['P'].sel(aIBi=aIBi).isel(t=0).values - qds['Pph'].sel(aIBi=aIBi).isel(t=0).values) / mI
    #     y0 = [x0, v0]
    #     data = np.concatenate((xVals, vVals))
    #     if ind == 0:
    #         p0 = [1e-3]
    #         lowerbound = [0]
    #         upperbound = [np.inf]

    #     else:
    #         p0 = [gammaVals_Pos[ind - 1]]
    #         # lowerbound = [gammaVals_Pos[ind - 1], 0]
    #         lowerbound = [0]
    #         upperbound = [np.inf]
    #     popt, cov = curve_fit(lambda t, gamma: yint(t, gamma, betaVals_Fixed[ind], y0), tVals, data, p0=p0, bounds=(lowerbound, upperbound))
    #     gopt = popt[0]
    #     y0Vals_Pos[ind] = y0; gammaVals_Pos[ind] = gopt; betaVals[ind] = betaVals_Fixed[ind]
    #     gVals_Pos[ind], phiVals_Pos[ind] = gphiVals(gopt, betaVals[ind], omega_Imp_x, omega_BEC_osc, xB0)

    #     fitvals = yint(tVals, gammaVals_Pos[ind], betaVals[ind], y0Vals_Pos[ind])
    #     xfit = fitvals[0:tVals.size]
    #     vfit = fitvals[tVals.size:]
    #     xI_DatArray_Pos[ind] = xVals
    #     vI_DatArray_Pos[ind] = vVals
    #     xI_FitArray_Pos[ind] = xfit
    #     vI_FitArray_Pos[ind] = vfit
    #     R2_Array_Pos[ind] = r2_score(xVals, xfit)
    #     MSErr_Array_Pos[ind] = mean_squared_error(xVals, xfit)

    #     xI_DatArray_LAB_Pos[ind] = qds_Pos['XLab'].sel(aIBi=aIBi).values
    #     xI_FitArray_LAB_Pos[ind] = xfit + xBEC
    #     P = qds_Pos['P'].sel(aIBi=aIBi).isel(t=0).values
    #     Pph = qds_Pos['Pph'].sel(aIBi=aIBi).isel(t=0).values
    #     msVals_Pos[ind] = mI * P / (P - Pph)

    # gNeg_da = xr.DataArray(gammaVals_Neg, coords=[(1 / aIBiVals_Neg) / a0_th], dims=['aIB'])
    # gPos_da = xr.DataArray(gammaVals_Pos, coords=[(1 / aIBiVals_Pos) / a0_th], dims=['aIB'])

    # gNeg_da.to_netcdf(datafilepath + 'gammaNeg.nc')
    # gPos_da.to_netcdf(datafilepath + 'gammaPos.nc')

    # # # # FIG 7 - QUAD FIT TO DISP PARAMETER

    # weakLim = 50  # in units of a0

    # gNeg_da = xr.open_dataarray(datafilepath + 'gammaNeg.nc')
    # gPos_da = xr.open_dataarray(datafilepath + 'gammaPos.nc')

    # aIBVals_Neg = gNeg_da.coords['aIB'].values  # in units of a0
    # aIBVals_Pos = gPos_da.coords['aIB'].values  # in units of a0
    # gammaVals_Neg = gNeg_da.values
    # gammaVals_Pos = gPos_da.values

    # def p2Fit(aIB, param):
    #     return param * aIB**2

    # weakMask = np.abs(aIBVals_Neg) <= weakLim
    # aIBVals_NegWeak = aIBVals_Neg[weakMask]
    # gammaVals_NegWeak = gammaVals_Neg[weakMask]

    # daIB = aIBVals_NegWeak[1] - aIBVals_NegWeak[0]
    # # pguess = np.diff(gammaVals_NegWeak, n=2)[0] / daIB
    # pguess = np.gradient(gammaVals_NegWeak, aIBVals_NegWeak)[0]

    # popt, cov = curve_fit(p2Fit, aIBVals_NegWeak, gammaVals_NegWeak, p0=pguess)
    # p2param = popt[0]
    # print(pguess, p2param)

    # p2FitVals_Neg = p2Fit(aIBVals_Neg, p2param)

    # weakMask = np.abs(aIBVals_Pos) <= weakLim
    # aIBVals_PosWeak = aIBVals_Pos[weakMask]
    # gammaVals_PosWeak = gammaVals_Pos[weakMask]

    # daIB = aIBVals_PosWeak[1] - aIBVals_PosWeak[0]
    # # pguess = np.diff(gammaVals_PosWeak, n=2)[0] / daIB
    # pguess = np.gradient(gammaVals_PosWeak, aIBVals_PosWeak)[0]

    # popt, cov = curve_fit(p2Fit, aIBVals_PosWeak, gammaVals_PosWeak, p0=pguess)
    # p2param = popt[0]
    # print(pguess, p2param)

    # p2FitVals_Pos = p2Fit(aIBVals_Pos, p2param)

    # fig7, ax7 = plt.subplots()

    # aIBVals_Pos = aIBVals_Pos[::-1]; gammaVals_Pos = gammaVals_Pos[::-1]; p2FitVals_Pos = p2FitVals_Pos[::-1]

    # ax7.plot(-1 * aIBVals_Neg, gammaVals_Neg, 'g-', label=r'$a_{\rm IB}<0$')
    # ax7.plot(-1 * aIBVals_Neg[0:10], gammaVals_Neg[0:10], color='g', marker='s', mew=2, markerfacecolor='None', linestyle='None')

    # ax7.plot(aIBVals_Pos, gammaVals_Pos, 'r-', label=r'$a_{\rm IB}>0$')
    # ax7.plot(aIBVals_Pos[0:10], gammaVals_Pos[0:10], color='r', marker='x', mew=2, linestyle='None')

    # # ax7.plot(-1 * aIBVals_Neg, p2FitVals_Neg, color='b', linestyle=':', lw=2, label=r'$\gamma_{fit}^{Neg}=($' + '{:.1e}'.format(p2param) + r'$)a_{IB}^{2}$')
    # ax7.plot(-1 * aIBVals_Neg, p2FitVals_Neg, color='g', linestyle='dashed', lw=2, label='Quad Fit')
    # # ax7.plot(-1 * aIBVals_Neg[0:10], p2FitVals_Neg[0:10], color='b', marker='s', mew=2, markerfacecolor='None', linestyle='None')

    # # ax7.plot(aIBVals_Pos, p2FitVals_Pos, color='#650021', linestyle=':', lw=2, label=r'$\gamma_{fit}^{Pos}=($' + '{:.1e}'.format(p2param) + r'$)a_{IB}^{2}$')
    # ax7.plot(aIBVals_Pos, p2FitVals_Pos, color='r', linestyle=':', lw=2, label=r'Quad Fit')
    # # ax7.plot(aIBVals_Pos[0:10], p2FitVals_Pos[0:10], color='#650021', marker='x', mew=2, linestyle='None')

    # # ax7.plot(-1 * aIBVals_NegWeak[0] * np.ones(aIBVals_Neg.size), np.linspace(np.min(p2FitVals_Neg), np.max(p2FitVals_Neg), p2FitVals_Neg.size), 'y:', label=r'$a_{IB}=$' + '{:.1f}'.format(aIBVals_NegWeak[0]) + r' [$a_{0}$]')
    # # ax7.plot(-1 * aIBVals_NegWeak[-1] * np.ones(aIBVals_Neg.size), np.linspace(np.min(p2FitVals_Neg), np.max(p2FitVals_Neg), p2FitVals_Neg.size), 'm:', label=r'$a_{IB}=$' + '{:.1f}'.format(aIBVals_NegWeak[-1]) + r' [$a_{0}$]')
    # # ax7.plot(aIBVals_PosWeak[0] * np.ones(aIBVals_Pos.size), np.linspace(np.min(p2FitVals_Pos), np.max(p2FitVals_Pos), p2FitVals_Pos.size), 'y:', label=r'$a_{IB}=$' + '{:.1f}'.format(aIBVals_PosWeak[0]) + r' [$a_{0}$]')
    # # ax7.plot(aIBVals_PosWeak[-1] * np.ones(aIBVals_Pos.size), np.linspace(np.min(p2FitVals_Pos), np.max(p2FitVals_Pos), p2FitVals_Pos.size), 'm:', label=r'$a_{IB}=$' + '{:.1f}'.format(aIBVals_PosWeak[-1]) + r' [$a_{0}$]')

    # ax7.set_xscale('log')
    # ax7.set_yscale('log')
    # ax7.set_xlim([0.9 * np.min(aIBVals_Pos), 1e3])
    # ax7.set_xlim([0.9 * np.min(aIBVals_Pos), 2e3])
    # ax7.set_xlabel(r'$|a_{\rm IB}|/a_{0}$', fontsize=labelsize)
    # ax7.set_ylabel(r'Dissipation Constant $\gamma$')

    # ax7.legend(loc=2, ncol=2, fontsize=legendsize)
    # # handles, labels = ax7.get_legend_handles_labels()
    # # fig7.legend(handles, labels, ncol=2, loc='lower center', fontsize=legendsize)
    # # fig7.subplots_adjust(bottom=0.2, top=0.97, right=0.95, left=0.15)
    # fig7.subplots_adjust(bottom=0.2, top=0.97, right=0.95, left=0.2)

    # fig7.set_size_inches(3.9, 3.1)
    # fig7.savefig(figdatapath + '/Fig7.pdf')

    # plt.show()
