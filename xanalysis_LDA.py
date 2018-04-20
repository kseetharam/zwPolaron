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

if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    # gParams

    # (Lx, Ly, Lz) = (30, 30, 30)
    (Lx, Ly, Lz) = (20, 20, 20)
    # (Lx, Ly, Lz) = (10, 10, 10)
    (dx, dy, dz) = (0.2, 0.2, 0.2)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    aBB = 0.011
    P0 = 0.1
    tfin = 20
    dP_rat = 0.5
    print('dP/(mI*c): {0}'.format(dP_rat))

    # Toggle parameters

    toggleDict = {'Location': 'home', 'Dynamics': 'real', 'Interaction': 'on', 'InitCS': 'steadystate', 'InitCS_datapath': '', 'Coupling': 'twophonon', 'Grid': 'spherical',
                  'F_ext': 'on', 'BEC_density': 'on'}

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
        innerdatapath = innerdatapath + '_cart_extForce'
    elif toggleDict['Grid'] == 'spherical':
        innerdatapath = innerdatapath + '_spherical_extForce'

    if toggleDict['BEC_density'] == 'on':
        innerdatapath = innerdatapath + '_BECden'
    elif toggleDict['BEC_density'] == 'off':
        innerdatapath = innerdatapath

    if toggleDict['InitCS'] == 'file':
        innerdatapath = innerdatapath + '_ImDynStart_P_{:.1f}'.format(P0)
    elif toggleDict['InitCS'] == 'steadystate':
        innerdatapath = innerdatapath + '_SteadyStart_P_{:.1f}'.format(P0)

    if toggleDict['Coupling'] == 'frohlich':
        innerdatapath = innerdatapath + '_froh'
    elif toggleDict['Coupling'] == 'twophonon':
        innerdatapath = innerdatapath

    # # Concatenate Individual Datasets

    ds_list = []; F_list = []; aIBi_list = []
    for ind, filename in enumerate(os.listdir(innerdatapath)):
        if filename == 'LDA_Dataset_sph.nc':
            continue
        ds = xr.open_dataset(innerdatapath + '/' + filename)
        mIc = ds.attrs['mI'] * ds.attrs['nu']
        if ds.attrs['Delta_P'] / ds.attrs['Fext_mag'] < 0.005:
            print('EXCLUDED: ' + filename)
            continue
        if ds.attrs['Delta_P'] != dP_rat * mIc:
            continue

        print(filename)
        ds = ds.sel(t=slice(0, tfin))
        ds_list.append(ds)
        F_list.append(ds.attrs['Fext_mag'])
        aIBi_list.append(ds.attrs['aIBi'])

    dP = ds_list[0].attrs['Fext_mag'] * ds_list[0].attrs['TF']
    s = sorted(zip(aIBi_list, F_list, ds_list))
    g = itertools.groupby(s, key=lambda x: x[0])

    aIBi_keys = []; aIBi_groups = []; aIBi_ds_list = []
    for key, group in g:
        aIBi_keys.append(key)
        aIBi_groups.append(list(group))

    for ind, group in enumerate(aIBi_groups):
        aIBi = aIBi_keys[ind]
        _, F_list_temp, ds_list_temp = zip(*group)
        ds_temp = xr.concat(ds_list_temp, pd.Index(F_list_temp, name='F'))
        aIBi_ds_list.append(ds_temp)

    ds_tot = xr.concat(aIBi_ds_list, pd.Index(aIBi_keys, name='aIBi'))
    del(ds_tot.attrs['Fext_mag']); del(ds_tot.attrs['aIBi']); del(ds_tot.attrs['gIB']); del(ds_tot.attrs['TF'])
    ds_tot.attrs['Delta_P'] = dP
    ds_tot.to_netcdf(innerdatapath + '/LDA_Dataset_sph_dP_{:.2f}mIc.nc'.format(dP_rat))

    # # # Analysis of Total Dataset

    aIBi = -1.24
    qds = xr.open_dataset(innerdatapath + '/LDA_Dataset_sph_dP_{:.2f}mIc.nc'.format(dP_rat))
    attrs = qds.attrs
    dP = attrs['Delta_P']
    mI = attrs['mI']
    nu = attrs['nu']
    xi = attrs['xi']
    Ptot = dP + P0
    Fscale = 2 * np.pi * nu / xi**2
    tscale = xi / nu
    qds_aIBi = qds.sel(aIBi=aIBi).dropna('F')
    FVals = qds_aIBi['F'].values
    tVals = qds_aIBi['t'].values
    dt = tVals[1] - tVals[0]
    ts = tVals / tscale
    v0 = (P0 - qds_aIBi['Pph'].isel(F=0, t=0).values) / mI
    print('mI*c: {0}'.format(mI * nu))

    # # PHONON NUMBER VS TIME

    # for Find, F in enumerate(FVals):
    #     fig, ax = plt.subplots()
    #     ds = qds_aIBi.sel(F=F)['Nph'].sel(t=slice(0, 20 * tscale))
    #     ts = ds['t'].values / tscale
    #     ax.plot(ts, ds.values, label='')
    #     ax.plot(((dP / F) / tscale) * np.ones(ts.size), np.linspace(0, ds.max('t'), ts.size), 'g--', label=r'$TF$')
    #     ax.legend()
    #     ax.set_ylabel('Nph')
    #     ax.set_xlabel(r'$t$ [$\frac{\xi}{c}$]')
    #     ax.set_xscale('log')
    #     ax.set_yscale('log')
    #     ax.set_title(r'$F$' + '={:.2f} '.format(F / Fscale) + r'[$\frac{2 \pi c}{\xi^{2}}$]')
    #     plt.show()

    # # IMPURITY AND PHONON MOMENTUM VS TIME

    # for Find, F in enumerate(FVals):
    #     fig, ax = plt.subplots()
    #     qds_aIBi_F = qds_aIBi.sel(F=F)
    #     ax.plot(ts, qds_aIBi_F['P'].values, label=r'$P$')
    #     ax.plot(ts, (qds_aIBi_F['P'] - qds_aIBi_F['Pph']).values, label=r'$P_{I}$')
    #     ax.plot(ts, qds_aIBi_F['Pph'].values, label=r'$P_{ph}$')

    #     ax.plot(((dP / F) / tscale) * np.ones(ts.size), np.linspace(0, qds_aIBi_F['P'].max('t'), ts.size), 'g--', label=r'$T_{F}$')
    #     ax.plot(ts, Ptot * np.ones(ts.size), 'r--', label=r'$P_{0}+F \cdot T_{F}$')
    #     ax.legend()
    #     ax.set_ylim([-0.1 * Ptot, 1.1 * Ptot])
    #     ax.set_ylabel('Momentum')
    #     ax.set_xlabel(r'$t$ [$\frac{\xi}{c}$]')
    #     ax.set_title(r'$F$' + '={:.2f} '.format(F / Fscale) + r'[$\frac{2 \pi c}{\xi^{2}}$]')
    #     plt.show()

    # # POSITION VS TIME

    # x_ds = qds_aIBi['X']
    # for Find, F in enumerate(FVals):
    #     fig, ax = plt.subplots()
    #     ax.plot(ts, x_ds.sel(F=F).values, label='')
    #     ax.plot(((dP / F) / tscale) * np.ones(ts.size), np.linspace(0, x_ds.sel(F=F).max('t'), ts.size), 'g--', label=r'$T_{F}$')
    #     ax.legend()
    #     ax.set_ylabel(r'$<X>$')
    #     ax.set_xlabel(r'$t$ [$\frac{\xi}{c}$]')
    #     ax.set_title(r'$F$' + '={:.2f} '.format(F / Fscale) + r'[$\frac{2 \pi c}{\xi^{2}}$]')
    #     plt.show()

    # # VELOCITY VS TIME

    # v_ds = (qds_aIBi['X'].diff('t'))/dt.rename('v')
    # ts = v_ds['t'].values / tscale
    # for Find, F in enumerate(FVals):
    #     fig, ax = plt.subplots()
    #     ax.plot(ts, v_ds.sel(F=F).values, label='')
    #     ax.plot(((dP / F) / tscale) * np.ones(ts.size), np.linspace(0, v_ds.sel(F=F).max('t'), ts.size), 'g--', label=r'$T_{F}$')
    #     ax.legend()
    #     ax.set_ylabel(r'$v=\frac{d<X>}{dt}$')
    #     ax.set_xlabel(r'$t$ [$\frac{\xi}{c}$]')
    #     ax.set_title(r'$F$' + '={:.2f} '.format(F / Fscale) + r'[$\frac{2 \pi c}{\xi^{2}}$]')
    #     plt.show()

    # # VELOCITY AND EFFECTIVE MASS VS FORCE

    # x_ds = qds_aIBi['X']
    # FM_Vals = []; vf_Vals = []; ms_Vals = []
    # for Find, F in enumerate(FVals):
    #     if(F / Fscale < 1):
    #         continue
    #     FM_Vals.append(F)
    #     TF = dP / F
    #     XTail = x_ds.sel(F=F).sel(t=slice(TF + 2 * tscale, TF + 3 * tscale))
    #     tTail = XTail.coords['t']
    #     [vf, const] = np.polyfit(tTail.values, XTail.values, deg=1)
    #     vf_Vals.append(vf)
    #     ms_Vals.append(dP / (vf - v0))

    # FM_Vals = np.array(FM_Vals); vf_Vals = np.array(vf_Vals); ms_Vals = np.array(ms_Vals)
    # vf_ave = np.average(vf_Vals)
    # ms_ave = np.average(ms_Vals)

    # print(vf_ave, ms_ave / mI)

    # fig, ax = plt.subplots()

    # # ax.plot(FM_Vals / Fscale, vf_Vals, 'r-')
    # # ax.set_ylim([0.975 * vf_ave, 1.025 * vf_ave])
    # # ax.set_ylabel(r'$v_{f}=\frac{d<X>}{dt}|_{t=\infty}$')
    # # ax.set_xlabel(r'$F$ [$\frac{2 \pi c}{\xi^{2}}$]')
    # # ax.set_xscale('log')
    # # ax.set_title('Final (average) impurity velocity')

    # ax.plot(FM_Vals / Fscale, ms_Vals / mI, 'b-')
    # ax.set_ylim([0.975 * ms_ave / mI, 1.025 * ms_ave / mI])
    # ax.set_ylabel(r'$\frac{m^{*}}{m_{I}}=\frac{1}{m_{I}} (\frac{F \cdot T_{F}}{v_{f}})$')
    # ax.set_xlabel(r'$F$ [$\frac{2 \pi c}{\xi^{2}}$]')
    # # ax.set_xscale('log')
    # ax.set_title('Polaron Mass Enhancement vs. Applied Force ($P=0.1$)')

    # plt.show()

    # # EFFECTIVE MASS CALCULATION AND COMPARISON

    # aIBi_Vals = qds['aIBi'].values
    # vf_AVals = np.zeros(aIBi_Vals.size)
    # ms_AVals = np.zeros(aIBi_Vals.size)
    # for aind, aIBi in enumerate(aIBi_Vals):
    #     qds_aIBi = qds.sel(aIBi=aIBi).dropna('F')
    #     FVals = qds_aIBi['F'].values
    #     tVals = qds_aIBi['t'].values
    #     x_ds = qds_aIBi['X']
    #     v0 = (P0 - qds_aIBi['Pph'].isel(F=0, t=0).values) / mI
    #     FM_Vals = []; vf_Vals = []; ms_Vals = []
    #     for Find, F in enumerate(FVals):
    #         if(F / Fscale < 1):
    #             continue
    #         FM_Vals.append(F)
    #         TF = dP / F
    #         XTail = x_ds.sel(F=F).sel(t=slice(TF + 1 * tscale, TF + 2 * tscale))
    #         tTail = XTail.coords['t']
    #         [vf, const] = np.polyfit(tTail.values, XTail.values, deg=1)
    #         vf_Vals.append(vf)
    #         ms_Vals.append(dP / (vf - v0))
    #         # ms_Vals.append(Ptot / vf)

    #     vf_AVals[aind] = np.average(np.array(vf_Vals))
    #     ms_AVals[aind] = np.average(np.array(ms_Vals))

    # # Steady state calc

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

    # mI = 1.7
    # mB = 1
    # n0 = 1
    # gBB = (4 * np.pi / mB) * aBB
    # nu = pfs.nu(mB, n0, gBB)
    # xi = (8 * np.pi * n0 * aBB)**(-1 / 2)

    # Nsteps = 1e2
    # aSi_tck, PBint_tck = pfs.createSpline_grid(Nsteps, kgrid, mI, mB, n0, gBB)
    # SS_ms_Avals = np.zeros(aIBi_Vals.size)

    # for Aind, aIBi in enumerate(aIBi_Vals):
    #     DP = pfs.DP_interp(0, P0, aIBi, aSi_tck, PBint_tck)
    #     aSi = pfs.aSi_interp(DP, aSi_tck)
    #     PB_Val = pfs.PB_interp(DP, aIBi, aSi_tck, PBint_tck)
    #     SS_ms_Avals[Aind] = pfs.effMass(P0, PB_Val, mI)

    # mE = ms_AVals / mI
    # SS_mE = SS_ms_Avals / mI
    # print('Force Protocol: {0}'.format(mE))
    # print('Steady State: {0}'.format(SS_mE))
    # mE_diff = np.abs(mE - SS_mE) / SS_mE
    # print('Percentage Error: {0}'.format(mE_diff * 100))

    # fig, ax = plt.subplots()
    # ax.plot(aIBi_Vals, mE, 'ro', label='Force Protocol Calculation')
    # ax.plot(aIBi_Vals, SS_mE, 'bo', label='Analytical Steady State Calculation')
    # ax.legend()
    # ax.set_ylabel(r'$\frac{m^{*}}{m_{I}}$')
    # ax.set_xlabel(r'$a_{IB}^{-1}$')
    # ax.set_title('Polaron Mass Enhancement vs. Interaction Strength ($P=0.1$)')
    # plt.show()

    # # COMBINING EFFECTIVE MASS

    # mE_da = xr.DataArray(mE, coords=[aIBi_Vals], dims=['aIBi'])
    # SSmE_da = xr.DataArray(SS_mE, coords=[aIBi_Vals], dims=['aIBi'])
    # mE_ds = xr.Dataset({'mE_file': mE_da, 'SSmE': SSmE_da}, coords={'aIBi': aIBi_Vals})
    # mE_ds.to_netcdf(datapath + '/extF_mE.nc')

    # with xr.open_dataset(datapath + '/extF_mE.nc') as mE_ds:
    #     mE_ds['mE_s'] = mE_da
    #     mE_ds_new = mE_ds.copy(deep=True)
    # mE_ds_new.to_netcdf(datapath + '/extF_mE.nc')

    # with xr.open_dataset(datapath + '/extF_mE.nc') as mE_ds:
    #     fig, ax = plt.subplots()
    #     aIBi_Vals = mE_ds['aIBi'].values
    #     mE = mE_ds['mE_file']
    #     mE_s = mE_ds['mE_s']
    #     ax.plot(aIBi_Vals, mE, 'ro', label='Force Protocol Imdyn State')
    #     ax.plot(aIBi_Vals, mE_s, 'go', label='Force Protocol Steady State')
    #     ax.plot(aIBi_Vals, SS_mE, 'bo', label='Analytical Steady State')
    #     ax.legend()
    #     ax.set_ylabel(r'$\frac{m^{*}}{m_{I}}$')
    #     ax.set_xlabel(r'$a_{IB}^{-1}$')
    #     ax.set_title('Polaron Mass Enhancement vs. Interaction Strength ($P=0.1$)')
    #     plt.show()
