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
    # ds_tot.attrs['Delta_P'] = dP
    # ds_tot.to_netcdf(innerdatapath + '/LDA_Dataset_sph.nc')

    # # # Analysis of Total Dataset

    aIBi = 0.1
    qds = xr.open_dataset(innerdatapath + '/LDA_Dataset_sph.nc')
    attrs = qds.attrs
    dP = attrs['Delta_P']
    mI = attrs['mI']
    Fscale = attrs['nu'] / attrs['xi']**2
    FVals = qds['F'].values
    tVals = qds['t'].values
    qds_aIBi = qds.sel(aIBi=aIBi)
    print(attrs['xi'] / attrs['nu'])

    # # TOTAL MOMENTUM VS TIME

    # for Find, F in enumerate(FVals):
    #     fig, ax = plt.subplots()
    #     qds_aIBi.sel(F=F)['P'].plot(ax=ax, label='')
    #     ax.plot((dP / F) * np.ones(tVals.size), np.linspace(0, qds_aIBi.sel(F=F)['P'].max('t'), tVals.size), 'g--', label=r'$TF$')
    #     ax.plot(tVals, dP * np.ones(tVals.size), 'r--', label=r'$\Delta P=F \cdot T_{F}$')
    #     ax.legend()
    #     ax.set_xlim([0, 20])
    #     ax.set_ylabel('P')
    #     ax.set_xlabel('t')
    #     ax.set_title(r'$\frac{F}{\eta}$' + '={0} with '.format(F / Fscale) + r'$\eta=\frac{c}{\xi^{2}}$')
    #     plt.show()

    # # PHONON NUMBER VS TIME

    # for Find, F in enumerate(FVals):
    #     fig, ax = plt.subplots()
    #     qds_aIBi.sel(F=F)['Nph'].plot(ax=ax, label='')
    #     ax.plot((dP / F) * np.ones(tVals.size), np.linspace(0, qds_aIBi.sel(F=F)['Nph'].max('t'), tVals.size), 'g--', label=r'$TF$')
    #     ax.legend()
    #     ax.set_xlim([0, 25])
    #     ax.set_ylabel('Nph')
    #     ax.set_xlabel('t')
    #     ax.set_title(r'$\frac{F}{\eta}$' + '={0} with '.format(F / Fscale) + r'$\eta=\frac{c}{\xi^{2}}$')
    #     plt.show()

    # IMPURITY AND PHONON MOMENTUM VS TIME

    for Find, F in enumerate(FVals):
        fig, ax = plt.subplots()
        qds_aIBi_F = qds_aIBi.sel(F=F)
        qds_aIBi_F['P'].plot(ax=ax, label=r'$P$')
        (qds_aIBi_F['P'] - qds_aIBi_F['Pph']).plot(ax=ax, label=r'$P_{I}$')
        qds_aIBi_F['Pph'].plot(ax=ax, label=r'$P_{ph}$')
        ax.plot((dP / F) * np.ones(tVals.size), np.linspace(0, qds_aIBi_F['P'].max('t'), tVals.size), 'g--', label=r'$T_{F}$')
        ax.plot(tVals, dP * np.ones(tVals.size), 'r--', label=r'$\Delta P=F \cdot T_{F}$')
        ax.legend()
        ax.set_xlim([0, 25])
        ax.set_ylim([-0.1 * dP, 1.1 * dP])
        ax.set_ylabel('Momentum')
        ax.set_xlabel('t')
        ax.set_title(r'$\frac{F}{\eta}$' + '={0} with '.format(F / Fscale) + r'$\eta=\frac{c}{\xi^{2}}$')
        plt.show()

    # # POSITION VS TIME

    # x_ds = qds_aIBi['X']
    # for Find, F in enumerate(FVals):
    #     fig, ax = plt.subplots()
    #     x_ds.sel(F=F).plot(ax=ax, label='')
    #     ax.plot((dP / F) * np.ones(tVals.size), np.linspace(0, x_ds.sel(F=F).max('t'), tVals.size), 'g--', label=r'$T_{F}$')
    #     ax.legend()
    #     ax.set_xlim([0, 25])
    #     ax.set_ylabel(r'$<X>$')
    #     ax.set_xlabel('t')
    #     ax.set_title(r'$\frac{F}{\eta}$' + '={0} with '.format(F / Fscale) + r'$\eta=\frac{c}{\xi^{2}}$')
    #     plt.show()

    # # VELOCITY VS TIME

    # v_ds = (qds_aIBi['X'].diff('t')).rename('v')
    # for Find, F in enumerate(FVals):
    #     fig, ax = plt.subplots()
    #     v_ds.sel(F=F).plot(ax=ax, label='')
    #     ax.plot((dP / F) * np.ones(tVals.size), np.linspace(0, v_ds.sel(F=F).max('t'), tVals.size), 'g--', label=r'$T_{F}$')
    #     ax.legend()
    #     ax.set_xlim([0, 25])
    #     ax.set_ylabel(r'$v=\frac{d<X>}{dt}$')
    #     ax.set_xlabel('t')
    #     ax.set_title(r'$\frac{F}{\eta}$' + '={0} with '.format(F / Fscale) + r'$\eta=\frac{c}{\xi^{2}}$')
    #     plt.show()

    # # VELOCITY AND EFFECTIVE MASS VS FORCE

    # x_ds = qds_aIBi['X'].dropna('t')
    # numPoints = 10
    # vf_Vals = np.zeros(FVals.size)
    # ms_Vals = np.zeros(FVals.size)
    # for Find, F in enumerate(FVals):
    #     XTail = x_ds.sel(F=F).isel(t=np.arange(-1 * numPoints, 0))
    #     tTail = XTail.coords['t']
    #     [vf_Vals[Find], const] = np.polyfit(tTail.values, XTail.values, deg=1)
    #     ms_Vals[Find] = dP / vf_Vals[Find]

    # vf_ave = np.average(vf_Vals)
    # ms_ave = np.average(ms_Vals)

    # print(vf_ave, ms_ave)

    # fig, ax = plt.subplots()

    # # ax.plot(FVals / Fscale, vf_Vals, 'r-')
    # # ax.set_ylim([0.975 * vf_ave, 1.025 * vf_ave])
    # # ax.set_ylabel(r'$v_{f}=\frac{d<X>}{dt}|_{t=\infty}$')
    # # ax.set_xlabel(r'$\frac{F}{\eta}$' + ' with ' + r'$\eta=\frac{c}{\xi^{2}}$')
    # # ax.set_xscale('log')
    # # ax.set_title('Final (average) impurity velocity')

    # ax.plot(FVals / Fscale, ms_Vals / mI, 'b-')
    # ax.set_ylim([0.975 * ms_ave / mI, 1.025 * ms_ave / mI])
    # ax.set_ylabel(r'$\frac{m^{*}}{m_{I}}=\frac{1}{m_{I}} (\frac{F \cdot T_{F}}{v_{f}})$')
    # ax.set_xlabel(r'$\frac{F}{\eta}$' + ' with ' + r'$\eta=\frac{c}{\xi^{2}}$')
    # ax.set_xscale('log')
    # ax.set_title('Polaron Mass Enhancement vs. Applied Force ($P=0.1$)')

    # plt.show()

    # # EFFECTIVE MASS CALCULATION AND COMPARISON

    # aIBi_Vals = qds['aIBi'].values
    # vf_AVals = np.zeros(aIBi_Vals.size)
    # ms_AVals = np.zeros(aIBi_Vals.size)
    # for aind, aIBi in enumerate(aIBi_Vals):
    #     x_ds = qds.sel(aIBi=aIBi)['X'].dropna('t')
    #     numPoints = 10
    #     vf_Vals = np.zeros(FVals.size)
    #     ms_Vals = np.zeros(FVals.size)
    #     for Find, F in enumerate(FVals):
    #         XTail = x_ds.sel(F=F).isel(t=np.arange(-1 * numPoints, 0))
    #         tTail = XTail.coords['t']
    #         [vf_Vals[Find], const] = np.polyfit(tTail.values, XTail.values, deg=1)
    #         ms_Vals[Find] = dP / vf_Vals[Find]

    #     vf_AVals[aind] = np.average(vf_Vals)
    #     ms_AVals[aind] = np.average(ms_Vals)

    # # Manual input for high interaction strength

    # aIBi_Large = aIBi_Vals[aIBi_Vals > 0]
    # F_fit = 5.02 * Fscale
    # for aLind, aIBi in enumerate(aIBi_Large):
    #     x_ds = qds.sel(aIBi=aIBi).sel(F=F)['X'].dropna('t')
    #     XTail = x_ds.sel(t=slice(3, 4))
    #     tTail = XTail.coords['t']
    #     ind = -1 * len(aIBi_Large) + aLind
    #     [vf_AVals[ind], const] = np.polyfit(tTail.values, XTail.values, deg=1)
    #     ms_AVals[ind] = dP / vf_AVals[ind]

    # # Steady state calc

    # (Lx, Ly, Lz) = (20, 20, 20)
    # (dx, dy, dz) = (0.2, 0.2, 0.2)

    # # (Lx, Ly, Lz) = (21, 21, 21)
    # # (dx, dy, dz) = (0.25, 0.25, 0.25)

    # NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    # NGridPoints_desired = (1 + 2 * Lx / dx) * (1 + 2 * Lz / dz)
    # Ntheta = 50
    # Nk = np.ceil(NGridPoints_desired / Ntheta)

    # theta_max = np.pi
    # thetaArray, dtheta = np.linspace(0, theta_max, Ntheta, retstep=True)

    # # k_max = np.sqrt((np.pi / dx)**2 + (np.pi / dy)**2 + (np.pi / dz)**2)
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
    # aBB = 0.062
    # gBB = (4 * np.pi / mB) * aBB
    # nu = pfs.nu(gBB)
    # xi = (8 * np.pi * n0 * aBB)**(-1 / 2)

    # Nsteps = 1e2
    # pfs.createSpline_grid(Nsteps, kgrid, mI, mB, n0, gBB)

    # aSi_tck = np.load('aSi_spline_sph.npy')
    # PBint_tck = np.load('PBint_spline_sph.npy')

    # P = 0.1
    # SS_ms_Avals = np.zeros(aIBi_Vals.size)

    # for Aind, aIBi in enumerate(aIBi_Vals):
    #     DP = pfs.DP_interp(0, P, aIBi, aSi_tck, PBint_tck)
    #     aSi = pfs.aSi_interp(DP, aSi_tck)
    #     PB_Val = pfs.PB_interp(DP, aIBi, aSi_tck, PBint_tck)
    #     # Pcrit = PCrit_grid(kgrid, aIBi, mI, mB, n0, gBB)
    #     # En = Energy(P, PB_Val, aIBi, aSi, mI, mB, n0)
    #     # nu_const = nu(gBB)
    #     SS_ms_Avals[Aind] = pfs.effMass(P, PB_Val, mI)
    #     # gIB = g(kgrid, aIBi, mI, mB, n0, gBB)
    #     # Nph = num_phonons(kgrid, aIBi, aSi, DP, mI, mB, n0, gBB)
    #     # Z_factor = z_factor(kgrid, aIBi, aSi, DP, mI, mB, n0, gBB)

    # mE = ms_AVals / mI
    # SS_mE = SS_ms_Avals / mI
    # print('Force Protocol: {0}'.format(mE))
    # print('Steady State: {0}'.format(SS_mE))
    # mE_diff = np.abs(mE - SS_mE) / mE
    # print('Percentage Error: {0}'.format(mE_diff * 100))

    # fig, ax = plt.subplots()
    # ax.plot(aIBi_Vals, mE, 'ro', label='Force Protocol Calculation')
    # ax.plot(aIBi_Vals, SS_mE, 'bo', label='Analytical Steady State Calculation')
    # ax.legend()
    # ax.set_ylabel(r'$\frac{m^{*}}{m_{I}}$')
    # ax.set_xlabel(r'$a_{IB}^{-1}$')
    # ax.set_title('Polaron Mass Enhancement vs. Interaction Strength ($P=0.1$)')
    # plt.show()
