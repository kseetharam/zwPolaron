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

    # Toggle parameters

    toggleDict = {'Location': 'work', 'Dynamics': 'real', 'Coupling': 'twophonon', 'Grid': 'spherical', 'InitCS': 'steadystate'}

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
        innerdatapath = innerdatapath + '_spherical_extForce'

    if toggleDict['InitCS'] == 'file':
        homogdatapath = innerdatapath + '_ImDynStart_P_{:.1f}'.format(P0)
        dendatapath = innerdatapath + '_BECden_ImDynStart_P_{:.1f}'.format(P0)
    elif toggleDict['InitCS'] == 'steadystate':
        homogdatapath = innerdatapath + '_SteadyStart_P_{:.1f}'.format(P0)
        dendatapath = innerdatapath + '_BECden_SteadyStart_P_{:.1f}'.format(P0)

    # # # Analysis of Total Dataset

    aIBi = -1.77
    dPList = [0.50, 1.54, 3.00]
    dP_sub = dPList[0]
    dP_sup = dPList[2]
    qds_subsonic_homog = xr.open_dataset(homogdatapath + '/LDA_Dataset_sph_dP_{:.2f}mIc.nc'.format(dP_sub))
    qds_subsonic_den = xr.open_dataset(dendatapath + '/LDA_Dataset_sph_dP_{:.2f}mIc.nc'.format(dP_sub))
    qds_supersonic_homog = xr.open_dataset(homogdatapath + '/LDA_Dataset_sph_dP_{:.2f}mIc.nc'.format(dP_sup))
    qds_supersonic_den = xr.open_dataset(dendatapath + '/LDA_Dataset_sph_dP_{:.2f}mIc.nc'.format(dP_sup))

    attrs = qds_subsonic_homog.attrs
    mI = attrs['mI']
    mB = attrs['mB']
    n0 = attrs['n0']
    gBB = attrs['gBB']
    nu = attrs['nu']
    xi = attrs['xi']
    Ptot_sub = dP_sub + P0
    Ptot_sup = dP_sup + P0
    Fscale = 2 * np.pi * nu / xi**2
    tscale = xi / nu
    FVals = qds_subsonic_homog['F'].values
    tVals = qds_subsonic_homog['t'].values
    dt = tVals[1] - tVals[0]
    ts = tVals / tscale
    v0 = (P0 - qds_subsonic_homog['Pph'].isel(F=0, t=0).sel(aIBi=aIBi).values) / mI
    print('mI*c: {0}'.format(mI * nu))

    # # VELOCITY VS TIME

    # v_ds = (qds_aIBi['X'].diff('t')).rename('v')
    # ts = v_ds['t'].values / tscale
    # for Find, F in enumerate(FVals):
    #     fig, ax = plt.subplots()
    #     ax.plot(ts, v_ds.sel(F=F).values / dt - v0, label='')
    #     ax.plot(((dP / F) / tscale) * np.ones(ts.size), np.linspace(0, v_ds.sel(F=F).max('t') / dt - v0, ts.size), 'g--', label=r'$T_{F}$')
    #     ax.legend()
    #     ax.set_ylabel(r'$v=\frac{d<X>}{dt}$')
    #     ax.set_xlabel(r'$t$ [$\frac{\xi}{c}$]')
    #     ax.set_title(r'$F$' + '={:.2f} '.format(F / Fscale) + r'[$\frac{2 \pi c}{\xi^{2}}$]')
    #     plt.show()

    # # VELOCITY AND EFFECTIVE MASS VS FORCE

    # x_ds = qds_aIBi['X']
    # v_ds = (qds_aIBi['X'].diff('t') / dt).rename('v')
    # FM_Vals = []; vf_Vals = []; ms_Vals = []
    # for Find, F in enumerate(FVals):
    #     if(F / Fscale < 1):
    #         continue
    #     FM_Vals.append(F)
    #     TF = dP / F
    #     XTail = x_ds.sel(F=F).sel(t=slice(TF + 2 * tscale, TF + 3 * tscale))
    #     tTail = XTail.coords['t']
    #     [vf, const] = np.polyfit(tTail.values, XTail.values, deg=1)
    #     vf = vf - v0
    #     vf_direct = v_ds.sel(F=F).sel(t=TF + dt, method='nearest').values - v0
    #     print(vf, vf_direct)
    #     vf = vf_direct
    #     vf_Vals.append(vf)
    #     ms_Vals.append(dP / vf)

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
    # # ax.set_ylim([0.975 * ms_ave / mI, 1.025 * ms_ave / mI])
    # ax.set_ylabel(r'$\frac{m^{*}}{m_{I}}=\frac{1}{m_{I}} (\frac{F \cdot T_{F}}{v_{f}})$')
    # ax.set_xlabel(r'$F$ [$\frac{2 \pi c}{\xi^{2}}$]')
    # # ax.set_xscale('log')
    # ax.set_title('Polaron Mass Enhancement vs. Applied Force ($P=0.1$)')

    # plt.show()

    # EFFECTIVE MASS CALCULATION AND COMPARISON

    def effMass(qds):
        dP = qds.attrs['Delta_P']
        mI = qds.attrs['mI']
        aIBi_Vals = qds['aIBi'].values
        vf_AVals = np.zeros(aIBi_Vals.size)
        mE_AVals = np.zeros(aIBi_Vals.size)
        for aind, aIBi in enumerate(aIBi_Vals):
            qds_aIBi = qds.sel(aIBi=aIBi).dropna('F')
            FVals = qds_aIBi['F'].values
            tVals = qds_aIBi['t'].values
            x_ds = qds_aIBi['X']
            v_ds = (qds_aIBi['X'].diff('t') / dt).rename('v')
            v0 = (P0 - qds_aIBi['Pph'].isel(F=0, t=0).values) / mI
            FM_Vals = []; vf_Vals = []; ms_Vals = []
            for Find, F in enumerate(FVals):
                if(F / Fscale < 1):  # why skipping the first force value?
                    continue
                FM_Vals.append(F)
                TF = dP / F
                XTail = x_ds.sel(F=F).sel(t=slice(TF + 1 * tscale, TF + 2 * tscale))
                tTail = XTail.coords['t']

                # fig, ax = plt.subplots()
                # ax.plot(v_ds.coords['t'].values, v_ds.sel(F=F).values)
                # plt.show()

                [vf, const] = np.polyfit(tTail.values, XTail.values, deg=1)
                vf_direct = v_ds.sel(F=F).sel(t=TF + dt, method='nearest').values
                # vf = vf_direct
                vf_Vals.append(vf)
                ms_Vals.append(dP / (vf - v0))
                # ms_Vals.append(Ptot / vf)

            vf_AVals[aind] = np.average(np.array(vf_Vals))
            mE_AVals[aind] = np.average(np.array(ms_Vals)) / mI
        return mE_AVals, aIBi_Vals

    mE_subsonic_homog, aIBi_subsonic_homog = effMass(qds_subsonic_homog)
    mE_subsonic_den, aIBi_subsonic_den = effMass(qds_subsonic_den)
    mE_supersonic_homog, aIBi_supersonic_homog = effMass(qds_supersonic_homog)
    mE_supersonic_den, aIBi_supersonic_den = effMass(qds_supersonic_den)

    # Steady state calc

    aIBi_Vals = aIBi_subsonic_homog
    NGridPoints_desired = (1 + 2 * Lx / dx) * (1 + 2 * Lz / dz)
    Ntheta = 50
    Nk = np.ceil(NGridPoints_desired / Ntheta)

    theta_max = np.pi
    thetaArray, dtheta = np.linspace(0, theta_max, Ntheta, retstep=True)

    k_max = ((2 * np.pi / dx)**3 / (4 * np.pi / 3))**(1 / 3)

    k_min = 1e-5
    kArray, dk = np.linspace(k_min, k_max, Nk, retstep=True)
    if dk < k_min:
        print('k ARRAY GENERATION ERROR')

    kgrid = Grid.Grid("SPHERICAL_2D")
    kgrid.initArray_premade('k', kArray)
    kgrid.initArray_premade('th', thetaArray)

    Nsteps = 1e2
    aSi_tck, PBint_tck = pfs.createSpline_grid(Nsteps, kgrid, mI, mB, n0, gBB)
    SS_ms_Avals = np.zeros(aIBi_Vals.size)

    for Aind, aIBi in enumerate(aIBi_Vals):
        DP = pfs.DP_interp(0, P0, aIBi, aSi_tck, PBint_tck)
        aSi = pfs.aSi_interp(DP, aSi_tck)
        PB_Val = pfs.PB_interp(DP, aIBi, aSi_tck, PBint_tck)
        SS_ms_Avals[Aind] = pfs.effMass(P0, PB_Val, mI)

    mE_steadystate = SS_ms_Avals / mI
    print('Percentage Error (Sub Homog): {0}'.format(100 * np.abs(mE_subsonic_homog - mE_steadystate) / mE_steadystate))
    print('Percentage Error (Sub BECden): {0}'.format(100 * np.abs(mE_subsonic_den - mE_steadystate) / mE_steadystate))
    print('Percentage Error (Sup Homog): {0}'.format(100 * np.abs(mE_supersonic_homog - mE_steadystate) / mE_steadystate))
    print('Percentage Error (Sup BECden): {0}'.format(100 * np.abs(mE_supersonic_den - mE_steadystate) / mE_steadystate))

    fig1, axes1 = plt.subplots(nrows=1, ncols=2)
    axes1[1].plot(aIBi_Vals, mE_subsonic_den, 'rx', label='Force Protocol (Harmonic BEC Trap)')
    axes1[1].plot(aIBi_Vals, mE_subsonic_homog, 'go', markerfacecolor='none', label='Force Protocol (Homogenous BEC)')
    axes1[1].plot(aIBi_Vals, mE_steadystate, 'bs', markerfacecolor='none', label='Analytical Steady State')
    axes1[1].legend()
    axes1[1].set_ylabel(r'$\frac{m^{*}}{m_{I}}$')
    axes1[1].set_xlabel(r'$a_{IB}^{-1}$')
    axes1[1].set_title('Polaron Mass Enhancement (Subsonic Case)')

    fig2, axes2 = plt.subplots(nrows=1, ncols=2)
    axes2[1].plot(aIBi_Vals, mE_supersonic_den, 'rx', label='Force Protocol (Harmonic BEC Trap)')
    axes2[1].plot(aIBi_Vals, mE_supersonic_homog, 'go', markerfacecolor='none', label='Force Protocol (Homogenous BEC)')
    axes2[1].plot(aIBi_Vals, mE_steadystate, 'bs', markerfacecolor='none', label='Analytical Steady State')
    axes2[1].legend()
    axes2[1].set_ylabel(r'$\frac{m^{*}}{m_{I}}$')
    axes2[1].set_xlabel(r'$a_{IB}^{-1}$')
    axes2[1].set_title('Polaron Mass Enhancement (Supersonic Case)')
    plt.show()

    # Note: initial P_tot = 0.1 for all force protocols above (and P=0.1 for analytical steady state)
