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

    print(0.1 / (mI * nu), mI / mB, attrs['k_mag_cutoff'] * xi, 1 / (aIBi * xi))

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
                # XTail = x_ds.sel(F=F).sel(t=slice(TF + 1 * tscale, TF + 2 * tscale))
                XTail = x_ds.sel(F=F).sel(t=slice(TF + 2 * tscale, TF + 6 * tscale))
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

    dP_subsonic = qds_subsonic_homog.attrs['Delta_P']
    dP_supersonic = qds_supersonic_homog.attrs['Delta_P']

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
    aIBi_Vals_steadystate = np.linspace(np.min(aIBi_Vals), 0.25, 30)
    print(aIBi_Vals)
    SS_ms_Avals = np.zeros(aIBi_Vals_steadystate.size)

    for Aind, aIBi in enumerate(aIBi_Vals_steadystate):
        DP = pfs.DP_interp(0, P0, aIBi, aSi_tck, PBint_tck)
        aSi = pfs.aSi_interp(DP, aSi_tck)
        PB_Val = pfs.PB_interp(DP, aIBi, aSi_tck, PBint_tck)
        SS_ms_Avals[Aind] = pfs.effMass(P0, PB_Val, mI)

    mE_steadystate = SS_ms_Avals / mI

    SS_ms_Avals_data = np.zeros(aIBi_Vals.size)
    for Aind, aIBi in enumerate(aIBi_Vals):
        DP = pfs.DP_interp(0, P0, aIBi, aSi_tck, PBint_tck)
        aSi = pfs.aSi_interp(DP, aSi_tck)
        PB_Val = pfs.PB_interp(DP, aIBi, aSi_tck, PBint_tck)
        SS_ms_Avals_data[Aind] = pfs.effMass(P0, PB_Val, mI)

    mE_steadystate_data = SS_ms_Avals_data / mI

    print('Percentage Error (Sub Homog): {0}'.format(100 * np.abs(mE_subsonic_homog - mE_steadystate_data) / mE_steadystate_data))
    print('Percentage Error (Sub BECden): {0}'.format(100 * np.abs(mE_subsonic_den - mE_steadystate_data) / mE_steadystate_data))
    print('Percentage Error (Sup Homog): {0}'.format(100 * np.abs(mE_supersonic_homog - mE_steadystate_data) / mE_steadystate_data))
    print('Percentage Error (Sup BECden): {0}'.format(100 * np.abs(mE_supersonic_den - mE_steadystate_data) / mE_steadystate_data))

    # VELOCITY VS TIME

    aIBi = -5
    v_ds_subsonic_homog = (qds_subsonic_homog.sel(aIBi=aIBi)['X'].diff('t')).rename('v')
    FVals_sub = v_ds_subsonic_homog.coords['F'].values
    # Find_subsonic_homog = np.argwhere(FVals_sub > Fscale)[0][0]
    Find_subsonic_homog = 0
    F_sub = FVals_sub[Find_subsonic_homog]
    vs_subsonic_homog = (v_ds_subsonic_homog.isel(F=Find_subsonic_homog).values / dt - v0) / nu
    ts_subsonic_homog = v_ds_subsonic_homog.coords['t'].values / tscale

    v_ds_supersonic_homog = (qds_supersonic_homog.sel(aIBi=aIBi)['X'].diff('t')).rename('v')
    Find_supersonic_homog = np.argwhere(v_ds_supersonic_homog.coords['F'].values > Fscale)[0][0]
    FVals_super = v_ds_supersonic_homog.coords['F'].values
    # Find_supersonic_homog = np.argwhere(FVals_super > Fscale)[0][0]
    Find_supersonic_homog = 0
    F_super = FVals_super[Find_supersonic_homog]
    vs_supersonic_homog = (v_ds_supersonic_homog.isel(F=Find_supersonic_homog).values / dt - v0) / nu
    ts_supersonic_homog = v_ds_supersonic_homog.coords['t'].values / tscale

    fig1, axes1 = plt.subplots(nrows=1, ncols=2)
    axes1[0].plot(ts_subsonic_homog, vs_subsonic_homog, label='')
    axes1[0].plot(((dP_subsonic / F_sub) / tscale) * np.ones(ts.size), np.linspace(0, np.max(vs_subsonic_homog), ts.size), 'g--', label=r'$T_{F}$')
    # axes1[0].plot(ts_subsonic_homog, np.ones(ts_subsonic_homog.size), 'r--', label=r'$c_{BEC}$')
    axes1[0].legend()
    axes1[0].set_ylabel(r'$\frac{1}{c_{BEC}}\frac{d<X>}{dt}$')
    axes1[0].set_xlabel(r'$t$ [$\frac{\xi}{c}$]')
    axes1[0].set_title('Average Impurity Velocity (' + r'$F$' + '={:.2f} '.format(F_sub / Fscale) + r'[$\frac{2 \pi c}{\xi^{2}}$]' + ')')

    # axes1[1].plot(aIBi_Vals, mE_subsonic_den, 'go', mew=1, ms=10, label='Force Protocol (Harmonic BEC Trap)')
    axes1[1].plot(aIBi_Vals, mE_subsonic_homog, 'rx', mew=1, ms=10, markerfacecolor='none', label='Force Protocol (Homogenous BEC)')
    axes1[1].plot(aIBi_Vals_steadystate, mE_steadystate, 'b-', mew=1, ms=10, markerfacecolor='none', label='Analytical Steady State')
    axes1[1].plot(aIBi_Vals, mE_steadystate_data, 'bs', mew=1, ms=10, markerfacecolor='none', label='')
    axes1[1].legend()
    axes1[1].set_ylim([0, 19.5])
    axes1[1].set_ylabel(r'$\frac{m^{*}}{m_{I}}$')
    axes1[1].set_xlabel(r'$a_{IB}^{-1}$')
    axes1[1].set_title('Polaron Mass Enhancement (Subsonic Case)')

    fig2, axes2 = plt.subplots(nrows=1, ncols=2)
    axes2[0].plot(ts_supersonic_homog, vs_supersonic_homog, label='')
    axes2[0].plot(((dP_supersonic / F_super) / tscale) * np.ones(ts.size), np.linspace(0, np.max(vs_supersonic_homog), ts.size), 'g--', label=r'$T_{F}$')
    axes2[0].plot(ts_supersonic_homog, np.ones(ts_supersonic_homog.size), 'r--', label=r'$c_{BEC}$')
    axes2[0].legend()
    axes2[0].set_ylabel(r'$\frac{1}{c_{BEC}}\frac{d<X>}{dt}$')
    axes2[0].set_xlabel(r'$t$ [$\frac{\xi}{c}$]')
    axes2[0].set_title('Average Impurity Velocity (' + r'$F$' + '={:.2f} '.format(F_super / Fscale) + r'[$\frac{2 \pi c}{\xi^{2}}$]' + ')')

    # axes2[1].plot(aIBi_Vals, mE_supersonic_den, 'go', mew=1, ms=10, label='Force Protocol (Harmonic BEC Trap)')
    axes2[1].plot(aIBi_Vals, mE_supersonic_homog, 'rx', mew=1, ms=10, markerfacecolor='none', label='Force Protocol (Homogenous BEC)')
    axes2[1].plot(aIBi_Vals_steadystate, mE_steadystate, 'b-', mew=1, ms=10, markerfacecolor='none', label='Analytical Steady State')
    axes2[1].plot(aIBi_Vals, mE_steadystate_data, 'bs', mew=1, ms=10, markerfacecolor='none', label='')
    axes2[1].legend()
    axes2[1].set_ylim([0, 19.5])
    axes2[1].set_ylabel(r'$\frac{m^{*}}{m_{I}}$')
    axes2[1].set_xlabel(r'$a_{IB}^{-1}$')
    axes2[1].set_title('Polaron Mass Enhancement (Supersonic Case)')

    fig3, axes3 = plt.subplots()
    x_ds_subsonic_homog = qds_subsonic_homog.sel(aIBi=aIBi).isel(F=Find_subsonic_homog)['X']
    x_subsonic_homog = x_ds_subsonic_homog.values
    axes3.plot(x_ds_subsonic_homog.coords['t'].values / tscale, x_subsonic_homog)
    axes3.plot(x_ds_subsonic_homog.coords['t'].values[0] / tscale, x_subsonic_homog[0], color='#5ca904', marker='x', mew=1, ms=15, markerfacecolor='none',)
    axes3.plot(x_ds_subsonic_homog.coords['t'].values[-1] / tscale, x_subsonic_homog[-1], color='#8f1402', marker='x', mew=1, ms=15, markerfacecolor='none',)
    axes3.set_ylabel(r'$<X>$')
    axes3.set_xlabel(r'$t$ [$\frac{\xi}{c}$]')

    fig4, axes4 = plt.subplots()
    mu = 1.5; sig = 0.3
    gaussian = np.exp(-0.5 * (ts_subsonic_homog - mu)**2 / (2 * sig**2))
    axes4.plot(ts_subsonic_homog, gaussian)
    axes4.set_ylabel(r'$F$')
    axes4.set_xlabel(r'$t$')

    plt.show()

    # Note: initial P_tot = 0.1 for all force protocols above (and P=0.1 for analytical steady state)
