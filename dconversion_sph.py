import numpy as np
import pandas as pd
import xarray as xr
import os
import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    # gParams

    (Lx, Ly, Lz) = (21, 21, 21)
    (dx, dy, dz) = (0.375, 0.375, 0.375)

    # (Lx, Ly, Lz) = (21, 21, 21)
    # (dx, dy, dz) = (0.25, 0.25, 0.25)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    aIBi = -2
    datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/spherical'.format(NGridPoints_cart)
    outputdatapath = '/media/kis/Storage/Dropbox/VariationalResearch/DataAnalysis/newdata/NGridPoints_{:.2E}/dynamics_spherical/aIBi_{:.2f}'.format(NGridPoints_cart, aIBi)

    # datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/spherical'.format(NGridPoints_cart)
    # outputdatapath = '/home/kis/Dropbox/VariationalResearch/DataAnalysis/newdata/NGridPoints_{:.2E}/dynamics_spherical/aIBi_{:.2f}'.format(NGridPoints_cart, aIBi)

    if os.path.isdir(outputdatapath) is False:
        os.mkdir(outputdatapath)

    # # Individual Datasets

    for ind, filename in enumerate(os.listdir(datapath)):
        if filename == 'quench_Dataset_sph.nc':
            continue
        ds = xr.open_dataset(datapath + '/' + filename)

        if(ds.attrs['aIBi'] != aIBi):
            continue

        print(filename)
        P = ds.attrs['P']
        tGrid = ds.coords['t'].values
        PVec = P * np.ones(tGrid.size)
        CSAmp = (ds['Real_CSAmp'] + 1j * ds['Imag_CSAmp']).values
        CSAmp = CSAmp.reshape(CSAmp.size)

        # generates data file with columns representing P, t, Phase, Phonon Momentum, !Momentum Dispersion, Phonon Number, Re(Dynamical Overlap), Im(Dynamical Overlap)
        data = np.concatenate((aIBiVec[:, np.newaxis], PVec[:, np.newaxis], tGrid[:, np.newaxis], Real_DynOv_Vec[:, np.newaxis], Imag_DynOv_Vec[:, np.newaxis]), axis=1)
        np.savetxt(outputdatapath + '/quench_P_%.2f.dat' % P, data)

    # # steady state

    # aIBi = -1
    # datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/steadystate_spherical/quench_st_Dataset_sph.nc'.format(NGridPoints_cart)
    # outputdatapath = '/home/kis/Dropbox/VariationalResearch/DataAnalysis/newdata/NGridPoints_{:.2E}/steadystate_spherical'.format(NGridPoints_cart)

    # ds = xr.open_dataset(datapath)
    # dsa = ds.sel(aIBi=aIBi)

    # data = np.concatenate((dsa.coords['P'].values[:, np.newaxis], dsa['PB'].values[:, np.newaxis], dsa['NB'].values[:, np.newaxis], dsa['Z_factor'].values[:, np.newaxis], dsa['Energy'].values[:, np.newaxis], dsa['effMass'].values[:, np.newaxis], dsa['Pcrit'].values[:, np.newaxis]), axis=1)
    # np.savetxt(outputdatapath + '/quench_aIBi_{:.2f}.dat'.format(aIBi), data)
