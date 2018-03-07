import numpy as np
import pandas as pd
import xarray as xr
import os
import itertools

if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    # gParams

    # (Lx, Ly, Lz) = (21, 21, 21)
    # (dx, dy, dz) = (0.375, 0.375, 0.375)

    (Lx, Ly, Lz) = (21, 21, 21)
    (dx, dy, dz) = (0.25, 0.25, 0.25)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    aIBi = -1
    # datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}/spherical'.format(NGridPoints_cart)
    outputdatapath = '/media/kis/Storage/Dropbox/VariationalResearch/DataAnalysis/newdata/NGridPoints_{:.2E}/dynamics_spherical/aIBi_{:.2f}'.format(NGridPoints_cart, aIBi)

    if os.path.isdir(outputdatapath) is False:
        os.mkdir(outputdatapath)

    # # Concatenate Individual Datasets

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
        Phase_Vec = ds['Phase'].values
        PB_Vec = ds['PB'].values
        NB_Vec = ds['NB'].values
        Real_DynOv_Vec = ds['Real_DynOv'].values
        Imag_DynOv_Vec = ds['Imag_DynOv'].values

        # generates data file with columns representing P, t, Phase, Phonon Momentum, !Momentum Dispersion, Phonon Number, Re(Dynamical Overlap), Im(Dynamical Overlap)
        data = np.concatenate((PVec[:, np.newaxis], tGrid[:, np.newaxis], Phase_Vec[:, np.newaxis], PB_Vec[:, np.newaxis], np.zeros(tGrid.size)[:, np.newaxis], NB_Vec[:, np.newaxis], Real_DynOv_Vec[:, np.newaxis], Imag_DynOv_Vec[:, np.newaxis]), axis=1)
        np.savetxt(outputdatapath + '/quench_P_%.2f.dat' % P, data)
