import numpy as np
import pandas as pd
import xarray as xr
import os


if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    # gParams

    (Lx, Ly, Lz) = (20, 20, 20)
    (dx, dy, dz) = (0.2, 0.2, 0.2)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    # datapath = '/home/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    # datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)
    datapath = '/n/regal/demler_lab/kis/ZwierleinExp_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)

    innerdatapath = datapath + '/redyn_spherical'
    outputdatapath = datapath + '/invRF'

    mI = 1.7

    # # Individual Datasets

    for ind, filename in enumerate(os.listdir(innerdatapath)):
        if filename == 'quench_Dataset_sph.nc':
            continue
        with xr.open_dataset(innerdatapath + '/' + filename) as ds:
            aIBi = ds.attrs['aIBi']
            P = ds.attrs['P']
            tgrid = ds.coords['t'].values
            aIBiVec = aIBi * np.ones(tgrid.size)
            PVec = P * np.ones(tgrid.size)
            St = np.exp(1j * P**2 / (2 * mI)) * (ds['Real_DynOv'].values + 1j * ds['Imag_DynOv'].values)
            ReDynOv_da = xr.DataArray(np.real(St), coords=[tgrid], dims=['t'])
            ImDynOv_da = xr.DataArray(np.imag(St), coords=[tgrid], dims=['t'])
            St_ds = xr.Dataset({'Real_DynOv': ReDynOv_da, 'Imag_DynOv': ImDynOv_da}, coords=ds.coords, attrs=ds.attrs)
            St_ds.to_netcdf(outputdatapath + '/P_{:.3f}_aIBi_{:.2f}.nc'.format(P, aIBi))
