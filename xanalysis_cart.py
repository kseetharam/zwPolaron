import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":

    # # Initialization

    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    # gParams
    (Lx, Ly, Lz) = (20, 20, 20)
    (dx, dy, dz) = (0.5, 0.5, 0.5)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    datapath = '/media/kis/Storage/Dropbox/VariationalResearch/HarvardOdyssey/genPol_data/NGridPoints_{:.2E}'.format(NGridPoints_cart)

    ds_list = []; P_list = []; aIBi_list = []; mI_list = []
    for ind, filename in enumerate(os.listdir(datapath)):
        ds = xr.open_dataset(datapath + '/' + filename)
        ds_list.append(ds)
        P_list.append(ds.attrs['P'])
        aIBi_list.append(ds.attrs['aIBi'])
        mI_list.append(ds.attrs['mI'])

    ds = ds_list[0]; P = P_list[0]; aIBi = aIBi_list[0]; mI = mI_list[0]

    # ds['NB'].plot()
    S_mag = np.sqrt(ds['Real_DynOv']**2 + ds['Imag_DynOv']**2)
    # S_mag.plot()
    # ds['nPI_mag'].isel(t=-1).plot()
    ds['nPI_xz_slice'].isel(t=3).plot()
    plt.show()
