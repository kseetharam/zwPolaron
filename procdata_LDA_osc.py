import numpy as np
import xarray as xr
import os

if __name__ == "__main__":

    # # Initialization

    # gParams

    (Lx, Ly, Lz) = (20, 20, 20)
    (dx, dy, dz) = (0.2, 0.2, 0.2)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)

    aBB = 0.013

    # Toggle parameters

    toggleDict = {'CS_Dyn': 'on', 'PosScat': 'off'}
    dParams = {'f_BEC_osc': 500, 'f_Imp_x': 1000, 'a_osc': 0.5, 'X0': 0.0, 'P0': 0.6}

    # ---- SET OUTPUT DATA FOLDER ----

    datapath = '/n/regal/demler_lab/kis/ZwierleinExp_data/aBB_{:.3f}/NGridPoints_{:.2E}'.format(aBB, NGridPoints_cart)
    if toggleDict['PosScat'] == 'on':
        innerdatapath = datapath + '/BEC_osc/PosScat'
    else:
        innerdatapath = datapath + '/BEC_osc'
    if toggleDict['CS_Dyn'] == 'off':
        innerdatapath = innerdatapath + '/NoCSdyn_fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])
    else:
        innerdatapath = innerdatapath + '/fBEC={:d}_fImp={:d}_aosc={:.1f}_X0={:.1f}_P0={:.1f}'.format(dParams['f_BEC_osc'], dParams['f_Imp_x'], dParams['a_osc'], dParams['X0'], dParams['P0'])

    outputdatapath = innerdatapath + '_tF'
    if os.path.isdir(outputdatapath) is False:
        os.mkdir(outputdatapath)

    def selectLast(filename, innerdatapath, outputdatapath):
        ds = xr.open_dataset(innerdatapath + '/' + filename)
        dsL = ds.isel(t=-1)
        dsL.to_netcdf(outputdatapath + '/' + filename)
        return

    for ind, filename in enumerate(os.listdir(innerdatapath)):
        selectLast(filename, innerdatapath, outputdatapath)
