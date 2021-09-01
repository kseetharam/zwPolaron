import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import pf_dynamic_sph
from scipy.io import savemat, loadmat

if __name__ == "__main__":

    # Initialization

    matplotlib.rcParams.update({'font.size': 12})
    labelsize = 13
    legendsize = 12

    datapath = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/2021/NoPolPot'
    figdatapath = '/Users/kis/KIS Dropbox/Kushal Seetharam/ZwierleinExp/2021/figures'

    # Load experimental data

    expData = loadmat('/Users/kis/KIS Dropbox/Kushal Seetharam/ZwierleinExp/2021/data/oscdata/dataToExport.mat')['dataToExport']  # experimental data
    # print(expData)

    aIBexp_Vals = expData['aBFs'][0][0][0]
    tVals_exp = expData['relVel_time'][0][0][0]
    V_exp = expData['relVel'][0][0]
    c_BEC_exp = expData['speedOfSound_array'][0][0][0]

    # Load simulation data

    inda = 4
    aIB = aIBexp_Vals[inda]; print('aIB: {0}a0'.format(aIB))

    qds = xr.open_dataset(datapath + '/aIB_{0}a0_1.nc'.format(aIB))

    expParams = pf_dynamic_sph.Zw_expParams_2021()
    L_exp2th, M_exp2th, T_exp2th = pf_dynamic_sph.unitConv_exp2th(expParams['n0_BEC_scale'], expParams['mB'])

    attrs = qds.attrs
    mI = attrs['mI']; mB = attrs['mB']; nu = attrs['nu']; xi = attrs['xi']; gBB = attrs['gBB']; tscale = xi / nu
    omega_BEC_osc = attrs['omega_BEC_osc']; omega_Imp_x = attrs['omega_Imp_x']; a_osc = attrs['a_osc']; X0 = attrs['X0']; P0 = attrs['P0']
    c_BEC_um_Per_ms = (nu * T_exp2th / L_exp2th) * (1e6 / 1e3)  # speed of sound in um/ms
    # print(c_BEC_exp[inda], c_BEC_um_Per_ms)
    tVals = 1e3 * qds['t'].values / T_exp2th  # time grid for simulation data in ms
    V = qds['V'].values * (T_exp2th / L_exp2th) * (1e6 / 1e3)
    xBEC = pf_dynamic_sph.x_BEC_osc(tVals, omega_BEC_osc, 1, a_osc)

    xL_bareImp = (xBEC[0] + X0) * np.cos(omega_Imp_x * tVals) + (P0 / (omega_Imp_x * mI)) * np.sin(omega_Imp_x * tVals)  # gives the lab frame trajectory time trace of a bare impurity (only subject to the impurity trap) that starts at the same position w.r.t. the BEC as the polaron and has the same initial total momentum
    vL_bareImp = np.gradient(xL_bareImp, tVals)
    aL_bareImp = np.gradient(np.gradient(xL_bareImp, tVals), tVals)

    # #############################################################################################################################
    # # RELATIVE VELOCITY
    # #############################################################################################################################

    fig, ax = plt.subplots()
    ax.plot(tVals_exp, V_exp[inda], 'kd', label='Experiment')
    ax.plot(tVals, V, label='Simulation')
    ax.fill_between(tVals_exp, -c_BEC_exp[inda], c_BEC_exp[inda], facecolor='red', alpha=0.1)
    ax.legend()
    plt.show()
