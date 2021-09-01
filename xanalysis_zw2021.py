import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import pf_dynamic_sph
from scipy.io import savemat, loadmat
from scipy import interpolate
from scipy.optimize import curve_fit

if __name__ == "__main__":

    # Initialization

    matplotlib.rcParams.update({'font.size': 12})
    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

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
    NaV_exp = expData['Na_vel'][0][0]
    omega_Na = np.array([465.418650581347, 445.155256942448, 461.691943131414, 480.899902898451, 448.655522184374, 465.195338759998, 460.143258369460, 464.565377197007, 465.206177963899, 471.262139163205, 471.260672147216, 473.122081065092, 454.649394420577, 449.679107889662, 466.770887179217, 470.530355145510, 486.615655444221, 454.601540658640])   # in rad*Hz
    Na_displacement = np.array([26.2969729628679, 22.6668334850173, 18.0950989598699, 20.1069898676222, 14.3011351453467, 18.8126473489499, 17.0373115356076, 18.6684373282353, 18.8357213162278, 19.5036039713438, 21.2438389441807, 18.2089748680659, 18.0433963046778, 8.62940156299093, 16.2007030552903, 23.2646987822343, 24.1115616621798, 28.4351972435186])  # initial position of the BEC (in um)
    phi_Na = np.array([-0.2888761, -0.50232022, -0.43763589, -0.43656233, -0.67963017, -0.41053479, -0.3692152, -0.40826816, -0.46117853, -0.41393032, -0.53483635, -0.42800711, -0.3795508, -0.42279337, -0.53760432, -0.4939509, -0.47920687, -0.51809527])  # phase of the BEC oscillation in rad
    gamma_Na = np.array([4.97524294, 14.88208436, 4.66212187, 6.10297397, 7.77264927, 4.5456649, 4.31293083, 7.28569606, 8.59578888, 3.30558254, 8.289436, 4.14485229, 7.08158476, 4.84228082, 9.67577823, 11.5791718, 3.91855863, 10.78070655])  # decay rate of the BEC oscillation in Hz

    # Load simulation data

    inda = 4
    aIB = aIBexp_Vals[inda]; print('aIB: {0}a0'.format(aIB))

    qds = xr.open_dataset(datapath + '/aIB_{0}a0_long.nc'.format(aIB))

    expParams = pf_dynamic_sph.Zw_expParams_2021()
    L_exp2th, M_exp2th, T_exp2th = pf_dynamic_sph.unitConv_exp2th(expParams['n0_BEC_scale'], expParams['mB'])

    attrs = qds.attrs
    mI = attrs['mI']; mB = attrs['mB']; nu = attrs['nu']; xi = attrs['xi']; gBB = attrs['gBB']; tscale = xi / nu
    omega_BEC_osc = attrs['omega_BEC_osc']; omega_Imp_x = attrs['omega_Imp_x']; a_osc = attrs['a_osc']; X0 = attrs['X0']; P0 = attrs['P0']
    c_BEC_um_Per_ms = (nu * T_exp2th / L_exp2th) * (1e6 / 1e3)  # speed of sound in um/ms
    # print(c_BEC_exp[inda], c_BEC_um_Per_ms)
    tVals = 1e3 * qds['t'].values / T_exp2th  # time grid for simulation data in ms
    V = qds['V'].values * (T_exp2th / L_exp2th) * (1e6 / 1e3)
    xBEC = pf_dynamic_sph.x_BEC_osc(qds['t'].values, omega_BEC_osc, 1, a_osc); xBEC_conv = 1e6 * xBEC / L_exp2th
    vBEC = np.gradient(xBEC, qds['t'].values); vBEC_conv = (vBEC * T_exp2th / L_exp2th) * (1e6 / 1e3)
    FBEC = pf_dynamic_sph.F_BEC_osc(qds['t'].values, omega_BEC_osc, 1, a_osc, mI)

    xL_bareImp = (xBEC[0] + X0) * np.cos(omega_Imp_x * tVals) + (P0 / (omega_Imp_x * mI)) * np.sin(omega_Imp_x * tVals)  # gives the lab frame trajectory time trace of a bare impurity (only subject to the impurity trap) that starts at the same position w.r.t. the BEC as the polaron and has the same initial total momentum
    vL_bareImp = np.gradient(xL_bareImp, tVals)
    aL_bareImp = np.gradient(np.gradient(xL_bareImp, tVals), tVals)

    # # #############################################################################################################################
    # # # FIT BEC OSCILLATION
    # # #############################################################################################################################

    # phiVals = []
    # gammaVals = []
    # for ind in np.arange(18):
    #     print(aIBexp_Vals[ind])
    #     NaV = NaV_exp[ind]
    #     nanmask = np.isnan(NaV)
    #     NaV_nanfill = np.interp(tVals_exp[nanmask], tVals_exp[~nanmask], NaV[~nanmask])
    #     NaV[nanmask] = NaV_nanfill
    #     NaV_tck = interpolate.splrep(tVals_exp, NaV, s=0)
    #     tVals_interp = np.linspace(0, 100, 1000)
    #     NaV_interp = interpolate.splev(tVals_interp, NaV_tck, der=0)
    #     aOsc_interp = interpolate.splev(tVals_interp, NaV_tck, der=1)

    #     def v_decayOsc(t, phi, gamma):
    #         # takes time values in s, phi in rad, gamma in Hz, and outputs velocity in m/s
    #         omega = omega_Na[ind]
    #         A = Na_displacement[ind] * 1e-6 / np.cos(phi)
    #         return -1 * A * np.exp(-1 * gamma * t) * (gamma * np.cos(omega * t + phi) + omega * np.sin(omega * t + phi))

    #     popt, cov = curve_fit(v_decayOsc, tVals_exp * 1e-3, NaV * 1e-6 / 1e-3, p0=np.array([0, 0]))
    #     phiFit = popt[0]; gammaFit = popt[1]
    #     phiVals.append(phiFit); gammaVals.append(gammaFit)
    #     print(omega_Na[ind] / (2 * np.pi), gammaFit, phiFit / np.pi)
    #     NaV_cf = v_decayOsc(tVals_interp * 1e-3, phiFit, gammaFit) * 1e6 / 1e3  # converts m/s velocity into um/ms

    #     def a_decayOsc(t, omega, x0, phi, gamma):
    #         # takes t in s, omega in radHz, x0 (=initial Na displacement) in m, phi in rad, gamma in Hz and outputs acceleration in m/s^2
    #         return x0 * np.cos(phi) * np.exp(-1 * gamma * t) * ((gamma**2 - omega**2) * np.cos(omega * t + phi) + 2 * gamma * omega * np.sin(omega * t + phi))

    #     a_cf = a_decayOsc(tVals_interp * 1e-3, omega_Na[ind], Na_displacement[ind] * 1e-6, phiFit, gammaFit) * 1e6 / 1e6  # converts m/s^2 velocity into um/ms^2

    #     fig2, ax2 = plt.subplots()
    #     ax2.plot(tVals_exp, NaV, 'kd-')
    #     # ax2.plot(tVals_interp, NaV_interp, 'r-')
    #     ax2.plot(tVals_interp, NaV_cf, 'g-')
    #     ax2.plot(tVals_interp, aOsc_interp, 'b-')
    #     ax2.plot(tVals_interp, a_cf, 'r-')

    #     # ax2.plot(dt_BEC + tVals, vBEC_conv)
    #     # if ind == inda:
    #     #     ax2.plot(tVals, vBEC_conv)
    #     ax2.plot()
    #     plt.show()
    # print(np.array(phiVals))
    # print(np.array(gammaVals))

    # #############################################################################################################################
    # # RELATIVE VELOCITY
    # #############################################################################################################################

    # dt_imp = tVals_exp[np.argmax(V_exp[inda][tVals_exp < 20])] - tVals[np.argmax(V[tVals < 20])]
    # dt_BEC = tVals_exp[np.argmax(NaV_exp[inda][tVals_exp < 20])] - tVals[np.argmax(vBEC_conv[tVals < 20])]
    # print(dt_imp, dt_BEC)

    fig, ax = plt.subplots()
    ax.plot(tVals_exp, V_exp[inda], 'kd', label='Experiment')
    ax.plot(tVals, V, label='Simulation')
    ax.fill_between(tVals_exp, -c_BEC_exp[inda], c_BEC_exp[inda], facecolor='red', alpha=0.1)
    ax.set_ylabel(r'velocity ($\mu$m/ms)')
    ax.set_xlabel(r'time (ms)')
    ax.set_title(r'$a_\mathrm{BF}=$' + '{0}'.format(aIB) + r'$a_\mathrm{Bohr}$')
    ax.legend()

    plt.show()
