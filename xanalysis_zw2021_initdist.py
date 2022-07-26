import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import pf_dynamic_sph
import pf_static_sph
from scipy.io import savemat, loadmat
from scipy import interpolate
from scipy.optimize import curve_fit
import os

if __name__ == "__main__":

    # Create kgrid

    import Grid
    (Lx, Ly, Lz) = (20, 20, 20)
    (dx, dy, dz) = (0.2, 0.2, 0.2)
    NGridPoints_desired = (1 + 2 * Lx / dx) * (1 + 2 * Lz / dz)
    Ntheta = 50
    Nk = np.ceil(NGridPoints_desired / Ntheta).astype(int)
    theta_max = np.pi
    thetaArray, dtheta = np.linspace(0, theta_max, Ntheta, retstep=True)
    k_max = ((2 * np.pi / dx)**3 / (4 * np.pi / 3))**(1 / 3)
    k_min = 1e-5
    kArray, dk = np.linspace(k_min, k_max, Nk, retstep=True)
    kgrid = Grid.Grid("SPHERICAL_2D")
    kgrid.initArray_premade('k', kArray)
    kgrid.initArray_premade('th', thetaArray)

    # Initialization

    matplotlib.rcParams.update({'font.size': 12})
    # matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    labelsize = 12
    legendsize = 10

    true2D = True

    aIBexp_Vals = np.array([-1000, -750, -500, -375, -250, -125, -60, -20, 0, 20, 50, 125, 175, 250, 375, 500, 750, 1000])

    inda = 14

    datapath = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/2021/gaussianTrap/PolPot/smarterPP/initdist_2D/aIB_{0}a0'.format(aIBexp_Vals[inda])
    # datapath = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/2021/gaussianTrap/PolPot/smarterPP/initdist_negMu'
    # datapath = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/2021/gaussianTrap/PolPot/smarterPP/initdist_P_P0'
    # datapath = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/ZwierleinExp_data/2021/gaussianTrap/PolPot/smarterPP/initdist_P_P0_Y_Y0'

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
    RTF_BEC_X = np.array([8.48469347093994, 8.11111072629368, 8.89071272031954, 8.57125199684266, 9.00767433275159, 9.65522167387697, 9.39241266912852, 9.23956650925869, 8.66153179309422, 9.14179769236378, 8.84900230929328, 8.94534024135962, 8.98248647105392, 8.81871271135454, 8.92241777405925, 9.11802005065468, 8.49295023977057, 8.81270137636933])  # Thomas-Fermi radius of BEC in x-direction (given in um)
    RTF_BEC_Y = np.array([11.4543973014280, 11.4485027292274, 12.0994087866866, 11.1987472415996, 12.6147755284164, 13.0408759297917, 12.8251948079726, 12.4963915490121, 11.6984708883771, 12.1884624646191, 11.7981246004719, 11.8796464214276, 12.4136593404667, 12.3220325703494, 12.0104329130883, 12.1756670927480, 10.9661042681457, 12.1803009563806])  # Thomas-Fermi radius of BEC in direction of oscillation (given in um)

    expParams = pf_dynamic_sph.Zw_expParams_2021()
    L_exp2th, M_exp2th, T_exp2th = pf_dynamic_sph.unitConv_exp2th(expParams['n0_BEC_scale'], expParams['mB'])

    RTF_BEC_Y_th = RTF_BEC_Y * 1e-6 * L_exp2th  # BEC oscillation amplitude (carries units of position)
    RTF_BEC_X_th = RTF_BEC_X * 1e-6 * L_exp2th  # BEC oscillation amplitude (carries units of position)

    # Load simulation data

    aIBList = [-1000, -750, -500, -375, -250, -125, -60, -20, 0, 20, 50, 125, 175, 250, 375, 500, 750, 1000]

    qds_List = []
    V_List = []
    param_List = []
    # vBEC_List = []
    # xBEC_List = []
    X_List = []
    Y_List = []
    PX_List = []
    filename_List = []

    for ind, filename in enumerate(os.listdir(datapath)):
        # print(filename)
        if filename == '.DS_Store':
            continue
        # if filename == 'aIB_-375a0_632.nc':
        #     continue
        # if filename =='aIB_-375a0_1.nc':
        #     continue

        qds = xr.open_dataset(datapath + '/' + filename)
        # print(qds)

        attrs = qds.attrs
        mI = attrs['mI']; mB = attrs['mB']; nu = attrs['nu']; xi = attrs['xi']; gBB = attrs['gBB']; tscale = xi / nu; aIBi = attrs['aIBi']
        # omega_BEC_osc = attrs['omega_BEC_osc']; phi_BEC_osc = attrs['phi_BEC_osc']; gamma_BEC_osc = attrs['gamma_BEC_osc']; amp_BEC_osc = attrs['amp_BEC_osc']; omega_Imp_x = attrs['omega_Imp_x']; X0 = attrs['X0']; P0 = attrs['P0']
        c_BEC_um_Per_ms = (nu * T_exp2th / L_exp2th) * (1e6 / 1e3)  # speed of sound in um/ms
        # print(c_BEC_exp[inda], c_BEC_um_Per_ms)
        tVals = 1e3 * qds['t'].values / T_exp2th  # time grid for simulation data in ms
        V = qds['V'].values * (T_exp2th / L_exp2th) * (1e6 / 1e3)
        X0 = attrs['X0']; Y0 = attrs['Y0']
        if np.any(np.isnan(V)):
            print(filename)
            continue
        if true2D:
            X = qds['X'].values * 1e6 / L_exp2th
            Y = qds['Y'].values * 1e6 / L_exp2th
            # if np.max(np.abs(Y)) > 20 * RTF_BEC_Y[inda]:
            #     print('RUNAWAY', filename)
            #     continue
            # if np.isclose(np.mean(V[int(0.9*tVals.size)::]), 0):
            #     print('FLAT', filename)
            #     continue
            PX0 = attrs['PX0']
            PY0 = attrs['PY0']
            param_List.append((X0, Y0, PX0, PY0))
            X_List.append(X)
            Y_List.append(Y)
            PX_List.append(qds['PX'].values)

        else:
            P0 = attrs['P0']
            param_List.append((X0, Y0, P0))

        # if Y0 < 3:
        #     continue

        # if Y0 > -5:
        #     continue

        # xBEC = pf_dynamic_sph.x_BEC_osc_zw2021(qds['t'].values, omega_BEC_osc, gamma_BEC_osc, phi_BEC_osc, amp_BEC_osc); xBEC_conv = 1e6 * xBEC / L_exp2th
        # vBEC = pf_dynamic_sph.v_BEC_osc_zw2021(qds['t'].values, omega_BEC_osc, gamma_BEC_osc, phi_BEC_osc, amp_BEC_osc); vBEC_conv = (vBEC * T_exp2th / L_exp2th) * (1e6 / 1e3)
        # vBEC = pf_dynamic_sph.v_BEC_osc_zw2021(np.linspace(0, 100, 1000) * 1e-3 * T_exp2th, omega_BEC_osc, gamma_BEC_osc, phi_BEC_osc, amp_BEC_osc); vBEC_conv = (vBEC * T_exp2th / L_exp2th) * (1e6 / 1e3)

        qds_List.append(qds)
        V_List.append(V)
        filename_List.append(filename)

        # vBEC_List.append(vBEC_conv)
        # xBEC_List.append(xBEC_conv)

    V_mean = np.mean(np.array(V_List), axis=0)
    print(len(V_List))
    print(tVals.size)
    print(V_mean.shape)

    # # #############################################################################################################################
    # # # FIT BEC OSCILLATION
    # # #############################################################################################################################

    # phiVals = []
    # gammaVals = []
    # for ind in np.arange(18):
    #     if ind != 4:
    #         continue
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
    #     # ax2.plot(tVals_interp, NaV_cf, 'r-')
    #     ax2.plot(np.linspace(0, 100, 1000), vBEC_conv, 'r-')

    #     # ax2.plot(tVals_interp, aOsc_interp, 'b-')
    #     # ax2.plot(tVals_interp, a_cf, 'r-')

    #     # ax2.plot(dt_BEC + tVals, vBEC_conv)
    #     # if ind == inda:
    #     #     ax2.plot(tVals, vBEC_conv)
    #     ax2.plot()
    #     plt.show()

    # # print(np.array(phiVals))
    # # print(np.array(gammaVals))

    # #############################################################################################################################
    # # RELATIVE VELOCITY
    # #############################################################################################################################

    # fig0, ax0 = plt.subplots()
    # # ax0.plot(tVals, qds_List[inda]['X'].values * 1e6 / L_exp2th, label='Simulation')
    # ax0.plot(tVals, qds_List[inda]['Y'].values * 1e6 / L_exp2th, label='Simulation')
    # ax0.fill_between(tVals, -RTF_BEC_Y[inda], RTF_BEC_Y[inda], facecolor='orange', alpha=0.1, label='Thomas-Fermi radius')
    # ax0.hlines(-3*RTF_BEC_Y[inda],np.min(tVals),np.max(tVals),'k','--')
    # ax0.hlines(3*RTF_BEC_Y[inda],np.min(tVals),np.max(tVals),'k','--')
    # ax0.set_ylabel(r'Relative impurity position ($\mu$m)')
    # ax0.set_xlabel(r'Time (ms)')
    # # ax0.set_title(r'$a_\mathrm{BF}=$' + '{0}'.format(aIB) + r'$a_\mathrm{Bohr}$')
    # ax0.legend()

    # inda = 1
    c_rat = c_BEC_exp[inda]/c_BEC_um_Per_ms
    print(c_rat)
    # v_rat = np.max(V_exp[inda][0:int(tVals.size/4)])/np.max(V_mean[0:int(tVals.size/4)])
    # v_rat = 1.2
    v_rat = 1.0
    print(v_rat)
    V_shift = np.mean(V_mean[int(tVals.size/2)::])
    print(V_shift)
    V_mean = V_mean - V_shift

    aIB = aIBexp_Vals[inda]
    fig, ax = plt.subplots()
    ax.plot(tVals_exp, V_exp[inda], 'kd-', label='Experiment')
    ax.plot(tVals, v_rat*V_mean, label='Simulation')
    ax.fill_between(tVals_exp, -c_BEC_exp[inda], c_BEC_exp[inda], facecolor='red', alpha=0.1, label='Subsonic regime')
    ax.set_ylabel(r'Impurity velocity ($\mu$m/ms)')
    ax.set_xlabel(r'Time (ms)')
    ax.set_title(r'$a_\mathrm{BF}=$' + '{0}'.format(aIB) + r'$a_\mathrm{Bohr}$')
    ax.legend()
    # ax.set_ylim([-20, 20])

    fig2, ax2 = plt.subplots()
    # ax2.plot(tVals_exp, V_exp[inda], 'kd-', label='Experiment')
    # paramSlice = param_List[0:-1:75]
    paramSlice = param_List
    # for indv, V in enumerate(V_List[0:-1:75]):
    for indv, V in enumerate(V_List):
        if true2D:
            X0, Y0, PX0, PY0 = param_List[indv]
            ax2.plot(tVals, V/c_BEC_exp[inda], label='X0: {:.1f}, Y0: {:.1f}, PX0: {:.2f}, PY0: {:.2f}, ind: {:d}'.format(X0, Y0, PX0, PY0, indv))
            # ax2.plot(tVals[int(0.75*tVals.size)::], V[int(0.75*tVals.size)::], label='X0: {:.1f}, Y0: {:.1f}, PX0: {:.2f}, PY0: {:.2f}, ind: {:d}'.format(X0, Y0, PX0, PY0, indv))
            # if np.max(np.abs(V[0::])) > 3 * c_BEC_exp[inda]:
            # if np.max(np.abs(V[0::])) > 100 * c_BEC_exp[inda]:
            # if np.max(np.abs(V[0::])) > 50 * c_BEC_exp[inda]:
            #     print(indv, filename_List[indv])
            #     ax2.plot(tVals, V, label='X0: {:.1f}, Y0: {:.1f}, PX0: {:.2f}, PY0: {:.2f}, ind: {:d}'.format(X0, Y0, PX0, PY0, indv))
        else:
            X0, Y0, P0 = paramSlice[indv]
            ax2.plot(tVals, V, label='X0: {:.1f}, Y0: {:.1f}, P0: {:.2f}'.format(X0, Y0, P0))
    # ax2.fill_between(tVals_exp, -c_BEC_exp[inda], c_BEC_exp[inda], facecolor='red', alpha=0.1, label='Subsonic regime')
    ax2.set_ylabel(r'Impurity velocity ($\mu$m/ms)')
    ax2.set_xlabel(r'Time (ms)')
    ax2.set_title(r'$a_\mathrm{BF}=$' + '{0}'.format(aIB) + r'$a_\mathrm{Bohr}$')
    # ax2.legend()
    # ax2.set_ylim([-20, 20])
    ax2.set_ylabel(r'Impurity velocity ($\mu$m/ms)')
    ax2.set_xlabel(r'Time (ms)')

    if true2D:
        # ax2.legend()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        fig5, ax5, = plt.subplots()
        dt = 1e-3*(tVals[1]-tVals[0])
        fVals = np.fft.fftshift(np.fft.fftfreq(tVals.size) / (dt))

        for indx, X in enumerate(X_List):
            X0, Y0, PX0, PY0 = param_List[indx]
            FTVals = np.abs(np.fft.fftshift(dt * np.fft.fft(np.fft.fftshift(X))))
            # print(np.allclose(np.heaviside(1 - X_List[indx] ** 2 / RTF_BEC_X_th[inda] ** 2 - Y_List[indx] ** 2 / RTF_BEC_Y_th[inda] ** 2, 1 / 2),1))
            # print(fVals[fVals>0][np.argmax(FTVals[fVals>0])])

            ax3.plot(tVals, X_List[indx] / 1e6 * L_exp2th, label='X0: {:.1f}, Y0: {:.1f}, PX0: {:.2f}, PY0: {:.2f}'.format(X0, Y0, PX0, PY0))
            # ax3.fill_between(tVals, -RTF_BEC_X[inda] / 1e6 * L_exp2th, RTF_BEC_X[inda] / 1e6 * L_exp2th, facecolor='orange', alpha=0.1, label='')
            # ax3.legend()
            ax4.plot(tVals, PX_List[indx], label='X0: {:.1f}, Y0: {:.1f}, PX0: {:.2f}, PY0: {:.2f}'.format(X0, Y0, PX0, PY0))
            # ax4.legend()

            # indmax = np.argmax(FTVals[fVals>0])
            # if fVals[fVals>0][indmax] < 100:
            #     print(fVals[fVals>0][indmax])
            #     print(filename_List[indx])

            ax5.plot(fVals[fVals>0], FTVals[fVals>0], label='X0: {:.1f}, Y0: {:.1f}, PX0: {:.2f}, PY0: {:.2f}'.format(X0, Y0, PX0, PY0))
            ax5.set_xlabel('Hz')
            ax5.set_xlim([0,500])
            # ax5.legend()

        fig6, ax6 = plt.subplots()
        ax6.fill_between(tVals, -RTF_BEC_Y[inda], RTF_BEC_Y[inda], facecolor='orange', alpha=0.1)
        ax6.hlines(-3*RTF_BEC_Y[inda],np.min(tVals),np.max(tVals),'k','--')
        ax6.hlines(3*RTF_BEC_Y[inda],np.min(tVals),np.max(tVals),'k','--')
        ax6.set_ylabel(r'Relative impurity position ($\mu$m)')
        ax6.set_xlabel(r'Time (ms)')
        # ax6.set_ylim([-20*RTF_BEC_Y[inda], 20*RTF_BEC_Y[inda]])
        ax6.set_ylim([-4*RTF_BEC_Y[inda], 4*RTF_BEC_Y[inda]])
        # ax6.legend()
        for indy, Y in enumerate(Y_List):
            X0, Y0, PX0, PY0 = param_List[indy]
            ax6.plot(tVals, Y, label='X0: {:.1f}, Y0: {:.1f}, PX0: {:.2f}, PY0: {:.2f}'.format(X0, Y0, PX0, PY0))





    plt.show()
