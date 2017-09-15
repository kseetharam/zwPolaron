import numpy as np
import Grid
import polaron_functions as pf
import os
from timeit import default_timer as timer
import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":

    start = timer()
    matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

    # ---- SET SPARAMS ----
    mB = 1
    mI = 0.5 * mB
    n0 = 1
    aBB = 0.05
    alpha = 2.1

    gBB = (4 * np.pi / mB) * aBB
    xi = 1 / (np.sqrt(2 * mB * gBB * n0))
    nu = pf.nu(gBB * n0 / mB)

    sParams = [mI, mB, n0, gBB, nu, xi, alpha]

    # ---- INITIALIZE GRIDS ----

    kcutoff = 20 / xi
    dk = 0.05

    Ntheta = 50
    dtheta = np.pi / (Ntheta - 1)

    NGridPoints = Ntheta * kcutoff / dk

    kgrid = Grid.Grid("SPHERICAL_2D")
    kgrid.initArray('k', dk, kcutoff, dk)
    kgrid.initArray('th', dtheta, np.pi, dtheta)

    xmax = 1 / dk
    dx = 1 / kcutoff

    xgrid = Grid.Grid("SPHERICAL_2D")
    xgrid.initArray('x', 0, xmax, dx)
    xgrid.initArray('th', dtheta, np.pi, dtheta)

    # ---- SET GPARAMS ----

    # dt = 1e-1
    # NtPoints = 94
    # tMax = dt * np.exp(dt * (NtPoints - 1))
    # tGrid = np.zeros(NtPoints)
    # for n in range(NtPoints):
    #     tGrid[n] = dt * np.exp(dt * n)

    tMax = 50 * xi / nu
    dt = .01 * xi / nu

    # tGrid = np.concatenate((np.arange(0, 1 + dt1, dt1), np.arange(1 + dt2, tMax + dt2, dt2)))
    tGrid = np.arange(0, tMax + dt, dt)

    gParams = [kgrid, xgrid, tGrid]

    # ---- SET CPARAMS (RANGE OVER MULTIPLE P VALUES) ----

    P = 0.5 * nu * mI
    aSi = pf.aSi_grid(kgrid, P, mI, mB, n0, gBB)
    # aSi = 0  # for Frohlich model
    aIBi = -1 * (1 / np.sqrt(alpha)) * ((32 * np.pi**2 * n0) / (mB * gBB))**(1 / 4) + aSi
    # aIBi = 1 * (1 / np.sqrt(alpha)) * ((32 * np.pi**2 * n0) / (mB * gBB))**(1 / 4) + aSi

    # g = pf.g(kgrid, 0, aIBi, mI, mB, n0, gBB)
    Pg = pf.PCrit_grid(kgrid, 0, aIBi, mI, mB, n0, gBB)
    Pc = pf.PCrit_inf(kcutoff, aIBi, mI, mB, n0, gBB)

    print('xi: %.3f, nu: %.3f, aIBi: %.3f, aSi: %.3f, P: %.3f' % (xi, nu, aIBi, aSi, P))
    print('Pc_inf: %.3f, Pc_g: %.3f' % (Pc, Pg))

    cParams = [P, aIBi]

    # # ---- COMPUTE DATA ----

    dirpath = os.path.dirname(os.path.realpath(__file__))
    outerdatapath = dirpath + '/trajdata' '/NGridPoints_%.2E' % NGridPoints
    datapath = outerdatapath + '/neg' + '/twophonon'
    if os.path.isdir(datapath) is False:
        os.mkdir(datapath)

    # paramInfo = 'kcutoff - {:.2f}, dk - {:.3f}, Ntheta - {:d}, NGridPoints - {:.2E}, tMax - {:.1f}, dt1 - {:.3f}, dt2 - {:.3f} NtPoints - {:d}\nmI - {:.1f}, mB - {:.1f}, n0 - {:0.1f}, gBB - {:0.3f}\naIBi - {:.2f}, gIB - {:0.3f}, PCrit_grid - {:.5f}, PCrit_true - {:0.5f}, NPVals - {:d}'.format(kcutoff, dk, Ntheta, NGridPoints, tMax, dt1, dt2, tGrid.size, mI, mB, n0, gBB, aIBi, g, Pg, Pc, NPVals)
    # with open(datapath + '/paramInfo.txt', 'w') as f:
    #     f.write(paramInfo)

    tGrid_N, Traj_Vec_N = pf.quenchDynamics_Traj(cParams, gParams, sParams, datapath)

    # # ---- POST-PROCESSING AND PLOTTING ----

    # Traj_Vec = np.zeros(tVec.size, dtype=float)

    # for ind, t in enumerate(tVec):
    #     PI_slice = PI_Vec[0:ind + 1]
    #     Traj_Vec[ind] = np.trapz(PI_slice, dx=dt) / mI

    fig, ax = plt.subplots()
    ax.plot(tGrid_N, Traj_Vec_N, 'r-', label='Interacting')
    ax.plot(tGrid_N, (P * tGrid / mI) / xi, 'b-', label='Free')
    ax.legend()
    # axN.set_xlim([0, 2])
    # ax.set_ylim([0, 0.8])
    ax.set_xlabel(r'Time ($t$)')
    ax.set_ylabel(r'Average Impurity Position ($x(t)$)')
    ax.set_title('Impurity Trajectory')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    plt.show()
