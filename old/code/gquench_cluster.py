import numpy as np
import Grid
import polaron_functions as pf
import os
from timeit import default_timer as timer


if __name__ == "__main__":

    start = timer()

    # ---- INITIALIZE GRIDS ----

    # kcutoff = 10
    # dk = 0.5

    # Ntheta = 50
    # dtheta = np.pi / (Ntheta - 1)

    # NGridPoints = Ntheta * kcutoff / dk

    # kgrid = Grid.Grid("SPHERICAL_2D")
    # kgrid.initArray('k', dk, kcutoff, dk)
    # kgrid.initArray('th', dtheta, np.pi, dtheta)

    # xmax = 1 / dk
    # dx = 1 / kcutoff

    # xgrid = Grid.Grid("SPHERICAL_2D")
    # xgrid.initArray('x', 0, xmax, dx)
    # xgrid.initArray('th', dtheta, np.pi, dtheta)

    xmax = 7
    xmin = 0
    dx = 0.05

    kcutoff = np.pi / dx
    dk = np.pi / xmax
    print('kcutoff: %.2f, dk: %.2f' % (kcutoff, dk))

    Ntheta = 50
    dtheta = np.pi / Ntheta

    NGridPoints = Ntheta * kcutoff / dk

    xgrid = Grid.Grid("SPHERICAL_2D")
    xgrid.initArray('x', xmin, xmax, dx)
    xgrid.initArray('th', 0, np.pi, dtheta)

    kgrid = Grid.Grid("SPHERICAL_2D")
    kgrid.initArray('k', dk, kcutoff, dk)
    kgrid.initArray('th', 0, np.pi, dtheta)

    # ---- SET GPARAMS ----

    # dt = 1e-1
    # NtPoints = 94
    # tMax = dt * np.exp(dt * (NtPoints - 1))
    # tGrid = np.zeros(NtPoints)
    # for n in range(NtPoints):
    #     tGrid[n] = dt * np.exp(dt * n)

    tMax = 10
    dt1 = 0.1
    dt2 = dt1
    # tGrid = np.concatenate((np.arange(0, 1 + dt1, dt1), np.arange(1 + dt2, tMax + dt2, dt2)))
    tGrid = np.arange(0, tMax + dt1, dt1)

    PB_multiplier = 25
    gParams = [kgrid, xgrid, tGrid, PB_multiplier]

    # ---- SET SPARAMS ----
    mI = 1
    mB = 1
    n0 = 1
    gBB = (4 * np.pi / mB) * 0.05
    sParams = [mI, mB, n0, gBB]

    # ---- SET CPARAMS (RANGE OVER MULTIPLE P VALUES) ----

    aIBi = -2
    g = pf.g(kgrid, 0, aIBi, mI, mB, n0, gBB)
    Pg = pf.PCrit_grid(kgrid, 0, aIBi, mI, mB, n0, gBB)
    Pc = pf.PCrit_inf(kcutoff, aIBi, mI, mB, n0, gBB)

    # NPVals = 40
    # PVals = np.linspace(0, 3 * Pc, NPVals)
    NPVals = 4
    PVals = np.linspace(0.1, .95 * Pg, NPVals)
    cParams_List = [[P, aIBi] for P in PVals]
    print('Pc_inf: %.3f, Pc_g: %.3f' % (Pc, Pg))

    # ---- SET OUTPUT DATA FOLDER, CREATE PARAMETER SET, AND SAVE PARAMETER INFO FILE ----

    dirpath = os.path.dirname(os.path.realpath(__file__))
    outer_datapath = dirpath + '/clusterdata/aIBi_%.2f' % aIBi
    datapath = outer_datapath + '/NGridPoints_%.2E' % NGridPoints

    # if os.path.isdir(outer_datapath) is False:
    #     os.mkdir(outer_datapath)
    # if os.path.isdir(datapath) is False:
    #     os.mkdir(datapath)
    #     os.mkdir(datapath + '/Dist')

    paramInfo = 'kcutoff - {:.2f}, dk - {:.3f}, Ntheta - {:d}, NGridPoints - {:.2E}, tMax - {:.1f}, dt1 - {:.3f}, dt2 - {:.3f} NtPoints - {:d}\nmI - {:.1f}, mB - {:.1f}, n0 - {:0.1f}, gBB - {:0.3f}\naIBi - {:.2f}, gIB - {:0.3f}, PCrit_grid - {:.5f}, PCrit_true - {:0.5f}, NPVals - {:d}'.format(kcutoff, dk, Ntheta, NGridPoints, tMax, dt1, dt2, tGrid.size, mI, mB, n0, gBB, aIBi, g, Pg, Pc, NPVals)
    # with open(datapath + '/paramInfo.txt', 'w') as f:
    #     f.write(paramInfo)

    # # ---- COMPUTE DATA ----

    # taskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    # taskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    # if(taskCount != NPVals):
    #     print('ERROR: TASK COUNT MISMATCH')
    # else:
    #     os.mkdir(datapath + '/Dist/P_%.2f' % PVals[taskID])
    #     pf.quenchDynamics(cParams_List[taskID], gParams, sParams, datapath)

    # end = timer()
    # print('Task ID: {:d}, P: {:.2f}, Time: {:.2f}'.format(taskID, PVals[taskID], end - start))

# if os.path.isdir(datapath + '/PosSpace/P_%.2f' % PVals[4]) is False:
#     os.mkdir(datapath + '/PosSpace/P_%.2f' % PVals[4])


datapath = dirpath + '/data' + '/theta' + '/NGridPoints_%.2E' % NGridPoints
if os.path.isdir(datapath) is False:
    os.mkdir(datapath)

pf.quenchDynamics(cParams_List[3], gParams, sParams, datapath)
