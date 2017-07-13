import numpy as np
import Grid
import polaron_functions as pf
import os
from timeit import default_timer as timer


if __name__ == "__main__":

    start = timer()

    # ---- INITIALIZE GRID ----

    kcutoff = 20
    dk = 0.05

    Ntheta = 100
    dtheta = np.pi / (Ntheta - 1)

    NGridPoints = Ntheta * kcutoff / dk

    grid_space = Grid.Grid("SPHERICAL_2D")
    grid_space.initArray('k', dk, kcutoff, dk)
    grid_space.initArray('th', dtheta, np.pi, dtheta)

    # ---- SET GPARAMS ----

    # dt = 1e-1
    # NtPoints = 94
    # tMax = dt * np.exp(dt * (NtPoints - 1))
    # tGrid = np.zeros(NtPoints)
    # for n in range(NtPoints):
    #     tGrid[n] = dt * np.exp(dt * n)

    tMax = 100
    dt1 = 1e-2
    dt2 = 1e-2
    tGrid = np.concatenate((np.arange(0, 1 + dt1, dt1), np.arange(1 + dt2, tMax + dt2, dt2)))
    gParams = [grid_space, tGrid]

    # ---- SET SPARAMS ----
    mI = 1
    mB = 1
    n0 = 1
    gBB = (4 * np.pi / mB) * 0.05
    sParams = [mI, mB, n0, gBB]

    # ---- SET CPARAMS (RANGE OVER MULTIPLE P VALUES) ----

    aIBi = 4
    g = pf.g(grid_space, 0, aIBi, mI, mB, n0, gBB)
    Pg = pf.PCrit_grid(grid_space, 0, aIBi, mI, mB, n0, gBB)
    Pc = pf.PCrit_inf(kcutoff, aIBi, mI, mB, n0, gBB)

    NPVals = 40
    PVals = np.linspace(0, 3 * Pc, NPVals)
    cParams_List = [[P, aIBi] for P in PVals]

    # ---- SET OUTPUT DATA FOLDER, CREATE PARAMETER SET, AND SAVE PARAMETER INFO FILE ----

    dirpath = os.path.dirname(os.path.realpath(__file__))
    outer_datapath = dirpath + '/clusterdata/aIBi_%.2f' % aIBi
    if os.path.isdir(outer_datapath) is False:
        os.mkdir(outer_datapath)
    datapath = outer_datapath + '/NGridPoints_%.2E' % NGridPoints
    if os.path.isdir(datapath) is False:
        os.mkdir(datapath)

    paramInfo = 'kcutoff: {:d}, dk: {:.3f}, Ntheta: {:d}, NGridPoints: {:.2E}, tMax: {:.1f}, dt1: {:.3f}, dt2: {:.3f} NtPoints: {:d}\nmI: {:.1f}, mB: {:.1f}, n0: {:0.1f}, gBB: {:0.3f}\naIBi: {:.2f}, gIB: {:0.3f}, PCrit_grid: {:.5f}, PCrit_true: {:0.5f}, NPVals: {:d}'.format(kcutoff, dk, Ntheta, NGridPoints, tMax, dt1, dt2, tGrid.size, mI, mB, n0, gBB, aIBi, g, Pg, Pc, NPVals)
    with open(datapath + '/paramInfo.txt', 'w') as f:
        f.write(paramInfo)

    # # ---- COMPUTE DATA ----

    taskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    taskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    if(taskCount != NPVals):
        print('ERROR: TASK COUNT MISMATCH')
    else:
        pf.quenchDynamics(cParams_List[taskID], gParams, sParams, datapath)

    end = timer()
    print('Task ID: {:d}, P: {:.2f}, Time: {:.2f}'.format(taskID, PVals[taskID], end - start))
