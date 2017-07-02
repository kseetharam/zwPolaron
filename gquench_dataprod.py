import numpy as np
import Grid
import polaron_functions as pf
import multiprocessing as mp
import itertools as it
import os
from timeit import default_timer as timer


if __name__ == "__main__":

    # ---- INITIALIZE GRID ----

    kcutoff = 10
    dk = 0.05

    Ntheta = 50
    dtheta = np.pi / (Ntheta - 1)

    NGridPoints = Ntheta * kcutoff / dk

    grid_space = Grid.Grid("SPHERICAL_2D")
    grid_space.initArray('k', dk, kcutoff, dk)
    grid_space.initArray('th', dtheta, np.pi, dtheta)

    # ---- SET GPARAMS ----

    dt = 5e-2
    NtPoints = 100
    tMax = dt * np.exp(dt * NtPoints)
    tGrid = np.zeros(NtPoints)
    for n in range(NtPoints):
        tGrid[n] = dt * np.exp(dt * n)

    gParams = [grid_space, tGrid]

    # ---- SET SPARAMS ----
    mI = 1
    mB = 1
    n0 = 1
    gBB = (4 * np.pi / mB) * 0.05
    sParams = [mI, mB, n0, gBB]

    # ---- SET CPARAMS (RANGE OVER MULTIPLE P VALUES) ----

    aIBi = -4
    Pg = pf.PCrit_grid(grid_space, 0, aIBi, mI, mB, n0, gBB)
    Pc = pf.PCrit_inf(aIBi, mI, mB, n0, gBB)

    NPVals = 64
    PVals = np.linspace(0, 4 * Pc, NPVals)
    cParams_List = [[P, aIBi] for P in PVals]

    # ---- SET OUTPUT DATA FOLDER, CREATE PARAMETER SET, AND SAVE PARAMETER INFO FILE ----

    dirpath = os.path.dirname(os.path.realpath(__file__))
    outer_datapath = dirpath + '/data/production_data/aIBi_%.2f' % aIBi
    if os.path.isdir(outer_datapath) is False:
        os.mkdir(outer_datapath)
    datapath = outer_datapath + '/NGridPoints_%.2E' % NGridPoints
    if os.path.isdir(datapath) is False:
        os.mkdir(datapath)

    paramsIter = zip(cParams_List, it.repeat(gParams), it.repeat(sParams), it.repeat(datapath))

    paramInfo = 'kcutoff: {:d}, dk: {:.3f}, Ntheta: {:d}, NGridPoints: {:.2E}, tMax: {:.1f}, dt: {:.2f}, NtPoints: {:d}\nmI: {:.1f}, mB: {:.1f}, n0: {:0.1f}, gBB: {:0.3f}\naIBi: {:.2f}, PCrit_grid: {:.3f}, PCrit_true: {:0.3f}, NPVals: {:d}'.format(kcutoff, dk, Ntheta, NGridPoints, tMax, dt, NtPoints, mI, mB, n0, gBB, aIBi, Pg, Pc, NPVals)
    with open(datapath + '/paramInfo.txt', 'w') as f:
        f.write(paramInfo)

    # ---- COMPUTE DATA (MULTIPROCESSING) ----

    start = timer()

    num_cores = min(mp.cpu_count(), NPVals)
    print("Running on %d cores" % num_cores)

    with mp.Pool(num_cores) as pool:
        pool.starmap(pf.quenchDynamics, paramsIter)

    end = timer()
    print(end - start)
