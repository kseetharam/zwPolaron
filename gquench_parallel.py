from timeit import default_timer as timer
import numpy as np
import os
import Grid
import CoherentState
import PolaronHamiltonian
from polrabi.staticfm import PCrit
import multiprocessing as mp
import itertools as it
from joblib import Parallel, delayed


def spectFunc(t_Vec, S_Vec):
    # spectral function (Fourier Transform of dynamical overlap)
    tstep = t_Vec[1] - t_Vec[0]
    N = t_Vec.size
    tdecay = 3
    decayFactor = np.exp(-1 * t_Vec / tdecay)
    # decayFactor = 1
    sf = 2 * np.real(np.fft.ifft(S_Vec * decayFactor))
    omega = 2 * np.pi * np.fft.fftfreq(N, d=tstep)
    return omega, sf


def dynamics(cParams, gParams, sParams):
    # takes parameters, performs dynamics, and outputs desired observables
    [P, aIBi] = cParams
    [grid_space, tMax, dt] = gParams
    [mI, mB, n0, gBB] = sParams

    # Initialization CoherentState
    cs = CoherentState.CoherentState(grid_space)
    # Initialization PolaronHamiltonian
    Params = [P, aIBi, mI, mB, n0, gBB]
    ham = PolaronHamiltonian.PolaronHamiltonian(cs, Params)
    # Time evolution
    tVec = np.arange(0, tMax, dt)
    PB_Vec = np.zeros(tVec.size, dtype=float)
    NB_Vec = np.zeros(tVec.size, dtype=float)
    DynOv_Vec = np.zeros(tVec.size, dtype=complex)

    for ind, t in enumerate(tVec):
        PB_Vec[ind] = cs.get_PhononMomentum()
        NB_Vec[ind] = cs.get_PhononNumber()
        DynOv_Vec[ind] = cs.get_DynOverlap()
        cs.evolve(dt, ham)

    freqVec, SpectFunc_Vec = spectFunc(tVec, DynOv_Vec)

    # Save Data
    dirpath = os.path.dirname(os.path.realpath(__file__))

    # data = [ham.Params, tVec, freqVec, PB_Vec, NB_Vec, DynOv_Vec, SpectFunc_Vec]
    # np.save(dirpath + '/pdata/gquench_aIBi:%.2f_P:%.2f.npy' % (aIBi, P), data)

    PVec = P * np.ones(freqVec.size)
    sfDat = np.concatenate((PVec[:, np.newaxis], freqVec[:, np.newaxis], SpectFunc_Vec[:, np.newaxis]), axis=1)
    np.save(dirpath + '/spectdata/gquench_aIBi_%.2f_P_%.2f.npy' % (aIBi, P), sfDat)
    # np.savetxt(dirpath + '/mm/gquench_aIBi_%.2f_P_%.2f.dat' % (aIBi, P), sfDat)


if __name__ == "__main__":

    # Initialization Grid
    kcutoff = 10
    dk = 0.05

    Ntheta = 50
    dtheta = np.pi / (Ntheta - 1)

    grid_space = Grid.Grid("SPHERICAL_2D")
    grid_space.initArray('k', dk, kcutoff, dk)
    grid_space.initArray('th', dtheta, np.pi, dtheta)

    # Set time evolution parameters
    tMax = 60
    dt = 1e-1
    gParams = [grid_space, tMax, dt]

    # Set sparams
    mI = 1
    mB = 1
    n0 = 1
    gBB = (4 * np.pi / mB) * 0.05
    sParams = [mI, mB, n0, gBB]

    # create range of cParam values (P,aIBi)

    NaIBiVals = 6  # must be an even number
    posarray = np.linspace(1.5, 5, NaIBiVals / 2)
    aIBiVals = 0.1 + np.concatenate((-1 * posarray[::-1], posarray), axis=0)
    Pc = PCrit(np.max(np.absolute(aIBiVals)), gBB, mI, mB, n0)

    aIBi = -2
    Pc = PCrit(aIBi, gBB, mI, mB, n0)
    print(Pc)

    NPVals = 24
    PVals = np.linspace(0, 2 * Pc, NPVals)

    cParams_List = [[P, aIBi] for P in PVals]
    # cParams_List = [[P, aIBi] for aIBi in aIBiVals for P in PVals]

    # create iterable over all tuples of function arguments for dynamics()

    paramsIter = zip(cParams_List, it.repeat(gParams), it.repeat(sParams))

    # compute data (parallel) - multiprocessing

    start = timer()

    num_cores = min(mp.cpu_count(), NPVals * NaIBiVals)
    print("Running on %d cores" % num_cores)

    with mp.Pool(num_cores) as pool:
        # pool = mp.Pool()
        pool.starmap(dynamics, paramsIter)
        # pool.close()
        # pool.join()

    end = timer()
    print(end - start)

    # compute data (parallel) - joblib

    # start = timer()

    # num_cores = min(mp.cpu_count(), NPVals)
    # print("Running on %d cores" % num_cores)
    # results1 = Parallel(n_jobs=num_cores)(delayed(dynamics)(*p) for p in paramsIter)

    # end = timer()
    # print(end - start)

    # compute data (serial) - for loop

    # start = timer()

    # for z in paramsIter:
    #     dynamics(*z)

    # end = timer()
    # print(end - start)

    # compute data(serial) - starmap

    # start = timer()

    # for i in it.starmap(dynamics, paramsIter):
    #     i

    # end = timer()
    # print(end - start)
