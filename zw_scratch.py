import numpy as np
import Grid
import pf_static_sph as pfs
import os
from timeit import default_timer as timer


if __name__ == "__main__":

    start = timer()

    # ---- INITIALIZE GRIDS ----

    (Lx, Ly, Lz) = (20, 20, 20)
    (dx, dy, dz) = (0.2, 0.2, 0.2)

    # (Lx, Ly, Lz) = (21, 21, 21)
    # (dx, dy, dz) = (0.25, 0.25, 0.25)

    NGridPoints_cart = (1 + 2 * Lx / dx) * (1 + 2 * Ly / dy) * (1 + 2 * Lz / dz)
    NGridPoints_desired = (1 + 2 * Lx / dx) * (1 + 2 * Lz / dz)
    Ntheta = 50
    Nk = np.ceil(NGridPoints_desired / Ntheta)

    theta_max = np.pi
    thetaArray, dtheta = np.linspace(0, theta_max, Ntheta, retstep=True)

    # k_max = np.sqrt((np.pi / dx)**2 + (np.pi / dy)**2 + (np.pi / dz)**2)
    k_max = ((2 * np.pi / dx)**3 / (4 * np.pi / 3))**(1 / 3)

    k_min = 1e-5
    kArray, dk = np.linspace(k_min, k_max, Nk, retstep=True)
    if dk < k_min:
        print('k ARRAY GENERATION ERROR')

    kgrid = Grid.Grid("SPHERICAL_2D")
    kgrid.initArray_premade('k', kArray)
    kgrid.initArray_premade('th', thetaArray)

    gParams = [kgrid]

    NGridPoints = kgrid.size()

    print('dk: {0}'.format(dk))
    print('UV cutoff: {0}'.format(k_max))
    print('NGridPoints: {0}'.format(NGridPoints))

    # print('Perfect \int ep: {0}'.format(k_max**5 / (5 * (2 * np.pi)**2)))

    # Basic parameters

    mI = 1.7
    mB = 1
    n0 = 1
    aBB = 0.062
    gBB = (4 * np.pi / mB) * aBB
    nu = pfs.nu(gBB)
    xi = np.sqrt(8 * np.pi * n0 * aBB)

    # Interpolation

    Nsteps = 1e2
    pfs.createSpline_grid(Nsteps, kgrid, mI, mB, n0, gBB)

    aSi_tck = np.load('aSi_spline_sph.npy')
    PBint_tck = np.load('PBint_spline_sph.npy')

    sParams = [mI, mB, n0, gBB, aSi_tck, PBint_tck]

    # # ---- SINGLE FUNCTION RUN ----

    runstart = timer()

    P = 0.1
    # aIBi = -0.32
    aIBi_Vals = np.array([-5.0, -1.17, -0.5])

    for Aind, aIBi in enumerate(aIBi_Vals):
        DP = pfs.DP_interp(0, P, aIBi, aSi_tck, PBint_tck)
        aSi = pfs.aSi_interp(DP, aSi_tck)
        PB_Val = pfs.PB_interp(DP, aIBi, aSi_tck, PBint_tck)
        # Pcrit = PCrit_grid(kgrid, aIBi, mI, mB, n0, gBB)
        # En = Energy(P, PB_Val, aIBi, aSi, mI, mB, n0)
        # nu_const = nu(gBB)
        eMass = pfs.effMass(P, PB_Val, mI)
        # gIB = g(kgrid, aIBi, mI, mB, n0, gBB)
        # Nph = num_phonons(kgrid, aIBi, aSi, DP, mI, mB, n0, gBB)
        # Z_factor = z_factor(kgrid, aIBi, aSi, DP, mI, mB, n0, gBB)
        end = timer()
        print('aIBi: {0}, m*/mI: {1}'.format(aIBi, eMass / mI))
        # print('aSi-aIBi: {0}'.format(aSi - aIBi))
