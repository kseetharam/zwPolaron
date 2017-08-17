from timeit import default_timer as timer
import numpy as np
import os
import Grid
import CoherentState
import PolaronHamiltonian
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

start = timer()
# Initialization Grid
kcutoff = 10
dk = 0.1

Ntheta = 30
dtheta = np.pi / (Ntheta - 1)

# in momentum space
grid_mom = Grid.Grid("SPHERICAL_2D")
grid_mom.initArray('k', dk, kcutoff, dk)
grid_mom.initArray('th', dtheta, np.pi, dtheta)

# in coordinate space
grid_coord = Grid.Grid("SPHERICAL_2D")
grid_coord.initArray('x', kcutoff**(-1), dk**(-1), kcutoff**(-1))
grid_coord.initArray('th', dtheta, np.pi, dtheta)


# Initialization CoherentState
cs = CoherentState.CoherentState(grid_mom, grid_coord)

# Initialization PolaronHamiltonian

mI = 1
mB = 1
n0 = 1
gBB = (4 * np.pi / mB) * 0.05
P = 2.
aIBi = -2

Params = [P, aIBi, mI, mB, n0, gBB]
ham = PolaronHamiltonian.PolaronHamiltonian(cs, Params)


# Time evolution
tMax = 3.
dt = 0.5


tVec = np.arange(0, tMax, dt)
PB_Vec = np.zeros(tVec.size, dtype=float)
NB_Vec = np.zeros(tVec.size, dtype=float)
DynOv_Vec = np.zeros(tVec.size, dtype=complex)

settings = ['g-', 'r-', 'b-', 'g--', 'r--', 'b--']
figMomDist, axMomDist = plt.subplots()
axMomDist.set_xlabel('$P$')
axMomDist.set_ylabel('$n_{k}^{bos}$')
axMomDist.set_title('MomDistr')


# figCoordDist, axCoordDist = plt.subplots()
# axCoordDist.set_xlabel('$X$')
# axCoordDist.set_ylabel('$n_{x}^{bos}$')
# axCoordDist.set_title('CoordDist')


for ind, t in enumerate(tVec):
    # PB_Vec[ind] = cs.get_PhononMomentum()
    # NB_Vec[ind] = cs.get_PhononNumber()
    # DynOv_Vec[ind] = cs.get_DynOverlap()

    # figMomDist.savefig('quench_PhononNumber.pdf')

    # dirpath = os.path.dirname(os.path.realpath(__file__))
    # np.save(dirpath + '/datatest/momdistr_aIBi:%.2f_P:%.2f_t:%.2f.npy' % (aIBi, P, t), data)

    momdistr = cs.get_MomentumDistribution()
    momdistr0 = np.abs(momdistr.reshape(len(grid_mom.arrays['k']), len(grid_mom.arrays['th']))[:, 0])
    axMomDist.plot(grid_mom.arrays['k'], momdistr0, settings[ind])

    # coorddist = cs.get_PositionDistribution()
    # coorddist0 = np.real(coorddist.reshape(len(grid_coord.arrays['x']), len(grid_coord.arrays['th']))[:, 0])
    # axCoordDist.plot(grid_coord.arrays['x'], coorddist0, settings[ind])

    cs.evolve(dt, ham)


# freqVec, SpectFunc_Vec = spectFunc(tVec, DynOv_Vec)

# # save data
# data = [ham.Params, tVec, PB_Vec, NB_Vec, DynOv_Vec]

# dirpath = os.path.dirname(os.path.realpath(__file__))
# np.save(dirpath + '/data/tgquench_aIBi:%.2f_P:%.2f.npy' % (aIBi, P), data)

end = timer()

print("time:" + str(end - start))

plt.show()

# figN, axN = plt.subplots()
# axN.plot(tVec, NB_Vec, 'k-')
# axN.set_xlabel('Time ($t$)')
# axN.set_ylabel('$N_{ph}$')
# axN.set_title('Number of Phonons')
# figN.savefig('quench_PhononNumber.pdf')

# def dynamics(cs, ham, tMax, dt):
#     # takes parameters, performs dynamics, and outputs desired observables
#     tVec = np.arange(0, tMax, dt)

#     PB_Vec = np.zeros(tVec.size, dtype=float)
#     NB_Vec = np.zeros(tVec.size, dtype=float)
#     DynOv_Vec = np.zeros(tVec.size, dtype=complex)

#     for ind, t in enumerate(tVec):

#         PB_Vec[ind] = cs.get_PhononMomentum()
#         NB_Vec[ind] = cs.get_PhononNumber()
#         DynOv_Vec[ind] = cs.get_DynOverlap()

#         cs.evolve(dt, ham)


#     # save data
#     data = [cs.Params, tVec, PB_Vec, NB_Vec, DynOv_Vec]

#     dirpath = os.path.dirname(os.path.realpath(__file__))
#     np.save(dirpath + '/data/gquench_aIBi:%.2f_P:%.2f.npy' % (aIBi, P), data)


# calculate dynamics

# print(trapz(A_Vec, freq_Vec))

# figN, axN = plt.subplots()
# axN.plot(tVec, NB_Vec, 'k-')
# axN.set_xlabel('Time ($t$)')
# axN.set_ylabel('$N_{ph}$')
# axN.set_title('Number of Phonons')
# figN.savefig('quench_PhononNumber.pdf')

# figPB, axPB = plt.subplots()
# axPB.plot(tVec, PB_Vec, 'b-')
# axPB.set_xlabel('Time ($t$)')
# axPB.set_ylabel('$P_{B}$')
# axPB.set_title('Phonon Momentum')
# figPB.savefig('quench_PhononMomentum.pdf')

# figp, axp = plt.subplots()
# axp.plot(tVec, np.sign(phi_Vec) * np.remainder(np.abs(phi_Vec), 2 * np.pi) / np.pi, 'r-')
# axp.set_xlabel('Time ($t$)')
# axp.set_ylabel(r'$\frac{\phi(t)}{\pi}$')
# axp.set_title('Global Phase')
# figp.savefig('quench_GlobalPhase.pdf')

# fig, axes = plt.subplots(nrows=1, ncols=2)
# axes[0].plot(tVec, np.abs(S_Vec), 'k-')
# axes[0].set_xlabel('Time ($t$)')
# axes[0].set_ylabel(r'$\left|S(t)\right|$')
# axes[0].set_title('Dynamical Overlap')


# axes[1].plot(freqVec, A_Vec, 'k-')
# axes[1].set_xlim([-30, 30])
# axes[1].set_ylim([0, 0.1])
# axes[1].set_xlabel(r'Frequency ($\omega$)')
# axes[1].set_ylabel(r'$A(\omega)$')
# axes[1].set_title(r'Spectral Function')
# fig.savefig('quench_DynOverlap&SpectFunction.pdf')
