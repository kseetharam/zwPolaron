from timeit import default_timer as timer
import numpy as np
import os
import Grid
import CoherentState
import PolaronHamiltonian
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})


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


# Initialization Grid
kcutoff = 10
dk = 0.05

Ntheta = 100
dtheta = np.pi / (Ntheta - 1)

grid_space = Grid.Grid("SPHERICAL_2D")
grid_space.initArray('k', dk, kcutoff, dk)
grid_space.initArray('th', dtheta, np.pi, dtheta)

# Initialization CoherentState
cs = CoherentState.CoherentState(grid_space)

# Initialization PolaronHamiltonian

mI = 1
mB = 1
n0 = 1
gBB = (4 * np.pi / mB) * 0.05
P = 3
aIBi = -2

Params = [P, aIBi, mI, mB, n0, gBB]
ham = PolaronHamiltonian.PolaronHamiltonian(cs, Params)


# Time evolution
tMax = 60
dt = 1e-1

start = timer()

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

# save data
data = [ham.Params, tVec, freqVec, PB_Vec, NB_Vec, DynOv_Vec, SpectFunc_Vec]

dirpath = os.path.dirname(os.path.realpath(__file__))
np.save(dirpath + '/data/tgquench_aIBi:%.2f_P:%.2f.npy' % (aIBi, P), data)

end = timer()

print(end - start)

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

plt.show()
