import numpy as np
import polaron_functions as pf
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})

# datapath
dirpath = os.path.dirname(os.path.realpath(__file__))
NGridP = '/NGridPoints_2.24E+04'
outerdatapath = dirpath + '/trajdata' + NGridP
ndatapath_f = outerdatapath + '/neg' + '/frohlich'
ndatapath_tp = outerdatapath + '/neg' + '/twophonon'

pdatapath_f = outerdatapath + '/pos' + '/frohlich'
pdatapath_tp = outerdatapath + '/pos' + '/twophonon'

# params

mB = 1
n0 = 1
aBB = 0.05
gBB = (4 * np.pi / mB) * aBB
nu = pf.nu(gBB * n0 / mB)

mI = 0.5 * mB
alpha = 6
P = 0.5 * nu * mI
xi = 1 / (np.sqrt(2 * mB * gBB * n0))

# data
ndatfile_f = ndatapath_f + '/traj_mI_%.2f_alpha_%.2f.dat' % (mI, alpha)
ndatfile_tp = ndatapath_tp + '/traj_mI_%.2f_alpha_%.2f.dat' % (mI, alpha)
pdatfile_f = pdatapath_f + '/traj_mI_%.2f_alpha_%.2f.dat' % (mI, alpha)
pdatfile_tp = pdatapath_tp + '/traj_mI_%.2f_alpha_%.2f.dat' % (mI, alpha)


tVec_f_n, xVec_f_n = np.loadtxt(ndatfile_f, unpack=True)
tVec_tp_n, xVec_tp_n = np.loadtxt(ndatfile_tp, unpack=True)
tVec_f_p, xVec_f_p = np.loadtxt(pdatfile_f, unpack=True)
tVec_tp_p, xVec_tp_p = np.loadtxt(pdatfile_tp, unpack=True)

xVec_free = (P * tVec_f_n / mI) / nu
# plot
fig, axes = plt.subplots(nrows=1, ncols=2)

axes[0].plot(tVec_tp_n, xVec_tp_n, 'r-', label='Two Phonon')
axes[0].plot(tVec_f_n, xVec_f_n, 'b-', label='Frohlich')
axes[0].plot(tVec_f_n, xVec_free, 'g-', label='Free')
axes[0].legend()
axes[0].set_xlabel(r'Time ($t$)')
axes[0].set_ylabel(r'Average Impurity Position ($x(t)$)')
axes[0].set_title('Negative Side')
# axes[0].set_xscale('log')
# axes[0].set_yscale('log')

axes[1].plot(tVec_tp_p, xVec_tp_p, 'r-', label='Two Phonon')
axes[1].plot(tVec_f_p, xVec_f_p, 'b-', label='Frohlich')
axes[1].plot(tVec_f_p, xVec_free, 'g-', label='Free')
axes[1].legend()
axes[1].set_xlabel(r'Time ($t$)')
axes[1].set_ylabel(r'Average Impurity Position ($x(t)$)')
axes[1].set_title('Positive Side')
# axes[1].set_xscale('log')
# axes[1].set_yscale('log')

fig.suptitle(r'Impurity Trajectory ($m_I=%.2f$, $\alpha=%.2f$)' % (mI, alpha))
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(outerdatapath + '/fig/traj_mI_%.2f_alpha_%.2f.pdf' % (mI, alpha))
plt.show()
