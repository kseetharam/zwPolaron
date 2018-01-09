import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from polaron_functions import spectFunc

if __name__ == "__main__":

   # # Initialization

    matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})
    NiceBlue = '#0087BD'
    NiceRed = '#C40233'
    NiceGreen = '#009F6B'
    NiceYellow = '#FFD300'
    fontsize = 16
    # load data
    Color = NiceRed
    aIBi = -10
    NGridPoints = 4e4

    dirpath = os.path.dirname(os.path.realpath(__file__))
    datapath = dirpath + '/data/production_data/aIBi_%.2f/NGridPoints_%.2E' % (aIBi, NGridPoints)

    for ind, filename in enumerate(os.listdir(datapath)):
        if(filename == 'paramInfo.txt'):
            continue
        P, t, S_Re, S_Imag = np.loadtxt(datapath + '/' + filename, usecols=(0, 1, 6, 7), unpack=True)
        # if(ind == 0):
        #     sfDat = dat
        # else:
        #     sfDat = np.concatenate((sfDat, dat), axis=0)


# fig, axes = plt.subplots(nrows=1, ncols=2)
# axes[0].plot(tVec, np.abs(DynOv_Vec), color=Color, lw=1, linestyle='-')
# axes[0].set_xlabel('Time, $t$', fontsize=fontsize)
# axes[0].set_ylabel(r'$\left|S(t)\right|$', fontsize=fontsize)
# axes[0].set_title('Dynamical Overlap')


# axes[1].plot(freqVec, SpectFunc_Vec, color=Color, lw=2, linestyle='-')
# # axes[1].set_xlim([-200, 100])
# # axes[1].set_ylim([0, 0.1])
# axes[1].set_xlabel(r'Frequency, $\omega$', fontsize=fontsize)
# axes[1].set_ylabel(r'$A(\omega)$', fontsize=fontsize)
# axes[1].set_title(r'Spectral Function')
# fig.tight_layout()
# fig.savefig(dirpath + '/figures/quench_DynOverlap&SpectFunction_aIBi:%.2f_P:%.2f.pdf' % (aIBi, P))

# plt.show()
