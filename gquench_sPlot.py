import numpy as np
import os


dirpath = os.path.dirname(os.path.realpath(__file__))
datapath = dirpath + '/spectdata'


for ind, filename in enumerate(os.listdir(datapath)):
    print(filename)
    dat = np.load(datapath + '/' + filename)
    if(ind == 0):
        sfDat = dat
    else:
        sfDat = np.concatenate((sfDat, dat), axis=0)


np.savetxt(dirpath + '/mm/sfDat.dat', sfDat)
