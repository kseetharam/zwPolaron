import numpy as np
from scipy.integrate import simps
import Grid as g
# import CoherentState as CS


# define parameters of the grid
k_max = 2
dk = 0.01
Ntheta = 100
dtheta = np.pi / (Ntheta - 1)

# create grid
# test init
grid_space = g.Grid("SPHERICAL_2D")
grid_space.init1d('k', dk, k_max, dk)
grid_space.init1d('th', dtheta, np.pi, dtheta)

# print(grid_space.arrays)
# print(grid_space.coordinate_system)

names = ['k', 'th']
k = grid_space.return_array1d('k')
th = grid_space.return_array1d('th')
# print(k)
# print(th)

# test function_prod
functions = [lambda k: k**2, np.cos]
mat = grid_space.function_prod(names, functions)
man = np.outer(k**2, np.cos(th))
# print(man.reshape(man.size) - mat)

# test dV
Volume = grid_space.dV()
# check that integration on the grid is not too bad
print("Inegral over 3d volume: " + str(sum(Volume) * dk * dtheta))
print("1/(6 pi^2) R^3: " + str(1. / (6 * np.pi**2) * k_max**3))
