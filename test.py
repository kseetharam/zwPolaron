import numpy as np
import coherent_states as cs


# make a separate library file with functions for polaron

kcutoff = 10
dk = 0.05

Ntheta = 50
dtheta = np.pi / (Ntheta - 1)

grid_space = cs.Grid()
grid_space.init1d('k', dk, kcutoff, dk)
grid_space.init1d('th', dtheta, np.pi, dtheta)


grid_space.print_arrays('k')
grid_space.print_arrays('th')
