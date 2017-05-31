import numpy as np
import coherent_states as cs

kcutoff = 10
dk = 0.05

Ntheta = 50
dtheta = np.pi / (Ntheta - 1)

gridk = cs.Grid()
gridk.init1d('k', dk, kcutoff, dk)
gridk.init1d('th', dtheta, np.pi, dtheta)
gridk.print_arrays('k')
gridk.print_arrays('th')