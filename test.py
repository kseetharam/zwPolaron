import numpy as np
import Grid as g

kcutoff = 10
dk = 0.05

Ntheta = 50
dtheta = np.pi / (Ntheta - 1)

grid_space = g.Grid("SPHERICAL_2D")
grid_space.init1d('k', dk, kcutoff, dk)
grid_space.init1d('th', dtheta, np.pi, dtheta)

names = ['k', 'th']

functions = [lambda k: k**2, np.cos]

mat = grid_space.function_prod(names, functions)


k = grid_space.return_array1d('k')
th = grid_space.return_array1d('th')
print(k)
print(th)


man = np.outer(k**2, np.cos(th))

# print((k.size, th.size, k.size * th.size, mat.size))
print(man.reshape(man.size) - mat)
