import coherent_states as cs

kcutoff = 10
dk = 0.05

gridk = cs.Grid()
gridk.init1d('k', dk, kcutoff, dk)
gridk.print_arrays('k')