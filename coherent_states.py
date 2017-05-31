import numpy as np


class Grid():

    def __init__(self):

        # creating an empty array for storing the grid
        self.arrays = {}

    def init1d(self, name, grid_min, grid_max, grid_step):

        # we create a 1d grid
        grid_1d = np.arange(grid_min, grid_max, grid_step)
        self.arrays[name] = grid_1d
        #self.grid_difference_1d = np.array()

    def print_arrays(self, name):
        print(self.arrays[name])


class Coherent():
#""" This is a class that stores information about coherent state """

    def __init__(self, grid):
        self.variational = np.array()
