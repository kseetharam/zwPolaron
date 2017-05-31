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

    def apply_function(list_of_names, list_of_functions):
        # apply function to an array

        outer_mat = list_of_functions[0](self.arrays[list_of_names[0]])

        if(len(list_of_names) == 1):
            return outer_mat

        for ind, name in enumerate(list_of_names[1:]):
            temp = list_of_functions[ind](self.arrays[name])
            outer_mat = np.outer(outer_mat, temp)

        return outer_mat.reshape(outer_mat.size)


class Coherent():
    #""" This is a class that stores information about coherent state """

    def __init__(self, grid):
        self.variational = np.array()
