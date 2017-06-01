import numpy as np


class Grid:

    def __init__(self, coordinate_system):

        # creating an empty array for storing the grid
        self.arrays = {}
        self.coordinate_system = coordinate_system

    def init1d(self, name, grid_min, grid_max, grid_step):
        # we create a 1d grid
        grid_1d = np.arange(grid_min, grid_max, grid_step)
        self.arrays[name] = grid_1d
        # self.grid_difference_1d = np.array()

    def return_array1d(self, name):
        return self.arrays[name]

    def print_arrays(self, name):
        print(self.arrays[name])

    def apply_function(self, list_of_names, list_of_functions):
        # apply function to an array
        # checking : list_of_keys = self.arrays.keys()

        outer_mat = list_of_functions[0](self.arrays[list_of_names[0]])

        if(len(list_of_names) == 1):
            return outer_mat

        for ind, name in enumerate(list_of_names[1:]):
            temp = list_of_functions[ind](self.arrays[name])
            outer_mat = np.outer(outer_mat, temp)

        return outer_mat.reshape(outer_mat.size)
        # doesn't confirm with the test

    def dV(self):
        list_of_names = self.arrays.keys()
        coordinate_system = self.coordinate_system
        if coordinate_system == "SPHERICAL_2D":
            list_of_functions = [lambda k: (2 * np.pi)**(-1) * k**2, np.sin]

        output = self.apply_function(list_of_names, list_of_functions)
        return output
        # use simps method for integration since there is no dk and dth yet
