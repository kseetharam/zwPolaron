import numpy as np
import collections


class Grid:

    COORDINATE_SYSTEMS = ["CARTESIAN_3D", "SPHERICAL_2D", "1D_NoZero", "1D"]

    def __init__(self, coordinate_system):
        # Initialization of the Grid using the hash table.
        # The hash table self.arrays contains an information about
        # the grid graphs in d dimentions.
        # Thereafter the length of the self.arrays can be less or equal to d.
        # By defining the coordinate system we later initialize the volume elements

        self.arrays = {}
        self.arrays_diff = {}
        self.coordinate_system = coordinate_system

    def initArray(self, name, grid_min, grid_max, grid_step):
        # creates 1 dimentional grid graph with equidistant spacing
        # and adds this array to the hash table self.arrays[name]
        # for 1D, only grid_max matters -> creates symmetry aray around 0 with max value grid_max on either side
        if(self.coordinate_system == '1D_NoZero'):
            posarray = np.arange(grid_step, grid_max + grid_step, grid_step)
            array = np.concatenate((-1 * posarray[::-1], posarray), axis=0)
        else:
            array = np.arange(grid_min, grid_max + grid_step, grid_step)

        self.arrays_diff[name] = grid_step
        self.arrays[name] = array

    def getArray(self, name):
        # with a given key returns a corrsponding 1 dimentional array
        return self.arrays[name]

    def function_prod(self, list_of_unit_vectors, list_of_functions):
        # calculates a function on a grid when function is a product of function
        # acting independently on each coordinate
        # example:  list_of_unit_vectors = [x1, x2]
        # assume that f = f1 (x1) * f2 (x2)
        # then we can use: list_of_functions = [f1, f2, ...]
        # NOTE_ys: do we need to generalize it?

        # check that the order of keys in the list_of_unit_vectors is ordered
        # the same way as in self.arrays.keys
        # otherwise throw an error
        nC = collections.Counter(list_of_unit_vectors)
        sC = collections.Counter(self.arrays.keys())
        if(nC != sC):
            print('INVALID LIST OF NAMES')
            return

        outer_mat = list_of_functions[0](self.arrays[list_of_unit_vectors[0]])

        if(len(list_of_unit_vectors) == 1):
            return outer_mat

        for ind, name in enumerate(list_of_unit_vectors[1:]):
            temp = list_of_functions[ind + 1](self.arrays[name])
            outer_mat = np.outer(outer_mat, temp)

        return outer_mat.reshape(outer_mat.size)

    def diffArray(self, name):
        # returns a 1D array of values of the step difference for the 'name' variable
        # e.g. 'name' = k gives an array where each element is dk
        # normalizes the first and last values to avoid double counting

        grid_diff = self.arrays_diff[name] * np.ones(len(self.arrays[name]))
        grid_diff[0] = 0.5 * grid_diff[0]
        grid_diff[-1] = 0.5 * grid_diff[-1]
        return grid_diff

    def dV(self):
        # create an infinitisimal element of the volume that corresponds to the
        # given coordinate_system

        list_of_unit_vectors = list(self.arrays.keys())
        coordinate_system = self.coordinate_system

        # create dk, dtheta and modify it
        grid_diff = self.diffArray(list_of_unit_vectors[0])

        if(coordinate_system == "1D"):
            return grid_diff
        elif(coordinate_system == "1D_NoZero"):
            grid_diff[int(grid_diff.size / 2)] = (1 / 2) * grid_diff[int(grid_diff.size / 2)]
            grid_diff[int(-1 + grid_diff.size / 2)] = (1 / 2) * grid_diff[int(-1 + grid_diff.size / 2)]
            return grid_diff
        else:
            for ind, name in enumerate(list_of_unit_vectors[1:]):
                temp_grid_diff = self.diffArray(name)
                grid_diff = np.outer(grid_diff, temp_grid_diff)

        if coordinate_system == "SPHERICAL_2D":
            list_of_functions = [lambda k: (2 * np.pi)**(-2) * k**2, np.sin]

        if coordinate_system == "CARTESIAN_3D":
            prefac = (2 * np.pi)**(-3)
        else:
            prefac = self.function_prod(list_of_unit_vectors, list_of_functions)

        # print(len(output))
        return prefac * grid_diff.reshape(grid_diff.size)
        # use simps method for integration since there is no dk and dth yet

    def size(self):
        # this method returns the number of grid points
        list_of_unit_vectors = list(self.arrays.keys())
        grid_size = 1
        for unit in list_of_unit_vectors:
            grid_size = grid_size * len(self.arrays[unit])
        return grid_size

    def integrateFunc(self, functionValues, name):
        # takes an array of values of some function on the grid
        # where the function has been constructed from an outer product
        # of grid arrays in order
        # e.g. grid with arrays k & theta, 'functionValues' = array of k^2 * cos(th) values
        # returns array of values representing function integrated over variable 'name'
        # SHOULD PROBABLY GENERALIZE THIS TO 'name' BEING A LIST AND INTEGRATING IN ANY NUMBER OF VARIABLES

        coordinate_system = self.coordinate_system
        list_of_unit_vectors = list(self.arrays.keys())
        # finding out which index in grid variable keys is the integration variable
        # in order to get array reshaping correct - can probably do this better with Counter
        varInd = -1
        for ind, var in enumerate(list_of_unit_vectors):
            if var == name:
                varInd = ind
        if varInd == -1:
            print('INVALID LIST OF NAMES')
            return

        # 2D spherical grid case

        if coordinate_system == "SPHERICAL_2D":
            k_array = self.arrays[list_of_unit_vectors[0]]
            th_array = self.arrays[list_of_unit_vectors[1]]
            functionValues_mat = functionValues.reshape((len(k_array), len(th_array)))
            grid_diff = self.diffArray(list_of_unit_vectors[varInd])
            if varInd == 0:
                # what is the appropriate prefactor griddiff vector? aka k^2*dk or sin(th)*dtheta?
                functionValues_mat = np.transpose(functionValues_mat)  # transpose so we can integrate over 'k'
                prefactor = (2 * np.pi)**(-2) * k_array**2
            else:
                prefactor = np.sin(th_array)

            return np.dot(functionValues_mat, prefactor * grid_diff)

        # 3D cartesian grid case - ****NOT COMPLETE

        if coordinate_system == "CARTESIAN_3D":
            kx_array = self.arrays[list_of_unit_vectors[0]]
            ky_array = self.arrays[list_of_unit_vectors[1]]
            kz_array = self.arrays[list_of_unit_vectors[2]]

            functionValues_mat = functionValues.reshape((len(kx_array), len(ky_array), len(kz_array)))
            grid_diff = self.diffArray(list_of_unit_vectors[varInd])
            if varInd == 0:
                functionValues_mat = np.transpose(functionValues_mat)  # transpose so we can integrate over 'k'
                prefactor = (2 * np.pi)**(-2) * k_array**2
            else:
                prefactor = np.sin(th_array)

            return np.dot(functionValues_mat, prefactor * grid_diff)
