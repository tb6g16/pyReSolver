# This file contains the function definition for several utility functions
# required by the Trajectory operations performed.

import numpy as np

def func2curve(curve_func, modes):
    disc = 2*(modes - 1)
    curve_array = np.zeros([disc, np.shape(curve_func(0))[0]])
    s = np.linspace(0, 2*np.pi*(1 - 1/disc), disc)
    for i in range(disc):
        curve_array[i, :] = curve_func(s[i])
    return curve_array

def list2array(my_list):
    array = np.zeros([len(my_list), *np.shape(my_list[0])], dtype = complex)
    for i in range(len(my_list)):
        array[i] = my_list[i]
    return array

def array2list(array):
    my_list = [None]*np.shape(array)[0]
    for i in range(len(my_list)):
        if type(array[i]) != np.ndarray:
            temp = np.array([array[i]])
        else:
            temp = array[i]
        my_list[i] = temp
    return my_list
