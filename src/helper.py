import numpy as np


def float_comp(array, value, equal=False, greater=False, less=False):
    # all true
    if all([equal, greater, less]):
        raise ValueError("Wrong set of arguments for comparison")

    is_equal = np.isclose(array, np.asfarray(value))
    is_not_equal = np.logical_not(is_equal)

    if not equal and (all([greater, less]) or not any([greater, less])):
        return is_not_equal
    elif greater:
        is_greater = array > value
        if equal:
            return np.logical_or(is_greater, is_equal)
        else:
            return np.logical_and(is_greater, is_not_equal)
    elif less:
        is_less = array < value
        if equal:
            return np.logical_or(is_less, is_equal)
        else:
            return np.logical_and(is_less, is_not_equal)
    else:
        return is_equal
