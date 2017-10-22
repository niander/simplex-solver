import numpy as np

from helper import float_comp


def pivoting(mat, line, column):
    mat = np.asmatrix(mat, np.float64)
    piv = mat[line, column]
    mat[line, :] = mat[line, :] / piv
    for i in np.arange(0, mat.shape[0]):
        if i != line:
            f = mat[i, column]
            mat[i, :] = mat[i, :] - (mat[line, :] * f)
    return mat


def is_pivot_column(columns):
    columns = np.asarray(columns, np.float64)
    ret = np.empty(columns.shape[1], np.bool)
    for j in np.arange(0, columns.shape[1]):
        line_equal_one = np.isclose(columns[:, j], [1.0])
        # only one line with 1.0
        ret[j] = line_equal_one.sum() == 1
        # all other lines with 0.0
        ret[j] = ret[j] and np.all(float_comp(columns[np.logical_not(line_equal_one), j], 0.0, equal=True))
    return ret
