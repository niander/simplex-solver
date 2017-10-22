import numpy as np


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
        column_pivot = np.isclose(columns[:, j], [1.0])
        ret[j] = column_pivot.sum() == 1 and np.allclose(columns[np.logical_not(column_pivot), j], [0.0])
    return ret
