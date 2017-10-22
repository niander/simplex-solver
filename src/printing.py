import numpy as np
import sympy as sy


def matrix_pretty(matrix):
    matrix = np.asmatrix(matrix)
    sym_mat = sy.Matrix(matrix)
    sym_mat = sy.nsimplify(sym_mat, rational=True, tolerance=0.0001)
    return sy.pretty(sym_mat)


def number_pretty(num):
    num = float(num)
    num = sy.nsimplify(num, rational=True, tolerance=0.0001)
    return sy.pretty(num)
