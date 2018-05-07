import numpy as np


class Tableau:
    def __init__(self, mat):
        self._mat = None
        self.mat = mat

    @property
    def mat(self):
        return self._mat

    @mat.setter
    def mat(self, value):
        self._mat = np.asmatrix(value, np.float64)

    @property
    def num_cons(self):
        return self.mat.shape[0] - 1  # minus (-c)^t line

    @property
    def num_vars(self):
        return self.A.shape[1]

    @property
    def yt(self):
        return self._mat[0, 0:self.num_cons]

    @property
    def ct(self):
        return self._mat[0, self.num_cons:-1]

    @property
    def obj(self):
        return self._mat[0, -1:]

    @property
    def op(self):
        return self._mat[1:, 0:self.num_cons]

    @property
    def op_mat(self):
        return self._mat[:, 0:self.num_cons]

    @property
    def A(self):
        return self._mat[1:, self.num_cons:-1]

    @property
    def b(self):
        return self._mat[1:, -1]

    def add_constraint(self, cons):
        cons = np.asmatrix(cons, np.float64).reshape(1, -1)
        add_var = (cons.shape[1] - 1) - self.num_vars
        if add_var > 1:
            raise ValueError("Constraint with too many variables")

        new_mat = np.asmatrix(np.zeros((self.mat.shape[0] + 1, self.mat.shape[1] + 1 + add_var), np.float64))
        # y^t and op
        new_mat[0:-1, 0:self.num_cons] = self.op_mat
        new_mat[-1, 0:self.num_cons] = 0.0
        new_mat[0:-1, self.num_cons] = 0.0
        new_mat[-1, self.num_cons] = 1.0
        # ct
        new_mat[0, (self.num_cons + 1):-(1 + add_var)] = self.ct
        # obj value
        new_mat[0, -1] = self.obj
        # b
        new_mat[1:-1, -1] = self.b
        # A
        new_mat[1:-1, (self.num_cons + 1):-(1 + add_var)] = self.A
        new_mat[-1, (self.num_cons + 1):] = cons
        if add_var > 0:
            new_mat[0:-1, self.num_cons + self.num_vars + add_var] = 0.0
        self.mat = new_mat
