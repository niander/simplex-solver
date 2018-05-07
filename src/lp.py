import numpy as np


class LinearProgramming:
    def __init__(self):
        self._A = None
        self._b = None
        self._c = None

    def set_lp(self, A, b, c, is_fpi=False):
        self._is_fpi = is_fpi
        self._A = np.asmatrix(A, np.float64)
        self._c = np.asmatrix(c, np.float64).reshape(-1, 1)
        self._b = np.asmatrix(b, np.float64).reshape(-1, 1)
        if self._c.shape[0] != self._A.shape[1]:
            raise ValueError("A and c have incompatible shape size")
        if self._b.shape[0] != self._A.shape[0]:
            raise ValueError("A and b have incompatible shape size")

    @property
    def num_variables(self):
        return self.A.shape[1]

    @property
    def A(self):
        return self._A

    @property
    def c(self):
        return self._c

    @property
    def b(self):
        return self._b

    @property
    def A_fpi(self):
        if self._is_fpi:
            return self.A
        freevarsmat = np.asmatrix(np.identity(self.A.shape[0], np.float64))
        return np.hstack((self.A, freevarsmat))

    @property
    def c_fpi(self):
        if self._is_fpi:
            return self.c
        freevarsc = np.asmatrix(np.zeros((self.A.shape[0], 1), np.float64))
        return np.vstack((self.c, freevarsc))

    @property
    def b_fpi(self):
        return self.b
