import os

import numpy as np

import gausselim
import printing
from helper import float_comp
from lp import LinearProgramming
from tableau import Tableau


class SimplexSolver:
    def __init__(self, lp):
        if not isinstance(lp, LinearProgramming):
            raise ValueError("lp is not an object of type LinearProgramming")
        self.lp = lp
        self.lp_type = None  # bounded, unbounded or infeasible
        self._certificate = None

        self._tableau = self._build_tableau()

    @property
    def solution(self):
        self._test_lp_bounded()
        base_columns = SimplexSolver._get_base_columns_indx(self._tableau)
        solmat = self._tableau.A[:, base_columns]
        sol = np.asmatrix(np.zeros((self._tableau.A.shape[1], 1), np.float64))
        sol[base_columns] = solmat.T * self._tableau.b
        return sol

    @property
    def orig_problem_solution(self):
        sol = self.solution
        return sol[0:self.lp.num_variables]

    @property
    def objvalue(self):
        self._test_lp_bounded()
        return self._tableau.obj[0, 0]

    @property
    def certificate(self):
        if self._certificate is None:
            raise Exception("Simplex has not been executed yet")
        else:
            return self._certificate

    def run_simplex(self, fout=None):
        # need auxiliary tableau?
        if np.any(float_comp(self._tableau.ct, 0.0, less=True)) and \
                np.any(float_comp(self._tableau.b, 0.0, less=True)):
            aux_tableau = self._build_aux_tableau()
            num_cons = aux_tableau.A.shape[0]
            base_columns = np.arange(-num_cons, 0)
            SimplexSolver._update_ct_to_canonical_form(aux_tableau, base_columns, fout)
            self._run_simplex(aux_tableau, fout)
            if np.all(float_comp(aux_tableau.obj, 0.0, equal=False)):  # Infeasible LP
                self.lp_type = "infeasible"
                self._certificate = aux_tableau.yt.T
                return
            else:
                num_cons = aux_tableau.A.shape[0]
                # updating the original tableau with the contents of the aux tableau
                self._tableau.mat[1:, :] = np.hstack((aux_tableau.mat[1:, :-(num_cons + 1)],
                                                      aux_tableau.b))
                bcols = SimplexSolver._get_base_columns_indx(aux_tableau)
                SimplexSolver._update_ct_to_canonical_form(self._tableau, bcols, fout)

        last_pivot = self._run_simplex(self._tableau, fout)
        if last_pivot is None:  # Bounded LP
            self.lp_type = "bounded"
            self._certificate = self._tableau.yt.T
        else:  # Unbounded  or Infeasible LP
            if last_pivot[0] is not None:  # Infeasible LP
                self.lp_type = "infeasible"
                self._certificate = self._tableau.op[last_pivot[0], :].T
            elif last_pivot[1] is not None:  # Unbounded LP
                self.lp_type = "unbounded"
                self._certificate = self._calc_unbounded_certificate(last_pivot[1])
            else:
                raise Exception("Result from simplex not expected")

    def _build_tableau(self):
        ncons = self.lp.A_fpi.shape[0]
        opmat = np.vstack((np.zeros((1, ncons)),
                           np.identity(ncons)))
        opmat = np.asmatrix(opmat, np.float64)

        lpmat = np.vstack((np.hstack((self.lp.c_fpi.T * (-1), [[0.0]])),
                           np.hstack((self.lp.A_fpi, self.lp.b_fpi))))
        tableau = np.asmatrix(np.hstack((opmat, lpmat)))

        return Tableau(tableau)

    def _build_aux_tableau(self):
        num_cons = self._tableau.A.shape[0]
        aux_c = np.zeros_like(self._tableau.ct)
        aux_c = np.asmatrix(np.hstack((np.zeros((1, num_cons)),
                                       aux_c,
                                       np.ones((1, num_cons)))),
                            np.float64)
        b_neglines = np.where(float_comp(self._tableau.b, 0.0, less=True))[0]
        opmat = np.asmatrix(np.identity(num_cons), np.float64)
        aux_opA = np.hstack((opmat, self._tableau.A))
        aux_opA[b_neglines, :] = aux_opA[b_neglines, :] * (-1)
        aux_opA = np.hstack((aux_opA, np.identity(aux_opA.shape[0])))
        aux_b = np.copy(self._tableau.b)
        aux_b[b_neglines, :] = aux_b[b_neglines, :] * (-1)

        aux_tab_mat = np.vstack((np.hstack((aux_c, [[0.0]])),
                                 np.hstack((aux_opA, aux_b))))

        aux_tab = Tableau(aux_tab_mat)

        return aux_tab

    def _test_lp_bounded(self):
        if self.lp_type is None:
            raise Exception("Simplex has not been executed yet")
        if self.lp_type != "bounded":
            raise Exception("LP is not bounded")

    def _calc_unbounded_certificate(self, A_column):
        nvars = self._tableau.A.shape[1]
        certificate = np.asmatrix(np.zeros((nvars, 1), np.float64))
        certificate[A_column] = 1.0
        base_columns = SimplexSolver._get_base_columns_indx(self._tableau)
        certificate[base_columns] = - self._tableau.A[:, base_columns].T * self._tableau.A[:, A_column]
        return certificate

    @staticmethod
    def _run_simplex(tableau, fout=None):
        pivot = SimplexSolver._choose_pivot(tableau)
        while pivot is not None and pivot[0] is not None and pivot[1] is not None:
            # print(pivot)
            tableau.mat = gausselim.pivoting(tableau.mat,
                                             pivot[0] + 1,
                                             pivot[1] + tableau.op.shape[1])
            if fout is not None:
                SimplexSolver._write_matrix_to_file(fout, tableau.mat)
            pivot = SimplexSolver._choose_pivot(tableau)
        if pivot is not None:  # Unbounded or Infeasible LP
            # If Primal -> Unbounded, if Dual -> Infeasible
            return pivot
        else:  # Bounded LP
            return None

    @staticmethod
    def _choose_pivot(tableau):
        if np.alltrue(float_comp(tableau.ct, 0.0, greater=True, equal=True)):
            return SimplexSolver._choose_pivot_dual(tableau)
        if np.alltrue(float_comp(tableau.b, 0.0, greater=True, equal=True)):
            return SimplexSolver._choose_pivot_primal(tableau)

    @staticmethod
    def _choose_pivot_dual(tableau):
        i = 0
        while i < tableau.b.shape[0] and \
                float_comp(tableau.b[i, 0], 0.0, greater=True, equal=True):
            i += 1
        if i < tableau.b.shape[0]:
            negat = np.where(float_comp(tableau.A[i, :].flat, 0.0, less=True))
            negat = negat[0]
            if negat.size == 0:  # infeasible
                return i, None
            else:
                j = negat[np.argmin(tableau.ct[0, negat] / (-1 * tableau.A[i, negat]))]
                return i, j
        else:
            return None  # No element in b negative

    @staticmethod
    def _choose_pivot_primal(tableau):
        j = 0
        while j < tableau.ct.shape[1] and \
                float_comp(tableau.ct[0, j], 0.0, greater=True, equal=True):
            j += 1
        if j < tableau.ct.shape[1]:
            posit = np.where(float_comp(tableau.A[:, j].flat, 0.0, greater=True))
            posit = posit[0]
            if posit.size == 0:  # unbounded
                return None, j  # return the last chosen column
            else:
                i = posit[np.argmin(tableau.b[posit, 0] / tableau.A[posit, j])]
                return i, j
        else:
            return None  # No element in (-c)^t negative

    @staticmethod
    def _get_base_columns_indx(tableau):
        base_columns = np.where(float_comp(tableau.ct, 0.0, equal=True))[1]
        base_columns = base_columns[gausselim.is_pivot_column(tableau.A[:, base_columns])]
        return base_columns

    @staticmethod
    def _update_ct_to_canonical_form(tableau, base_columns, fout=None):
        pivot_lines = np.arange(0, tableau.A.shape[0]) * tableau.A[:, base_columns]
        pivot_lines = pivot_lines.round().astype(int) + 1  # first line
        for i, j in zip(pivot_lines.flat, base_columns):
            tableau.mat[0, :] = tableau.mat[0, :] - (tableau.mat[i, :] * tableau.ct[0, j])
            if fout is not None:
                SimplexSolver._write_matrix_to_file(fout, tableau.mat)

    @staticmethod
    def _write_matrix_to_file(file, mat):
        SimplexSolver._write_to_file(file, printing.matrix_pretty(mat) + os.linesep)

    @staticmethod
    def _write_to_file(file, text):
        file.write(text)
