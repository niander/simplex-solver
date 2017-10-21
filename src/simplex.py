from lp import LinearProgramming
import gausselim
import numpy as np
import sympy


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
        sol = np.asarray(np.zeros((self._tableau.A.shape[1], 1), np.float64))
        sol[base_columns] = solmat.T * self._tableau.b
        return sol

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

    def run_simplex(self, output_file=None):
        if np.any(self._tableau.ct < 0.0) and np.any(self._tableau.b < 0.0):
            aux_tableau = self._build_aux_tableau()
            self._run_simplex(aux_tableau, output_file)
            if not np.allclose(aux_tableau.obj, [0.0]):  # Infeasible LP
                self.lp_type = "infeasible"
                self._certificate = aux_tableau.yt.T
                return
            else:
                num_cons = aux_tableau.A.shape[0]
                # updating the original tableau with the contents of the aux tableau
                self._tableau.mat[1:, :] = np.hstack((aux_tableau.mat[1:, :-(num_cons + 1)],
                                                      aux_tableau.b))
                bcols = SimplexSolver._get_base_columns_indx(aux_tableau)
                SimplexSolver._update_ct_to_canonical_form(self._tableau, bcols)

                # get lines indx for pivoting
                # nlines = self._tableau.A.shape[0]
                # pivot_lines = np.arange(0, nlines) * aux_tableau.A[:, bcols]
                # for i, j in zip(pivot_lines, bcols):
                #     i += self._tableau.ct.shape[0]  # always 1
                #     j += self._tableau.op.shape[1]
                #     self._tableau.mat = gausselim.pivoting(self._tableau.mat, i, j)

        last_pivot = self._run_simplex(self._tableau)
        if last_pivot is None:  # Bounded LP
            self.lp_type = "bounded"
            self._certificate = self._tableau.yt.T
        else:  # Unbounded  or Infeasible LP
            if last_pivot[0] is not None:  # Infeasible LP
                self.lp_type = "infeasible"
                self._certificate = self._tableau.yt.T
            elif last_pivot[1] is not None:  # Unbounded LP
                self.lp_type = "unbounded"
                self._certificate = self._calc_unbounded_certificate(last_pivot[1])
            else:
                raise Exception("Result from simplex not expected")

    def _run_simplex(self, tableau, output_file=None):
        print(tableau.mat)
        pivot = SimplexSolver._choose_pivot(tableau)
        while pivot is not None and pivot[0] is not None:
            print(pivot)
            tableau.mat = gausselim.pivoting(tableau.mat, pivot[0], pivot[1])
            print("-------------------------------")
            print(tableau.mat)
            pivot = SimplexSolver._choose_pivot(tableau)
        if pivot is not None:  # Unbounded or Infeasible LP
            # If Simplex -> Unbounded, if Dual -> Infeasible
            return pivot
        else:  # Bounded LP
            return None

    def _build_tableau(self):
        ncons = self.lp.A_fpi.shape[0]
        opmat = np.vstack((np.zeros((1, ncons)),
                           np.identity(ncons)))
        opmat = np.asmatrix(opmat, np.float64)

        lpmat = np.vstack((np.hstack((self.lp.c_fpi.T * (-1), [[0.0]])),
                           np.hstack((self.lp.A_fpi, self.lp.b_fpi))))
        tableau = np.asmatrix(np.hstack((opmat, lpmat)))

        return Tableau(tableau)

        # self._tab_yt = self._tableau[0, 0:ncons]
        # self._tab_op = self._tableau[1:, 0:ncons]
        # self._tab_ct = self._tableau[0,ncons:-1]
        # self._tab_obj = self._tableau[0,-1:]
        # self._tab_A = self._tableau[1:,ncons:-1]
        # self._tab_b = self._tableau[1:,-1]

    def _test_lp_bounded(self):
        if self.lp_type is None:
            raise Exception("Simplex has not been executed yet")
        if self.lp_type != "bounded":
            raise Exception("LP is not bounded")

    @staticmethod
    def _choose_pivot(tableau):
        if np.alltrue(tableau.ct >= 0.0):
            return SimplexSolver._choose_pivot_dual(tableau)
        if np.alltrue(tableau.b >= 0.0):
            return SimplexSolver._choose_pivot_primal(tableau)

    @staticmethod
    def _choose_pivot_dual(tableau):
        i = 0
        while i < tableau.b.shape[0] and tableau.b[i, 0] >= 0.0:
            i += 1
        if i < tableau.b.shape[0]:
            negat = np.where(np.logical_and(
                tableau.A[i, :].flat < 0.0,
                np.logical_not(np.isclose(tableau.A[i, :].flat, [0.0]))
            ))
            negat = negat[0]
            if negat.size == 0:  # infeasible
                return (i, None)
            else:
                j = negat[np.argmin(tableau.ct[0, negat] / (-1 * tableau.A[i, negat]))]
                i += tableau.ct.shape[0]  # always 1
                j += tableau.op.shape[1]  # add opt matrix columns
                return (i, j)
        else:
            return None  # No element in b negative

    @staticmethod
    def _choose_pivot_primal(tableau):
        j = 0
        while j < tableau.ct.shape[1] and tableau.ct[0, j] >= 0.0:
            j += 1
        if j < tableau.ct.shape[1]:
            posit = np.where(np.logical_and(
                tableau.A[:, j].flat > 0.0,
                np.logical_not(np.isclose(tableau.A[:, j].flat, [0.0]))
            ))
            posit = posit[0]
            if posit.size == 0:  # unbounded
                return (None, j)  # return the last chosen column
            else:
                i = posit[np.argmin(tableau.b[posit, 0] / tableau.A[posit, j])]
                i = tableau.ct.shape[0] + i  # always 1
                j = tableau.op.shape[1] + j  # add opt matrix columns
                return (i, j)
        else:
            return None  # No element in (-c)^t negative

    def _build_aux_tableau(self):
        num_cons = self._tableau.A.shape[0]
        aux_c = np.zeros_like(self._tableau.ct)
        aux_c = np.asmatrix(np.hstack((np.zeros((1, num_cons)),
                                       aux_c,
                                       np.ones((1, num_cons)))),
                            np.float64)
        b_neglines = np.where(self._tableau.b < 0.0)[0]
        opmat = np.asmatrix(np.identity(num_cons), np.float64)
        aux_opA = np.hstack((opmat, self._tableau.A))
        aux_opA[b_neglines, :] = aux_opA[b_neglines, :] * (-1)
        aux_opA = np.hstack((aux_opA, np.identity(aux_opA.shape[0])))
        aux_b = np.copy(self._tableau.b)
        aux_b[b_neglines, :] = aux_b[b_neglines, :] * (-1)

        aux_tab_mat = np.vstack((np.hstack((aux_c, [[0.0]])),
                                 np.hstack((aux_opA, aux_b))))

        aux_tab = Tableau(aux_tab_mat)
        num_vars = self._tableau.A.shape[1]
        pivot_columns = num_vars + np.arange(0, num_cons)
        SimplexSolver._update_ct_to_canonical_form(aux_tab, pivot_columns)
        return aux_tab

    @staticmethod
    def _get_base_columns_indx(tableau):
        base_columns = np.where(np.isclose(tableau.ct, [0.0]))[1]
        base_columns = base_columns[gausselim.is_pivot_column(tableau.A[:, base_columns])]
        return base_columns

    def _calc_unbounded_certificate(self, column):
        nvars = self._tableau.A.shape[1]
        certificate = np.asmatrix(np.zeros((nvars, 1), np.float64))
        certificate[column] = 1.0
        base_columns = SimplexSolver._get_base_columns_indx(self._tableau)
        certificate[base_columns] = - self._tableau.A[:, base_columns].T * self._tableau.A[:, column]
        return certificate

    @staticmethod
    def _update_ct_to_canonical_form(tableau, base_columns):
        # pivot_lines = np.where(np.isclose(tableau.A[:, pivot_columns], [1.0]))[0]
        pivot_lines = np.arange(0, tableau.A.shape[0]) * tableau.A[:, base_columns]
        pivot_lines = pivot_lines.round().astype(int) + 1  # first line
        for i, j in zip(pivot_lines.flat, base_columns):
            tableau.mat[0, :] = tableau.mat[0, :] - (tableau.mat[i, :] * tableau.ct[0, j])


class Tableau:
    def __init__(self, tableau):
        self.mat = tableau
        self._nlines = tableau.shape[0] - 1  # minus (-c)^t line

    @property
    def mat(self):
        return self._tableau

    @mat.setter
    def mat(self, value):
        self._tableau = np.asmatrix(value, np.float64)

    @property
    def yt(self):
        return self._tableau[0, 0:self._nlines]

    @property
    def ct(self):
        return self._tableau[0, self._nlines:-1]

    @property
    def obj(self):
        return self._tableau[0, -1:]

    @property
    def op(self):
        return self._tableau[1:, 0:self._nlines]

    @property
    def A(self):
        return self._tableau[1:, self._nlines:-1]

    @property
    def b(self):
        return self._tableau[1:, -1]


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    lp = LinearProgramming()
    # lp.set_lp([[3,-2], [6,-2]], [2,2], [3,2]) # unbounded
    lp.set_lp([[3, 2], [6, 2]], [2, 2], [2, 1])
    # lp.set_lp([[-3,-2], [-6,-2]], [-2,-2], [-2,-1]) # dual -> bounded
    # lp.set_lp([[3, 1], [2, 2]], [-1, 2], [1, 1]) # aux -> infeasible
    # lp.set_lp([[3, -1], [2, 2]], [-1, 2], [1, 1]) # aux -> bounded
    ss = SimplexSolver(lp)
    print(ss._tableau.mat)
    print("#############################")
    # print(ss._choose_pivot_primal())
    ss.run_simplex()
    if ss.lp_type == "bounded":
        print(ss.solution)
    elif ss.lp_type == "unbounded":
        print("unbounded:")
        print(ss.certificate)
        # print(ss.solution)
    elif ss.lp_type == "infeasible":
        print("infeasible:")
        print(ss.certificate)
        # print(ss.solution)
