from copy import deepcopy

import numpy as np

import gausselim
from helper import float_comp
from intsimplex import IntSimplexSolver


class CuttingPlaneSolver(IntSimplexSolver):
    def __init__(self, ilp):
        super().__init__(ilp)
        self._int_tableau = self._tableau

    def run_solver(self, fout=None):
        super().run_simplex(fout)
        if self.lp_type == "bounded":
            rlx_tableau = deepcopy(self._tableau)
            rlx_certificate = deepcopy(self.certificate)
            self._run_cuttingplane(fout)
            self._tableau = rlx_tableau
            self._certificate = rlx_certificate

    def _run_cuttingplane(self, fout=None):
        fract_sol_pivot = self._get_first_fraction_solution_pivot()
        if fract_sol_pivot is not None:
            cut_plane = self._tableau.A[fract_sol_pivot[0], :]
            cut_plane = np.hstack((cut_plane, [[1.0]], self._tableau.b[fract_sol_pivot[0], :]))
            cut_plane = np.floor(cut_plane)
            self._tableau.add_constraint(cut_plane)
            self._tableau.mat = gausselim.pivoting(self._tableau.mat,
                                                   fract_sol_pivot[0] + 1,
                                                   fract_sol_pivot[1] + self._tableau.op.shape[1])
            if fout is not None:
                super()._write_matrix_to_file(fout, self._tableau.mat)
            super().run_simplex(fout)
            if self.lp_type != "bounded":
                # raise Exception("LP from linear relaxation is feasible and bounded but ILP is infeasible")
                self.lp_type = "infeasible"
            else:
                self._run_cuttingplane(fout)

    def _get_first_fraction_solution_pivot(self):
        sol = self.orig_problem_solution
        int_test = float_comp(sol - np.round(sol), 0.0, equal=True)
        if np.all(int_test):
            return None
        else:
            col_idx = np.where(np.logical_not(int_test).flat)[0][0]  # first fraction
            # base_columns = SimplexSolver._get_base_columns_indx(self._tableau)
            # row_idx = np.where(float_comp(self._tableau.A[:, base_columns[col_idx]], 1.0, equal=True).flat)[0][0]
            row_idx = np.where(float_comp(self._tableau.A[:, col_idx], 1.0, equal=True).flat)[0][0]
            # return pivot location for canonical fraction solution constraint
            return row_idx, col_idx
