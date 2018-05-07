from copy import deepcopy

import numpy as np

import gausselim
from helper import float_comp
from intsimplex import IntSimplexSolver


class BranchAndBoundSolver(IntSimplexSolver):
    def __init__(self, ilp):
        super().__init__(ilp)

    def run_solver(self, fout=None):
        super().run_simplex(fout)
        if self.lp_type == "bounded":
            self._int_tableau = self._run_solver_recursive(None, fout)
            if self._int_tableau is None:
                # raise Exception("LP from linear relaxation is feasible and bounded but ILP is infeasible")
                self.lp_type = "infeasible"
            else:
                self.lp_type = "bounded"

    def _run_solver_recursive(self, best_solution, fout=None):
        super().run_simplex(fout)
        if self.lp_type == "bounded":
            fract_sol_idx = self._get_fraction_solution_idx()
            if fract_sol_idx is None:
                if best_solution is None or self.objvalue > best_solution.obj[0, 0]:
                    return self._tableau
            else:
                if best_solution is None or self.objvalue > best_solution.obj[0, 0]:  # if not prunning
                    orig_tableau = self._tableau
                    orig_certificate = self._certificate

                    sol = self.orig_problem_solution
                    row_pivot = np.where(float_comp(self._tableau.A[:, fract_sol_idx], 1.0, equal=True).flat)[0][0]
                    new_cons = np.zeros((1, orig_tableau.num_vars + 2), np.float64)
                    new_cons[0, -2] = 1.0

                    new_cons[0, fract_sol_idx] = 1.0
                    new_cons[0, -1] = np.floor(sol[fract_sol_idx])
                    new_tableau_less = deepcopy(orig_tableau)
                    new_tableau_less.add_constraint(new_cons)
                    new_tableau_less.mat = gausselim.pivoting(new_tableau_less.mat,
                                                              row_pivot + 1,
                                                              fract_sol_idx + new_tableau_less.op.shape[1])
                    self._tableau = new_tableau_less
                    best_solution = self._run_solver_recursive(best_solution, fout)

                    new_cons[0, fract_sol_idx] = -1.0
                    new_cons[0, -1] = -np.ceil(sol[fract_sol_idx])
                    new_tableau_greater = deepcopy(orig_tableau)
                    new_tableau_greater.add_constraint(new_cons)
                    new_tableau_greater.mat = gausselim.pivoting(new_tableau_greater.mat,
                                                                 row_pivot + 1,
                                                                 fract_sol_idx + new_tableau_greater.op.shape[1])
                    self._tableau = new_tableau_greater
                    best_solution = self._run_solver_recursive(best_solution, fout)

                    self._tableau = orig_tableau
                    self._certificate = orig_certificate

        return best_solution

    def _get_fraction_solution_idx(self):
        sol = self.orig_problem_solution
        int_test = float_comp(sol - np.round(sol), 0.0, equal=True)
        if np.all(int_test):
            return None
        else:
            return np.where(np.logical_not(int_test).flat)[0][0]  # first fraction solution
