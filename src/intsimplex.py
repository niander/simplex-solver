from ilp import IntLinearProgramming
from simplex import SimplexSolver


class IntSimplexSolver(SimplexSolver):
    def __init__(self, ilp):
        if not isinstance(ilp, IntLinearProgramming):
            raise ValueError("ilp is not an object of type IntLinearProgramming")
        super().__init__(ilp)
        self._int_tableau = None

    @property
    def int_solution(self):
        return self._change_tableau_and_get_attr('solution')

    @property
    def int_orig_problem_solution(self):
        return self._change_tableau_and_get_attr('orig_problem_solution')

    @property
    def int_objvalue(self):
        return self._change_tableau_and_get_attr('objvalue')

    def _change_tableau_and_get_attr(self, attr_name):
        rlx_tableau = self._tableau
        self._tableau = self._int_tableau
        attr = getattr(self, attr_name)
        self._tableau = rlx_tableau
        return attr
