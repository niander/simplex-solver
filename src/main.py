import argparse
import sys

import numpy as np

import printing
from branchandbound import BranchAndBoundSolver
from cuttingplane import CuttingPlaneSolver
from ilp import IntLinearProgramming

parser = argparse.ArgumentParser(description='Simplex Solver')
parser.add_argument('fin', metavar='input-file-path', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                    help='input file path (default: stdin)')
parser.add_argument('fout', metavar='output-file-path', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
                    help='output file path (default: stdout)')


def read_file_in(fin):
    with fin:
        type = fin.readline().rstrip()
        m = fin.readline().rstrip()
        n = fin.readline().rstrip()
        mat = fin.readline().rstrip()
    return type, m, n, mat


def write_file_out(fout, text):
    with fout:
        fout.write(text)


def create_lp_from_input(lines_a, columns_a, matrix):
    if not str.isdigit(lines_a):
        raise Exception('m (number of lines) is expected to be an integer')
    if not str.isdigit(columns_a):
        raise Exception('n (number of columns) is expected to be an integer')

    lines_a = int(lines_a) + 1
    columns_a = int(columns_a) + 1

    try:
        matrix = np.matrix(eval(matrix))
    except SyntaxError as err:
        raise Exception('incorrect format for given matrix') from err

    if matrix.shape[0] != lines_a or matrix.shape[1] != columns_a:
        raise Exception('given matrix (dim: {!s}) has different specified dimensions: {!s}'
                        .format(matrix.shape, (lines_a, columns_a)))

    lp = IntLinearProgramming()
    lp.set_lp(matrix[1:, :-1], matrix[1:, -1], matrix[0, :-1])

    return lp


if __name__ == '__main__':
    args = parser.parse_args()

    type, m, n, mat = read_file_in(args.fin)

    lp = create_lp_from_input(m, n, mat)
    if type == '0':
        ss = CuttingPlaneSolver(lp)
    elif type == '1':
        ss = BranchAndBoundSolver(lp)
    else:
        raise Exception('unkown option for integer solver')
    ss.run_solver(args.fout)

    if ss.lp_type == 'infeasible':
        result = 'PI inviável'
    elif ss.lp_type == 'unbounded':
        result = 'PI ilimitada, aqui está um certificado '
        result += printing.matrix_pretty(ss.certificate.flat)
    elif ss.lp_type == 'bounded':
        result = ' Solução ótima xi = {}, com valor objetivo {}. A solução ' \
                 'ótima da relaxação linear é x = {}, com valor objetivo {} e ' \
                 'certificado y = {}'.format(printing.matrix_pretty(ss.int_orig_problem_solution.flat),
                                             printing.number_pretty(ss.int_objvalue),
                                             printing.matrix_pretty(ss.orig_problem_solution.flat),
                                             printing.number_pretty(ss.objvalue),
                                             printing.matrix_pretty(ss.certificate.flat))
    else:
        raise Exception('Unknown lp type returned')

    write_file_out(args.fout, result)
