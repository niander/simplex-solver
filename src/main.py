from simplex import SimplexSolver
from lp import LinearProgramming


def read_file_in():
    raise NotImplementedError()


def write_file_out():
    raise NotImplementedError()


if __name__ == '__main__':
    lp = read_file_in()
    ss = SimplexSolver(lp)
    ss.run_simplex(output_file)

    if ss.lp_type == "infeasible":
        result = "PL inviável, aqui está um certificado "
        result += str(ss.certificate.A1)
    elif ss.lp_type == "unbounded":
        result = "PL ilimitada, aqui está um certificado "
        result += str(ss.certificate.A1)
    elif ss.lp_type == "bounded":
        result = " Solução ótima x = {!s}, com valor objetivo {:.3f}, e certificado " \
                 "y = {!s}".format(ss.solution.A1, ss.objvalue, ss.certificate.A1)
    else:
        raise Exception("Unknown lp type returned")

    print(result)
