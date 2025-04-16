import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import math

class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

Point = tuple[float, float]

x = sp.symbols('x')

FUNCTION: sp.Expr = x**2 / (sp.sqrt(4 + x**2))

def factorial(k: int) -> int:
    if k < 0:
        raise TypeError('Can find factorial of only positive numbers.')

    if k == 0:
        return 1

    return k * factorial(k - 1)

def find_derivatives(func: sp.Expr, degree: int) -> list[sp.Expr]:
    """ Returns a list of derivatives up to given degree """
    result: list[sp.Expr] = [func]
    for k in range(1, degree + 1):
        result.append(result[k - 1].diff(x))

    return result

def get_taylor(func: sp.Expr, x0: float, degree: int) -> sp.Poly:
    """ Builds a Taylor series of degree n at the point x0"""
    derivatives: list[sp.Expr] = find_derivatives(func, degree)
    coeffs = [ derivatives[k].subs(x, x0).evalf() / factorial(k) for k in range(degree + 1) ]
    result: sp.Expr = 0
    for k in range(degree + 1):
        result += coeffs[k] * (x - x0)**k

    result = sp.simplify(result)
    return sp.poly(result)


if __name__ == "__main__":
    polynomial: sp.Poly = get_taylor(FUNCTION, x0=0, degree=8)  

    print(f"{colors.HEADER}RESULTS{colors.ENDC}")
    print(f"\t{colors.OKBLUE}Taylor polynomial:{colors.ENDC} {sp.pretty(polynomial)}")

    # Plot
    X_plot = [i * 0.1 for i in range(-500, 500)]
    Y_poly = [polynomial.subs(x, i).evalf() for i in X_plot]
    Y_func = [FUNCTION.subs(x, i).evalf() for i in X_plot]

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Newtonian Approximation and Original Function')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.plot(X_plot, Y_poly, color='g', label='Newton approximation')
    plt.plot(X_plot, Y_func, color='b', label='Original function')
    plt.scatter([0], [FUNCTION.subs(x, 0).evalf()], s=80, marker='o', color='r', label='Local point')
    plt.legend()
    plt.show()
