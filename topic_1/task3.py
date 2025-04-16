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

FUNCTION: sp.Expr = 4*x - x**2 - 2*sp.cos(x - 2)

def get_divided_differences(points: list[Point]):
    X = [point[0] for point in points]
    Y = [point[1] for point in points]

    size = len(X)
    D = np.zeros((size, size))
    D[:, 0] = Y
    for j in range(1, size): 
        for k in range(j, size): 
            D[k, j] = (D[k, j - 1] - D[k - 1, j - 1]) / (X[k] - X[k - j]) 
    C = np.diag(D)

    return C, D

def get_newton(points: list[Point]) -> sp.Poly:
    """ Builds a Newtonian polynomial in one variable given list of points"""
    X = [point[0] for point in points]
    C, D = get_divided_differences(points)

    result: sp.Expr = C[0]   # Free coefficient 
    for i in range(1, len(C)):
        factor = 1
        for j in range(i):
            factor *= (x - X[j])
        result += (C[i] * factor)

    result = sp.simplify(result)
    return sp.poly(result)


if __name__ == "__main__":
    MIN = -10
    MAX = 10
    STEP = math.floor((MAX - MIN) / 6)

    X = [i for i in range(MIN, MAX, STEP)]
    Y = [FUNCTION.subs(x, i).evalf() for i in X]

    points: list[Point] = [(X[i], Y[i]) for i in range(len(X))]
    polynomial: sp.Poly = get_newton(points)  

    print(f"{colors.HEADER}RESULTS{colors.ENDC}")
    print(f"\t{colors.OKBLUE}Input points:{colors.ENDC} {points}")
    print(f"\t{colors.OKBLUE}Newtonian polynomial:{colors.ENDC} {sp.pretty(polynomial)}")

    # Plot
    special_X = X
    special_Y = [polynomial.subs({x: i}).evalf() for i in special_X]
    X_plot = [i * 0.1 for i in range(-500, 500)]
    Y = [polynomial.subs({x: i}).evalf() for i in X_plot]
    Y_func = [FUNCTION.subs(x, i).evalf() for i in X_plot]

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Newtonian Approximation and Original Function')
    plt.xlim(-50, 50)
    plt.ylim(-200, 25)
    plt.plot(X_plot, Y, color='g', label='Newton approximation')
    plt.plot(X_plot, Y_func, color='b', label='Original function')
    plt.scatter(special_X, special_Y, s=80, marker='o', color='r', label='Interpolated points')
    plt.legend()
    plt.show()
