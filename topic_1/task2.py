import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

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
    points: list[Point] = [
        (0.284, -3.856), (0.883, -3.953), (1.384, -5.112), (1.856, -7.632), (2.644, -8.011)
    ]

    polynomial: sp.Poly = get_newton(points)  

    print(f"{colors.HEADER}RESULTS{colors.ENDC}")
    print(f"\t{colors.OKBLUE}Input points:{colors.ENDC} {points}")
    print(f"\t{colors.OKBLUE}Newtonian polynomial:{colors.ENDC} {sp.pretty(polynomial)}")

    # Plot
    special_X = [point[0] for point in points]
    special_Y = [polynomial.subs({x: i}).evalf() for i in special_X]
    X = [i * 0.1 for i in range(-20, 50)]
    Y = [polynomial.subs({x: i}).evalf() for i in X]

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Newtonian Approximation')
    plt.xlim(-2, 5)
    plt.ylim(-20, 100)
    plt.scatter(special_X, special_Y, s=80, marker='o', color='r')
    plt.plot(X, Y)
    plt.show()
