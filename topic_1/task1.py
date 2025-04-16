import sympy as sp
import matplotlib.pyplot as plt

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

def get_lagrange(points: list[Point]) -> sp.Poly:
    """ Builds a Lagrangian polynomial in one variable given list of points"""
    X = [point[0] for point in points]
    Y = [point[1] for point in points]
    degree = len(points)
    result: sp.Expr = 0

    for i in range(degree):
        term = Y[i]
        for j in range(degree):
            if i == j:
                continue
            term *= (x - X[j])
            term /= (X[i] - X[j])

        result += term
    result = sp.simplify(result)

    return sp.poly(result)

if __name__ == "__main__":
    points: list[Point] = [
        (-4, -4), (-2, -1), (1, 2), (2, 0)
    ]
    outsider = -0.5 # X coordinate of a point between points[1] and points[2]

    polynomial: sp.Poly = get_lagrange(points)

    print(f"{colors.HEADER}RESULTS{colors.ENDC}")
    print(f"\t{colors.OKBLUE}Input points:{colors.ENDC} {points}")
    print(f"\t{colors.OKBLUE}Lagrangian polynomial:{colors.ENDC} {sp.pretty(polynomial)}")

    # Plot
    special_X = [point[0] for point in points]
    special_X.append(outsider)
    special_Y = [polynomial.subs({x: i}).evalf() for i in special_X]
    X = [i * 0.1 for i in range(-60, 50)]
    Y = [polynomial.subs({x: i}).evalf() for i in X]

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Lagrangian Approximation')
    plt.xlim(-6, 6)
    plt.ylim(-20, 20)
    plt.scatter(special_X, special_Y, s=80, marker='o', color='r')
    plt.plot(X, Y)
    plt.show()
