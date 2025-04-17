import sympy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
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

def find_second_derivatives(points: list[Point], g0: float, gn: float, bc_type: str = "clamped") -> list[float]:
    n = len(points)
    X = [point[0] for point in points]
    Y = [point[1] for point in points]
    H = [X[i] - X[i - 1] for i in range(1, n)] # X_{i} - X_{i - 1}
    L = [Y[i] - Y[i - 1] for i in range(1, n)] # Y_{i} - Y_{i - 1}
    
        # Build the n+1 by n+1 matrix A
    A = np.zeros((n, n))
    b = np.zeros(n)

    # Interior rows
    for i in range(1, n-1):
        A[i, i-1] = H[i-1]
        A[i, i  ] = 2*(H[i-1] + H[i])
        A[i, i+1] = H[i]
        b[i] = L[i]/H[i] - L[i-1]/H[i-1]

    if bc_type == 'clamped':
        # first row  (p'(x0) = g0)
        A[0,0] = 2*H[0]
        A[0,1] = H[0]
        b[0]   = (L[0]/H[0] - g0)
        # last row   (p'(xn) = gn)
        A[-1,-2] = H[-1]
        A[-1,-1] = 2*H[-1]
        b[-1]    = (gn - L[-1]/H[-1])
    elif bc_type == 'natural':
        # natural: force s0 = 0, sN = 0
        A[0,0]   = 1.0
        A[0,1]   = 0.0
        b[0]     = 0.0

        A[-1,-1] = 1.0
        A[-1,-2] = 0.0
        b[-1]    = 0.0
    else:
        raise ValueError("bc_type must be 'clamped' or 'natural'")
    
    b *= 6 

    # Solve
    s = np.linalg.solve(A, b)
    return s


def get_cubic_spline(points: list[Point], g0: float, gn: float, bc_type: str = "clamped") -> sp.Piecewise:
    n = len(points)
    X = [point[0] for point in points]
    Y = [point[1] for point in points]
    H = [X[i] - X[i - 1] for i in range(1, n)] # X_{i} - X_{i - 1}
    S = find_second_derivatives(points, g0, gn, bc_type)
    polynomials: list[sp.Expr] = [0 for i in range(n)]

    for i in range(1, n):
        polynomials[i] = ( S[i - 1] * (X[i] - x)**3 ) / (6 * H[i - 1])
        polynomials[i] += ( S[i] * (x - X[i - 1])**3 ) / (6 * H[i - 1])
        polynomials[i] += ( ( Y[i - 1] / H[i - 1]) - ( S[i - 1] * H[i - 1] / 6 ) ) * (X[i] - x)
        polynomials[i] += ( ( Y[i] / H[i - 1]) - ( S[i] * H[i - 1] / 6 ) ) * (x - X[i - 1])
        polynomials[i] = sp.simplify(polynomials[i])

    pieces = []
    for i in range(1, n):
        a, b = X[i-1], X[i]
        pieces.append((
            polynomials[i],
            sp.And(x >= a, x <= b)
        ))
    piecewise = sp.Piecewise(*pieces)
    
    piecewise = sp.Piecewise(*pieces)
    return piecewise


if __name__ == "__main__":
    points: list[Point] = [
        (-4, -4), 
        (-2, -1),
        (1, 2),
        (2, 0)
    ]
    my_clamped = get_cubic_spline(points, g0 = 1, gn = 1, bc_type="clamped")
    my_natural = get_cubic_spline(points, g0 = 0, gn = 0, bc_type="natural")
    cubic_spline1 = CubicSpline([-4, -2, 1, 2], [-4, -1, 2, 0], bc_type=((1, 1.0), (1, 1.0)))
    cubic_spline2 = CubicSpline([-4, -2, 1, 2], [-4, -1, 2, 0], bc_type="natural")

    # Plot
    X_special = [-4, -2, 1, 2]
    Y_special = [-4, -1, 2, 0]
    X_plot = [i * 0.1 for i in range(-100, 100)]
    Y_func1 = [my_clamped.subs(x, i).evalf() for i in X_plot]
    Y_func2 = [my_natural.subs(x, i).evalf() for i in X_plot]

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cubic Spline')
    plt.xlim(-7, 7)
    plt.ylim(-5, 5)
    plt.plot(X_plot, Y_func1, color='b', label='My clamped with g0 = 1, gn = 1')
    plt.plot(X_plot, Y_func2, color='g', label='My natural with s0 = 0, sn = 0')
    plt.plot(X_plot, cubic_spline1(X_plot),  label="Scikit Clamped spline")
    plt.plot(X_plot, cubic_spline2(X_plot),  label="Scikit Natural spline", linestyle='--')
    plt.scatter(X_special, Y_special, s=50, marker='o', color='r', label='Given points')
    plt.legend()
    plt.show()
