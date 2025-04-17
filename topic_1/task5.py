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

# x coordinate, y coordinate, derivative at this point
ErmitPoint = tuple[float, float, float]

x = sp.symbols('x')


def get_ermit(p1: ErmitPoint, p2: ErmitPoint) -> sp.Poly:
    """ Builds a polynomial that connects two ErmitPoints"""
    x1,y1,g1=p1        
    x2,y2,g2=p2  
    h=x2-x1 

    result = y1 * (h + 2 * (x - x1)) * (x - x2) ** 2 / h ** 3 
    result += g1 * (x - x1) * (x - x2) ** 2 / h ** 2 
    result += g2 * (x - x2) * (x - x1) ** 2 / h ** 2 
    result += y2 * (h - 2 * (x - x2)) * (x - x1) ** 2 / h ** 3
    result = sp.simplify(result)
    result = sp.poly(result)

    return result


if __name__ == "__main__":
    points: list[ErmitPoint] = [
        (-4, -4, 1), 
        (-2, -1, 1),
        (1, 2, 1),
        (2, 0, 1)
    ]

    poly1: sp.Poly = get_ermit(points[0], points[1])
    poly2: sp.Poly = get_ermit(points[1], points[2])
    poly3: sp.Poly = get_ermit(points[2], points[3])  
    piecewise = sp.Piecewise(
        (-4, x <= -4),
        (poly1, x < -2),
        (poly2, x < 1),
        (poly3, x < 2),
        (0, True)
    )


    print(f"{colors.HEADER}RESULTS{colors.ENDC}")
    print(f"\t{colors.OKBLUE}Fist polynomial:{colors.ENDC} {sp.pretty(poly1)}")
    print(f"\t{colors.OKBLUE}Second polynomial:{colors.ENDC} {sp.pretty(poly2)}")
    print(f"\t{colors.OKBLUE}Third polynomial:{colors.ENDC} {sp.pretty(poly3)}")

    # Plot
    X_special = [-4, -2, 1, 2]
    Y_special = [-4, -1, 2, 0]
    X_plot = [i * 0.1 for i in range(-100, 100)]
    Y_func = [piecewise.subs(x, i).evalf() for i in X_plot]

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Ermit Interpolation')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.plot(X_plot, Y_func, color='b', label='Interpolated function')
    plt.scatter(X_special, Y_special, s=80, marker='o', color='r', label='Given points')
    plt.legend()
    plt.show()
