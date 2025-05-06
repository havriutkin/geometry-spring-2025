import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

# You already have this alias
Point = tuple[float, float]
t = sp.symbols('t')

def find_second_derivatives(
    points: list[Point],
    g0: float = 0,
    gn: float = 0,
    bc_type: str = "clamped"
) -> np.ndarray:
    n = len(points)
    X = [pt[0] for pt in points]
    Y = [pt[1] for pt in points]
    H = [X[i] - X[i-1] for i in range(1, n)]
    L = [Y[i] - Y[i-1] for i in range(1, n)]

    A = np.zeros((n, n))
    b = np.zeros(n)

    # interior rows (same for all BC)
    for i in range(1, n-1):
        A[i, i-1] = H[i-1]
        A[i, i]   = 2*(H[i-1] + H[i])
        A[i, i+1] = H[i]
        b[i] = (L[i]/H[i] - L[i-1]/H[i-1])

    if bc_type == 'clamped':
        # p'(x0)=g0, p'(xn)=gn
        A[0,0]   = 2*H[0]
        A[0,1]   = H[0]
        b[0]     = (L[0]/H[0] - g0)

        A[-1,-2] = H[-1]
        A[-1,-1] = 2*H[-1]
        b[-1]    = (gn - L[-1]/H[-1])

    elif bc_type == 'natural':
        # s0 = sn = 0
        A[0,0]   = 1
        b[0]     = 0
        A[-1,-1] = 1
        b[-1]    = 0

    elif bc_type == 'periodic':
        # enforce s0 = s_{n-1}
        A[0,0]  = 1
        A[0,-1] = -1
        b[0]    = 0

        # p'(t0)=p'(tn) gives this last row:
        A[-1,0]    = -2*H[0]
        A[-1,1]    = -H[0]
        A[-1,-2]   = -H[-1]
        A[-1,-1]   = -2*H[-1]
        b[-1]      = (L[-1]/H[-1] - L[0]/H[0])

    else:
        raise ValueError("bc_type must be 'clamped', 'natural', or 'periodic'")

    # scale and solve
    b *= 6
    return np.linalg.solve(A, b)


def get_cubic_spline(
    points: list[Point],
    g0: float = 0,
    gn: float = 0,
    bc_type: str = "clamped"
) -> sp.Piecewise:
    n = len(points)
    X = [pt[0] for pt in points]
    Y = [pt[1] for pt in points]
    H = [X[i] - X[i-1] for i in range(1, n)]
    S = find_second_derivatives(points, g0, gn, bc_type)

    pieces = []
    for i in range(1, n):
        x0, x1 = X[i-1], X[i]
        h = H[i-1]
        s0, s1 = S[i-1], S[i]
        y0, y1 = Y[i-1], Y[i]

        poly = (
            s0*(x1 - t)**3/(6*h)
            + s1*(t - x0)**3/(6*h)
            + ((y0/h) - s0*h/6)*(x1 - t)
            + ((y1/h) - s1*h/6)*(t - x0)
        )
        pieces.append((sp.simplify(poly), sp.And(t >= x0, t <= x1)))

    return sp.Piecewise(*pieces)


if __name__ == "__main__":
    quad = [(-4, 4), (2, 4), (3, -4), (-3, -2)]

    # parameter values 0,1,2,3,4 and close back to 0
    t_vals   = list(range(len(quad) + 1))
    extended = quad + [quad[0]]

    # build two point-lists (t, x) and (t, y)
    x_pts = [(t_vals[i], extended[i][0]) for i in range(len(extended))]
    y_pts = [(t_vals[i], extended[i][1]) for i in range(len(extended))]

    # symbolic piecewise splines with periodic BC
    px = get_cubic_spline(x_pts, bc_type="periodic")
    py = get_cubic_spline(y_pts, bc_type="periodic")

    # numeric evaluation
    import numpy as _np
    t_plot = _np.linspace(0, len(quad), 400)
    x_plot = [float(px.subs(t, ti).evalf()) for ti in t_plot]
    y_plot = [float(py.subs(t, ti).evalf()) for ti in t_plot]

    # plot original polygon
    poly_x = [p[0] for p in quad] + [quad[0][0]]
    poly_y = [p[1] for p in quad] + [quad[0][1]]

    plt.figure()
    plt.plot(poly_x, poly_y, 'ro-', label='Quadrilateral')
    plt.plot(x_plot, y_plot, 'b-',  label='Periodic Cubic Spline')
    plt.axis('equal')
    plt.legend()
    plt.xlabel('x'); plt.ylabel('y')
    plt.title('Closed Parametric Cubic Spline')
    plt.show()
