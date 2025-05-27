import sympy as sp

def bernstein_polyline(points, t_vals, t:sp.Symbol):
    """
    Given control points [(x0,y0),…,(xn,yn)] and parameter values t_vals=[t0,…,tn],
    returns symbolic parametric functions Rx(t), Ry(t) for the piecewise linear polyline
    using the Bernstein absolute-value formula.
    """

    a, w = sp.symbols('a w')

    P  = (w + sp.Abs(t - a) - sp.Abs(t - a - w)) / (2*w)
    Ql = (t - a - sp.Abs(t - a)) / 2
    Q  = (t - a + sp.Abs(t - a)) / 2

    n = len(points) - 1  
    Rx = points[0][0] + (points[1][0] - points[0][0]) * Ql.subs(a, t_vals[0])
    Ry = points[0][1] + (points[1][1] - points[0][1]) * Ql.subs(a, t_vals[0])

    # Sum over each interior segment
    for i in range(n):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        wi = t_vals[i+1] - t_vals[i]
        Rx += dx * P.subs({a: t_vals[i], w: wi})
        Ry += dy * P.subs({a: t_vals[i], w: wi})
    dx_last = points[-1][0] - points[-2][0]
    dy_last = points[-1][1] - points[-2][1]
    Rx += dx_last * Q.subs(a, t_vals[-1])
    Ry += dy_last * Q.subs(a, t_vals[-1])

    # Simplify and return symbolic expressions
    return sp.simplify(Rx), sp.simplify(Ry)


if __name__ == '__main__':
    points = [
        (0, 0),
        (5, 0),
        (5, 2),
        (2, 2),
        (2, 5),
        (5, 5),
        (5, 7),
        (0, 7),
    ]

    t_vals = [0, 1, 2, 3, 4, 5, 6, 7]

    u = sp.symbols('u')
    v = sp.symbols('v')

    phi, psi = bernstein_polyline(points, t_vals, u)

    X = sp.lambdify((u, v), sp.simplify(phi), 'numpy')
    Y = sp.lambdify((u, v), sp.simplify(v), 'numpy')
    Z = sp.lambdify(u, psi, 'numpy')

    import numpy as np
    import matplotlib.pyplot as plt

    # Create a grid in (u,v)
    uu_vals = np.linspace(t_vals[0], t_vals[-1], 200)
    vv_vals = np.linspace(0, 10,          200)

    # meshgrid (with 'ij' so that UU varies over uu_vals rows, VV over vv_vals columns)
    UU, VV = np.meshgrid(uu_vals, vv_vals, indexing='ij')

    XX = X(UU, VV)
    YY = Y(UU, VV)
    Z1d = Z(uu_vals)                               # shape (200,)
    ZZ  = np.tile(Z1d[:,None], (1, vv_vals.size))  # Repeat Z across v-axis, shape (200, 200)

    # Plot
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        XX, YY, ZZ,
        rstride=5, cstride=5,
        cmap='viridis',
        edgecolor='k',
        alpha=0.8,
    )
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Surface of Translation')
    ax.legend()
    plt.tight_layout()
    plt.show()



