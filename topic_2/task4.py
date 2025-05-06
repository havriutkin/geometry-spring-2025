import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# 1) Input points and boundary derivatives
points = [(-4, -4), (-2, -1), (1, 2), (2, 0)]
g0, gn = 1, 1  # clamped: first derivative at ends = 1

# Helpers: second derivatives solver and piecewise spline builder
def find_second_derivatives(pts, g0=0, gn=0, bc_type="clamped"):
    n = len(pts)
    X = [p[0] for p in pts]
    Y = [p[1] for p in pts]
    H = [X[i] - X[i-1] for i in range(1, n)]
    L = [Y[i] - Y[i-1] for i in range(1, n)]

    A = np.zeros((n, n))
    b = np.zeros(n)

    # interior
    for i in range(1, n-1):
        A[i, i-1] = H[i-1]
        A[i, i]   = 2*(H[i-1] + H[i])
        A[i, i+1] = H[i]
        b[i] = (L[i]/H[i] - L[i-1]/H[i-1])

    if bc_type == 'clamped':
        A[0,0], A[0,1] = 2*H[0], H[0]
        b[0]           = (L[0]/H[0] - g0)
        A[-1,-2], A[-1,-1] = H[-1], 2*H[-1]
        b[-1]              = (gn - L[-1]/H[-1])
    elif bc_type == 'natural':
        A[0,0], b[0]     = 1, 0
        A[-1,-1], b[-1]  = 1, 0
    else:
        raise ValueError("bc_type must be 'clamped' or 'natural'")

    b *= 6
    return np.linalg.solve(A, b)

def get_cubic_spline(pts, g0=0, gn=0, bc_type="clamped"):
    x = sp.symbols('x')
    n = len(pts)
    X = [p[0] for p in pts]
    Y = [p[1] for p in pts]
    H = [X[i] - X[i-1] for i in range(1, n)]
    S = find_second_derivatives(pts, g0, gn, bc_type)

    pieces = []
    for i in range(1, n):
        x0, x1 = X[i-1], X[i]
        h = H[i-1]
        s0, s1 = S[i-1], S[i]
        y0, y1 = Y[i-1], Y[i]

        poly = (s0*(x1 - x)**3)/(6*h) + (s1*(x - x0)**3)/(6*h) \
             + ((y0/h) - s0*h/6)*(x1 - x) \
             + ((y1/h) - s1*h/6)*(x - x0)
        pieces.append((sp.simplify(poly), sp.And(x >= x0, x <= x1)))

    return sp.Piecewise(*pieces)

# Build both clamped and natural splines
pw_clamped = get_cubic_spline(points, g0, gn, "clamped")
pw_natural = get_cubic_spline(points, 0, 0, "natural")
s_clamped  = find_second_derivatives(points, g0, gn, "clamped")
s_natural  = find_second_derivatives(points, 0, 0, "natural")

# Extract p1 and pn for each
x = sp.symbols('x')
p1_c = pw_clamped.args[0][0]
pn_c = pw_clamped.args[-1][0]
p1_n = pw_natural.args[0][0]
pn_n = pw_natural.args[-1][0]

X_vals = [p[0] for p in points]
# Formula (6)
def build_closed_formula(p1, pn, s):
    expr = (p1 + pn)/2
    n = len(X_vals) - 1
    for k in range(1, n):
        term = ((s[k+1] - s[k])/(X_vals[k+1] - X_vals[k])
              - (s[k]   - s[k-1])/(X_vals[k]   - X_vals[k-1]))
        expr += term * sp.Abs(x - X_vals[k])**3 / 12
    return sp.simplify(expr)

expr_clamped = build_closed_formula(p1_c, pn_c, s_clamped)
expr_natural = build_closed_formula(p1_n, pn_n, s_natural)

print("Clamped spline formula S_c(x) =", expr_clamped)
print("Natural spline formula S_n(x) =", expr_natural)

# Plot both formulas vs piecewise
f_sc = sp.lambdify(x, expr_clamped, 'numpy')
f_sn = sp.lambdify(x, expr_natural, 'numpy')
f_pc = sp.lambdify(x, pw_clamped,   'numpy')
f_pn = sp.lambdify(x, pw_natural,   'numpy')

xx = np.linspace(min(X_vals)-1, max(X_vals)+1, 500)
plt.figure()
plt.plot(xx, f_sc(xx), 'b-',   linewidth=2, label='Clamped formula')
plt.plot(xx, f_pc(xx), 'b--',  linewidth=1, label='Clamped piecewise')
plt.plot(xx, f_sn(xx), 'g-',   linewidth=2, label='Natural formula')
plt.plot(xx, f_pn(xx), 'g--',  linewidth=1, label='Natural piecewise')
plt.scatter(X_vals, [p[1] for p in points], color='red', zorder=10)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cubic Spline (defect=1) — Clamped vs Natural')
plt.grid(True)
plt.show()
