import sympy as sp
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from sympy.plotting import plot_implicit

# Symbols and executors
x, y = sp.symbols('x y', real=True)
ir = lambda U, V: (U + V - sp.Abs(U - V)) / 2
ur = lambda U, V: (U + V + sp.Abs(U - V)) / 2

# Define three quadrilaterals
patches = [
    [(-2, 0), (-1, 2), (-3, 6), (-5, 6)],
    [(-2, 0), (-1, 2), ( 1, 2), ( 2, 0)],
    [ (2, 0), ( 1, 2), ( 3, 6), ( 5, 6)]
]

def quad_region(quad):
    # Centroid
    cx = sum(p[0] for p in quad) / 4
    cy = sum(p[1] for p in quad) / 4
    halfspaces = []
    for P, Q in zip(quad, quad[1:]+quad[:1]):
        x1, y1 = P
        x2, y2 = Q
        A = y2 - y1
        B = -(x2 - x1)
        C = -(A*x1 + B*y1)
        L = A*x + B*y + C
        if L.subs({x:cx, y:cy}) < 0:
            L = -L
        halfspaces.append(L)
    return reduce(ir, halfspaces)

# Build implicit for each quad and union
f1 = quad_region(patches[0])
f2 = quad_region(patches[1])
f3 = quad_region(patches[2])
omega_V = ur(ur(f1, f2), f3)
omega_V = sp.simplify(omega_V)

f = sp.lambdify((x, y), omega_V, 'numpy')

# Create grid of points
x_vals = np.linspace(-6, 6, 500)
y_vals = np.linspace(0, 7, 500)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

# Plot the region where omega_V > 0 
plt.figure(figsize=(6, 6))
plt.contourf(X, Y, Z, levels=[0, Z.max()], colors=['#aee9f5'])
plt.contour(X, Y, Z, levels=[0], colors='black')
plt.title("Shaded region where omega_V > 0")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.show()

