import numpy as np
import sympy as sp
from functools import reduce
import matplotlib.pyplot as plt

# Symbolic setup for V-shape and L-shape in x-y
x, y, z = sp.symbols('x y z', real=True)
ir = lambda U, V: (U + V - sp.Abs(U - V)) / 2  # intersection of regions
ur = lambda U, V: (U + V + sp.Abs(U - V)) / 2  # union of regions

# Helper for quadrilateral region (intersection of halfspaces)
def quad_region(quad):
    cx = sum(pt[0] for pt in quad) / len(quad)
    cy = sum(pt[1] for pt in quad) / len(quad)
    regs = []
    for P, Q in zip(quad, quad[1:]+quad[:1]):
        x1, y1 = P; x2, y2 = Q
        A = y2 - y1
        B = -(x2 - x1)
        C = -(A*x1 + B*y1)
        L = A*x + B*y + C
        if L.subs({x:cx, y:cy}) < 0:
            L = -L
        regs.append(L)
    return reduce(ir, regs)

# Build digit "4" using three rectangle segments
segments4 = [
    [(-6, 3), (-6, 6), (-5, 6), (-5, 3)],  # left vertical bar
    [(-3, 0), (-3, 6), (-2, 6), (-2, 0)],  # right vertical bar
    [(-6, 3), (-2, 3), (-2, 4), (-6, 4)],  # middle horizontal bar
]
f4_patches = [quad_region(q) for q in segments4]
gamma_4 = reduce(ur, f4_patches)

# Build digit "2" using five segments (seven-segment style)
segments2 = [
    [( 2, 5), ( 6, 5), ( 6, 6), ( 2, 6)],  # top horizontal bar
    [( 5, 4), ( 6, 4), ( 6, 6), ( 5, 6)],  # upper-right vertical bar
    [( 2, 3), ( 6, 3), ( 6, 4), ( 2, 4)],  # middle horizontal bar
    [( 2, 0), ( 3, 0), ( 3, 3), ( 2, 3)],  # lower-left vertical bar
    [( 2, 0), ( 6, 0), ( 6, 1), ( 2, 1)],  # bottom horizontal bar
]
f2_patches = [quad_region(q) for q in segments2]
gamma_2 = reduce(ur, f2_patches)

# Union of "4" and "2"
gamma_shape = ur(gamma_4, gamma_2)
final_shape = gamma_shape

# Output the implicit equation (contour, i.e., final_shape(x, y) = 0)
print("Implicit equation for the contour of the region in the shape of the number 42:")

f = sp.lambdify((x, y), final_shape, 'numpy')

# Create a grid of points
x_vals = np.linspace(-10, 10, 200)
y_vals = np.linspace(-1, 7, 200)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, levels=[0, np.max(Z)], colors=['lightblue'])
plt.contour(X, Y, Z, levels=[0], colors='black')
plt.title("Answer to the question of the meaning of life")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.show()
