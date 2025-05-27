import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# 1) SYMBOLIC SETUP
x, y = sp.symbols('x y', real=True)

# 2) Half-space identifiers
omega1 = y                    # y >= 0
omega2 = 2 - x - y            # 2 - x - y >= 0
omega3 = 2 + x - y            # 2 + x - y >= 0

# 3) Intersection executor ir(u,v) = (u+v - |u-v|)/2
ir = lambda u, v: (u + v - sp.Abs(u - v)) / 2

# 4) Build identifier for intersection of three half-spaces
omega12 = ir(omega1, omega2)
omega = ir(omega12, omega3)
omega_simp = sp.simplify(omega)

# 5) Print the implicit identifier and contour equation
print("Implicit identifier ω(x,y) for the region:")
sp.pprint(omega_simp)
print("\nContour is given by ω(x,y) = 0")

# 6) Lambdify for numerical evaluation
f = sp.lambdify((x, y), omega_simp, 'numpy')

# 7) Create grid and evaluate
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

# 8) Plot filled region (ω>0) and contour (ω=0)
plt.figure(figsize=(6,6))
plt.contourf(X, Y, Z, levels=[0, Z.max()], colors=['lightgray'])
contours = plt.contour(X, Y, Z, levels=[0], colors='black', linewidths=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Region and contour: y≥0, 2−x−y≥0, 2+x−y≥0')
plt.gca().set_aspect('equal')
plt.show()
