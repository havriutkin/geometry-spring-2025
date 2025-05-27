import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


plt.close('all')

u, v = sp.symbols('u v', real=True)

y_down = (u + 1)**2
y_up   = u + 1

# Solve intersections
solutions = sp.solve(sp.simplify(y_down - y_up), u)
u_vals = sorted([sol.evalf() for sol in solutions if sol.is_real])
u0, u1 = float(u_vals[0]), float(u_vals[1])
print("Intersection u-values:", u_vals)

# x(u,v) and y(u,v)
x_expr = u
y_expr = y_down*(1 - v) + y_up*v

# z = x^2 - y^2
z_expr = x_expr**2 - y_expr**2

# Print symbolic formulas
print(f"x(u,v) = {sp.simplify(x_expr)}")
print(f"y(u,v) = {sp.simplify(y_expr)}")
print(f"z(u,v) = {sp.simplify(z_expr)}")

# Lambdify for numeric plotting
fX = sp.lambdify((u, v), x_expr, 'numpy')
fY = sp.lambdify((u, v), y_expr, 'numpy')
fZ = sp.lambdify((u, v), z_expr, 'numpy')

# Plot
uu = np.linspace(u0, u1, 120)
vv = np.linspace(0, 1,   60)
UU, VV = np.meshgrid(uu, vv, indexing='ij')

# Evaluate surface
XX = fX(UU, VV)
YY = fY(UU, VV)
ZZ = fZ(UU, VV)

fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection='3d')

# Plot the hyperboloid patch
ax.plot_surface(
    XX, YY, ZZ,
    rstride=4, cstride=4,
    cmap='viridis',
    edgecolor='none',
    alpha=0.8
)

# Overlay the boundary curves
# v=0 (lower)
yy0 = (uu + 1)**2
zz0 = uu**2 - yy0**2
ax.plot(uu, yy0, zz0, 'k-', lw=2, label='y = (x+1)^2')

# v=1 (upper)
yy1 = uu + 1
zz1 = uu**2 - yy1**2
ax.plot(uu, yy1, zz1, 'k-', lw=2, label='y = x+1')

ax.set_xlabel('X (u)')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Task 3.3.3: Hyperboloid over region between (x+1)^2 and x+1')
ax.legend()
plt.tight_layout()
plt.show()
