import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


u, v = sp.symbols('u v', real=True)

# Triangle A, B, C
Ax, Ay, Az = 0, -3, 6
Bx, By, Bz = -12, -3, -3
Cx, Cy, Cz = -9, 5, -6

# Boundary curves
# r0(u) : straight line from A to B
x0 = Ax*(1 - u) + Bx*u
y0 = Ay*(1 - u) + By*u
z0 = Az*(1 - u) + Bz*u

# r1(u) : just at point C
x1 = Cx
y1 = Cy
z1 = Cz

# Ruled-surface formula for v0=0, v1=1:
X_sym = x0*(1 - v) + x1*v
Y_sym = y0*(1 - v) + y1*v
Z_sym = z0*(1 - v) + z1*v

# Lambdify for numeric plotting
X = sp.lambdify((u, v), X_sym, 'numpy')
Y = sp.lambdify((u, v), Y_sym, 'numpy')
Z = sp.lambdify((u, v), Z_sym, 'numpy')


# Create parameter grid on [0,1]×[0,1]
uu = np.linspace(0, 1, 50)
vv = np.linspace(0, 1, 50)
UU, VV = np.meshgrid(uu, vv, indexing='ij')

# Evaluate surface points
XX = X(UU, VV)
YY = Y(UU, VV)
ZZ = Z(UU, VV)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot 
ax.plot_surface(
    XX, YY, ZZ,
    rstride=1, cstride=1,
    cmap='viridis',
    alpha=0.7,
    edgecolor='none'
)

# Overlay triangle edges
u_line = np.linspace(0, 1, 100)
v_line = np.linspace(0, 1, 100)

# Edge AB: v=0
ax.plot(X(u_line, 0), Y(u_line, 0), Z(u_line, 0), 'k-', lw=2)
# Edge AC: u=0
ax.plot(X(0, v_line), Y(0, v_line), Z(0, v_line), 'k-', lw=2)
# Edge BC: u=1
ax.plot(X(1, v_line), Y(1, v_line), Z(1, v_line), 'k-', lw=2)

# Scatter the three vertices
ax.scatter([Ax, Bx, Cx], [Ay, By, Cy], [Az, Bz, Cz],
           color='red', s=50, label='Vertices A, B, C')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Triangle Surface')
ax.legend()

plt.tight_layout()
plt.show()
