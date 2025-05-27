import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1) SYMBOLIC SETUP
u, v, t = sp.symbols('u v t', real=True)

# Given A, B, C
A = (1, -2,  3)
B = (2, -1,  2)
C = (3, -4,  5)

# Compute D opposite A: D = B + C - A
D = (B[0] + C[0] - A[0],
     B[1] + C[1] - A[1],
     B[2] + C[2] - A[2])

# Parametric parallelogram region via ruled-surface formula
# r0(u): segment A->B
x0 = A[0]*(1-u) + B[0]*u
y0 = A[1]*(1-u) + B[1]*u
z0 = A[2]*(1-u) + B[2]*u

# r1(u): segment C->D
x1 = C[0]*(1-u) + D[0]*u
y1 = C[1]*(1-u) + D[1]*u
z1 = C[2]*(1-u) + D[2]*u

# Surface r(u,v) = r0(u)*(1-v) + r1(u)*v
X_sym = x0*(1-v) + x1*v
Y_sym = y0*(1-v) + y1*v
Z_sym = z0*(1-v) + z1*v

# Lambdify for numeric evaluation
X = sp.lambdify((u, v), X_sym, 'numpy')
Y = sp.lambdify((u, v), Y_sym, 'numpy')
Z = sp.lambdify((u, v), Z_sym, 'numpy')

# Parametric contour using Bernstein polyline
def bernstein_polyline_3d(points, t_vals, t_sym):
    a, w = sp.symbols('a w', real=True)
    P  = (w + sp.Abs(t_sym - a) - sp.Abs(t_sym - a - w)) / (2*w)
    Ql = (t_sym - a - sp.Abs(t_sym - a)) / 2
    Q  = (t_sym - a + sp.Abs(t_sym - a)) / 2

    n = len(points) - 1
    Rx = points[0][0] + (points[1][0] - points[0][0]) * Ql.subs(a, t_vals[0])
    Ry = points[0][1] + (points[1][1] - points[0][1]) * Ql.subs(a, t_vals[0])
    Rz = points[0][2] + (points[1][2] - points[0][2]) * Ql.subs(a, t_vals[0])

    for i in range(n):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        dz = points[i+1][2] - points[i][2]
        wi = t_vals[i+1] - t_vals[i]
        Rx += dx * P.subs({a: t_vals[i], w: wi})
        Ry += dy * P.subs({a: t_vals[i], w: wi})
        Rz += dz * P.subs({a: t_vals[i], w: wi})

    dx_last = points[-1][0] - points[-2][0]
    dy_last = points[-1][1] - points[-2][1]
    dz_last = points[-1][2] - points[-2][2]
    Rx += dx_last * Q.subs(a, t_vals[-1])
    Ry += dy_last * Q.subs(a, t_vals[-1])
    Rz += dz_last * Q.subs(a, t_vals[-1])

    return sp.simplify(Rx), sp.simplify(Ry), sp.simplify(Rz)

# Define contour points: A->B->D->C->A
pts_contour = [A, B, D, C, A]
t_vals_cont = [0, 1, 2, 3, 4]

Rx_c, Ry_c, Rz_c = bernstein_polyline_3d(pts_contour, t_vals_cont, t)
fRx_c = sp.lambdify(t, Rx_c, 'numpy')
fRy_c = sp.lambdify(t, Ry_c, 'numpy')
fRz_c = sp.lambdify(t, Rz_c, 'numpy')


# Parallelogram surface
uu = np.linspace(0, 1, 50)
vv = np.linspace(0, 1, 50)
UU, VV = np.meshgrid(uu, vv, indexing='ij')
XX = X(UU, VV)
YY = Y(UU, VV)
ZZ = Z(UU, VV)

# Contour curve
tt = np.linspace(0, 4, 400)
XC = fRx_c(tt)
YC = fRy_c(tt)
ZC = fRz_c(tt)

# Plotting
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Filled parallelogram
ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1,
                color='orange', alpha=0.6, edgecolor='none')

# Contour edges
ax.plot(XC, YC, ZC, 'k-', lw=2, label='Parallelogram contour')

# Plot vertices
verts = np.array([A, B, D, C])
ax.scatter(verts[:,0], verts[:,1], verts[:,2],
           color='red', s=50, label='A, B, D, C')

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('Parallelogram region & contour')
ax.legend()
plt.tight_layout()
plt.show()
