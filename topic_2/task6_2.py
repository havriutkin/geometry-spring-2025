import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

t = sp.symbols('t', real=True)
a, w = sp.symbols('a w', real=True)

P_expr  = (w + sp.Abs(t - a) - sp.Abs(t - a - w)) / (2*w)
Ql_expr = (t - a - sp.Abs(t - a)) / 2
Q_expr  = (t - a + sp.Abs(t - a)) / 2

verts = [
    (-4,  2,  6),  # A1
    ( 2, -3,  0),  # A2
    (10,  5,  8),  # A3
    (-4,  2,  6),  # A1
    ( 5,  2, -4),  # A4
    (10,  5,  8),  # A3
    ( 2, -3,  0),  # A2
    ( 5,  2, -4),  # A4
]
t_vals = list(range(len(verts)))  # [0..7]

r0x, r0y, r0z = verts[0]
v_lx = verts[1][0] - r0x
v_ly = verts[1][1] - r0y
v_lz = verts[1][2] - r0z
v_rx = verts[-1][0] - verts[-2][0]
v_ry = verts[-1][1] - verts[-2][1]
v_rz = verts[-1][2] - verts[-2][2]

Rx = r0x + v_lx*Ql_expr.subs(a, t_vals[0])
Ry = r0y + v_ly*Ql_expr.subs(a, t_vals[0])
Rz = r0z + v_lz*Ql_expr.subs(a, t_vals[0])

for (x0, y0, z0), (x1, y1, z1), t0, t1 in zip(verts, verts[1:], t_vals, t_vals[1:]):
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    seg = P_expr.subs({a: t0, w: t1 - t0})
    Rx += dx * seg
    Ry += dy * seg
    Rz += dz * seg

Rx += v_rx*Q_expr.subs(a, t_vals[-1])
Ry += v_ry*Q_expr.subs(a, t_vals[-1])
Rz += v_rz*Q_expr.subs(a, t_vals[-1])

Rx = sp.simplify(Rx)
Ry = sp.simplify(Ry)
Rz = sp.simplify(Rz)

f_x = sp.lambdify(t, Rx, 'numpy')
f_y = sp.lambdify(t, Ry, 'numpy')
f_z = sp.lambdify(t, Rz, 'numpy')

tt = np.linspace(0, 7, 500)
xx = f_x(tt); yy = f_y(tt); zz = f_z(tt)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xx, yy, zz, 'b-', linewidth=2)
vx, vy, vz = zip(*verts[:-1])
ax.scatter(vx, vy, vz, color='red', s=50)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
plt.show()
