import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

t = sp.symbols('t')
a, w = sp.symbols('a w', real=True)
P = (w + sp.Abs(t - a) - sp.Abs(t - a - w)) / (2*w)

# original eight corners of the "C"
verts = [
    (0, 0),
    (5, 0),
    (5, 2),
    (2, 2),
    (2, 5),
    (5, 5),
    (5, 7),
    (0, 7),
]
# append the first again to close the loop
verts.append(verts[0])
# now there are 9 points ⇒ 8 segments
t_vals = list(range(len(verts)))  # [0,1,...,8]

Rx = 0
Ry = 0
for (x0, y0), (x1, y1), t0, t1 in zip(verts, verts[1:], t_vals, t_vals[1:]):
    dx = x1 - x0
    dy = y1 - y0
    Rx += dx * P.subs({a: t0, w: (t1 - t0)})
    Ry += dy * P.subs({a: t0, w: (t1 - t0)})

Rx = sp.simplify(Rx)
Ry = sp.simplify(Ry)

print("x(t) =", Rx)
print("y(t) =", Ry)

f_x = sp.lambdify(t, Rx, 'numpy')
f_y = sp.lambdify(t, Ry, 'numpy')

tt = np.linspace(t_vals[0], t_vals[-1], 800)
xp = f_x(tt)
yp = f_y(tt)

plt.figure(figsize=(5,5))
plt.plot(xp, yp, 'b-', linewidth=2)
px, py = zip(*verts[:-1])    # drop the last duplicate for plotting red markers
plt.scatter(px, py, color='red', zorder=10)
plt.axis('equal')
plt.title("Closed Parametric Polyline")
plt.xlabel("x"); plt.ylabel("y")
plt.grid(True)
plt.show()
