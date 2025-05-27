import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

t = sp.symbols('t', real=True)

points = [(-4, 4), (2, 4), (3, -4), (-3, -2)]
t_vals = [0, 1, 2, 3, 4]   

# Extract xᵢ, yᵢ and enforce periodic closure
x_vals = [p[0] for p in points] + [points[0][0]]
y_vals = [p[1] for p in points] + [points[0][1]]
n      = len(points)           # 4 segments
h      = [t_vals[i+1] - t_vals[i] for i in range(n)]  # all 1 here

# Prepare unknown second-derivatives s₀…s₃ (periodic: s₄ = s₀)
s_vars = sp.symbols('s0:'+str(n))

# Build the 4×4 system As=qs for x, then As = qy for y
eqs_x = []
# interior i=1..n-2
for i in range(1, n-1):
    eqs_x.append(
        h[i-1]*s_vars[i-1]
      + 2*(h[i-1]+h[i])*s_vars[i]
      + h[i]*s_vars[i+1]
      - 6*((x_vals[i+1]-x_vals[i])/h[i]
           - (x_vals[i]-x_vals[i-1])/h[i-1])
    )
# periodic at i=0
eqs_x.insert(0,
    h[n-1]*s_vars[n-1]
  + 2*(h[n-1]+h[0])*s_vars[0]
  + h[0]*s_vars[1]
  - 6*((x_vals[1]-x_vals[0])/h[0]
       - (x_vals[0]-x_vals[n-1])/h[n-1])
)
# periodic at i=n-1
eqs_x.append(
    h[n-2]*s_vars[n-2]
  + 2*(h[n-2]+h[n-1])*s_vars[n-1]
  + h[n-1]*s_vars[0]
  - 6*((x_vals[n]-x_vals[n-1])/h[n-1]
       - (x_vals[n-1]-x_vals[n-2])/h[n-2])
)
sol_x = sp.solve(eqs_x, s_vars)

# same for y
eqs_y = []
for i in range(1, n-1):
    eqs_y.append(
        h[i-1]*s_vars[i-1]
      + 2*(h[i-1]+h[i])*s_vars[i]
      + h[i]*s_vars[i+1]
      - 6*((y_vals[i+1]-y_vals[i])/h[i]
           - (y_vals[i]-y_vals[i-1])/h[i-1])
    )
eqs_y.insert(0,
    h[n-1]*s_vars[n-1]
  + 2*(h[n-1]+h[0])*s_vars[0]
  + h[0]*s_vars[1]
  - 6*((y_vals[1]-y_vals[0])/h[0]
       - (y_vals[0]-y_vals[n-1])/h[n-1])
)
eqs_y.append(
    h[n-2]*s_vars[n-2]
  + 2*(h[n-2]+h[n-1])*s_vars[n-1]
  + h[n-1]*s_vars[0]
  - 6*((y_vals[n]-y_vals[n-1])/h[n-1]
       - (y_vals[n-1]-y_vals[n-2])/h[n-2])
)
sol_y = sp.solve(eqs_y, s_vars)

# Build second-derivative lists s0, ..., s4
s_x = [sol_x[v] for v in s_vars] + [sol_x[s_vars[0]]]
s_y = [sol_y[v] for v in s_vars] + [sol_y[s_vars[0]]]

# p1(t) and pn(t) using the standard cubic-Hermite formula
def build_segment_polynomial(vals, s_list, idx):
    a = vals[idx]
    h_i = h[idx]
    b = (vals[idx+1] - vals[idx])/h_i - (2*s_list[idx] + s_list[idx+1])*h_i/6
    c = s_list[idx]/2
    d = (s_list[idx+1] - s_list[idx])/(6*h_i)
    return a + b*(t - t_vals[idx]) + c*(t - t_vals[idx])**2 + d*(t - t_vals[idx])**3

p1_x  = build_segment_polynomial(x_vals, s_x, 0)
pn_x  = build_segment_polynomial(x_vals, s_x, n-1)
p1_y  = build_segment_polynomial(y_vals, s_y, 0)
pn_y  = build_segment_polynomial(y_vals, s_y, n-1)

# use (1):  S(t) = ½[p₁(t)+pₙ(t)] + 1/12 ∑ Bᵢ·|t - tᵢ|³
P_x = (p1_x + pn_x)/2
P_y = (p1_y + pn_y)/2

Sx = P_x
Sy = P_y
for i in range(1, n):
    Bx = (s_x[i+1] - s_x[i])/h[i] - (s_x[i] - s_x[i-1])/h[i-1]
    By = (s_y[i+1] - s_y[i])/h[i] - (s_y[i] - s_y[i-1])/h[i-1]
    Sx += Bx*sp.Abs(t - t_vals[i])**3/12
    Sy += By*sp.Abs(t - t_vals[i])**3/12

Sx = sp.simplify(Sx)
Sy = sp.simplify(Sy)

f_Sx = sp.lambdify(t, Sx, 'numpy')
f_Sy = sp.lambdify(t, Sy, 'numpy')

# Parametric R(t) Bernstein piecewise formula
a, w = sp.symbols('a w', real=True)
P_expr  = (w + sp.Abs(t - a) - sp.Abs(t - a - w)) / (2*w)
Ql_expr = (t - a - sp.Abs(t - a)) / 2
Q_expr  = (t - a + sp.Abs(t - a)) / 2

# Prepare equations
verts = [(x_vals[i], y_vals[i]) for i in range(n+1)]
r0x, r0y = verts[0]
vlx = verts[1][0] - r0x
vly = verts[1][1] - r0y
vrx = verts[-1][0] - verts[-2][0]
vry = verts[-1][1] - verts[-2][1]

Rx = r0x + vlx*Ql_expr.subs(a, t_vals[0])
Ry = r0y + vly*Ql_expr.subs(a, t_vals[0])

for (x0,y0),(x1,y1),ti,ti1 in zip(verts, verts[1:], t_vals, t_vals[1:]):
    dx, dy = x1 - x0, y1 - y0
    Rx += dx*P_expr.subs({a: ti, w: ti1 - ti})
    Ry += dy*P_expr.subs({a: ti, w: ti1 - ti})

Rx += vrx*Q_expr.subs(a, t_vals[-1])
Ry += vry*Q_expr.subs(a, t_vals[-1])

Rx = sp.simplify(Rx)
Ry = sp.simplify(Ry)

f_Rx = sp.lambdify(t, Rx, 'numpy')
f_Ry = sp.lambdify(t, Ry, 'numpy')

# Plot everything
tt  = np.linspace(0, 4, 400)
xs  = f_Sx(tt); ys  = f_Sy(tt)
xr  = f_Rx(tt); yr  = f_Ry(tt)

plt.figure(figsize=(8,6))
plt.plot(xs, ys, 'b-', label='Cubic spline S(t)')
plt.plot(xr, yr, 'r--', label='Quadrilateral R(t)')
px, py = zip(*points)
plt.scatter(px, py, c='k', s=50, label='Control points')
plt.axis('equal')
plt.legend()
plt.xlabel('x'); plt.ylabel('y')
plt.title('Task 2.7: Closed cubic spline & quadrilateral')
plt.grid(True)
plt.show()
