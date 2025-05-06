import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

x = sp.symbols('x')

# Define the breakpoints and functions f_i
xs = [0, 1, 2, 3]
f0 = sp.Lambda(x, 0)
f1 = sp.Lambda(x, x)
f2 = sp.Lambda(x, x*(3 - x) - 1)
f3 = sp.Lambda(x, 3 - x)
f4 = sp.Lambda(x, 0)
fs = [f0, f1, f2, f3, f4]

# Define Π, Q, Ql
a, w = sp.symbols('a w', real=True)
Pi = sp.Lambda((x,a,w), (w + sp.Abs(x - a) - sp.Abs(x - a - w)) / 2)
Q  = sp.Lambda((x,a),       (x - a + sp.Abs(x - a)) / 2)
Ql = sp.Lambda((x,a),       (x - a - sp.Abs(x - a)) / 2)

# Build formula
n = len(xs) - 1
expr = fs[0]( xs[0] + Ql(x, xs[0]) )
for i in range(1, n+1):
    a_i_1 = xs[i-1]
    w_i   = xs[i] - xs[i-1]
    term  = fs[i]( a_i_1 + Pi(x, a_i_1, w_i) ) - fs[i](a_i_1)
    expr += term
expr += fs[n+1]( xs[n] + Q(x, xs[n]) ) - fs[n+1](xs[n])

expr = sp.simplify(expr)

print("Closed-form y(x) =", expr)

# Piecewise function for comparison
pw = sp.Piecewise(
    (fs[0](x), x <= xs[0]),
    (fs[1](x), (x > xs[0]) & (x < xs[1])),
    (fs[2](x), (x >= xs[1]) & (x < xs[2])),
    (fs[3](x), (x >= xs[2]) & (x < xs[3])),
    (fs[4](x), x >= xs[3])
)

# Numeric lambdify and plot
f_closed = sp.lambdify(x, expr, 'numpy')
f_pw     = sp.lambdify(x, pw,  'numpy')

xx = np.linspace(-1, 4, 500)
y1 = f_closed(xx)
y2 = f_pw(xx)

plt.figure()
plt.plot(xx, y1, linewidth=3, label='Formula (3)')
plt.plot(xx, y2, linewidth=1, linestyle='--', label='Piecewise')
# scatter the knots
for xi, fi in zip(xs, fs[1:-1]):
    plt.scatter([xi], [fi(xi)], color='red')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Continuous Piecewise Function via Formula (3)')
plt.grid(True)
plt.show()
