import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Given nodes, values, and derivatives
X = [0, 1, 2, 3]
Y = [0, 1, 1, 0]
G = [1, 1, -1, -1]
x = sp.symbols('x')

# Build the two endpoint Hermite cubics p1(x), p_n(x)
def hermite_segment(i):
    x0, x1 = X[i], X[i+1]
    y0, y1 = Y[i], Y[i+1]
    g0, g1 = G[i], G[i+1]
    h = x1 - x0
    t = (x - x0)/h
    H00 =  2*t**3 - 3*t**2 + 1
    H10 =      t**3 - 2*t**2 + t
    H01 = -2*t**3 + 3*t**2
    H11 =      t**3 -   t**2
    return sp.simplify(H00*y0 + H10*h*g0 + H01*y1 + H11*h*g1)

p1 = hermite_segment(0)
pn = hermite_segment(len(X)-2)

# Compute Pi¹ from formula (8)
H = [X[i] - X[i-1] for i in range(1, len(X))]
Pi1 = []
for i in range(1, len(X)-1):
    h_i   = H[i-1]
    h_ip1 = H[i]
    A = 3*((Y[i+1]-Y[i])/h_ip1**2 + (Y[i]-Y[i-1])/h_i**2)
    B = ( G[i+1] + 2*G[i] )/h_ip1 + ( G[i-1] + 2*G[i] )/h_i
    C = (( G[i] + G[i+1] )/h_ip1**2
        - 2*(Y[i+1]-Y[i])/h_ip1**3
        + 2*(Y[i]  -Y[i-1])/h_i**3
        -   (G[i-1]+G[i])/h_i**2)
    Pi1.append( sp.Rational(1,2)*(A - B + C*(x - X[i])) )

# Assemble SΔᴴ(x) via (7)
expr = (p1 + pn)/2
for i, P1 in enumerate(Pi1, start=1):
    expr += P1*(x - X[i])*sp.Abs(x - X[i])
expr = sp.simplify(expr)

print("Closed-form Hermite SΔᴴ(x) =", expr)

# Plot both closed and piecewise
pieces = [(hermite_segment(i), sp.And(x >= X[i], x <= X[i+1]))
          for i in range(len(X)-1)]
pw = sp.Piecewise(*pieces)

f_closed = sp.lambdify(x, expr, 'numpy')
f_pw     = sp.lambdify(x, pw,   'numpy')
xx = np.linspace(min(X)-0.5, max(X)+0.5, 400)

plt.plot(xx, f_closed(xx), 'b-',   linewidth=2, label='Formula (7)')
plt.plot(xx, f_pw(xx),     'r--',  linewidth=1, label='Piecewise Hermite')
plt.scatter(X, Y, color='black', zorder=10, label='Points')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Piecewise-Cubic Hermite (Closed-Form vs Piecewise)')
plt.grid(True)
plt.show()
