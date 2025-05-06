import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

Point = tuple[float, float]
x = sp.symbols('x')

def bernstein_piecewise(X: list[float], Y: list[float]) -> sp.Expr:
    n = len(X)
    # endpoints: 0 and n-1, interiors: 1..n-2
    # first two “ray” terms:
    term1 = Y[0] + ((Y[1] - Y[0])*(x - X[0]))/(X[1] - X[0])
    term2 = Y[n-2] + ((Y[n-1] - Y[n-2])*(x - X[n-2]))/(X[n-1] - X[n-2])
    expr  = sp.Rational(1,2)*(term1 + term2)

    # sum over all interior k=1..n-2
    for k in range(1, n-1):
        slope_next = (Y[k+1] - Y[k])/(X[k+1] - X[k])
        slope_prev = (Y[k]   - Y[k-1])/(X[k]   - X[k-1])
        expr += sp.Rational(1,2)*(slope_next - slope_prev)*sp.Abs(x - X[k])

    return sp.simplify(expr)


# Plot
X = [-4, -2, 1, 2]
Y = [-4, -1, 2, 0]
eq = bernstein_piecewise(X, Y)
print(f"Bernstein equations: {eq}")

# numeric lambdify
f = sp.lambdify(x, eq, 'numpy')

xp = np.linspace(min(X)-1, max(X)+1, 400)
yp = f(xp)

plt.scatter(X, Y, color='red', s=50, label='vertices')
plt.plot(xp, yp, color='blue', label='Bernstein polyline')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Explicit Bernstein Formula')
plt.grid(True)
plt.show()
