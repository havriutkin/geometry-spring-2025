import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

X = [-4, -2, 1, 2]
Y = [-4, -1,  2, 0]
n = len(X)

def least_squares_coeffs(X, Y, m):
    # Precompute sums U_k = sum(x_i^k) for k = 0..2m, and V_k = sum(x_i^k * y_i) for k = 0..m
    U = [sum(x**k for x in X) for k in range(2*m + 1)]
    V = [sum((X[i]**k) * Y[i] for i in range(n)) for k in range(m + 1)]

    # Build the (m+1)x(m+1) system A⋅a = b
    A = np.zeros((m+1, m+1))
    for i in range(m+1):
        for j in range(m+1):
            A[i, j] = U[i + j]
    b = np.array(V)

    # Solve for a = [a0, a1, …, am]
    return np.linalg.solve(A, b)

# Compute line and parabola coefficients
a_line = least_squares_coeffs(X, Y, m=1)
a_parabola = least_squares_coeffs(X, Y, m=2)

x = sp.symbols('x')
a_line_rounded     = [round(float(ai), 6) for ai in a_line]
a_parabola_rounded = [round(float(ai), 6) for ai in a_parabola]

phi_line     = a_line_rounded[0] + a_line_rounded[1]*x
phi_parabola = (
    a_parabola_rounded[0]
    + a_parabola_rounded[1]*x
    + a_parabola_rounded[2]*x**2
)

print("Least-squares line:     y =", sp.simplify(phi_line))
print("Least-squares parabola: y =", sp.simplify(phi_parabola))

# Deviation Δ = sqrt( sum( (y_i - φ(x_i))^2 ) )
def delta(coeffs, X, Y):
    # coeffs = [a0, a1, ...]
    def f(x): 
        return sum(coeffs[k] * x**k for k in range(len(coeffs)))
    return np.sqrt(sum((f(np.array(X)) - np.array(Y))**2))

dev_line   = round(delta(a_line_rounded,     X, Y), 3)
dev_para   = round(delta(a_parabola_rounded, X, Y), 3)

print(f"Deviation of line:     {dev_line}")
print(f"Deviation of parabola: {dev_para}")

# Plot
x_plot = np.linspace(min(X)-1, max(X)+1, 300)
y_line = a_line_rounded[0] + a_line_rounded[1]*x_plot
y_para = (
    a_parabola_rounded[0]
    + a_parabola_rounded[1]*x_plot
    + a_parabola_rounded[2]*x_plot**2
)

plt.scatter(X, Y, color='red', marker='o', label='Data points')
plt.plot(x_plot, y_line,      label='Least-squares line')
plt.plot(x_plot, y_para, linestyle='--', label='Least-squares parabola')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear and Quadratic Approximation')
plt.grid(True)
plt.show()
