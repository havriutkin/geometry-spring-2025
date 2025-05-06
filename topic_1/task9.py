import numpy as np
import matplotlib.pyplot as plt

X = [0, 1, 2, 3, 4, 5]
Y = [1.75, -0.44, -0.76, 0.11, 2.47, 2.22]
n = len(X)

# Basis functions φ₀, φ₁, φ₂, φ₃
def phi0(x):      return 1
def phi1(x):      return x
def phi2(x):      return np.sin(x)
def phi3(x):      return np.exp(-x)

phis = [phi0, phi1, phi2, phi3]
m = len(phis) - 1  # = 3

# Build matrices U and V
U = np.zeros((m+1, m+1))
V = np.zeros(m+1)

for i in range(m+1):
    for j in range(m+1):
        U[i, j] = sum(phis[i](xk) * phis[j](xk) for xk in X)
    V[i] = sum(Y[k] * phis[i](X[k]) for k in range(n))

# Solve for coefficients 
a = np.linalg.solve(U, V)
a0, a1, a2, a3 = a.round(4)

print(f"y = {a0:.4f} + {a1:.4f}*x + {a2:.4f}*sin(x) + {a3:.4f}*exp(-x)")

# Deviation
def deviation(a, X, Y):
    def f(x):
        return a[0] + a[1]*x + a[2]*np.sin(x) + a[3]*np.exp(-x)
    return np.sqrt(sum((f(np.array(X)) - np.array(Y))**2))

Δ = deviation([a0, a1, a2, a3], X, Y)
print(f"Deviation Δ = {Δ:.3f}")

# Plot
x_plot = np.linspace(min(X)-0.5, max(X)+0.5, 300)
y_fit  = a0 + a1*x_plot + a2*np.sin(x_plot) + a3*np.exp(-x_plot)

plt.scatter(X, Y, color='red', marker='o', label='Data points')
plt.plot(x_plot, y_fit, color='blue', label='LS fit: 1, x, sin(x), e⁻ˣ')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Least Squares Approximation')
plt.grid(True)
plt.show()
