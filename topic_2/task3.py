import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define the piecewise polynomials p1, p2, p3 and their knots x1, x2
x = sp.symbols('x')
x1, x2 = 0, 2

p1 = sp.Lambda(x, 2*x)
p2 = sp.Lambda(x, x*(x-1)*(x-2))
p3 = sp.Lambda(x, 2*(x-2))

# Compute P(x) = ½ (p1 + p3)
P = sp.simplify( (p1(x) + p3(x)) / 2 )

# Set the smoothness orders and h_k
m1 = m2 = 1
h1 = h2 = 3

# Build P1^N(x) and P2^N(x) via formula (5), then P1(x), P2(x)
#    helper: q-th derivative
def deriv(f, q, at):
    return float(sp.diff(f(x), x, q).subs(x, at))

# k = 1 at x1=0
coeffs1 = []
for q in range(m1+1, h1+1):  # q = 2,3
    Δq = deriv(p2, q, x1) - deriv(p1, q, x1)
    coeffs1.append( Δq / sp.factorial(q) * (x - x1)**(q - m1 - 1) )
P1N = sp.simplify( sp.Rational(1,2) * sum(coeffs1) )
P1  = sp.simplify( P1N * (x - x1) )

# k = 2 at x2=2
coeffs2 = []
for q in range(m2+1, h2+1):  # q = 2,3
    Δq = deriv(p3, q, x2) - deriv(p2, q, x2)
    coeffs2.append( Δq / sp.factorial(q) * (x - x2)**(q - m2 - 1) )
P2N = sp.simplify( sp.Rational(1,2) * sum(coeffs2) )
P2  = sp.simplify( P2N * (x - x2) )

expr = sp.simplify( P + P1*sp.Abs(x - x1) + P2*sp.Abs(x - x2) )
print("Closed-form PΔ(x) =", expr)

# Build the piecewise for checking
piecewise = sp.Piecewise(
    (p1(x), x <= x1),
    (p2(x), (x > x1) & (x < x2)),
    (p3(x),            x >= x2)
)

# Numeric lambdify & plot both
f_closed = sp.lambdify(x, expr, 'numpy')
f_pw     = sp.lambdify(x, piecewise, 'numpy')

xx = np.linspace(-3, 5, 600)
y1 = f_closed(xx)
y2 = f_pw(xx)

plt.figure()
plt.plot(xx, y1, linewidth=3, label='Formula (2)')
plt.plot(xx, y2, linewidth=1, linestyle='--', label='Piecewise')
plt.scatter([x1, x2], [p1(x1), p3(x2)], color='red', s=50, label='Knots')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Continuous Piecewise Polynomial via |x−x_k| Representation')
plt.grid(True)
plt.show()
