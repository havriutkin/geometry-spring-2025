import sympy as sp
from sympy.plotting import plot_implicit
import matplotlib.pyplot as plt

# Symbols
x, y = sp.symbols('x y', real=True)

# Stripe identifiers for |x|≤1 and |y|≤1 (square of side 2)
omega_x = 1 - sp.Abs(x)
omega_y = 1 - sp.Abs(y)

# Intersection executor ir(u,v) = (u+v - |u-v|)/2
u, v = sp.symbols('u v', real=True)
ir = lambda U, V: (U + V - sp.Abs(U - V)) / 2

# Square identifier
omega_sq = ir(omega_x, omega_y)

# Circle identifier & its complement, with new radius r=0.5
r = 1         # radius = 1
omega_c     = r**2 - (x**2 + y**2)
omega_not_c = -omega_c         # positive outside the smaller disk

# Final region: square ∩ (outside the circle)
omega_region = ir(omega_sq, omega_not_c)

# Plot the **filled** region ω_region > 0, shaded gray
plt.close('all')
plot_implicit(
    omega_region > 0,           # fill where omega_region is positive
    (x, -1.5, 1.5),
    (y, -1.5, 1.5),
    aspect=(1,1),
    border_color="k",
    n=400,
    color='gray',               # shading color
    alpha=0.7                   # semi‐transparent
)
