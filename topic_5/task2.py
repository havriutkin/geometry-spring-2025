import numpy as np
import matplotlib.pyplot as plt

# Define the three convex quadrilateral patches (x,z only; y=0)
patches = [
    [(-2, 0), (-5, 6), (-3, 6), (-1, 2)],  # left
    [(-2, 0), (-1, 2), (1, 2), (2, 0)],     # center
    [(2, 0), (1, 2), (3, 6), (5, 6)]        # right
]

# Parameter grids
nu, nv = 80, 80
u = np.linspace(0, 1, nu)
v = np.linspace(0, 1, nv)
UU, VV = np.meshgrid(u, v, indexing='ij')

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for quad in patches:
    A, B, C, D = quad
    Ax, Az = A; Bx, Bz = B
    Cx, Cz = C; Dx, Dz = D

    # interpolate along the two opposite edges
    XB = Ax + UU * (Bx - Ax)
    ZB = Az + UU * (Bz - Az)
    XD = Dx + UU * (Cx - Dx)
    ZD = Dz + UU * (Cz - Dz)

    # blend between those boundary‐curves
    X = (1 - VV) * XB + VV * XD
    Z = (1 - VV) * ZB + VV * ZD
    Y = np.zeros_like(X)

    ax.plot_surface(
        X, Y, Z,
        rstride=4, cstride=4,
        cmap='viridis',
        edgecolor='none',
        alpha=0.8
    )

# Overlay original V outline
boundary = [
    (-2, 0), (-5, 6), (-3, 6),
    (-1, 2), (1, 2), (3, 6),
    (5, 6), (2, 0), (-2, 0)
]
bx = [p[0] for p in boundary]
bz = [p[1] for p in boundary]
by = [0]*len(bx)
ax.plot(bx, by, bz, 'k-', lw=2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Ruled Surface Filling the Letter V')
ax.set_xlim(-6, 6)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 7)
plt.tight_layout()
plt.show()
