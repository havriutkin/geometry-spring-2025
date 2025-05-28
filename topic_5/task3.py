import numpy as np
import matplotlib.pyplot as plt

# Three convex quadrilateral patches in the XZ‐plane
# Each quad given as [A, B, C, D] in CCW order
patches = [
    [(-2, 0), (-5, 6), (-3, 6), (-1, 2)],  # left quad
    [(-2, 0), (-1, 2), ( 1, 2), ( 2, 0)],  # center quad
    [( 2, 0), ( 1, 2), ( 3, 6), ( 5, 6)]   # right quad
]

# Translation extent along Y
H = 1

# Sampling resolutions
nu, nv, nw = 100, 100, 40
u = np.linspace(0, 1, nu)
v = np.linspace(0, 1, nv)
w = np.linspace(0, H, nw)
UU, VV = np.meshgrid(u, v, indexing='ij')
WW = np.meshgrid(w, w, indexing='ij')[0]  # for extrusion

fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection='3d')

# Bilinear patch formula for quadrilateral [A, B, C, D]:
# r0(u) = A*(1-u) + B*u
# r1(u) = D*(1-u) + C*u
# patch: R(u,v) = r0(u)*(1-v) + r1(u)*v
for A, B, C, D in patches:
    Ax, Az = A
    Bx, Bz = B
    Cx, Cz = C
    Dx, Dz = D

    # compute boundary curves r0 and r1 as 2D arrays over u
    X0 = Ax + (Bx - Ax) * UU      # shape (nu, nv)
    Z0 = Az + (Bz - Az) * UU
    X1 = Dx + (Cx - Dx) * UU
    Z1 = Dz + (Cz - Dz) * UU

    # bilinear interpolation over (u,v)
    X_patch = X0 * (1 - VV) + X1 * VV
    Z_patch = Z0 * (1 - VV) + Z1 * VV

    # translate patch along Y to form front and back faces
    for y_val in (0, H):
        ax.plot_surface(
            X_patch, 
            y_val * np.ones_like(X_patch), 
            Z_patch,
            rstride=4, cstride=4,
            cmap='viridis', edgecolor='none', alpha=0.8
        )

# Extrude side walls by translating each edge along Y
for A, B, C, D in patches:
    # edges AB, BC, CD, DA
    for P1, P2 in [(A,B), (B,C), (C,D), (D,A)]:
        x1, z1 = P1
        x2, z2 = P2
        # line φ(u) = P1 + u*(P2-P1)
        X_edge = x1 + (x2 - x1) * np.outer(u, np.ones(nw))
        Z_edge = z1 + (z2 - z1) * np.outer(u, np.ones(nw))
        Y_edge = np.outer(np.ones(nu), w)  # shape (nu, nw)
        ax.plot_surface(
            X_edge, 
            Y_edge, 
            Z_edge,
            rstride=4, cstride=4,
            cmap='viridis', edgecolor='none', alpha=0.8
        )

# Outline of V at mid‐height
midY = H/2
boundary = [(-2,0),(-5,6),(-3,6),(-1,2),(1,2),(3,6),(5,6),(2,0),(-2,0)]
bx = [p[0] for p in boundary]
bz = [p[1] for p in boundary]
by = [midY] * len(bx)
ax.plot(bx, by, bz, 'k-', lw=2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Extruded 3D Surface of V using Surface of Translation')
ax.set_xlim(-6, 6)
ax.set_ylim(0, 5)
ax.set_zlim(0, 7)
plt.tight_layout()
plt.show()
