import numpy as np
import sympy as sp
import pyvista as pv
from functools import reduce

# 1) SYMBOLIC SETUP FOR V-SHAPE AND L-SHAPE IN X-Y
x, y, z = sp.symbols('x y z', real=True)
ir = lambda U, V: (U + V - sp.Abs(U - V)) / 2
ur = lambda U, V: (U + V + sp.Abs(U - V)) / 2

# Quadrilateral helper (intersection of halfspaces)
def quad_region(quad):
    cx = sum(pt[0] for pt in quad) / len(quad)
    cy = sum(pt[1] for pt in quad) / len(quad)
    regs = []
    for P, Q in zip(quad, quad[1:]+quad[:1]):
        x1, y1 = P; x2, y2 = Q
        A = y2 - y1
        B = -(x2 - x1)
        C = -(A*x1 + B*y1)
        L = A*x + B*y + C
        if L.subs({x:cx, y:cy}) < 0:
            L = -L
        regs.append(L)
    return reduce(ir, regs)

# V-shape patches (as before)
patches_V = [
    [(-2, 0), (-1, 2), (-3, 6), (-5, 6)],
    [(-2, 0), (-1, 2), ( 1, 2), ( 2, 0)],
    [( 2, 0), ( 1, 2), ( 3, 6), ( 5, 6)]
]
fV1 = quad_region(patches_V[0])
fV2 = quad_region(patches_V[1])
fV3 = quad_region(patches_V[2])
gamma_V = ur(ur(fV1, fV2), fV3)

# L-shape: vertical bar + foot bar
# Vertical bar of L
bar_height = 6
bar_width  = 1
x_offset   = 6  # position L to the right of V
vertical_bar = [
    (x_offset,      0),
    (x_offset,      bar_height),
    (x_offset+bar_width, bar_height),
    (x_offset+bar_width, 0)
]
# Foot of L (horizontal bar)
foot_length = 4
foot_thickness = 1
horizontal_bar = [
    (x_offset,            0),
    (x_offset+foot_length,0),
    (x_offset+foot_length,foot_thickness),
    (x_offset,            foot_thickness)
]

fL1 = quad_region(vertical_bar)
fL2 = quad_region(horizontal_bar)
gamma_L = ur(fL1, fL2)

# Combine V and L via union
gamma_shape = ur(gamma_V, gamma_L)

# 2) Z constraints: 0 ≤ z ≤ H
H = 3.0
gamma_z1 = z
gamma_z2 = H - z

# Full 3D identifier via intersection
gamma_xyz = ir(ir(gamma_shape, gamma_z1), gamma_z2)
gamma_xyz_s = sp.simplify(gamma_xyz)

# Convert symbolic expression to numerical function
f = sp.lambdify((x, y, z), gamma_xyz_s, 'numpy')

# Set up grid points for (x,y,z)
x_vals = np.linspace(-6, 12, 200)
y_vals = np.linspace(-6, 6, 200)
z_vals = np.linspace(-6, 6, 200)
# Use parameter indexing='ij' so that coordinates match the order (x,y,z)
X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
F = f(X, Y, Z)

# Get grid dimensions
nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)

# Create StructuredGrid:
# Form an array of points with shape (N,3)
points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = (nx, ny, nz)

# Add function data to the grid (data must be in Fortran order)
grid["values"] = F.flatten(order="F")

# Get isosurface for level 0 (i.e., ω(x,y,z)=0)
contours = grid.contour(isosurfaces=[0], scalars="values")

# Visualize the isosurface using PyVista
plotter = pv.Plotter()
plotter.add_mesh(contours, color="blue", opacity=0.5, label="ω(x,y,z)=0")
plotter.add_axes(xlabel='x', ylabel='y', zlabel='z')
plotter.add_legend()
plotter.add_title("Surface VL")
plotter.show()