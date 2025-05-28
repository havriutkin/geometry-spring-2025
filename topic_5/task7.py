import numpy as np
import sympy as sp
import pyvista as pv
from functools import reduce

# 1) SYMBOLIC SETUP FOR V-SHAPE IN X-Y
x, y, z = sp.symbols('x y z', real=True)
ir = lambda U, V: (U + V - sp.Abs(U - V)) / 2
ur = lambda U, V: (U + V + sp.Abs(U - V)) / 2

# Define the three quadrilateral patches (CCW order)
patches = [
    [(-2, 0), (-1, 2), (-3, 6), (-5, 6)],
    [(-2, 0), (-1, 2), ( 1, 2), ( 2, 0)],
    [ (2, 0), ( 1, 2), ( 3, 6), ( 5, 6)]
]

def quad_region(quad):
    cx = sum(pt[0] for pt in quad) / 4
    cy = sum(pt[1] for pt in quad) / 4
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

# Build symbolic γ_shape = union of the three quad regions
f1 = quad_region(patches[0])
f2 = quad_region(patches[1])
f3 = quad_region(patches[2])
gamma_shape = ur(ur(f1, f2), f3)

# 2) Z constraints: 0 ≤ z ≤ H
H = 3
gamma_z1 = z         # z ≥ 0
gamma_z2 = H - z     # z ≤ H

# Full 3D identifier via intersection
gamma_xyz = ir(ir(gamma_shape, gamma_z1), gamma_z2)
gamma_xyz_s = sp.simplify(gamma_xyz)

# Convert symbolic expression to numerical function
f = sp.lambdify((x, y, z), gamma_xyz_s, 'numpy')

# Set up grid points for (x,y,z)
x_vals = np.linspace(-6, 6, 200)
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
plotter.add_title("Surface of the body defined by the system of constraints")
plotter.show()
