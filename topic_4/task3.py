import numpy as np
import sympy as sp
from mayavi import mlab

# 1) SYMBOLIC SETUP
x, y, z = sp.symbols('x y z', real=True)

# 2) Half‐space identifiers for the triangle in the XY‐plane
ω1 = y               # y ≥ 0
ω2 = 2 - x - y       # 2 - x - y ≥ 0
ω3 = 2 + x - y       # 2 + x - y ≥ 0

# 3) Intersection executor ir(u,v) = ½(u + v - |u - v|)
ir = lambda u, v: (u + v - sp.Abs(u - v)) / 2

# 4) Build planar‐region identifier ω_xy = ir(ir(ω1,ω2),ω3)
ω12 = ir(ω1, ω2)
ω_xy = ir(ω12, ω3)

# 5) Z‐constraints: 0 ≤ z ≤ 3
ω_z1 = z            # z ≥ 0
ω_z2 = 3 - z        # 3 - z ≥ 0

# 6) Full 3D identifier ω_xyz = ir(ir(ω_xy, ω_z1), ω_z2)
ω_xyz = ir(ir(ω_xy, ω_z1), ω_z2)
ω_xyz = sp.simplify(ω_xyz)

# 7) Lambdify for fast numeric evaluation
fω = sp.lambdify((x, y, z), ω_xyz, 'numpy')

# 8) Sample a grid over the bounding box:
#    x∈[−2.5,2.5], y∈[−0.5,2.5], z∈[0,3]
nx, ny, nz =  80,  80,  60
xs = np.linspace(-2.5,  2.5, nx)
ys = np.linspace(-0.5,  2.5, ny)
zs = np.linspace( 0.0,  3.0, nz)
X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
V = fω(X, Y, Z)

# 9) Render the ω=0 isosurface with Mayavi
mlab.figure(size=(800, 600), bgcolor=(1,1,1))
mlab.contour3d(
    X, Y, Z, V,
    contours=[0.0],        # extract ω=0 surface
    opacity=0.7,
    colormap='Spectral'
)
mlab.axes(
    xlabel='x', ylabel='y', zlabel='z',
    ranges=(-2.5,2.5, -0.5,2.5, 0,3)
)
mlab.title('Body: y≥0,2−x−y≥0,2+x−y≥0,0≤z≤3')
mlab.show()
