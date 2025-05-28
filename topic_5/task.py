import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

x, y, z, t, u, v = sp.symbols('x y z t u v', real=True)

def P(a, w, sym=t):
    return (1 / (2 * w)) * (w + sp.Abs(sym - a) - sp.Abs(sym - a - w))


# Boolean operations
def ir(u, v):
    """Intersection (перетин) ~ мінімум двох виразів"""
    return (u + v - sp.Abs(u - v)) / 2

def ur(u, v):
    """Union (об’єднання) ~ максимум двох виразів"""
    return (u + v + sp.Abs(u - v)) / 2

def dr(u, v):
    """Difference (різниця) ~ u - v"""
    return (u - v - sp.Abs(u + v)) / 2

def strip(a, b, c, h):
    """
    Генерація ідентифікатора w для похилої смуги:
    lin = a*x + b*y + c,
    w = h*sqrt(a^2+b^2) - |lin|
    """
    lin = a * x + b * y + c
    w = h * sp.sqrt(a ** 2 + b ** 2) - sp.Abs(lin)
    return lin, w


def build_polyline_expr_nd(vertices, t_values=None, sym=t):
    n = len(vertices)
    d = len(vertices[0])  
    if t_values is None:
        t_values = list(range(n))
    

    exprs = [sp.sympify(vertices[0][j]) for j in range(d)]
    # Кусочно-лінійна інтерполяція між точками
    for i in range(1, n):
        dt = t_values[i] - t_values[i - 1]
        for j in range(d):
            d_coord = vertices[i][j] - vertices[i - 1][j]
            exprs[j] += d_coord * P(t_values[i - 1], dt, sym=sym)
    exprs = [sp.simplify(e) for e in exprs]
    return exprs, t_values


def build_ruled_surface(curve1, curve2):
    # Побудова параметричних рівнянь для кожної кривої
    exprs1, t_vals1 = build_polyline_expr_nd(curve1, sym=t)
    exprs2, t_vals2 = build_polyline_expr_nd(curve2, sym=t)

    # Нормалізація параметра: підставляємо t = u*(t_last - t_first)
    u_exprs1 = [sp.simplify(e.subs(t, u * (t_vals1[-1] - t_vals1[0]))) for e in exprs1]
    u_exprs2 = [sp.simplify(e.subs(t, u * (t_vals2[-1] - t_vals2[0]))) for e in exprs2]

    # Лінійна інтерполяція між кривими для параметра v ∈ [0,1]
    x_ruled = sp.simplify(u_exprs1[0] * (1 - v) + u_exprs2[0] * v)
    y_ruled = sp.simplify(u_exprs1[1] * (1 - v) + u_exprs2[1] * v)
    return x_ruled, y_ruled


print("Завдання 5.8")
# Letter H (shifted to x in [0,1.5])
# Left vertical part of letter H: x in [0,0.5], y in [0,4]
_, left_x = strip(1, 0, -0.25, 0.25)
_, left_y = strip(0, 1, -2, 2.0)
left_rect = ir(left_x, left_y)

# Right vertical part of letter H: x in [1,1.5], y in [0,4]
_, right_x = strip(1, 0, -1.25, 0.25)
_, right_y = strip(0, 1, -2, 2.0)
right_rect = ir(right_x, right_y)

# Middle horizontal bar of letter H: x in [0,1.5], y in [1.75,2.25]
_, mid_x = strip(1, 0, -0.75, 0.75)
_, mid_y = strip(0, 1, -2.0, 0.25)
mid_rect = ir(mid_x, mid_y)

# Final shape for H
gamma_shape = ur(ur(left_rect, right_rect), mid_rect)

# Letter A (shifted to the right, x in [2,4])
xA = x - 2  # shift A by +2 along x
# Left slanted stripe of A
_, a_left_band = strip(-0.5, 1, 0, 0.12)
_, a_left_y = strip(0, 1, -2, 2)
a_left = ir(a_left_band.subs(x, xA), a_left_y.subs(x, xA))
# Right slanted stripe of A
_, a_right_band = strip(0.5, 1, -4, 0.12)
_, a_right_y = strip(0, 1, -2, 2)
a_right = ir(a_right_band.subs(x, xA), a_right_y.subs(x, xA))
# Horizontal crossbar of A
_, a_bar_y = strip(0, 1, -2, 0.10)
_, a_bar_x = strip(1, 0, -2, 1.0)
a_bar = ir(a_bar_y.subs(x, xA), a_bar_x.subs(x, xA))
# Combine all parts of A
o_shape = ur(ur(a_left, a_right), a_bar)

# Letter V (shifted to the right, x in [5,7])
xV = x - 5  # shift V by +5 along x
# Left slant of V: y = -2x + 4
_, v_left_band = strip(2, 1, -4, 0.12)
_, v_left_y = strip(0, 1, -2, 2)
v_left = ir(v_left_band.subs(x, xV), v_left_y.subs(x, xV))
# Right slant of V: y = 2x - 12
_, v_right_band = strip(-2, 1, 12, 0.12)
_, v_right_y = strip(0, 1, -2, 2)
v_right = ir(v_right_band.subs(x, xV), v_right_y.subs(x, xV))
# Combine both slants to form V
p_shape = ur(v_left, v_right)

print("Contour equation for letters HAV:")
# Combine the shapes for H, A, and V using union
total_shape = ur(gamma_shape, o_shape)
total_shape = ur(total_shape, p_shape)
total_shape = sp.simplify(total_shape)
sp.pprint(sp.Eq(total_shape, 0))
print()
# For plotting the contour (level 0)
f = sp.lambdify((x, y), total_shape, 'numpy')

# Create a grid of points
x_vals = np.linspace(-1, 12, 1000)
y_vals = np.linspace(-1, 5, 500)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

# Plot the filled area representing the region
plt.figure(figsize=(6, 6))
# Fill the area where total_shape(x, y) >= 0 (function values are positive)
plt.contourf(X, Y, Z, levels=[0, np.max(Z)], colors=['lightblue'])
# Draw the boundary contour (total_shape = 0)
plt.contour(X, Y, Z, levels=[0], colors='black')
plt.title("Region in the shape of letters HAV")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.show()

# Identification functions for z constraints:
all_z1 = z  # z >= 0
all_z2 = 1 - z  # z <= 1  <=> 1 - z >= 0
# Build the full 3D identification function as intersection of the planar region and z interval
total_xyz = ir(ir(total_shape, all_z1), all_z2)
total_xyz_simpl = sp.simplify(total_xyz)

# Get the symbolic equation for the surface contour (ω(x,y,z)=0)
print("\nImplicit equation for the surface contour:")
sp.pprint(total_xyz_simpl)
print(" = 0")
print("Start of lambdify")
# Convert the symbolic expression to a numerical function
f = sp.lambdify((x, y, z), total_xyz_simpl, 'numpy')
print("End of lambdify")
# Set up the grid of points for (x, y, z)
x_vals = np.linspace(-1, 12, 500)
y_vals = np.linspace(-1, 5, 500)
z_vals = np.linspace(-1, 2, 200)
# Use indexing='ij' so coordinates match the order (x, y, z)
print("Start of meshgrid")
X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
print("Start of calculation F")
F = f(X, Y, Z)
print("End of Calculation F")
# Get grid dimensions
nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)

# Create StructuredGrid:
# Form the array of points with shape (N, 3)
points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = (nx, ny, nz)

# Add function data to the grid (data must be in Fortran order)
grid["values"] = F.flatten(order="F")

# Extract the isosurface for level 0 (i.e., ω(x,y,z)=0)
contours = grid.contour(isosurfaces=[0], scalars="values")

# Visualize the isosurface using PyVista
plotter = pv.Plotter()
plotter.add_mesh(contours, color="blue", opacity=0.5, label="ω(x,y,z)=0")
plotter.add_axes(xlabel='x', ylabel='y', zlabel='z')
plotter.add_legend()
plotter.add_title("Body surface defined by the system of constraints")
plotter.show()

print("Завдання 5.4")
# --- 1. Вертикальна частина літери Г ---

# Обмеження по x: x ∈ [0,1]
# Використовуємо смугу, центровану в x=0.5, з півшириною 0.5:
_, vertical_x = strip(1, 0, -0.5, 0.5)  # 0.5 - |x - 0.5|

# Обмеження по y: y ∈ [0,4]
# Використовуємо смугу, центровану в y=2, з піввисотою 2:
_, vertical_y = strip(0, 1, -2, 2.5)  # 2 - |y - 2|

# Перетин цих умов дає вертикальний прямокутник:
vertical_rect = ir(vertical_x, vertical_y)

# --- 2. Горизонтальна частина літери Г (верхня "шапка") ---

# Обмеження по x: x ∈ [0,3]
# Смужка, центрована в x=1.5 з півшириною 1.5:
_, horizontal_x = strip(1, 0, -1.5, 1.5)  # 1.5 - |x - 1.5|

# Обмеження по y: y ∈ [3,4]
# Смужка, центрована в y=3.5 з піввисотою 0.5:
_, horizontal_y = strip(0, 1, -4, 0.5)  # 0.5 - |y - 3.5|

# Перетин дає горизонтальний прямокутник:
horizontal_rect = ir(horizontal_x, horizontal_y)

# --- 3. Фінальна форма літери Г ---
# Об’єднуємо вертикальну та горизонтальну частини:
gamma_shape = ur(vertical_rect, horizontal_rect)
print("Рівняння контуру літери Г:")
sp.pprint(sp.Eq(sp.simplify(gamma_shape), 0))
print()
# Для побудови контуру (рівень 0)
f = sp.lambdify((x, y), gamma_shape, 'numpy')

# Створюємо сітку точок
x_vals = np.linspace(-1, 4, 500)
y_vals = np.linspace(-1, 5, 500)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

print("Завдання 5.6")
# --- Додатково: побудова неявного рівняння плоскої області ---
# Для області всередині контуру літери Г визначимо нерівність:
print("Неявне рівняння плоскої області у формі літери Г:")
print("Область визначається нерівністю:")
print(f"{sp.pretty(sp.simplify(gamma_shape))} >= 0")
print()
# Побудова заповненого графіка, що демонструє площу області
plt.figure(figsize=(6, 6))
# Заповнюємо область, де gamma_shape(x, y) >= 0 (значення функції додатні)
plt.contourf(X, Y, Z, levels=[0, np.max(Z)], colors=['lightblue'])
# Наносимо контур границі (gamma_shape = 0)
plt.contour(X, Y, Z, levels=[0], colors='black')
plt.title("Площа області літери Г")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.show()
print("Завдання 5.7")
# Ідентифікаційні функції для обмежень по z:
gamma_z1 = z  # z >= 0
gamma_z2 = 1 - z  # z <= 3  <=> 3 - z >= 0
# Побудова повної 3D ідентифікаційної функції як перетин області в площині та проміжку по z
gamma_xyz = ir(ir(gamma_shape, gamma_z1), gamma_z2)
gamma_xyz_simpl = sp.simplify(gamma_xyz)

# Отримання символьного рівняння контуру поверхні тіла (ω(x,y,z)=0)
print("\nНеявне рівняння контуру поверхні тіла:")
sp.pprint(gamma_xyz_simpl)
print(" = 0")

# Перетворення символьного виразу у числову функцію
f = sp.lambdify((x, y, z), gamma_xyz_simpl, 'numpy')

# Задаємо сітку точок для (x,y,z)
x_vals = np.linspace(-1, 4, 200)
y_vals = np.linspace(-1, 5, 200)
z_vals = np.linspace(-1, 4, 200)
# Використовуємо параметр indexing='ij', щоб координати відповідали порядку (x,y,z)
X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
F = f(X, Y, Z)

# Отримання розмірів сітки
nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)

# Створення StructuredGrid:
# Формуємо масив точок розмірності (N,3)
points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = (nx, ny, nz)

# Додаємо дані функції до сітки (дані мають бути у порядку Fortran)
grid["values"] = F.flatten(order="F")

# Отримання ізоповерхні для рівня 0 (тобто ω(x,y,z)=0)
contours = grid.contour(isosurfaces=[0], scalars="values")

# Візуалізація ізоповерхні за допомогою PyVista
plotter = pv.Plotter()
plotter.add_mesh(contours, color="blue", opacity=0.5, label="ω(x,y,z)=0")
plotter.add_axes(xlabel='x', ylabel='y', zlabel='z')
plotter.add_legend()
plotter.add_title("Поверхня тіла, задана системою обмежень")
plotter.show()

print("Завдання 5.5")
# =======================
# Побудова цифри 8
# =======================
# Верхній контур 8 (коло)
f8_top = 0.9 - sp.sqrt((x + 3) ** 2 + (y - 0.8) ** 2)
# Нижній контур 8 (коло)
f8_bot = 1 - sp.sqrt((x + 3) ** 2 + (y + 0.8) ** 2)
# Об’єднуємо два кола, щоб отримати цифру 8
digit8 = ur(f8_top, f8_bot)
# Прибираємо середину цифри 8
digit8 = dr(digit8, (0.65 - sp.sqrt((x + 3) ** 2 + (y - 0.8) ** 2)))
digit8 = dr(digit8, (0.75 - sp.sqrt((x + 3) ** 2 + (y + 0.8) ** 2)))
# =======================
# Побудова цифри 6
# =======================
# Основне кільце для 6 - збільшуємо радіус та покращуємо положення
# Нижнє кільце цифри 6
f6_lower_ring = 1 - sp.sqrt(x ** 2 + (y + 0.8) ** 2)
# Внутрішня частина нижнього кільця
f6_lower_inner = 0.75 - sp.sqrt(x ** 2 + (y + 0.8) ** 2)
# Формуємо нижнє кільце (зовнішнє - внутрішнє)
f6_lower = dr(f6_lower_ring, f6_lower_inner)

# Верхнє кільце цифри 6
f6_upper_ring = 2 - 0.5 - sp.sqrt((x - 1.22 + 0.5) ** 2 + (y - 0.15) ** 2)
# Внутрішня частина верхнього кільця
f6_upper_inner = 1.75 - 0.5 - sp.sqrt((x - 1.22 + 0.5) ** 2 + (y - 0.15) ** 2)
# Створюємо обмеження для видалення частини де y<0.5 і x>0.5
keep_region = ir(y + 0.15, 0.5 - x)  # True коли y≥0.5 & x≤0.5
# Формуємо верхнє кільце з видаленою частиною
f6_upper = ir(dr(f6_upper_ring, f6_upper_inner), keep_region)

# Об'єднуємо два кільця для отримання цифри 6
digit6 = ur(f6_lower, f6_upper)
# =======================
# Побудова цифри 2
# =======================

# 1) Верхня дуга (подібно до цифр "8" чи "6"):
#    Використовуємо "кільце" (зовнішнє коло радіуса ~1, внутрішнє коло ~0.7),
#    центр зміщений до (3, 1.0). Це все потрібно підлаштувати під загальну композицію.
f2_arc_outer = 1 - sp.sqrt((x - 3) ** 2 + (y - 1.0) ** 2)
f2_arc_inner = 0.8 - sp.sqrt((x - 3) ** 2 + (y - 1.0) ** 2)
# Формуємо кільце як різницю зовнішнього та внутрішнього кіл:
f2_top_arc = dr(f2_arc_outer, f2_arc_inner)

# Залишимо лише "верхню половину" дуги, щоб вона не «замикалася» внизу:
# Наприклад, нехай y >= 0.3 (можна змінити на інше значення).
f2_top_arc = ir(f2_top_arc, (y - 0.9))
f2_top_arc = ir(f2_top_arc, (x - 2.2))
# 2) Діагональ (похила «ніжка» двійки):
_, f2_diag = strip(-1.6, 1, 5.28, 0.1)
#    Можемо також обмежити діапазон y, аби діагональ не «вилазила» за межі верхньої дуги чи низу.
#    Наприклад, нехай y ≤ 0.3:
_, diag_y_lim = strip(0, 1, (-1.45 + 1.675) + 0.1, (1 + 1.575) / 2)
#    Тут strip(0, -1, c, h) означає a=0, b=-1 => лінія y= -c, half-width = h.
#    За потреби доберіть c та h точніше.
f2_diag = ir(f2_diag, diag_y_lim)

# 3) Нижня горизонтальна «підошва» (смуга біля y = -1):
#    Лінія y + 1 = 0 => (a=0, b=1, c=1).
#    Візьмемо half-width=0.3, щоб смуга покривала y від -1.3 до -0.7.
_, f2_bot_y = strip(0, 1, 1.8 - 0.125, 0.125)

#    Обмежимо x, скажімо, від 2.3 до 4.0 (центр 3.15, half=0.85):
_, f2_bot_x = strip(1, 0, -3.05, 0.855)
bottom_bar = ir(f2_bot_y, f2_bot_x)

# 4) Остаточне об’єднання всіх частин цифри «2»:
digit2 = ur(f2_top_arc, ur(f2_diag, bottom_bar))
# =======================
# Об’єднання цифр 8, 6 та 2
# =======================
# final_shape визначає неявне рівняння контуру для всього числа 862:
final_shape = ur(ur(digit8, digit6), digit2)

# Вивід неявного рівняння (контур, тобто final_shape(x,y)=0)
print("Неявне рівняння контуру області у формі числа 862:")
sp.pprint(sp.Eq(sp.simplify(final_shape), 0))
print("\nОбласть визначається нерівністю:")
print(f"{sp.pretty(sp.simplify(final_shape))} ≥ 0")

# =======================
# Побудова графіки
# =======================
# Функція для обчислення значень final_shape
f = sp.lambdify((x, y), final_shape, 'numpy')

# Створюємо сітку точок
x_vals = np.linspace(-6, 6, 600)
y_vals = np.linspace(-4, 4, 600)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

# Графік
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, levels=[0, np.max(Z)], colors=['lightblue'])
plt.contour(X, Y, Z, levels=[0], colors='black')
plt.title("Область у формі числа 862")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.show()

print("Завдання 5.1")
print("Параметричне рівняння контуру літери M:")
# Визначаємо вершини контуру літери M
# Точки у 2D-просторі (x, y)
vertices = [(0, 0), (0, 1), (0.5, 0.5), (1, 1), (1, 0), (0.8, 0), (0.8, 0.6), (0.5, 0.3), (0.2, 0.6), (0.2, 0), (0, 0)]
# Побудова символьних виразів для x(t) та y(t)
exprs_curve, t_vals = build_polyline_expr_nd(vertices, sym=t)

print("x(t) =", sp.pretty(exprs_curve[0]))
print("y(t) =", sp.pretty(exprs_curve[1]))

# Розбиваємо полігон на дві криві.
# Вибираємо спільні ендпоінти: (0,0) і (1,0)
upper_curve = [(0, 0), (0, 1), (0.5, 0.5), (1, 1), (1, 0)]
lower_curve = [(0, 0), (0.2, 0), (0.2, 0.6), (0.5, 0.3), (0.8, 0.6), (0.8, 0), (1, 0)]

# Побудова лінійчатої поверхні (ruled surface) між верхньою та нижньою кривими
x_ruled, y_ruled = build_ruled_surface(upper_curve, lower_curve)
print("Завдання 5.2")
print("x(u,v) =", sp.pretty(x_ruled))
print("y(u,v) =", sp.pretty(y_ruled))
# Перетворення символьних виразів у числові функції
f_x = sp.lambdify((u, v), x_ruled, "numpy")
f_y = sp.lambdify((u, v), y_ruled, "numpy")

# Генеруємо сітку параметрів u, v ∈ [0,1]
u_vals = np.linspace(0, 1, 100)
v_vals = np.linspace(0, 1, 100)
U, V = np.meshgrid(u_vals, v_vals)
X = f_x(U, V)
Y = f_y(U, V)

# Побудова графіка: зафарбована ruled поверхня + контури кривих для наочності
plt.figure(figsize=(7, 7))
# Створення триангуляції для зафарбування області
from matplotlib.tri import Triangulation

triang = Triangulation(U.flatten(), V.flatten())
facecolors = np.zeros(len(triang.triangles))
plt.tripcolor(X.flatten(), Y.flatten(), triang.triangles,
              facecolors=facecolors, cmap='Pastel1', alpha=0.5)

# Нанесення контурів кривих
upper_x = [pt[0] for pt in upper_curve]
upper_y = [pt[1] for pt in upper_curve]
plt.plot(upper_x, upper_y, 'b-', linewidth=2, label='Верхня крива')

lower_x = [pt[0] for pt in lower_curve]
lower_y = [pt[1] for pt in lower_curve]
plt.plot(lower_x, lower_y, 'b-', linewidth=2, label='Нижня крива')

plt.title("Лінійчата поверхня для букви M (розбиття на дві криві)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='upper right')
plt.axis('equal')
plt.show()
print("Завдання 5.3")
u_vals_sym = t_vals
x_expr = exprs_curve[0]
y_expr = exprs_curve[1]
X_expr = x_expr
Y_expr = y_expr
Z_expr = v
print("\nПараметричне рівняння поверхні перенесення:")
print("X(u, v) =")
sp.pprint(x_ruled)
print("\nY(u, v) =")
sp.pprint(y_ruled)
print("\nZ(u, v) =")
sp.pprint(Z_expr)

# Create numerical functions for computation
f_x = sp.lambdify(t, x_expr, 'numpy')  # For the original curve
f_y = sp.lambdify(t, y_expr, 'numpy')  # For the original curve
f_X = sp.lambdify((t, v), X_expr, 'numpy')  # For the ruled surface
f_Y = sp.lambdify((t, v), Y_expr, 'numpy')  # For the ruled surface
f_Z = sp.lambdify((t, v), Z_expr, 'numpy')  # For the z-coordinate

u_num = np.linspace(u_vals_sym[0], u_vals_sym[-1], 300)
x_num = f_x(u_num)
y_num = f_y(u_num)

# Обираємо параметричний інтервал для u (як для ламаної) та для v (довжина екструзії)
u_vals = np.linspace(u_vals_sym[0], u_vals_sym[-1], 100)
v_vals = np.linspace(-2, 2, 50)  # перенесення вздовж Z від -2 до 2
U, V = np.meshgrid(u_vals, v_vals)
# Compute numerical values from the lambdified functions
X_vals = f_X(U, V)
Y_vals = f_Y(U, V)
Z_vals = f_Z(U, V)

# Масиви U, V мають форму (n_v, n_u) – встановлюємо розміри як (n_u, n_v, 1)
n_u = len(u_vals)
n_v = len(v_vals)
points = np.column_stack([
    X_vals.T.ravel(order='F'),
    Y_vals.T.ravel(order='F'),
    Z_vals.T.ravel(order='F')
])
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = (n_u, n_v, 1)

# Створимо також генеруючу криву (при v = 0) як PolyData для порівняння:
u_line = np.linspace(u_vals_sym[0], u_vals_sym[-1], 300)
x_line = f_x(u_line)
y_line = f_y(u_line)
z_line = np.zeros_like(u_line)
curve_points = np.column_stack([x_line, y_line, z_line])
n_line = len(u_line)
# Створюємо масив з'єднувальних індексів для ламаної
lines = np.hstack(([n_line], np.arange(n_line)))
polyline = pv.PolyData()
polyline.points = curve_points
polyline.lines = lines

# Візуалізація за допомогою PyVista
plotter = pv.Plotter(window_size=(800, 600))
plotter.add_mesh(grid, opacity=0.8, cmap='viridis', show_scalar_bar=False)
plotter.add_mesh(polyline, color='red', line_width=5)
plotter.add_text("Поверхня перенесення", font_size=14)
plotter.camera_position = 'xy'
plotter.show()