import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


u = sp.symbols('u')
v = sp.symbols('v')
t = sp.symbols('t')

def P(a, w, sym=t):
    return (1 / (2 * w)) * (w + sp.Abs(sym - a) - sp.Abs(sym - a - w))

def build_polyline_expr(vertices, t_values=None, sym=t):
    n = len(vertices)
    # Якщо значення параметра не задані – рівномірний розподіл: 0, 1, 2, ..., n-1
    if t_values is None:
        t_values = list(range(n))
    if len(t_values) != n:
        raise ValueError("Кількість значень параметра має дорівнювати кількості вершин.")

    # Початкове положення – перша вершина
    x_expr = sp.sympify(vertices[0][0])
    y_expr = sp.sympify(vertices[0][1])
    # Додаємо внески для кожного відрізка ламаної
    for i in range(1, n):
        dt = t_values[i] - t_values[i - 1]
        dx = vertices[i][0] - vertices[i - 1][0]
        dy = vertices[i][1] - vertices[i - 1][1]
        x_expr += dx * P(t_values[i - 1], dt, sym=sym)
        y_expr += dy * P(t_values[i - 1], dt, sym=sym)
    return sp.simplify(x_expr), sp.simplify(y_expr), t_values

def build_polyline_expr_nd(vertices, t_values=None, sym=t):
    n = len(vertices)
    d = len(vertices[0])  # вимірність (наприклад, 3 для 3D)
    if t_values is None:
        t_values = list(range(n))
    if len(t_values) != n:
        raise ValueError("Кількість значень параметра має дорівнювати кількості вершин.")
    # Початкові вирази для кожної координати
    exprs = [sp.sympify(vertices[0][j]) for j in range(d)]
    # Кусочно-лінійна інтерполяція між точками
    for i in range(1, n):
        dt = t_values[i] - t_values[i - 1]
        for j in range(d):
            d_coord = vertices[i][j] - vertices[i - 1][j]
            exprs[j] += d_coord * P(t_values[i - 1], dt, sym=sym)
    exprs = [sp.simplify(e) for e in exprs]
    return exprs, t_values


print("Завдання 3.4. Параметричні рівняння поверхні подібних поперечних перерізів")
# Задаємо вершини ламаної L (напрямний профіль) в площині XZ.
# Кожна вершина представлена як (x, z)
L_vertices = [(0, 1), (0.75, 1), (0.75, 0), (0, 0)]
L_t_values = list(range(len(L_vertices)))  # Наприклад, [0, 1, 2, ..., 6]

# Задаємо вершини базового перерізу S (наприклад, квадрат) в площині XY.
# Щоб отримати замкнуту ламану, повторюємо першу вершину в кінці.
S_vertices = [(-1, 0), (-1, 1), (1, 1), (1, 0), (-1, 0)]
S_t_values = list(range(len(S_vertices)))  # Наприклад, [0, 1, 2, 3, 4]

# ===== Обчислення параметричних рівнянь ламаних =====
# Для ламаної L (профіль) – координати X та Z
L_expr, L_params = build_polyline_expr_nd(L_vertices, L_t_values, sym=u)
xp_expr = L_expr[0]  # x(u)
zp_expr = L_expr[1]  # z(u)
print("Параметричні рівняння профілю (ламана L):")
print("x(u) =")
sp.pprint(xp_expr)
print("\nz(u) =")
sp.pprint(zp_expr)
print()

# Для ламаної S (базовий переріз) – координати X та Y
S_expr, S_params = build_polyline_expr_nd(S_vertices, S_t_values, sym=v)
xb_expr = S_expr[0]  # xb(v)
yb_expr = S_expr[1]  # yb(v)
print("Параметричні рівняння базового перерізу (ламана S):")
print("xb(v) =")
sp.pprint(xb_expr)
print("\nyb(v) =")
sp.pprint(yb_expr)
print()

# Перетворення символьних виразів у числові функції
f_xp = sp.lambdify(u, xp_expr, 'numpy')
f_z = sp.lambdify(u, zp_expr, 'numpy')
f_xb = sp.lambdify(v, xb_expr, 'numpy')
f_yb = sp.lambdify(v, yb_expr, 'numpy')

# ===== Побудова 2D-графіка ламаної L =====
u_min, u_max = L_params[0], L_params[-1]
u_vals = np.linspace(u_min, u_max, 200, endpoint=True)
L_x_vals = f_xp(u_vals)
L_z_vals = f_z(u_vals)

plt.figure(figsize=(6, 4))
plt.plot(L_x_vals, L_z_vals, 'r-', lw=2, label="Ламана L (профіль)")
plt.xlabel("x")
plt.ylabel("z")
plt.title("Графік ламаної L (профіль) в площині XZ")
plt.legend()
plt.grid(True)
plt.show()

# ===== Побудова 2D-графіка ламаної S =====
v_min, v_max = S_params[0], S_params[-1]
v_vals = np.linspace(v_min, v_max, 200, endpoint=True)
S_x_vals = f_xb(v_vals)
S_y_vals = f_yb(v_vals)

plt.figure(figsize=(6, 4))
plt.plot(S_x_vals, S_y_vals, 'g-', lw=2, label="Ламана S (базовий переріз)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Графік ламаної S (базовий переріз) в площині XY")
plt.legend()
plt.grid(True)
plt.show()

u_sample = np.linspace(u_min, u_max, 10000, endpoint=True)
x_sample = f_xp(u_sample)
x0 = x_sample.max()
print("Автоматично обчислений масштабний коефіцієнт x0 =", x0)

# Побудова параметричних рівнянь для поверхні:
# x(u,v) = (x(u)/x0) * x_b(v)
# y(u,v) = (x(u)/x0) * y_b(v)
# z(u,v) = z(u)
x_expr = (xp_expr / x0) * xb_expr
y_expr = (xp_expr / x0) * yb_expr
z_expr = zp_expr

# Перетворення виразів у числові функції
f_x = sp.lambdify((u, v), x_expr, 'numpy')
f_y = sp.lambdify((u, v), y_expr, 'numpy')
f_z_surface = sp.lambdify(u, z_expr, 'numpy')

# Створення сітки параметрів для поверхні
num_u, num_v = 1210, 410
u_vals_3d = np.linspace(u_min, u_max, num_u, endpoint=True)
v_vals_3d = np.linspace(v_min, v_max, num_v, endpoint=True)
U, V = np.meshgrid(u_vals_3d, v_vals_3d, indexing='ij')

X = f_x(U, V)
Y = f_y(U, V)
Z = f_z_surface(U)

# Формування масиву точок для pyvista
points = np.empty(X.shape + (3,))
points[..., 0] = X
points[..., 1] = Y
points[..., 2] = Z

grid = pv.StructuredGrid()
grid.points = points.reshape(-1, 3)
grid.dimensions = X.shape[0], X.shape[1], 1

# Побудова кривої L (профіль) в 3D (оскільки L задана в площині XZ, використовуємо y=0)
L_curve_points = np.column_stack((L_x_vals, np.zeros_like(L_x_vals), L_z_vals))

# Визначаємо u0 – точку, в якій x(u) досягає максимального значення (x0)
u0 = u_sample[np.argmax(x_sample)]
# Побудова кривої S (базовий переріз) в 3D: беремо S в площині XY та фіксуємо z = z(u0)
S_curve_x = f_xb(v_vals)
S_curve_y = f_yb(v_vals)
S_curve_z = np.full_like(S_curve_x, f_z(u0))
S_curve_points = np.column_stack((S_curve_x, S_curve_y, S_curve_z))

# Створення plotter-а та додавання поверхні і кривих
plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True, color="lightblue", opacity=0.7)
plotter.add_lines(L_curve_points, color="red", width=3, label="Ламана L")
plotter.add_lines(S_curve_points, color="green", width=3, label="Ламана S")
plotter.add_legend()
plotter.show()