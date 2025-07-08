import sympy as sp
import torch
from direct_solver import solve_power_series
from pinn_evaluate import solve_and_plot

DTYPE = torch.float64

print("\n1. Legendre equation (l=5)")
print("-" * 40)

x = sp.symbols("x")

# (1 - x²) y'' - 2x y' + l(l+1) y = 0 with l = 5
l = 5
c_legendre = [
    l * (l + 1),  # coefficient of y
    -2 * x,  # coefficient of y'
    1 - x**2,  # coefficient of y''
]
f_legendre = sp.Integer(0)  # homogeneous

bcs_legendre = [
    (0.0, 0, 0.0),  # P_5(0) = 0
    (1.0, 0, 1.0),  # P_5(1) = 1
]

coeffs_legendre = solve_power_series(
    c_legendre,
    f_legendre,
    bcs_legendre,
    N=30,
    num_collocation=5000,
    bc_weight=100.0,
    recurrence_weight=100.0,
    dtype=DTYPE,
    x_left=-0.9,  # Avoid singularity at x = +-1
    x_right=0.9,
)

P5 = (63 * x**5 - 70 * x**3 + 15 * x) / 8

solve_and_plot(
    c_legendre,
    f_legendre,
    bcs_legendre,
    coeffs_legendre,
    file_prefix="legendre_",
    dtype=DTYPE,
    x_left=-0.9,
    x_right=0.9,
    analytic_expr=P5,
)

print("\n2. Airy equation")
print("-" * 40)

# y'' - x y = 0
c_airy = [
    -x,  # coefficient of y
    sp.Integer(0),  # coefficient of y'
    sp.Integer(1),  # coefficient of y'' (c_2(0) = 1 ✓)
]
f_airy = sp.Integer(0)

# Boundary conditions for Airy function Ai(x)
# Ai(0) ≈ 0.35502805388781723926
# Ai'(0) ≈ -0.25881940379280679840
Ai0 = 0.35502805388781723926
Aip0 = -0.25881940379280679840

bcs_airy = [
    (0.0, 0, Ai0),  # y(0) = Ai(0)
    (0.0, 1, Aip0),  # y'(0) = Ai'(0)
]

coeffs_airy = solve_power_series(
    c_airy,
    f_airy,
    bcs_airy,
    N=40,
    num_collocation=5000,
    bc_weight=100.0,
    recurrence_weight=75.0,
    dtype=DTYPE,
    x_left=-2.0,
    x_right=2.0,
)

solve_and_plot(
    c_airy,
    f_airy,
    bcs_airy,
    coeffs_airy,
    file_prefix="airy_",
    dtype=DTYPE,
    x_left=-2.0,
    x_right=2.0,
)

print("\n3. Hermite equation (n=6)")
print("-" * 40)

# y'' - 2x y' + 2n y = 0 with n = 6
n = 6
c_hermite = [
    2 * n,  # coefficient of y
    -2 * x,  # coefficient of y'
    sp.Integer(1),  # coefficient of y''
]
f_hermite = sp.Integer(0)

bcs_hermite = [
    (0.0, 0, -120.0),  # H_6(0) = -120
    (0.0, 1, 0.0),  # H'_6(0) = 0
]

coeffs_hermite = solve_power_series(
    c_hermite,
    f_hermite,
    bcs_hermite,
    N=30,
    num_collocation=5000,
    bc_weight=100.0,
    recurrence_weight=50.0,
    dtype=DTYPE,
    x_left=-2.0,
    x_right=2.0,
)

print("✓ Hermite equation solved successfully")

# Known solution: H_6(x) = 64x^6 - 480x^4 + 720x^2 - 120
H6 = 64 * x**6 - 480 * x**4 + 720 * x**2 - 120

solve_and_plot(
    c_hermite,
    f_hermite,
    bcs_hermite,
    coeffs_hermite,
    file_prefix="hermite_",
    dtype=DTYPE,
    x_left=-2.0,
    x_right=2.0,
    analytic_expr=H6,
)

print("\n4. Beam equation")
print("-" * 40)

# (1 + 0.3x²) y'''' = sin(2x)
c_beam = [
    sp.Integer(0),  # y
    sp.Integer(0),  # y'
    sp.Integer(0),  # y''
    sp.Integer(0),  # y'''
    1 + sp.Rational(3, 10) * x**2,  # y'''' (c_4(0) = 1 ✓)
]
f_beam = sp.sin(2 * x)

bcs_beam = [
    (0.0, 0, 0.0),  # y(0) = 0
    (0.0, 1, 0.0),  # y'(0) = 0
    (1.0, 2, 0.0),  # y''(1) = 0
    (1.0, 3, 0.0),  # y'''(1) = 0
]

coeffs_beam = solve_power_series(
    c_beam,
    f_beam,
    bcs_beam,
    N=40,
    num_collocation=8000,
    bc_weight=200.0,
    recurrence_weight=100.0,
    dtype=DTYPE,
)

print("✓ Beam equation solved successfully")

A = 0.25358522562474695
B = 0.6296763877330811
t = sp.symbols("t")
integrand = (x - t) ** 3 * sp.sin(2 * t) / (1 + 0.3 * t**2)
analytic_expr = (
    (B - A) / 2 * x**2 - B / 6 * x**3 + sp.Integral(integrand, (t, 0, x)) / 6  # type: ignore[operator]
)

solve_and_plot(
    c_beam,
    f_beam,
    bcs_beam,
    coeffs_beam,
    file_prefix="beam_",
    dtype=DTYPE,
    num_plot=501,
    analytic_expr=analytic_expr,
    vectorize=True,
)

print("Testing complete. Check the 'data/' directory for plots.")
