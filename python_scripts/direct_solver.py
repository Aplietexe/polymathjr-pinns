"""
Closed-form (least-squares) solver for the power-series Physics-Informed
Neural Network.

    minimise_a  ||P a - f||²  +  w_bc‖Q a - b‖²  +  w_rec‖R a - d‖²

where the matrices P, Q, R collect respectively the PDE residuals at the
collocation points, the boundary/initial conditions and the recurrence
consistency relations.
"""

import math

import numpy as np
import sympy as sp
import torch


def _factorials(n: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return tensor [0!, 1!, ..., n!] of shape (n+1,)."""
    return torch.tensor(
        [math.factorial(i) for i in range(n + 1)], dtype=dtype, device=device
    )


def _poly_derivative_row(
    x: float | torch.Tensor, n_max: int, k: int, fact: torch.Tensor
) -> torch.Tensor:
    r"""Return the row vector v such that (v · a) = y^{(k)}(x).

    Elements (for 0 ≤ j ≤ n_max):
        v_j = 0                         if j < k
        v_j = j! / (j-k)! * x^{j-k}     otherwise
    """

    j_idx = torch.arange(
        0, n_max + 1, dtype=torch.long, device=fact.device
    )  # (n_max+1,)
    if isinstance(x, torch.Tensor):
        x_t = x.to(dtype=fact.dtype)
    else:
        x_t = torch.tensor(x, dtype=fact.dtype, device=fact.device)

    coeff = torch.zeros(n_max + 1, dtype=fact.dtype, device=fact.device)

    mask = j_idx >= k
    j_masked = j_idx[mask]

    j_masked_f = j_masked.to(fact.dtype)

    coeff[mask] = fact[j_masked] / fact[j_masked - k] * (x_t ** (j_masked_f - k))

    return coeff  # shape (n_max+1,)


def _build_ode_system(
    c_list: list[sp.Expr],
    f_expr: sp.Expr,
    N: int,
    xs_collocation: np.ndarray,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (A_ode, y_ode) so that A_ode @ a ≈ y_ode enforces the ODE."""

    m = len(c_list) - 1
    fact = _factorials(N, dtype=dtype, device=device)

    # Pre-compute u^{(k)} rows for all k, then combine with c_k(x)
    rows: list[torch.Tensor] = []
    rhs: list[torch.Tensor] = []

    # Lambdify coefficients and RHS on numpy
    x_sym = sp.symbols("x")
    c_funcs = [sp.lambdify(x_sym, c) for c in c_list]
    f_func = sp.lambdify(x_sym, f_expr)

    c_vals_np = []
    for cf in c_funcs:
        val = cf(xs_collocation)
        # Handle constant coefficients by broadcasting to array shape (C,)
        if np.isscalar(val):
            val_arr = np.full(xs_collocation.shape, val)
        else:
            val_arr = np.asarray(val)
        c_vals_np.append(val_arr)

    f_vals_np = f_func(xs_collocation)  # (C,)
    # Handle scalar f(x) by broadcasting to array shape (C,)
    if np.isscalar(f_vals_np):
        f_vals_np = np.full(xs_collocation.shape, f_vals_np)
    else:
        f_vals_np = np.asarray(f_vals_np)

    # Normalize by sqrt(num_collocation) so that sum of squares equals mean
    scale = 1.0 / math.sqrt(xs_collocation.shape[0])

    for idx, x_i in enumerate(xs_collocation):
        # Build row for residual at x_i
        row_sum = torch.zeros(N + 1, dtype=dtype, device=device)
        for k in range(m + 1):
            deriv_row = _poly_derivative_row(float(x_i), N, k, fact)
            row_sum += (
                torch.as_tensor(c_vals_np[k][idx], dtype=dtype, device=device)
                * deriv_row
            )
        rows.append(row_sum * scale)
        rhs.append(
            torch.tensor(float(f_vals_np[idx]), dtype=dtype, device=device) * scale
        )

    A = torch.stack(rows, dim=0)  # (C, N+1)
    y = torch.stack(rhs, dim=0)  # (C,)
    return A, y


def _build_bc_system(
    bc_tuples: list[tuple[float, int, float]],
    N: int,
    *,
    bc_weight: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (A_bc, y_bc) for the boundary/initial conditions."""

    if not bc_tuples:
        # Return empty matrices with correct number of columns
        return torch.empty((0, N + 1), dtype=dtype, device=device), torch.empty(
            (0,), dtype=dtype, device=device
        )

    fact = _factorials(N, dtype=dtype, device=device)
    scale = math.sqrt(bc_weight)

    rows: list[torch.Tensor] = []
    rhs: list[torch.Tensor] = []

    for x0, k, val in bc_tuples:
        row = _poly_derivative_row(x0, N, k, fact) * scale
        rows.append(row)
        rhs.append(torch.tensor(val, dtype=dtype, device=device) * scale)

    A = torch.stack(rows, dim=0)
    y = torch.stack(rhs, dim=0)
    return A, y


def _build_recurrence_system(
    c_list: list[sp.Expr],
    f_expr: sp.Expr,
    N: int,
    *,
    rec_weight: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear system encoding the analytic recurrence.

    The recurrence gives, for each ℓ ≥ m,
        a_ℓ  −  Φ_ℓ(a_0, …, a_{ℓ-1})  = 0
    which is linear in the coefficients.  Each such equation yields one row.
    """

    if rec_weight == 0.0:
        return (
            torch.empty((0, N + 1), dtype=dtype, device=device),
            torch.empty((0,), dtype=dtype, device=device),
        )

    x = sp.symbols("x")
    m = len(c_list) - 1

    # Evaluate c_m(0)
    c_m0 = float(sp.N(c_list[m].subs(x, 0)))
    if c_m0 == 0:
        raise ValueError("x = 0 is a singular point (c_m(0)=0).")

    # Pre-compute factorials and Maclaurin coefficients
    fact_np = np.array([math.factorial(i) for i in range(N + m + 1)], dtype=float)
    ckj_arr = np.zeros((m + 1, N - m + 1), dtype=float)  # only need j ≤ n ≤ N-m

    for k in range(m + 1):
        for j in range(N - m + 1):
            ckj_arr[k, j] = float(sp.series(c_list[k], x, 0, j + 1).coeff(x, j))

    # RHS series coefficients of f(x)
    bn_arr = np.array(
        [float(sp.series(f_expr, x, 0, n + 1).coeff(x, n)) for n in range(N - m + 1)],
        dtype=float,
    )

    # Pre-compute prefactors  pref_n = n!/(n+m)!
    pref_arr = np.array(
        [math.factorial(n) / math.factorial(n + m) for n in range(N - m + 1)],
        dtype=float,
    )

    rows: list[torch.Tensor] = []
    rhs: list[torch.Tensor] = []
    scale = math.sqrt(rec_weight)

    for n in range(N - m + 1):
        ell = n + m  # coefficient index governed by this recurrence

        row_np = np.zeros(N + 1, dtype=float)
        row_np[ell] = 1.0  # a_ell term

        # Build T = Σ consts * a_prev[idx]
        for k in range(m + 1):
            for j in range(n + 1):
                if k == m and j == 0:
                    continue  # (k=m, j=0) term excluded
                idx = n - j + k
                const = ckj_arr[k, j] * fact_np[n - j + k] / fact_np[n - j]
                row_np[idx] += (pref_arr[n] / c_m0) * const

        y_val = (pref_arr[n] / c_m0) * bn_arr[n]

        rows.append(torch.tensor(row_np, dtype=dtype, device=device) * scale)
        rhs.append(torch.tensor(y_val, dtype=dtype, device=device) * scale)

    A = torch.stack(rows, dim=0)
    y = torch.stack(rhs, dim=0)
    return A, y


def solve_power_series(
    c_list: list[sp.Expr],
    f_expr: sp.Expr,
    bc_tuples: list[tuple[float, int, float]],
    *,
    N: int = 10,
    x_left: float = 0.0,
    x_right: float = 1.0,
    num_collocation: int = 1000,
    bc_weight: float = 100.0,
    recurrence_weight: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the coefficient vector a that minimizes the PINN loss.

    Parameters
    ----------
    c_list, f_expr
        Definitions of the linear ODE.
    bc_tuples
        Iterable of boundary/initial conditions (x0, k, value).
    N
        Truncation order of the Maclaurin series.
    x_left, x_right, num_collocation
        Collocation domain and number of interior points.
    bc_weight, recurrence_weight
        Relative weights of BC and recurrence terms (as in the original
        gradient-based training function).
    dtype, device
        Torch tensor dtype / device.

    Returns
    -------
    torch.Tensor
        Learned coefficient vector a of shape (N+1,) (on CPU).
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collocation grid (uniform)
    xs_coll = np.linspace(x_left, x_right, num_collocation)

    # Build each subsystem
    A_pde, y_pde = _build_ode_system(
        c_list, f_expr, N, xs_coll, dtype=dtype, device=device
    )
    A_bc, y_bc = _build_bc_system(
        bc_tuples, N, bc_weight=bc_weight, dtype=dtype, device=device
    )
    A_rec, y_rec = _build_recurrence_system(
        c_list, f_expr, N, rec_weight=recurrence_weight, dtype=dtype, device=device
    )

    # Stack to one least-squares problem
    A = torch.cat([A_pde, A_bc, A_rec], dim=0)
    y = torch.cat([y_pde, y_bc, y_rec], dim=0)

    # Solve least-squares:  minimise ||A a - y||²
    sol, *_ = torch.linalg.lstsq(A, y, rcond=torch.finfo(dtype).eps)
    print(torch.linalg.norm(A @ sol - y))

    return sol.detach().cpu()


if __name__ == "__main__":
    import sympy as sp
    import torch
    from pinn_evaluate import solve_and_plot

    DTYPE = torch.float64

    # ODE: y'''(x) = cos(π x)
    x = sp.symbols("x")
    c: list[sp.Expr] = [
        sp.Integer(0),
        sp.Integer(0),
        sp.Integer(0),
        sp.Integer(1),
    ]  # y''' term only
    f = sp.cos(sp.pi * x)  # right-hand side f(x) = cos(π x)

    # Boundary / initial conditions (x0, derivative order k, value)
    bcs = [
        (0.0, 0, 0.0),  # y(0) = 0
        (1.0, 0, -1.0),  # y(1) = cos(π) = -1
        (1.0, 1, 1.0),  # y'(1) = 1
    ]

    coeffs_direct = solve_power_series(
        c,
        f,
        bcs,
        N=40,
        recurrence_weight=100.0,
        bc_weight=100.0,
        num_collocation=5000,
        dtype=DTYPE,
    )

    print("Direct-solver coefficients:\n", coeffs_direct)

    # Pre-computed analytic solution for y'''(x) = cos(π x) with given BCs
    analytic_expr = (
        sp.pi * x * (-x + (sp.pi**2) * (2 * x - 3) + 1) - sp.sin(sp.pi * x)
    ) / (sp.pi**3)

    solve_and_plot(
        c,
        f,
        bcs,
        coeffs_direct,
        file_prefix="direct_",
        dtype=DTYPE,
        num_plot=501,
        analytic_expr=analytic_expr,
    )
