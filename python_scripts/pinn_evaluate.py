from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch


def solve_ode_sympy(
    c_list: list[sp.Expr],
    f_expr: sp.Expr,
    bc_tuples: list[tuple[float, int, float]],
) -> sp.Expr:
    """Solve ODE analytically with SymPy and return the solution expression.

    Parameters
    ----------
    c_list, f_expr, bc_tuples
        ODE description using the same conventions as in `train_power_series_pinn`.

    Returns
    -------
    sp.Expr
        The analytic solution expression y(x).
    """
    x = sp.symbols("x")
    y = sp.Function("y")

    # Build the left-hand side Σ c_k(x) y^{(k)}(x)
    lhs_terms = [c_list[k] * sp.diff(y(x), x, k) for k in range(len(c_list))]  # type: ignore[operator]
    lhs: sp.Expr = sum(lhs_terms)  # type: ignore[arg-type]
    ode_eq = sp.Eq(lhs - f_expr, 0)  # type: ignore[operator]

    # Build initial/boundary condition dictionary for dsolve
    ics: dict[sp.Expr, float] = {}
    for x0, k, val in bc_tuples:
        if k == 0:
            ics[y(x0)] = val  # type: ignore[index]
        else:
            ics[sp.diff(y(x), x, k).subs(x, x0)] = val  # type: ignore[index]

    try:
        sol_raw = sp.dsolve(ode_eq, y(x), ics=ics)  # type: ignore[arg-type]

        # dsolve may return either an Equality or a list thereof.
        if isinstance(sol_raw, (list, tuple)):
            sol_eq = sol_raw[0]
        else:
            sol_eq = sol_raw  # type: ignore[assignment]

        return sol_eq.rhs  # type: ignore[attr-defined]
    except Exception as err:
        raise RuntimeError(
            "SymPy failed to solve the ODE with the provided boundary conditions"
        ) from err


def solve_and_plot(
    c_list: list[sp.Expr],
    f_expr: sp.Expr,
    bc_tuples: list[tuple[float, int, float]],
    coeffs: np.ndarray | torch.Tensor,
    *,
    x_left: float = 0.0,
    x_right: float = 1.0,
    num_plot: int = 101,
    out_dir: str | Path = "data",
    file_prefix: str = "",
    figsize: tuple[int, int] = (6, 4),
    dtype: torch.dtype | None = None,
    analytic_expr: sp.Expr | None = None,
    vectorize: bool = False,
) -> None:
    """Solve the ODE analytically with SymPy and create diagnostic plots.

    Parameters
    ----------
    c_list, f_expr, bc_tuples
        ODE description using the same conventions as in `train_power_series_pinn`.
    coeffs
        Learned coefficient array *(N+1,)* - either NumPy or PyTorch tensor.
    x_left, x_right
        Plotting range.
    num_plot
        Number of sample points for the curves.
    out_dir
        Directory where PNG files will be saved.
    file_prefix
        Optional prefix for the output filenames (handy when comparing runs).
    figsize
        Matplotlib figure size.
    dtype
        PyTorch dtype to use for numpy precision (float32 -> np.float32, float64 -> np.float64).
    analytic_expr
        Optional pre-computed analytic solution expression. If None, will solve the ODE with SymPy.
    vectorize
        Whether to vectorize the lambdified analytic expression.
    """

    # Determine numpy dtype based on torch dtype
    if dtype is None:
        np_dtype = np.float64  # default
    elif dtype == torch.float32:
        np_dtype = np.float32
    elif dtype == torch.float64:
        np_dtype = np.float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Prepare numpy arrays
    if isinstance(coeffs, torch.Tensor):
        coeffs_np = coeffs.detach().cpu().numpy()
    else:
        coeffs_np = np.asarray(coeffs, dtype=np_dtype)

    N = coeffs_np.shape[0] - 1

    # Get analytic solution
    x = sp.symbols("x")

    if analytic_expr is None:
        analytic_expr = solve_ode_sympy(c_list, f_expr, bc_tuples)

    analytic_fn = (
        sp.lambdify(x, analytic_expr)
        if not vectorize
        else np.vectorize(sp.lambdify(x, analytic_expr))
    )

    # Generate evaluation grid and curves
    x_plot = np.linspace(x_left, x_right, num_plot, dtype=np_dtype)
    u_true = analytic_fn(x_plot)

    # Evaluate learned power-series solution
    powers = x_plot[:, None] ** np.arange(0, N + 1, dtype=np_dtype)[None, :]
    u_pred = powers @ coeffs_np

    # Error metrics
    error = np.abs(u_true - u_pred)
    error = np.maximum(error, 1e-20)  # avoid log(0)

    # Compute series once and read off coefficients
    series_expr = sp.series(analytic_expr, x, 0, N + 1).removeO().expand()
    coeff_true = np.array(
        [float(series_expr.coeff(x, k)) for k in range(N + 1)], dtype=np_dtype
    )
    coeff_error = np.abs(coeff_true - coeffs_np)
    coeff_error = np.maximum(coeff_error, 1e-20)

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # Plot 1: Solution comparison
    plt.figure(figsize=figsize)
    plt.plot(x_plot, u_pred, "b-", lw=2, label="PINN power-series")
    plt.plot(x_plot, u_true, "k--", lw=2, label="Analytic (SymPy)")
    plt.title("ODE solution comparison")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}solution_comparison.png", dpi=200)
    plt.close()

    # Plot 2: Absolute error
    plt.figure(figsize=figsize)
    plt.semilogy(x_plot, error, lw=2)
    plt.title("Absolute error of power-series solution")
    plt.xlabel("x")
    plt.ylabel("Error (log scale)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}error.png", dpi=200)
    plt.close()

    # Plot 3: Coefficient error
    plt.figure(figsize=figsize)
    plt.semilogy(np.arange(N + 1), coeff_error, "o-")
    plt.title("Error in learned coefficients")
    plt.xlabel("Coefficient index")
    plt.ylabel("Absolute error (log scale)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}coefficient_error.png", dpi=200)
    plt.close()

    print("\nPlots saved to", out_dir.as_posix())
    print("-", f"{file_prefix}solution_comparison.png")
    print("-", f"{file_prefix}error.png")
    print("-", f"{file_prefix}coefficient_error.png")
