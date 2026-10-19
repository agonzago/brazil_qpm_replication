"""Constrained Hodrick-Prescott trend filter.

Python port of MATLAB ``hp_filter_constrained.m`` (Andres Gonzalez,
``+gpm/+modeling`` toolbox).  The classical HP trend solves

    min_{tau} ||y - tau||^2 + lambda * ||D tau||^2

where ``D`` is the second-difference operator.  This module additionally
admits linear *equality* constraints on the trend,

    C tau = d                       (general form)
    tau[idx]            = v         (trend anchors)
    y[idx] - tau[idx]   = x         (cycle anchors  <=>  tau[idx] = y[idx] - x)

solved in closed form via the Schur complement (no QP solver needed).

The unconstrained problem reduces to ``(I + lambda * D'D) tau = y``; the
constrained problem augments this with Lagrange multipliers ``mu`` that
satisfy ``S mu = C tau0 - d`` with ``S = C A^{-1} C'`` and recovers
``tau = tau0 - A^{-1} C' mu``.  ``A`` is symmetric positive-definite and
pentadiagonal, so the banded sparse solve is O(T).

Typical usage:

    from qpm_toolbox.hp_filter import hp_filter
    cycle, trend = hp_filter(y, lam=1600)                     # unconstrained
    cycle, trend = hp_filter(y, lam=1600,
                             cycle_idx=[0, len(y)-1],
                             cycle_val=[0.0, 0.0])           # endpoints pinned
    cycle, trend = hp_filter(y, lam=1600,
                             trend_idx=[0], trend_val=[y[0]]) # left anchor on level

References
----------
- Hodrick & Prescott (1997), J. Money Credit Banking 29 (1): 1--16.
- Boyd & Vandenberghe (2004), *Convex Optimization*, Section 10.1.1
  (equality-constrained quadratic programs via Schur complement).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


__all__ = ["hp_filter", "hp_trend"]


def _second_diff_matrix(T: int) -> sp.csc_matrix:
    """(T-2, T) sparse banded second-difference operator.

    Row r (0 <= r <= T-3) contains [1, -2, 1] at columns r, r+1, r+2,
    so ``(D tau)[r] = tau[r] - 2*tau[r+1] + tau[r+2]``.
    """
    if T < 3:
        return sp.csc_matrix((0, T))
    m = T - 2
    diagonals = [
        np.ones(m),                     # column offset 0
        -2.0 * np.ones(m),              # column offset 1
        np.ones(m),                     # column offset 2
    ]
    return sp.diags(diagonals, offsets=[0, 1, 2], shape=(m, T), format="csc")


def _build_trend_anchors(T: int,
                         idx: Sequence[int],
                         val: Sequence[float]) -> tuple[sp.csc_matrix, np.ndarray]:
    """Build (C, d) for constraints tau[idx[i]] = val[i]."""
    idx = np.asarray(idx, dtype=int).ravel()
    val = np.asarray(val, dtype=float).ravel()
    if idx.shape != val.shape:
        raise ValueError("idx and val must have the same length")
    if np.any((idx < 0) | (idx >= T)):
        raise ValueError(f"all indices must lie in [0, {T - 1}]")
    m = idx.size
    rows = np.arange(m)
    data = np.ones(m)
    C = sp.csc_matrix((data, (rows, idx)), shape=(m, T))
    return C, val


def hp_filter(
    y: np.ndarray,
    lam: float = 1600.0,
    *,
    C: np.ndarray | sp.spmatrix | None = None,
    d: np.ndarray | None = None,
    trend_idx: Sequence[int] | None = None,
    trend_val: Sequence[float] | None = None,
    cycle_idx: Sequence[int] | None = None,
    cycle_val: Sequence[float] | None = None,
    return_multipliers: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constrained Hodrick-Prescott trend-cycle decomposition.

    Parameters
    ----------
    y :
        Data vector (T,).  NaN entries are not supported - drop or impute
        them upstream.
    lam :
        Smoothing parameter.  Standard quarterly value is 1600.  Use 6.25
        for annual and 129600 for monthly (Ravn-Uhlig).
    C, d :
        Optional general linear equality constraints ``C tau = d``.
        ``C`` may be dense ``ndarray`` or any ``scipy.sparse`` matrix of
        shape ``(m, T)``; ``d`` is ``(m,)``.
    trend_idx, trend_val :
        Time indices and target values pinning ``tau[trend_idx] = trend_val``.
    cycle_idx, cycle_val :
        Time indices and target values for the cycle.  Internally this
        becomes ``tau[cycle_idx] = y[cycle_idx] - cycle_val``.  Set
        ``cycle_val = 0`` to force cycle to zero (useful at endpoints).
    return_multipliers :
        If True, also return the Lagrange multipliers ``mu`` (shape ``(m,)``,
        empty if no constraints).

    Returns
    -------
    cycle :
        ``(T,)`` cycle = y - trend.
    trend :
        ``(T,)`` HP trend.
    multipliers :
        ``(m,)`` Lagrange multipliers (only when ``return_multipliers=True``).

    Notes
    -----
    Anchor types may be combined; they will be stacked into a single
    ``C`` matrix.  Duplicate constraints at the same time index are
    permitted but may produce a rank-deficient ``S``; the routine then
    falls back to a Tikhonov-regularised solve.
    """
    y = np.asarray(y, dtype=float).ravel()
    T = y.size
    if not (np.isscalar(lam) and np.isfinite(lam) and lam >= 0):
        raise ValueError("lam must be a finite non-negative scalar")

    # Build A = I + lam * D'D  (symmetric, positive-definite, banded).
    I = sp.eye(T, format="csc")
    if lam == 0:
        A = I
    else:
        D = _second_diff_matrix(T)
        A = (I + lam * (D.T @ D)).tocsc()

    # Unconstrained solve.
    tau0 = spla.spsolve(A, y)

    # Assemble constraint stack (if any).
    constraints: list[tuple[sp.csc_matrix, np.ndarray]] = []
    if C is not None:
        if d is None:
            raise ValueError("C provided without d")
        C_arr = C if sp.issparse(C) else sp.csc_matrix(np.asarray(C, dtype=float))
        d_arr = np.asarray(d, dtype=float).ravel()
        if C_arr.shape != (d_arr.size, T):
            raise ValueError(f"C shape {C_arr.shape} inconsistent with "
                             f"d size {d_arr.size} and T={T}")
        constraints.append((C_arr.tocsc(), d_arr))
    if trend_idx is not None and trend_val is not None:
        constraints.append(_build_trend_anchors(T, trend_idx, trend_val))
    if cycle_idx is not None and cycle_val is not None:
        idx_arr = np.asarray(cycle_idx, dtype=int).ravel()
        cyc_arr = np.asarray(cycle_val, dtype=float).ravel()
        if idx_arr.shape != cyc_arr.shape:
            raise ValueError("cycle_idx and cycle_val must have the same length")
        constraints.append(
            _build_trend_anchors(T, idx_arr, y[idx_arr] - cyc_arr))

    if not constraints:
        trend = tau0
        cycle = y - trend
        if return_multipliers:
            return cycle, trend, np.empty(0)
        return cycle, trend

    C_full = sp.vstack([c for c, _ in constraints], format="csc")
    d_full = np.concatenate([dd for _, dd in constraints])
    m = C_full.shape[0]

    # Schur complement: solve A Z = C^T for Z (T x m), then S = C Z (m x m).
    Z = spla.spsolve(A, C_full.T.toarray())
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    S = C_full @ Z                # (m, m), dense
    r = C_full @ tau0 - d_full    # (m,)

    try:
        mu = np.linalg.solve(S, r)
    except np.linalg.LinAlgError:
        # Singular S -- add a tiny ridge.  Happens for redundant constraints.
        mu = np.linalg.solve(S + 1e-12 * np.eye(m), r)

    trend = tau0 - Z @ mu
    cycle = y - trend
    if return_multipliers:
        return cycle, trend, mu
    return cycle, trend


def hp_trend(y: np.ndarray, lam: float = 1600.0, **kwargs) -> np.ndarray:
    """Convenience wrapper returning just the trend component."""
    _, trend = hp_filter(y, lam=lam, **kwargs)
    return trend
