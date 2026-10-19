"""Impulse response functions for classical and Bayesian VARs.

Identification schemes:

1. **Cholesky (recursive)**: Lower-triangular Cholesky decomposition
   of the residual covariance matrix.  Functions: ``irf_cholesky``,
   ``irf_bvar``.

2. **Sign restrictions** (Uhlig 2005): Identifies a single structural
   shock by drawing random impulse vectors on the K-sphere and
   retaining only those whose IRFs satisfy user-specified sign
   constraints at given horizons.  Function: ``irf_bvar_sign_restrictions``.

3. **Proxy / external instrument** (Mertens & Ravn 2013, Stock & Watson
   2012): Identifies a single structural shock using an external
   instrument (proxy) that is correlated with the target shock but
   uncorrelated with all other structural shocks.
   Function: ``irf_bvar_proxy``.

References:
    Uhlig (2005) "What are the effects of monetary policy on output?
    Results from an agnostic identification procedure", *Journal of
    Monetary Economics* 52(2).

    Mertens & Ravn (2013) "The dynamic effects of personal and
    corporate income tax changes in the United States", *American
    Economic Review* 103(4).

    Stock & Watson (2012) "Disentangling the channels of the 2007-2009
    recession", *Brookings Papers on Economic Activity*.

    Gertler & Karadi (2015) "Monetary policy surprises, credit costs,
    and economic activity", *American Economic Journal: Macroeconomics* 7(1).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .core import cholesky_identification, companion_matrix
from .types import BVARDraws, BVARIRFResult, IRFResult, VARModel


# ======================================================================== #
#  Private: single-draw IRF computation
# ======================================================================== #

def _irf_single(
    A: np.ndarray,
    Sigma: np.ndarray,
    K: int,
    p: int,
    horizon: int,
    shock_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Compute Cholesky-identified IRFs for one set of coefficients.

    Args:
        A: (Kp, K) lag coefficients.
        Sigma: (K, K) residual covariance.
        K: Number of variables.
        p: Lag order.
        horizon: Number of periods (IRF computed at h = 0, 1, ..., horizon).
        shock_indices: Which structural shocks to compute (0-based).
                       Default: all shocks [0, 1, ..., K-1].

    Returns:
        IR: (K, n_shocks, horizon+1) array.
            IR[resp, shock_idx, h] = response of variable *resp* to
            structural shock *shock_idx* at horizon *h*.
    """
    if shock_indices is None:
        shock_indices = list(range(K))
    n_shocks = len(shock_indices)

    C = cholesky_identification(Sigma)       # (K, K) lower triangular
    F = companion_matrix(A, K, p)            # (Kp, Kp)

    IR = np.zeros((K, n_shocks, horizon + 1))

    for si, s in enumerate(shock_indices):
        # Unit structural shock
        eps = np.zeros(K)
        eps[s] = 1.0
        u0 = C @ eps                         # reduced-form impact

        # Companion-form state: [Y_t; Y_{t-1}; ... ; Y_{t-p+1}]
        state = np.zeros(K * p)
        state[:K] = u0
        IR[:, si, 0] = u0

        for h in range(1, horizon + 1):
            state = F @ state
            IR[:, si, h] = state[:K]

    return IR


# ======================================================================== #
#  Classical IRFs
# ======================================================================== #

def irf_cholesky(
    model: VARModel,
    horizon: int,
    shock_indices: Optional[List[int]] = None,
) -> IRFResult:
    """Cholesky-identified impulse responses from a classical VAR.

    Args:
        model: Estimated VARModel (from :func:`classical.estimate_ols`
               or :func:`classical.estimate_stable`).
        horizon: Number of periods ahead (h = 0 .. horizon).
        shock_indices: 0-based indices of shocks to compute (default: all).

    Returns:
        IRFResult with *irf* array of shape (horizon+1, K_resp, K_shock).
    """
    K, p = model.K, model.p
    if shock_indices is None:
        shock_indices = list(range(K))

    IR = _irf_single(model.A, model.Sigma, K, p, horizon, shock_indices)
    # Transpose to (H+1, K_resp, K_shock)
    IR = IR.transpose(2, 0, 1)

    shock_names = None
    if model.var_names is not None:
        shock_names = [model.var_names[s] for s in shock_indices]

    return IRFResult(
        irf=IR,
        var_names=model.var_names,
        shock_names=shock_names,
        horizon=horizon,
    )


# ======================================================================== #
#  Bayesian IRFs
# ======================================================================== #

def irf_bvar(
    draws: BVARDraws,
    horizon: int,
    shock_indices: Optional[List[int]] = None,
) -> BVARIRFResult:
    """Cholesky-identified IRFs over BVAR posterior draws.

    Computes IRFs for each posterior draw of (B, Sigma) and returns
    the full distribution plus percentile summaries.

    Args:
        draws: BVARDraws from :func:`bayesian.estimate_bvar`.
        horizon: Number of periods ahead.
        shock_indices: 0-based shock indices (default: all).

    Returns:
        BVARIRFResult with draw-level IRFs and median / 16th / 84th
        percentile summaries.
    """
    K, p = draws.K, draws.p
    if shock_indices is None:
        shock_indices = list(range(K))
    n_shocks = len(shock_indices)

    all_ir = np.zeros((draws.n_draws, K, n_shocks, horizon + 1))

    offset = int(draws.constant)
    Kp = K * p
    for i in range(draws.n_draws):
        B_i = draws.B_draws[i]
        Sigma_i = draws.Sigma_draws[i]
        A_i = B_i[offset:offset + Kp, :]  # lag block only (excludes exog)
        all_ir[i] = _irf_single(A_i, Sigma_i, K, p, horizon, shock_indices)

    # Percentiles over draws (axis 0)
    median = np.percentile(all_ir, 50, axis=0)
    p16 = np.percentile(all_ir, 16, axis=0)
    p84 = np.percentile(all_ir, 84, axis=0)

    shock_names = None
    if draws.var_names is not None:
        shock_names = [draws.var_names[s] for s in shock_indices]

    return BVARIRFResult(
        irf=all_ir,
        median=median, p16=p16, p84=p84,
        var_names=draws.var_names,
        shock_names=shock_names,
        horizon=horizon,
    )


# ======================================================================== #
#  IRF normalisation
# ======================================================================== #

def normalise_irf(
    irf_result: BVARIRFResult,
    impulse_var: str,
    response_var: str,
    target_pp: float = 1.0,
    at_horizon: int = 0,
) -> BVARIRFResult:
    """Re-scale BVAR IRFs so that *response_var* moves by *target_pp*
    at *at_horizon* in response to the *impulse_var* shock (median).

    Useful for normalising to a "100 bp monetary-policy shock".

    Args:
        irf_result: BVARIRFResult to normalise.
        impulse_var: Name of the impulse variable (must be in shock_names).
        response_var: Name of the response variable (must be in var_names).
        target_pp: Desired size of *response_var* at *at_horizon*.
        at_horizon: Horizon at which to normalise.

    Returns:
        New BVARIRFResult with rescaled IRFs.
    """
    if irf_result.var_names is None or irf_result.shock_names is None:
        raise ValueError("Variable names are required for normalisation.")

    si = irf_result.shock_names.index(impulse_var)
    ri = irf_result.var_names.index(response_var)

    # Median response at the normalisation horizon
    ref = irf_result.median[ri, si, at_horizon]
    if abs(ref) < 1e-12:
        raise ValueError(
            f"Median response of {response_var} to {impulse_var} at h={at_horizon} "
            f"is near zero ({ref:.2e}); cannot normalise.")

    scale = target_pp / ref

    return BVARIRFResult(
        irf=irf_result.irf * scale,
        median=irf_result.median * scale,
        p16=irf_result.p16 * scale,
        p84=irf_result.p84 * scale,
        var_names=irf_result.var_names,
        shock_names=irf_result.shock_names,
        horizon=irf_result.horizon,
    )


# ======================================================================== #
#  Sign-restriction identification (Uhlig 2005)
# ======================================================================== #

@dataclass
class SignRestriction:
    """A sign restriction on an impulse response function.

    Constrains the response of *response* variable to the identified
    shock to have a given *sign* at each horizon in *horizons*.

    Attributes:
        response: Name of the response variable (must appear in var_names).
        sign: ``'+'`` (non-negative) or ``'-'`` (non-positive).
        horizons: List of horizons at which the constraint must hold.
                  For example, ``[0]`` for impact only, ``[0, 1, 2, 3]``
                  for the first year at quarterly frequency.
    """
    response: str
    sign: str       # '+' or '-'
    horizons: List[int]


def _irf_from_impulse(
    impulse: np.ndarray,
    F: np.ndarray,
    K: int,
    p: int,
    horizon: int,
) -> np.ndarray:
    """Compute the IRF path from an arbitrary reduced-form impulse vector.

    Unlike ``_irf_single`` (which uses the Cholesky factor), this function
    takes an arbitrary K-dimensional impulse vector and propagates it
    through the companion form.

    Args:
        impulse: (K,) impact effect on the K variables.
        F: (Kp, Kp) companion matrix.
        K, p: VAR dimensions.
        horizon: Number of periods (h = 0, 1, ..., horizon).

    Returns:
        ir: (K, horizon+1) response path.
    """
    ir = np.zeros((K, horizon + 1))
    state = np.zeros(K * p)
    state[:K] = impulse
    ir[:, 0] = impulse

    for h in range(1, horizon + 1):
        state = F @ state
        ir[:, h] = state[:K]

    return ir


def _check_sign_restrictions(
    ir: np.ndarray,
    var_names: List[str],
    restrictions: List[SignRestriction],
) -> bool:
    """Check whether IRFs satisfy all sign restrictions.

    Args:
        ir: (K, H+1) IRF path for the candidate shock.
        var_names: List of K variable names.
        restrictions: Sign restrictions to verify.

    Returns:
        ``True`` if every restriction is satisfied (weakly).
    """
    for restr in restrictions:
        ri = var_names.index(restr.response)
        for h in restr.horizons:
            if h >= ir.shape[1]:
                continue
            val = ir[ri, h]
            if restr.sign == '+' and val < 0:
                return False
            if restr.sign == '-' and val > 0:
                return False
    return True


def irf_bvar_sign_restrictions(
    draws: BVARDraws,
    horizon: int,
    restrictions: List[SignRestriction],
    n_rotations: int = 500,
    seed: Optional[int] = None,
    normalise_var: Optional[str] = None,
    normalise_target: float = 1.0,
    normalise_horizon: int = 0,
    shock_label: Optional[str] = None,
) -> BVARIRFResult:
    """Sign-restriction-identified IRFs over BVAR posterior draws.

    Implements the Uhlig (2005) agnostic identification procedure.
    For each posterior draw (B_i, Sigma_i):

    1. Compute the Cholesky factor P = chol(Sigma_i).
    2. Draw *n_rotations* random unit vectors q on the K-sphere.
    3. Form the candidate impulse vector a = P @ q.
    4. Compute the full IRF path from a.
    5. If the IRFs satisfy all *restrictions*, accept the draw.

    Both q and -q are tested (sign flip), so the effective number
    of candidate rotations is 2 x *n_rotations*.

    The output ``BVARIRFResult`` has ``n_shocks = 1`` (the identified
    shock) and ``n_draws = n_accepted`` (the number of admissible
    draw-rotation pairs).

    Args:
        draws: BVARDraws from :func:`bayesian.estimate_bvar`.
        horizon: Number of periods ahead (h = 0 .. horizon).
        restrictions: List of :class:`SignRestriction` objects
            defining the constraints on the identified shock.
        n_rotations: Number of random unit vectors per posterior draw.
        seed: Random seed for reproducibility.
        normalise_var: If specified, rescale each accepted IRF so that
            the response of this variable equals *normalise_target*
            at *normalise_horizon*.  Useful for normalising to a
            "100 bp monetary-policy shock".
        normalise_target: Target value for normalisation (default 1.0).
        normalise_horizon: Horizon at which to normalise (default 0).
        shock_label: Label for the identified shock in the output
            (default: *normalise_var* or ``"identified_shock"``).

    Returns:
        BVARIRFResult with the distribution of admissible IRFs.
        Shape conventions: ``irf`` has shape
        ``(n_accepted, K, 1, horizon+1)``.

    Raises:
        ValueError: If no rotations satisfy the sign restrictions.

    References:
        Uhlig (2005), Sec. 6.
    """
    K, p = draws.K, draws.p
    var_names = draws.var_names
    if var_names is None:
        raise ValueError("var_names are required for sign restrictions.")

    rng = np.random.default_rng(seed)
    offset = int(draws.constant)
    Kp = K * p

    accepted = []

    for i in range(draws.n_draws):
        B_i = draws.B_draws[i]
        Sigma_i = draws.Sigma_draws[i]
        A_i = B_i[offset:offset + Kp, :]

        P = cholesky_identification(Sigma_i)   # (K, K) lower-triangular
        F = companion_matrix(A_i, K, p)        # (Kp, Kp)

        for _ in range(n_rotations):
            # Random unit vector on the K-sphere (uniform Haar)
            q = rng.standard_normal(K)
            q /= np.linalg.norm(q)

            impulse = P @ q                    # (K,) candidate impact
            ir = _irf_from_impulse(impulse, F, K, p, horizon)

            # Try both orientations (q and -q)
            for sign_flip in (1.0, -1.0):
                ir_try = ir * sign_flip
                if not _check_sign_restrictions(ir_try, var_names,
                                                restrictions):
                    continue

                # Normalise if requested
                if normalise_var is not None:
                    ni = var_names.index(normalise_var)
                    ref = ir_try[ni, normalise_horizon]
                    if abs(ref) < 1e-12:
                        continue
                    ir_try = ir_try * (normalise_target / ref)

                accepted.append(ir_try)
                break  # accept one orientation per q, avoid double-counting

    n_total = draws.n_draws * n_rotations * 2
    n_accepted = len(accepted)
    if n_accepted == 0:
        raise ValueError(
            f"No rotations satisfied the sign restrictions "
            f"({n_total} candidates tested).  Consider relaxing "
            f"the restrictions or increasing n_rotations.")

    pct = 100 * n_accepted / (draws.n_draws * n_rotations)
    print(f"  Sign restrictions: {n_accepted:,} accepted / "
          f"{draws.n_draws * n_rotations:,} drawsxrotations "
          f"({pct:.1f}%)")

    # Stack into (n_accepted, K, 1, H+1) to match BVARIRFResult layout
    all_ir = np.array(accepted)[:, :, np.newaxis, :]

    median = np.percentile(all_ir, 50, axis=0)
    p16 = np.percentile(all_ir, 16, axis=0)
    p84 = np.percentile(all_ir, 84, axis=0)

    if shock_label is None:
        shock_label = normalise_var if normalise_var else "identified_shock"

    return BVARIRFResult(
        irf=all_ir,
        median=median, p16=p16, p84=p84,
        var_names=var_names,
        shock_names=[shock_label],
        horizon=horizon,
    )


# ======================================================================== #
#  Proxy / external-instrument identification (Mertens & Ravn 2013)
# ======================================================================== #

@dataclass
class ProxyVARDiagnostics:
    """Diagnostics for the proxy VAR first-stage regression.

    Attributes:
        f_stat_median: Median first-stage F-statistic across BVAR draws.
        f_stat_p16: 16th percentile of first-stage F.
        f_stat_p84: 84th percentile of first-stage F.
        f_stats: (n_draws,) array of all first-stage F-statistics.
        r2_median: Median first-stage R-squared.
        n_obs_instrument: Number of observations with valid instrument.
        n_obs_total: Total effective sample size (VAR residuals).
        instrument_coverage: Fraction of VAR sample covered by instrument.
    """
    f_stat_median: float
    f_stat_p16: float
    f_stat_p84: float
    f_stats: np.ndarray
    r2_median: float
    n_obs_instrument: int
    n_obs_total: int
    instrument_coverage: float


def irf_bvar_proxy(
    draws: BVARDraws,
    Y: np.ndarray,
    X: np.ndarray,
    instrument: np.ndarray,
    horizon: int,
    policy_var: str,
    normalise_target: float = 1.0,
    shock_label: Optional[str] = None,
    restrictions: Optional[List[SignRestriction]] = None,
) -> tuple:
    """Proxy-identified IRFs over BVAR posterior draws.

    Implements the Mertens & Ravn (2013) / Stock & Watson (2012)
    external-instrument identification.  For each posterior draw
    (B_i, Sigma_i):

    1. Compute reduced-form residuals: uhat = Y - X @ B_i.
    2. Restrict to observations where the instrument is not NaN.
    3. Regress each residual series on the instrument (OLS):
       b_k = Cov(uhat_k, z) / Var(z).
    4. The K-vector b is proportional to the first column of the
       structural impact matrix H.
    5. Normalise so that b[policy_idx] = normalise_target.
    6. Propagate through the companion form to get the full IRF.
    7. If *restrictions* are provided, discard draws whose IRFs
       violate the sign constraints (proxy + sign filter).

    The first-stage F-statistic (regression of the policy-variable
    residual on z) is reported for instrument-relevance testing.
    Staiger & Stock (1997) recommend F > 10.

    Args:
        draws: BVARDraws from :func:`bayesian.estimate_bvar`.
        Y: (T, K) response matrix (from :func:`core.build_design_matrices`).
        X: (T, k) regressor matrix (from :func:`core.build_design_matrices`).
        instrument: (T,) external instrument.  May contain NaNs for
            periods outside the instrument's coverage; these observations
            are excluded from the identification regressions.
        horizon: Number of periods ahead (h = 0 .. horizon).
        policy_var: Name of the policy variable whose shock is
            identified (must appear in draws.var_names).
        normalise_target: Target impact response of *policy_var*
            (default 1.0 = 100 bp if the variable is in percentage
            points).
        shock_label: Label for the identified shock (default:
            ``"proxy_<policy_var>"``).
        restrictions: Optional list of :class:`SignRestriction` objects.
            When provided, proxy-identified IRFs that violate any
            restriction are discarded.  This combines the external-
            instrument direction with sign-restriction filtering,
            keeping only economically sensible draws.

    Returns:
        (irf_result, diagnostics):
          - irf_result: BVARIRFResult with n_shocks = 1.
            Shape: ``irf`` is ``(n_accepted, K, 1, horizon+1)``
            where n_accepted <= n_draws.
          - diagnostics: ProxyVARDiagnostics with first-stage stats.

    Raises:
        ValueError: If policy_var not found, instrument has
            insufficient valid observations, or no draws survive
            sign filtering.

    References:
        Mertens & Ravn (2013), *AER* 103(4).
        Stock & Watson (2012), *BPEA*.
        Gertler & Karadi (2015), *AEJ: Macro* 7(1).
    """
    K, p = draws.K, draws.p
    var_names = draws.var_names
    if var_names is None:
        raise ValueError("var_names are required for proxy identification.")

    policy_idx = var_names.index(policy_var)

    instrument = np.asarray(instrument, dtype=np.float64)
    T = Y.shape[0]
    if instrument.shape[0] != T:
        raise ValueError(
            f"instrument has {instrument.shape[0]} obs, expected {T} "
            f"(same as Y).")

    # Valid observations: instrument is not NaN
    valid = ~np.isnan(instrument)
    n_valid = int(valid.sum())
    if n_valid < K + 2:
        raise ValueError(
            f"Only {n_valid} valid instrument observations; need at "
            f"least K+2 = {K + 2}.")

    z = instrument[valid]           # (n_valid,)
    z_dm = z - z.mean()             # demeaned
    z_var = z_dm @ z_dm / n_valid   # Var(z)

    if shock_label is None:
        shock_label = f"proxy_{policy_var}"

    offset = int(draws.constant)
    Kp = K * p

    all_ir = []
    f_stats_all = np.zeros(draws.n_draws)
    n_sign_rejected = 0

    for i in range(draws.n_draws):
        B_i = draws.B_draws[i]
        A_i = B_i[offset:offset + Kp, :]

        # Reduced-form residuals for this draw
        resid = Y - X @ B_i                     # (T, K)
        resid_valid = resid[valid, :]            # (n_valid, K)
        resid_dm = resid_valid - resid_valid.mean(axis=0)

        # Impact vector: b_k = Cov(uhat_k, z) / Var(z) for each k
        cov_uz = (resid_dm.T @ z_dm) / n_valid  # (K,)
        impact = cov_uz / z_var                  # (K,)

        # First-stage F: regression of policy residual on z
        u_pol = resid_dm[:, policy_idx]
        ss_total = u_pol @ u_pol
        fitted = impact[policy_idx] * z_dm
        ss_resid = (u_pol - fitted) @ (u_pol - fitted)
        if ss_resid > 0:
            f_stats_all[i] = ((ss_total - ss_resid) / 1) / (ss_resid / (n_valid - 2))
        else:
            f_stats_all[i] = np.inf

        # Normalise impact so policy_var response = normalise_target
        ref = impact[policy_idx]
        if abs(ref) < 1e-12:
            n_sign_rejected += 1
            continue
        impact = impact * (normalise_target / ref)

        # Propagate through companion form
        F = companion_matrix(A_i, K, p)
        ir = _irf_from_impulse(impact, F, K, p, horizon)

        # Sign-restriction filter (if provided)
        if restrictions is not None:
            if not _check_sign_restrictions(ir, var_names, restrictions):
                n_sign_rejected += 1
                continue

        all_ir.append(ir)

    n_accepted = len(all_ir)
    if n_accepted == 0:
        raise ValueError(
            f"No draws survived proxy identification"
            + (" + sign filtering" if restrictions else "")
            + f" ({draws.n_draws} tested)."
            + (" Consider relaxing the sign restrictions."
               if restrictions else ""))

    # Stack into (n_accepted, K, 1, H+1)
    all_ir = np.array(all_ir)[:, :, np.newaxis, :]

    median = np.percentile(all_ir, 50, axis=0)
    p16 = np.percentile(all_ir, 16, axis=0)
    p84 = np.percentile(all_ir, 84, axis=0)

    # First-stage R^2 (from the F-statistic, computed over all draws)
    f_valid = f_stats_all[f_stats_all > 0]
    r2_arr = f_valid / (f_valid + (n_valid - 2))

    diagnostics = ProxyVARDiagnostics(
        f_stat_median=float(np.median(f_stats_all)),
        f_stat_p16=float(np.percentile(f_stats_all, 16)),
        f_stat_p84=float(np.percentile(f_stats_all, 84)),
        f_stats=f_stats_all,
        r2_median=float(np.median(r2_arr)) if len(r2_arr) > 0 else 0.0,
        n_obs_instrument=n_valid,
        n_obs_total=T,
        instrument_coverage=n_valid / T,
    )

    print(f"  Proxy VAR identification ({shock_label}):")
    print(f"    Instrument obs: {n_valid} / {T} "
          f"({100 * n_valid / T:.1f}% coverage)")
    print(f"    First-stage F:  {diagnostics.f_stat_median:.1f} "
          f"[{diagnostics.f_stat_p16:.1f}, {diagnostics.f_stat_p84:.1f}]")
    print(f"    First-stage R^2: {diagnostics.r2_median:.4f}")
    if diagnostics.f_stat_median < 10:
        print(f"    WARNING: Median F < 10 - weak instrument concern "
              f"(Staiger & Stock 1997).")
    if restrictions is not None:
        pct = 100 * n_accepted / draws.n_draws
        print(f"    Sign filter: {n_accepted} / {draws.n_draws} accepted "
              f"({pct:.1f}%)")

    irf_result = BVARIRFResult(
        irf=all_ir,
        median=median, p16=p16, p84=p84,
        var_names=var_names,
        shock_names=[shock_label],
        horizon=horizon,
    )

    return irf_result, diagnostics
