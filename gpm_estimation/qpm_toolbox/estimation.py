"""NumPyro model function and prior map for MCMC estimation."""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from qpm_toolbox.kalman_filter import get_dtype
from qpm_toolbox.model import _LARGE_NEG

_PRIOR_MAP = {
    "normal": lambda a: dist.Normal(a.get("loc", 0.0),
                                     jnp.maximum(a.get("scale", 1.0), 1e-7)),
    "beta": lambda a: dist.Beta(jnp.maximum(a.get("concentration1", 1.0), 1e-7),
                                 jnp.maximum(a.get("concentration2", 1.0), 1e-7)),
    "gamma": lambda a: dist.Gamma(jnp.maximum(a.get("concentration", 1.0), 1e-7),
                                   rate=jnp.maximum(a.get("rate", 1.0), 1e-7)),
    "invgamma": lambda a: dist.InverseGamma(
        jnp.maximum(a.get("concentration", 1.0), 1e-7),
        rate=jnp.maximum(a.get("rate", 1.0), 1e-7)),
    "uniform": lambda a: dist.Uniform(a.get("low", 0.0), a.get("high", 1.0)),
    "halfnormal": lambda a: dist.HalfNormal(
        jnp.maximum(a.get("scale", 1.0), 1e-7)),
    "truncnorm": lambda a: dist.TruncatedNormal(
        loc=a.get("loc", 0.0),
        scale=jnp.maximum(a.get("scale", 1.0), 1e-7),
        low=a.get("low", -jnp.inf), high=a.get("high", jnp.inf)),
}


def numpyro_model(model_instance, user_priors, fixed_params,
                  ys, H_obs, init_x, init_P, presample=0,
                  H_t=None, shock_multipliers=None,
                  use_diffuse=False, Pinf=None, ur_mask=None,
                  lyap_iters=60, n_diffuse=10):
    """NumPyro model function for NUTS.

    Args:
        model_instance: DynareModelWithLambdified.
        user_priors: list of prior specification dicts.
        fixed_params: dict of fixed (non-estimated) parameters.
        ys: (T, n_obs) observations (None for prior predictive).
            When H_t is provided, NaN entries should be replaced by 0.
        H_obs: (n_obs, n_obs) measurement noise covariance.
        init_x: (n_aug,) initial state mean (ignored when use_diffuse=True).
        init_P: (n_aug, n_aug) initial state covariance (ignored when
            use_diffuse=True).
        presample: number of initial periods to discard from the
            log-likelihood (see KalmanFilter.log_likelihood).
            Ignored when use_diffuse=True - the exact diffuse LL
            automatically excludes the diffuse-phase observations.
        H_t: optional (T, n_obs, n_obs) per-period observation noise
            for missing-data handling.  Ignored when use_diffuse=True.
        shock_multipliers: optional (T, n_shock) per-period scale
            factors for structural shocks.  Entry [t,j] multiplies
            column j of the impact matrix R at time t.
            Ignored when use_diffuse=True.
        use_diffuse: if True, evaluate the exact diffuse log-likelihood
            (Durbin & Koopman 2012, Section 7.2) instead of the
            approximate DARE-based filter.  ``Pinf`` and ``ur_mask``
            must be supplied.
        Pinf: (n_aug, n_aug) diffuse initial covariance (projector onto
            the unit-root subspace).  Pre-compute once before MCMC with
            ``model.build_diffuse_matrices()``; this array is
            parameter-independent and remains constant throughout sampling.
        ur_mask: (n_aug,) boolean array - True for unit-root states.
            Pre-compute once with ``model.build_diffuse_matrices()``.
            Used to zero-mask the Lyapunov equation for ``Pstar``
            at each MCMC iteration.
        lyap_iters: number of doubling steps in the JAX Lyapunov solver
            for ``Pstar``.  Use ``check_lyap_convergence()`` to determine
            the appropriate value (typically 8-10 for QPM models).
        n_diffuse: static Python int - number of periods for the exact
            diffuse phase in ``log_likelihood_fast``.  Must be >= the
            true diffuse phase length.  Default 10 is ample for QPM models.
    """
    dt = get_dtype()
    params = {}
    estimated_names = {p["name"] for p in user_priors}

    for spec in user_priors:
        name = spec["name"]
        d_name = spec.get("prior", "").lower()
        args = {k: jnp.asarray(v, dtype=dt) for k, v in spec.get("args", {}).items()}
        if d_name not in _PRIOR_MAP:
            raise NotImplementedError(f"Prior '{d_name}' not implemented for '{name}'.")
        params[name] = numpyro.sample(name, _PRIOR_MAP[d_name](args))

    for name, value in fixed_params.items():
        if name not in estimated_names:
            params[name] = jnp.asarray(value, dtype=dt)

    if ys is not None:
        if use_diffuse:
            log_prob = model_instance.log_likelihood_diffuse(
                params, ys, H_obs,
                Pinf=Pinf, Pstar=None, ur_mask=ur_mask,
                lyap_iters=lyap_iters, n_diffuse=n_diffuse)
        else:
            log_prob = model_instance.log_likelihood(
                params, ys, H_obs, init_x, init_P, presample=presample,
                H_t=H_t, shock_multipliers=shock_multipliers)
        safe = jax.lax.cond(jnp.isfinite(log_prob),
                            lambda x: x,
                            lambda _: jnp.array(_LARGE_NEG, dtype=dt),
                            log_prob)
        numpyro.factor("log_likelihood", safe)
