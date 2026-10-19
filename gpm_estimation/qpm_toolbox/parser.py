# -*- coding: utf-8 -*-
"""
Enhanced Dynare Parser and State-Space Solver (JAX Compatible)

Parses Dynare-like models, solves using JAX-based SDA, builds augmented
state-space, computes IRFs.

Dtype flexibility: import ``get_dtype`` from ``kalman_filter`` or call
``get_dtype()`` here -- both respect ``jax.config.jax_enable_x64``.
"""

import re
import sympy
import numpy as np
from collections import OrderedDict
import copy
import os
import functools
from numpy.linalg import norm
from scipy.linalg import lu_factor, lu_solve, block_diag
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import jax.random
from jax.typing import ArrayLike
from typing import Tuple, Optional, List, Dict, Any
from jax import lax

from qpm_toolbox.kalman_filter import get_dtype


def _jax_eps():
    """Machine epsilon for the active dtype."""
    dt = get_dtype()
    return jnp.finfo(dt).eps

# --- Helper Functions (Mostly unchanged) ---

def plot_irfs(irf_values, var_names, horizon, title="Impulse Responses"):
    """ Simple IRF plotting function """
    # Convert JAX array to NumPy for plotting
    irf_values_np = np.asarray(irf_values)
    var_names_list = list(var_names) # Ensure it's a list

    num_vars = irf_values_np.shape[1]
    if num_vars == 0:
        print(f"No variables to plot for: {title}")
        return

    cols = 4 if num_vars > 9 else (3 if num_vars > 4 else (2 if num_vars > 1 else 1))
    rows = (num_vars + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(min(5*cols, 18), 3*rows), sharex=True, squeeze=False)
    axes = axes.flatten() # Flatten the axes array for easy iteration
    plt.suptitle(title, fontsize=14)
    time = np.arange(horizon)

    for i, var_name in enumerate(var_names_list):
        if i < len(axes): # Check if subplot exists
            ax = axes[i]
            ax.plot(time, irf_values_np[:, i], label=var_name)
            ax.axhline(0, color='black', linewidth=0.7, linestyle=':') # Zero line
            ax.set_title(var_name)
            ax.grid(True, linestyle='--', alpha=0.6)
            # Add x-label only to bottom row plots for clarity
            if (i // cols) == (rows - 1): # Check if it's in the last row
                ax.set_xlabel("Horizon")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    # plt.show() # Remove automatic showing, let the main script decide


def simulate_ssm_data(
    P: ArrayLike,
    R: ArrayLike,
    Omega: ArrayLike,
    T: int,
    key: jax.random.PRNGKey,
    state_init: Optional[ArrayLike] = None,
    measurement_noise_std: float = 0.0,
) -> Tuple[jax.Array, jax.Array]:
    """
    Simulate data from a linear state-space model using JAX:
      s_t = P @ s_{t-1} + R @ e_t
      y_t = Omega @ s_t + eta_t

    [Docstring omitted for brevity; see your original.]

    Returns:
        Tuple[jax.Array, jax.Array]:
          - states: Simulated states (s_1, ..., s_T), shape (T, n_state).
          - observables: Simulated observables (y_1, ..., y_T), shape (T, n_obs).
    """
    P_jax = jnp.asarray(P, dtype=get_dtype())
    R_jax = jnp.asarray(R, dtype=get_dtype())
    Omega_jax = jnp.asarray(Omega, dtype=get_dtype())

    n_state = P_jax.shape[0]
    n_shock = R_jax.shape[1] if R_jax.ndim == 2 and R_jax.shape[1] > 0 else 0
    n_obs = Omega_jax.shape[0] if Omega_jax.ndim == 2 and Omega_jax.shape[0] > 0 else 0

    if P_jax.shape != (n_state, n_state):
        raise ValueError(f"P matrix shape mismatch: expected ({n_state}, {n_state}), got {P_jax.shape}")
    if R_jax.shape != (n_state, n_shock):
        # Allow R to be empty if no shocks
        if not (n_shock == 0 and R_jax.shape in [(n_state,), (n_state, 0)]):
            raise ValueError(f"R matrix shape mismatch: expected ({n_state}, {n_shock}), got {R_jax.shape}")
        elif n_shock == 0: # Ensure R has correct shape even if empty
            R_jax = jnp.zeros((n_state, 0), dtype=get_dtype())

    if Omega_jax.shape != (n_obs, n_state):
        # Allow Omega to be empty if no obs
        if not (n_obs == 0 and Omega_jax.shape in [(0, n_state), (0,)]):
            raise ValueError(f"Omega matrix shape mismatch: expected ({n_obs}, {n_state}), got {Omega_jax.shape}")
        elif n_obs == 0: # Ensure Omega has correct shape even if empty
            Omega_jax = jnp.zeros((0, n_state), dtype=get_dtype())

    key_state, key_measure = jax.random.split(key)

    # Handle initial state
    if state_init is None:
        print(
            "[simulate_ssm_data] Warning: No state_init provided! Initial state s_0 defaults to zeros. "
            "This may be unsuitable for models with trends or non-zero steady states. "
            "Provide a specific state_init for accurate simulation."
        )
        s_previous = jnp.zeros(n_state, dtype=get_dtype())
    else:
        s_previous = jnp.asarray(state_init, dtype=get_dtype())
        if s_previous.shape != (n_state,):
            raise ValueError(f"state_init shape mismatch: expected ({n_state},), got {s_previous.shape}")

    # Draw state and measurement noises for all periods T
    state_shocks = jnp.zeros((T, n_shock), dtype=get_dtype())
    if n_shock > 0:
        state_shocks = jax.random.normal(key_state, shape=(T, n_shock), dtype=get_dtype())

    measurement_noise = jnp.zeros((T, n_obs), dtype=get_dtype())
    if measurement_noise_std > 0.0 and n_obs > 0:
        measurement_noise = jax.random.normal(key_measure, shape=(T, n_obs), dtype=get_dtype()) * measurement_noise_std

    def step(s_prev, inputs):
        """
        Args:
            s_prev: State from the previous period (s_{t-1}).
            inputs: Tuple containing shocks for the current period (e_t, eta_t).
        Returns:
            Tuple: (s_t, (s_t, y_t))
        """
        e_t, eta_t = inputs

        # State evolution: s_t = P @ s_{t-1} + R @ e_t
        s_t = P_jax @ s_prev
        if n_shock > 0:
            s_t += R_jax @ e_t # Add shock impact

        # Observation: y_t = Omega @ s_t + eta_t
        y_t = jnp.zeros(n_obs, dtype=get_dtype()) # Default if no obs
        if n_obs > 0:
            y_t = Omega_jax @ s_t
            if measurement_noise_std > 0.0:
                y_t += eta_t # Add measurement noise

        return s_t, (s_t, y_t)

    # Run the simulation using lax.scan
    _, (states_T, ys_T) = lax.scan(
        step, s_previous, (state_shocks, measurement_noise)
    )

    # states_T will have shape (T, n_state); ys_T will have shape (T, n_obs)
    return states_T, ys_T


def plot_simulation(sim_data, var_names, title="Simulated Data"):
    """Simple simulation plotting function"""
    sim_data_np = np.asarray(sim_data)  # Convert JAX array to NumPy for plotting
    var_names_list = list(var_names)
    num_vars = sim_data_np.shape[1]
    T = sim_data_np.shape[0]

    if num_vars == 0:
        print(f"No variables to plot for: {title}")
        return

    cols = 4 if num_vars > 9 else (3 if num_vars > 4 else (2 if num_vars > 1 else 1))
    rows = (num_vars + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(min(5 * cols, 18), 3 * rows), sharex=True, squeeze=False)
    axes = axes.flatten()
    plt.suptitle(title, fontsize=14)
    time = np.arange(T)

    for i, var_name in enumerate(var_names_list):
        if i < len(axes):
            ax = axes[i]
            ax.plot(time, sim_data_np[:, i], label=var_name)
            ax.set_title(var_name)
            ax.grid(True, linestyle='--', alpha=0.6)
            if (i // cols) == (rows - 1):
                ax.set_xlabel("Time")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_simulation_with_trends_matched(
    sim_observables: np.ndarray,
    obs_var_names: list,                   # ["L_GDP_OBS", ...]
    sim_states: np.ndarray,
    state_var_names: list,                 # e.g. ["L_GDP_GAP", ..., "L_GDP_TREND", ...]
    trend_state_names: list,               # e.g. ["L_GDP_TREND", "PI_TREND", ...]
    mapping: dict = None,                  # {"L_GDP_OBS": "L_GDP_TREND", ...}
    title="Simulated Observables and Trends"
):
    """
    Plots observables and their "main" trend side-by-side. If `mapping` is None, guesses by substring match.
    """
    sim_obs_np = np.asarray(sim_observables)
    sim_states_np = np.asarray(sim_states)
    time = np.arange(sim_obs_np.shape[0])

    if mapping is None:
        # Guess a mapping by substring; e.g., L_GDP_OBS -> L_GDP_TREND
        mapping = {}
        for obs in obs_var_names:
            for trend in trend_state_names:
                if obs.split('_')[0] in trend:
                    mapping[obs] = trend
                    break

    cols = 2
    rows = (len(obs_var_names) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(min(6 * cols, 18), 4 * rows), squeeze=False)
    axes = axes.flatten()
    plt.suptitle(title, fontsize=14)

    for i, obs_name in enumerate(obs_var_names):
        ax = axes[i]
        obs_idx = obs_var_names.index(obs_name)
        ax.plot(time, sim_obs_np[:, obs_idx], label=f"{obs_name}", linewidth=2)
        # Plot matching trend if found
        trend_name = mapping.get(obs_name)
        if trend_name and trend_name in state_var_names:
            trend_idx = state_var_names.index(trend_name)
            ax.plot(time, sim_states_np[:, trend_idx], label=f"{trend_name} (Trend)", linestyle='--', alpha=0.85)
        ax.legend()
        ax.grid(True)
    # Hide any empty subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

## Main functions of the parser 
def create_timed_symbol(base_name, time_shift):
    if time_shift == -1:
        return sympy.symbols(f"{base_name}_m1")
    elif time_shift == 1:
        return sympy.symbols(f"{base_name}_p1")
    elif time_shift == 0: 
        return sympy.symbols(base_name)
    else: 
        raise ValueError(f"Unexpected time shift {time_shift} for {base_name}")

def base_name_from_aux(aux_name):
    match_lead = re.match(r"aux_([a-zA-Z_]\w*)_lead_p\d+", aux_name)
    if match_lead: 
        return match_lead.group(1)
    match_lag = re.match(r"aux_([a-zA-Z_]\w*)_lag_m\d+", aux_name)
    if match_lag: 
        return match_lag.group(1)
    return aux_name

def symbolic_jacobian(equations, variables):
    num_eq = len(equations); num_var = len(variables)
    jacobian = sympy.zeros(num_eq, num_var)
    for i, eq in enumerate(equations):
        for j, var in enumerate(variables):
            jacobian[i, j] = sympy.diff(eq, var)
    return jacobian

def robust_lambdify(args, expr, modules='jax'):
    # Ensure expr is Sympy Matrix for consistency checks
    if not isinstance(expr, (sympy.Matrix, sympy.ImmutableMatrix)):
         if isinstance(expr, (int, float, np.number, jnp.number)):
              # If it's a scalar number, wrap it in a lambda
              return lambda *a: jnp.array(expr)
         elif isinstance(expr, (np.ndarray, jnp.ndarray)):
              # If it's already an array, wrap it
              return lambda *a: jnp.asarray(expr)
         else:
             # Try converting other types to Sympy Matrix if possible
             try: expr = sympy.Matrix(expr)
             except Exception: pass # Fall through if conversion fails


    if isinstance(expr, (sympy.Matrix, sympy.ImmutableMatrix)):
        if not expr.free_symbols:
            try:
                numerical_matrix = jnp.array(expr.tolist(), dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
                return lambda *a: numerical_matrix
            except (TypeError, ValueError) as e:
                 print(f"Warning: Could not convert symbol-free matrix {expr} to JAX array: {e}")

    try:
        # Use JAX backend
        # Need to map Sympy Matrix to jnp.array explicitly for some cases
        jax_modules = [{'ImmutableDenseMatrix': jnp.array, 'MutableDenseMatrix': jnp.array}, 'jax']
        return sympy.lambdify(args, expr, modules=jax_modules)
    except Exception as e:
        print(f"Error during lambdify (using JAX backend). Arguments: {args}")
        print(f"Expression causing error:\n{sympy.pretty(expr)}") # Pretty print expression
        raise e

def extract_declarations(model_string):
    declarations = {'var': [], 'varexo': [], 'parameters': [] }
    param_assignments = {}
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n'); cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " \n ".join(cleaned_lines)
    model_marker = re.search(r'\bmodel\b\s*;', processed_content, re.IGNORECASE | re.DOTALL)
    content_to_search = processed_content[:model_marker.start()] if model_marker else processed_content
    param_assignment_content_search_area = content_to_search
    block_matches = re.finditer(r'(?i)\b(var|varexo|parameters)\b(.*?)(?=\b(?:var|varexo|parameters|model)\b|$)', content_to_search, re.DOTALL | re.IGNORECASE)
    def process_block_content(content_str, block_type):
        if not content_str: return []
        content = content_str.strip(); first_semicolon_match = re.search(r';', content)
        if first_semicolon_match: content = content[:first_semicolon_match.start()].strip()
        content = content.replace('\n', ' '); names = []
        raw_names = re.split(r'[,\s]+', content)
        cleaned_names = [name for name in raw_names if name and re.fullmatch(r'[a-zA-Z_]\w*', name)]
        keywords = {'var', 'varexo', 'parameters', 'model', 'end'}; names = [n for n in cleaned_names if n not in keywords]
        return list(dict.fromkeys(names).keys())
    for match in block_matches:
        block_keyword = match.group(1).lower(); block_content_raw = match.group(2)
        extracted_names = process_block_content(block_content_raw, block_keyword)
        declarations[block_keyword].extend(extracted_names)
    final_declarations = {key: list(dict.fromkeys(lst).keys()) for key, lst in declarations.items()}
    assignment_matches = re.finditer(r'\b([a-zA-Z_]\w*)\b\s*=\s*([^;]+);', param_assignment_content_search_area)
    parameter_names_declared = final_declarations.get('parameters', [])
    for match in assignment_matches:
        name = match.group(1); value_str = match.group(2).strip()
        if name in parameter_names_declared:
            try: param_assignments[name] = float(value_str)
            except ValueError: print(f"Warning: Could not parse value '{value_str}' for param '{name}'.")
    return (final_declarations.get('var', []), final_declarations.get('varexo', []),
            parameter_names_declared, param_assignments)

def extract_model_equations(model_string):
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n'); cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " ".join(cleaned_lines)
    model_match = re.search(r'(?i)\bmodel\b\s*;(.*?)\bend\b\s*;', processed_content, re.DOTALL | re.IGNORECASE)
    if not model_match: raise ValueError("Could not find 'model; ... end;' block.")
    model_content = model_match.group(1)
    equations_raw = [eq.strip() for eq in model_content.split(';') if eq.strip()]
    processed_equations = []
    for line in equations_raw:
        if '=' in line:
            parts = line.split('=', 1)
            if len(parts) == 2: lhs, rhs = parts; processed_equations.append(f"({lhs.strip()}) - ({rhs.strip()})")
            else: print(f"Warning: Skipping malformed equation line: '{line}'")
        else: print(f"Warning: Equation '{line}' has no '='. Assuming 'expr = 0'."); processed_equations.append(line)
    return processed_equations



def parse_lambdify_and_order_model(model_string, verbose=True):
    """
    Parses the stationary part of the model, handles leads/lags,
    orders variables/equations, generates symbolic matrices for the
    quadratic form A P^2 + B P + C = 0, and returns lambdified JAX functions.

    Args:
        model_string (str): The full content of the Dynare file.
        verbose (bool): If True, prints progress information.

    Returns:
        tuple: Contains:
            - func_A, func_B, func_C, func_D: Lambdified JAX functions for ordered matrices.
            - ordered_vars (List[str]): Ordered list of stationary state variables.
            - shock_names (List[str]): List of stationary shock names.
            - param_names (List[str]): List of combined stationary parameter names.
            - param_assignments (Dict[str, float]): Dictionary of default parameter values.
            - symbolic_matrices_ordered (Dict[str, sympy.Matrix]): Ordered symbolic matrices.
            - initial_info (Dict): Information about the initial (unordered) symbolic setup.
    """
    if verbose:  print("--- Parsing Stationary Model Declarations ---")
    declared_vars, shock_names, param_names_declared, param_assignments_initial = extract_declarations(model_string)
    inferred_sigma_params = [f"sigma_{shk}" for shk in shock_names]
    stat_stderr_values = extract_stationary_shock_stderrs(model_string)
    #sigma_stderr_assignments = {f"sigma_{shk}": val for shk, val in stat_stderr_values.items()}
    sigma_stderr_assignments = stat_stderr_values.copy()

    # --- Step 1d: Combine parameter NAMES ---
    combined_param_names = list(dict.fromkeys(param_names_declared).keys())
    added_sigmas = []
    for p_sigma in inferred_sigma_params:
        if p_sigma not in combined_param_names:
            combined_param_names.append(p_sigma)
            added_sigmas.append(p_sigma)
    # Optional: Print added sigmas only if verbose and list is not empty
    # if verbose and added_sigmas: print(f"   Added inferred stationary sigma parameters: {added_sigmas}")

    # --- Step 1e: Combine parameter ASSIGNMENTS ---
    combined_param_assignments = sigma_stderr_assignments.copy()
    combined_param_assignments.update(param_assignments_initial)
    # Ensure defaults exist for added sigma params if not set
    for p_sigma in added_sigmas:
        if p_sigma not in combined_param_assignments:
             if verbose: print(f"Warning: Inferred sigma parameter '{p_sigma}' has no default value. Setting default to 1.0.")
             combined_param_assignments[p_sigma] = 1.0

    # --- Use the combined lists from now on ---
    param_names = combined_param_names
    param_assignments = combined_param_assignments

    if verbose:
        # Simplified verbose output
        print(f"Declared Variables ({len(declared_vars)}), Shocks ({len(shock_names)}), Declared Params ({len(param_names_declared)})")
        print(f"==> Final Parameter List for Stationary Model ({len(param_names)}): {param_names}")
        print(f"==> Combined Initial Parameter Assignments: {param_assignments}")

    if verbose: print("\n--- Parsing Stationary Model Equations ---")
    raw_equations = extract_model_equations(model_string)
    if verbose: print(f"Found {len(raw_equations)} equations in model block.")

    # --- Handling Leads/Lags & Auxiliaries ---
    if verbose:  print("\n--- Handling Leads/Lags & Auxiliaries ---")
    endogenous_vars = list(declared_vars)
    aux_variables = OrderedDict() # Stores definition string for each aux var
    processed_equations = list(raw_equations)
    var_time_regex = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]?\d+)\s*\)')

    eq_idx = 0
    while eq_idx < len(processed_equations):
        eq = processed_equations[eq_idx]
        eq_idx += 1
        modified_eq = eq
        matches = list(var_time_regex.finditer(eq))

        # Process matches in reverse order to avoid index issues
        for match in reversed(matches):
            base_name = match.group(1)
            time_shift = int(match.group(2))

            # Skip if not an endogenous variable or already processed aux
            if base_name not in endogenous_vars and base_name not in aux_variables:
                continue

            # --- Handle Leads > 1 ---
            if time_shift > 1:
                aux_needed_defs = []
                for k in range(1, time_shift):
                    aux_name = f"aux_{base_name}_lead_p{k}"
                    if aux_name not in aux_variables:
                        prev_var_for_def = base_name if k == 1 else f"aux_{base_name}_lead_p{k-1}"
                        # Definition: aux_k = prev_var(+1) => aux_k - prev_var(+1) = 0
                        def_eq_str = f"{aux_name} - {prev_var_for_def}(+1)"
                        aux_variables[aux_name] = def_eq_str
                        aux_needed_defs.append(def_eq_str)
                        if aux_name not in endogenous_vars:
                            endogenous_vars.append(aux_name)

                target_aux = f"aux_{base_name}_lead_p{time_shift-1}"
                replacement = f"{target_aux}(+1)" # Replace original var(t+s) with aux(t+1)
                start, end = match.span()
                modified_eq = modified_eq[:start] + replacement + modified_eq[end:]

                # Add the definition equations if they aren't already there
                for def_eq in aux_needed_defs:
                    if def_eq not in processed_equations:
                        processed_equations.append(def_eq)

            # --- Handle Lags < -1 ---
            elif time_shift < -1:
                aux_needed_defs = []
                for k in range(1, abs(time_shift)):
                    aux_name = f"aux_{base_name}_lag_m{k}"
                    if aux_name not in aux_variables:
                        prev_var_for_def = base_name if k == 1 else f"aux_{base_name}_lag_m{k-1}"
                        # Definition: aux_k = prev_var(-1) => aux_k - prev_var(-1) = 0
                        def_eq_str = f"{aux_name} - {prev_var_for_def}(-1)"
                        aux_variables[aux_name] = def_eq_str
                        aux_needed_defs.append(def_eq_str)
                        if aux_name not in endogenous_vars:
                            endogenous_vars.append(aux_name)

                target_aux = f"aux_{base_name}_lag_m{abs(time_shift)-1}"
                replacement = f"{target_aux}(-1)" # Replace original var(t-s) with aux(t-1)
                start, end = match.span()
                modified_eq = modified_eq[:start] + replacement + modified_eq[end:]

                # Add the definition equations if they aren't already there
                for def_eq in aux_needed_defs:
                    if def_eq not in processed_equations:
                        processed_equations.append(def_eq)

        if modified_eq != eq:
            processed_equations[eq_idx - 1] = modified_eq
            # if verbose: print(f"  Updated Eq {eq_idx-1}: {modified_eq}") # Can be very verbose

    initial_vars_ordered = list(endogenous_vars) # All vars including auxiliaries
    num_vars = len(initial_vars_ordered)
    num_eq = len(processed_equations)
    num_shocks = len(shock_names)

    if verbose: print(f"Total variables after processing leads/lags ({num_vars}): {initial_vars_ordered}")
    # if verbose: print(f"Total equations after processing leads/lags ({num_eq}):") # Avoid printing all eqs
    # for i, eq in enumerate(processed_equations): print(f"  Eq {i}: {eq}")

    if num_vars != num_eq:
        # Provide more context in the error message
        print("\nError Details:")
        print(f"  Original Declared Vars: {declared_vars}")
        print(f"  Auxiliary Vars Added: {list(aux_variables.keys())}")
        print(f"  Final Variable List ({num_vars}): {initial_vars_ordered}")
        print(f"\n  Original Equations: {raw_equations}")
        print(f"  Auxiliary Equations Added: {list(aux_variables.values())}")
        print(f"  Final Equation List ({num_eq}): {processed_equations}")
        raise ValueError(
            f"Stationary model not square after processing leads/lags: {num_vars} vars vs {num_eq} eqs."
        )
    if verbose: print("Stationary model is square.")

    # --- Symbolic Representation ---
    if verbose: print("\n--- Creating Symbolic Representation (Stationary Model) ---")
    param_syms = {p: sympy.symbols(p) for p in param_names}
    shock_syms = {s: sympy.symbols(s) for s in shock_names}
    var_syms = {} # Holds {'var': {'m1': sym, 't': sym, 'p1': sym}}
    all_syms_for_parsing = set(param_syms.values()) | set(shock_syms.values())
    for var in initial_vars_ordered:
        sym_m1 = create_timed_symbol(var, -1)
        sym_t  = create_timed_symbol(var, 0)
        sym_p1 = create_timed_symbol(var, 1)
        var_syms[var] = {'m1': sym_m1, 't': sym_t, 'p1': sym_p1}
        all_syms_for_parsing.update([sym_m1, sym_t, sym_p1])

    local_dict = {str(s): s for s in all_syms_for_parsing}
    # Add common math functions known by sympy
    local_dict.update({'log': sympy.log, 'exp': sympy.exp, 'sqrt': sympy.sqrt, 'abs': sympy.Abs})
    # Add others if your models use them: 'ln', 'Abs', etc.

    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                          implicit_multiplication_application, rationalize)
    # Define standard transformations for the parser
    transformations = (standard_transformations + (implicit_multiplication_application, rationalize))

    sym_equations = []
    if verbose: print("Parsing stationary equations into symbolic form...")
    for i, eq_str in enumerate(processed_equations):
        eq_str_sym = eq_str # Start with the string equation

        # Define a function to replace timed variables (var(time)) with their symbolic counterparts
        def replace_var_time(match):
            base_name, time_shift_str = match.groups()
            time_shift = int(time_shift_str)
            if base_name in shock_names:
                if time_shift == 0: return str(shock_syms[base_name])
                else: raise ValueError(f"Shock {base_name}({time_shift}) invalid in stationary model eq {i}: '{eq_str}'. Shocks are contemporaneous.")
            elif base_name in var_syms: # Endogenous or auxiliary variable
                if time_shift == -1: return str(var_syms[base_name]['m1'])
                if time_shift == 0:  return str(var_syms[base_name]['t'])
                if time_shift == 1:  return str(var_syms[base_name]['p1'])
                # Should not happen after auxiliary variable processing
                raise ValueError(f"Unexpected time shift {time_shift} for variable {base_name} in eq {i}: '{eq_str}' after aux processing.")
            elif base_name in param_syms:
                raise ValueError(f"Parameter {base_name}({time_shift}) is invalid syntax in eq {i}: '{eq_str}'. Parameters are time-invariant.")
            elif base_name in local_dict: # e.g. log, exp - leave as is
                return match.group(0)
            else: # Unknown symbol with time shift - potential error or undeclared var
                # Create a symbolic representation but warn
                timed_sym_str = str(create_timed_symbol(base_name, time_shift))
                if timed_sym_str not in local_dict:
                    print(f"Warning: Symbol '{base_name}' with time shift {time_shift} in eq {i} ('{eq_str}') is undeclared. Treating symbolically as '{timed_sym_str}'.")
                    local_dict[timed_sym_str] = sympy.symbols(timed_sym_str)
                return timed_sym_str

        # Apply the replacement using regex substitution
        eq_str_sym = var_time_regex.sub(replace_var_time, eq_str_sym)

        # Replace remaining base names (implicitly time t)
        # Sort by length descending to replace longer names first (e.g., ABC before AB)
        all_known_base_names = sorted(list(var_syms.keys()) + param_names + shock_names, key=len, reverse=True)
        for name in all_known_base_names:
            # Use word boundaries to avoid partial matches within other names
            pattern = r'\b' + re.escape(name) + r'\b'
            if name in var_syms: replacement = str(var_syms[name]['t'])
            elif name in param_syms: replacement = str(param_syms[name])
            elif name in shock_names: replacement = str(shock_syms[name])
            else: continue # Should not happen if list is correct
            eq_str_sym = re.sub(pattern, replacement, eq_str_sym)

        try:
            # Check for remaining undeclared symbols (likely errors) before parsing
            current_symbols = set(re.findall(r'\b([a-zA-Z_]\w*)\b', eq_str_sym))
            known_keys = set(local_dict.keys()) | {'log', 'exp', 'sqrt', 'abs'} # Add known functions
            # Exclude numbers
            unknown_symbols = {s for s in current_symbols if s not in known_keys and not re.fullmatch(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', s)}

            if unknown_symbols:
                # This is usually an error (typo, undeclared var/param)
                print(f"ERROR: Potential undeclared symbols found parsing eq {i} ('{eq_str_sym}'): {unknown_symbols}. Check model definition.")
                # Decide whether to raise error or try to continue by adding them
                # raise ValueError(f"Undeclared symbols found: {unknown_symbols}")
                print("Attempting to continue by adding symbols to local_dict.")
                for sym_str in unknown_symbols:
                    if sym_str not in local_dict: local_dict[sym_str] = sympy.symbols(sym_str)


            # Parse the string into a sympy expression
            sym_eq = parse_expr(eq_str_sym, local_dict=local_dict, transformations=transformations)
            sym_equations.append(sym_eq)
        except Exception as e:
            print(f"\n--- Error parsing stationary equation {i} ---")
            print(f"Original Eq String : '{eq_str}'")
            print(f"Processed Sym String: '{eq_str_sym}'")
            print(f"Local dict keys    : {sorted(local_dict.keys())}")
            print(f"Sympy error        : {e}")
            raise

    if verbose: print("Symbolic parsing completed.")

    # --- Generate Initial Symbolic Matrices A P^2 + B P + C = 0, D for shocks ---
    if verbose: print("\n--- Generating Initial Symbolic Matrices (A, B, C, D) ---")
    # Convention: A coeffs y(t+1), B coeffs y(t), C coeffs y(t-1)
    # This matches the convention needed for the `solve_quadratic_matrix_equation`
    # Note the definition change: A_quad = dF/dy_{t+1}, B_quad = dF/dy_t, C_quad = dF/dy_{t-1}
    sympy_A_quad = sympy.zeros(num_eq, num_vars) # Coeffs of y(t+1)
    sympy_B_quad = sympy.zeros(num_eq, num_vars) # Coeffs of y(t)
    sympy_C_quad = sympy.zeros(num_eq, num_vars) # Coeffs of y(t-1)
    sympy_D_quad = sympy.zeros(num_eq, num_shocks) # Coeffs of e(t) (multiplied by -1 for Q calc)

    # Get lists of symbolic variables for Jacobian calculation
    var_p1_syms = [var_syms[v]['p1'] for v in initial_vars_ordered]
    var_t_syms  = [var_syms[v]['t']  for v in initial_vars_ordered]
    var_m1_syms = [var_syms[v]['m1'] for v in initial_vars_ordered]
    shock_t_syms = [shock_syms[s] for s in shock_names]

    # Compute Jacobians
    for i, eq in enumerate(sym_equations):
        # Use Jacobian calculation for robustness with non-linear terms (though model is assumed linear)
        for j, var_p1 in enumerate(var_p1_syms): sympy_A_quad[i, j] = sympy.diff(eq, var_p1)
        for j, var_t  in enumerate(var_t_syms):  sympy_B_quad[i, j] = sympy.diff(eq, var_t)
        for j, var_m1 in enumerate(var_m1_syms): sympy_C_quad[i, j] = sympy.diff(eq, var_m1)
        # For D, we need coefficient of shock e(t). Derivative is dF/de(t). We store -dF/de(t).
        for k, shk_t in enumerate(shock_t_syms): sympy_D_quad[i, k] = -sympy.diff(eq, shk_t) # Note the minus sign

    # Store initial symbolic matrices (using solver convention A=coeff(p1), B=coeff(t), C=coeff(m1))
    # Keep a copy before reordering if needed
    initial_info = {
        'A': copy.deepcopy(sympy_A_quad), # Store based on quad eq: A P^2 + B P + C = 0 convention
        'B': copy.deepcopy(sympy_B_quad),
        'C': copy.deepcopy(sympy_C_quad),
        'D': copy.deepcopy(sympy_D_quad),
        'vars': list(initial_vars_ordered), # Keep original order info
        'eqs': list(processed_equations)   # Keep original order info
    }
    if verbose: print("Symbolic matrices A, B, C, D generated (for quadratic solver).")

    # --- Classify Variables (Graph-Based) ---
    if verbose: print("\n--- Classifying Variables for Ordering (Stationary Model) ---")

    # Step 1: Classify each variable purely from the Jacobian structure
    backward_only_vars = []    # Appears at t-1 only (no lead)
    forward_backward_vars = [] # Appears at t+1 and/or t-1 (dynamic)
    static_vars = []           # Appears only at t (no lead, no lag)

    for j, var in enumerate(initial_vars_ordered):
        has_lead = not sympy_A_quad.col(j).is_zero_matrix
        has_lag  = not sympy_C_quad.col(j).is_zero_matrix
        if has_lead:
            forward_backward_vars.append(var)
        elif has_lag:
            backward_only_vars.append(var)
        else:
            static_vars.append(var)

    if verbose:
        print(f"  Backward-only ({len(backward_only_vars)}): {backward_only_vars}")
        print(f"  Forward/Backward ({len(forward_backward_vars)}): {forward_backward_vars}")
        print(f"  Static ({len(static_vars)}): {static_vars}")

    # Step 2: Variable ordering = backward-only, then dynamic, then static
    ordered_vars = backward_only_vars + forward_backward_vars + static_vars
    assert len(ordered_vars) == len(initial_vars_ordered) and set(ordered_vars) == set(initial_vars_ordered), \
        f"Variable reordering lost/gained variables: {set(initial_vars_ordered) - set(ordered_vars)}"
    var_perm_indices = [initial_vars_ordered.index(v) for v in ordered_vars]
    if verbose: print(f"\nNew Variable Order ({len(ordered_vars)}): {ordered_vars}")

    # Step 3: Equation ordering via maximum bipartite matching
    # Build incidence: for each equation, which variables appear at t (in B_quad)?
    # The matching pairs each equation with the variable it "defines".
    if verbose: print("\n--- Equation-Variable Matching (Bipartite Graph) ---")

    # Build adjacency: eq_i -> set of var indices (in original ordering) where B_quad[i,j] != 0
    eq_to_var_adj = {}
    for i in range(num_eq):
        eq_to_var_adj[i] = set()
        for j in range(num_vars):
            if sympy_B_quad[i, j] != 0:
                eq_to_var_adj[i].add(j)

    # Hopcroft-Karp maximum bipartite matching
    def _hopcroft_karp(eq_adj, n_eq, n_var):
        """Find maximum matching in bipartite graph (equations -> variables).
        Returns: match_eq[i] = variable index matched to equation i (or -1)."""
        match_eq = [-1] * n_eq    # eq -> var
        match_var = [-1] * n_var  # var -> eq
        INF = float('inf')

        def bfs():
            dist = {}
            queue = []
            for eq in range(n_eq):
                if match_eq[eq] == -1:
                    dist[eq] = 0
                    queue.append(eq)
                else:
                    dist[eq] = INF
            found = False
            head = 0
            while head < len(queue):
                eq = queue[head]; head += 1
                for var in eq_adj.get(eq, set()):
                    next_eq = match_var[var]
                    if next_eq == -1:
                        found = True
                    elif dist.get(next_eq, INF) == INF:
                        dist[next_eq] = dist[eq] + 1
                        queue.append(next_eq)
            return found, dist

        def dfs(eq, dist):
            for var in eq_adj.get(eq, set()):
                next_eq = match_var[var]
                if next_eq == -1 or (dist.get(next_eq, INF) == dist[eq] + 1 and dfs(next_eq, dist)):
                    match_eq[eq] = var
                    match_var[var] = eq
                    return True
            dist[eq] = INF
            return False

        while True:
            found, dist = bfs()
            if not found:
                break
            for eq in range(n_eq):
                if match_eq[eq] == -1:
                    dfs(eq, dist)
        return match_eq, match_var

    match_eq, match_var = _hopcroft_karp(eq_to_var_adj, num_eq, num_vars)

    matching_size = sum(1 for m in match_eq if m != -1)
    if matching_size != num_eq:
        unmatched_eqs = [i for i, m in enumerate(match_eq) if m == -1]
        unmatched_vars = [initial_vars_ordered[j] for j, m in enumerate(match_var) if m == -1]
        raise RuntimeError(
            f"Bipartite matching failed: matched {matching_size}/{num_eq}. "
            f"Unmatched equations: {unmatched_eqs}, unmatched variables: {unmatched_vars}. "
            f"The model may be singular or have structural issues."
        )

    # --- Static Variable Elimination ---
    # Build var_name -> matched_eq_index mapping (in initial ordering)
    match_eq_for_var = {initial_vars_ordered[j]: match_var[j]
                        for j in range(num_vars) if match_var[j] != -1}

    if verbose: print("\n--- Static Variable Elimination ---")
    (sym_equations, initial_vars_ordered,
     elim_var_names, elim_subs_symbolic, func_S) = _eliminate_static_contemporaneous_vars(
        sym_equations, var_syms, initial_vars_ordered,
        sympy_A_quad, sympy_B_quad, sympy_C_quad, sympy_D_quad,
        match_eq_for_var, param_syms, shock_syms, param_names,
        verbose=verbose)

    if elim_var_names:
        # Reduced system: rebuild A, B, C, D, reclassify, and rematch.
        num_vars = len(initial_vars_ordered)
        num_eq   = len(sym_equations)

        var_p1_syms = [var_syms[v]['p1'] for v in initial_vars_ordered]
        var_t_syms  = [var_syms[v]['t']  for v in initial_vars_ordered]
        var_m1_syms = [var_syms[v]['m1'] for v in initial_vars_ordered]

        sympy_A_quad = sympy.zeros(num_eq, num_vars)
        sympy_B_quad = sympy.zeros(num_eq, num_vars)
        sympy_C_quad = sympy.zeros(num_eq, num_vars)
        sympy_D_quad = sympy.zeros(num_eq, num_shocks)

        for i, eq in enumerate(sym_equations):
            for j, s in enumerate(var_p1_syms): sympy_A_quad[i, j] = sympy.diff(eq, s)
            for j, s in enumerate(var_t_syms):  sympy_B_quad[i, j] = sympy.diff(eq, s)
            for j, s in enumerate(var_m1_syms): sympy_C_quad[i, j] = sympy.diff(eq, s)
            for k, s in enumerate(shock_t_syms): sympy_D_quad[i, k] = -sympy.diff(eq, s)

        # Reclassify
        backward_only_vars, forward_backward_vars, static_vars = [], [], []
        for j, var in enumerate(initial_vars_ordered):
            has_lead = not sympy_A_quad.col(j).is_zero_matrix
            has_lag  = not sympy_C_quad.col(j).is_zero_matrix
            if has_lead:               forward_backward_vars.append(var)
            elif has_lag:              backward_only_vars.append(var)
            else:                      static_vars.append(var)

        if verbose:
            print(f"  [Reduced] Backward-only ({len(backward_only_vars)}): {backward_only_vars}")
            print(f"  [Reduced] Forward/Backward ({len(forward_backward_vars)}): {forward_backward_vars}")
            print(f"  [Reduced] Static ({len(static_vars)}): {static_vars}")

        ordered_vars    = backward_only_vars + forward_backward_vars + static_vars
        var_perm_indices = [initial_vars_ordered.index(v) for v in ordered_vars]

        # Rematch
        eq_to_var_adj = {}
        for i in range(num_eq):
            eq_to_var_adj[i] = {j for j in range(num_vars) if sympy_B_quad[i, j] != 0}
        match_eq, match_var = _hopcroft_karp(eq_to_var_adj, num_eq, num_vars)
        matching_size = sum(1 for m in match_eq if m != -1)
        if matching_size != num_eq:
            raise RuntimeError(
                f"Bipartite matching failed on reduced system after elimination: "
                f"matched {matching_size}/{num_eq}.")

    # --- Build equation permutation ---
    eq_perm_indices = []
    for var in ordered_vars:
        j_orig = initial_vars_ordered.index(var)
        eq_idx = match_var[j_orig]
        assert eq_idx != -1, f"Variable '{var}' has no matched equation after elimination"
        eq_perm_indices.append(eq_idx)

    assert len(eq_perm_indices) == num_eq
    assert len(set(eq_perm_indices)) == num_eq

    if verbose:
        for i, var in enumerate(ordered_vars):
            eq_idx = eq_perm_indices[i]
            print(f"  {var:30s} <- eq {eq_idx}")

    # --- Reorder Symbolic Matrices ---
    if verbose: print("\n--- Reordering Symbolic Matrices (Stationary Model) ---")
    sympy_A_ord = sympy_A_quad.extract(eq_perm_indices, var_perm_indices)
    sympy_B_ord = sympy_B_quad.extract(eq_perm_indices, var_perm_indices)
    sympy_C_ord = sympy_C_quad.extract(eq_perm_indices, var_perm_indices)
    sympy_D_ord = sympy_D_quad.extract(eq_perm_indices, list(range(num_shocks)))

    symbolic_matrices_ordered = {'A': sympy_A_ord, 'B': sympy_B_ord, 'C': sympy_C_ord, 'D': sympy_D_ord}
    if verbose: print("Symbolic reordering complete.")

    # --- Lambdify ---
    if verbose: print("\n--- Lambdifying Ordered Matrices (Stationary Model) ---")
    param_sym_list = [param_syms[p] for p in param_names]

    func_A = robust_lambdify(param_sym_list, sympy_A_ord, modules='jax')
    func_B = robust_lambdify(param_sym_list, sympy_B_ord, modules='jax')
    func_C = robust_lambdify(param_sym_list, sympy_C_ord, modules='jax')
    func_D = robust_lambdify(param_sym_list, sympy_D_ord, modules='jax')
    if verbose: print("Lambdification successful.")

    # Return all necessary components.
    # New additions (backward-compatible - model.py unpacks by position):
    #   [10] elim_var_names    : list of eliminated variable names
    #   [11] func_S            : callable(params...) -> (n_elim x n_reduced) jnp array
    #   [12] elim_subs_symbolic: dict {var_name: sympy_expr} for reference
    return (func_A, func_B, func_C, func_D,
            ordered_vars, shock_names, param_names, param_assignments,
            symbolic_matrices_ordered, initial_info,
            elim_var_names, func_S, elim_subs_symbolic)

# --- END OF CORRECTED parse_lambdify_and_order_model FUNCTION ---

def extract_stationary_shock_stderrs(model_string):
    stderrs = {}
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n'); cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " ".join(cleaned_lines)
    match = re.search(r'(?i)\bshocks\b\s*;(.*?)\bend\b\s*;', processed_content, re.DOTALL | re.IGNORECASE)
    if not match: return {}
    content = match.group(1)
    stderr_matches = re.finditer(r'(?i)\bvar\s+([a-zA-Z_]\w*)\s*;\s*stderr\s+([^;]+);', content)
    parsed_count = 0
    for m in stderr_matches:
        shock_name = m.group(1); stderr_val_str = m.group(2).strip()
        try: stderr_val = float(stderr_val_str); param_name = f"sigma_{shock_name}"; stderrs[param_name] = stderr_val; parsed_count += 1
        except ValueError: print(f"Warning: Could not parse stderr value '{stderr_val_str}' for stat shock '{shock_name}'.")
    # if parsed_count > 0: print(f"   Parsed {parsed_count} stderr definitions from 'shocks;' block.") # Less verbose
    return stderrs


# --- SDA Solver with Implicit Differentiation (JAX) ---
#
# Forward pass:  lax.while_loop SDA with true early stopping (~8 iterations).
# Backward pass: implicit function theorem via adjoint Sylvester equation.
#
# Reference: The Successive Doubling Algorithm is adapted from DSGE.jl.
# The implicit differentiation follows Christianson (1994) / Blondel et al.
# (2022) "Efficient and Modular Implicit Differentiation".

def _eliminate_static_contemporaneous_vars(
        sym_equations, var_syms, initial_vars_ordered,
        sympy_A_quad, sympy_B_quad, sympy_C_quad, sympy_D_quad,
        match_eq_for_var,   # dict: var_name -> equation_index (in initial ordering)
        param_syms, shock_syms, param_names,
        verbose=False):
    """Eliminate purely contemporaneous static variables from the QME system.

    A variable ``var_j`` (and its matched equation ``eq_i``) is eliminable iff:
      1. Column j in A = 0   -- var_j never appears at t+1 anywhere
      2. Column j in C = 0   -- var_j never appears at t-1 anywhere
      3. Row eq_i in A = 0   -- the defining equation has no leads
      4. Row eq_i in C = 0   -- the defining equation has no lags

    These are true contemporaneous identities: ``var_j = linear_combo(other vars at t)``.
    Elimination substitutes the identity into all other equations, removing
    ``var_j`` and ``eq_i`` from the system.  The result has dimension (n-k) where
    k is the number of eliminated variables.

    Returns
    -------
    reduced_eqs       : list of sympy expressions (residuals = 0)
    reduced_vars      : list of remaining variable names (in original order)
    elim_var_names    : list of eliminated variable names (in elimination order)
    elim_subs_symbolic: dict {var_name: sympy_expr} giving the back-substitution
                        (each expr is in terms of time-t symbols of reduced vars)
    func_S            : callable(param_values...) -> jnp.ndarray of shape (n_elim, n_reduced)
    """
    # Work with mutable copies of the equation list and variable list
    eqs = list(sym_equations)
    vars_ordered = list(initial_vars_ordered)

    elim_var_names = []
    elim_subs_symbolic = {}   # {var_name: sympy_expr at time t}

    # Build a working copy of the A, B, C matrices as Python lists of lists
    # (We'll rebuild after elimination from the reduced equations, but we need
    #  the matrices to identify eliminable vars in each round.)
    A = [list(sympy_A_quad.row(i)) for i in range(sympy_A_quad.shape[0])]
    B = [list(sympy_B_quad.row(i)) for i in range(sympy_B_quad.shape[0])]
    C = [list(sympy_C_quad.row(i)) for i in range(sympy_C_quad.shape[0])]
    D = [list(sympy_D_quad.row(i)) for i in range(sympy_D_quad.shape[0])]

    # Build the eq->var and var->eq mappings as mutable dicts
    var_to_eq = dict(match_eq_for_var)   # var_name -> eq_index (current numbering)
    eq_to_var = {eq_i: vname for vname, eq_i in var_to_eq.items()}

    changed = True
    while changed:
        changed = False
        n = len(vars_ordered)

        for j, var in enumerate(vars_ordered):
            eq_i = var_to_eq.get(var, -1)
            if eq_i == -1:
                continue

            # Check all four conditions on the A, B, C matrices
            col_A_nonzero = any(A[i][j] != 0 for i in range(n))   # cond 1
            col_C_nonzero = any(C[i][j] != 0 for i in range(n))   # cond 2
            if col_A_nonzero or col_C_nonzero:
                continue

            row_A_nonzero = any(A[eq_i][k] != 0 for k in range(n))  # cond 3
            row_C_nonzero = any(C[eq_i][k] != 0 for k in range(n))  # cond 4
            if row_A_nonzero or row_C_nonzero:
                continue

            # This variable is eliminable.
            b_jj = B[eq_i][j]
            if b_jj == 0:
                continue   # Diagonal entry missing - skip (shouldn't happen)

            # Solve for var_j from its defining equation:
            # sum_k B[eq_i][k] * var_k_t = 0
            # => var_j_t = -(1 / b_jj) * sum_{k != j} B[eq_i][k] * var_k_t
            var_sym_t = var_syms[var]['t']
            subst_expr = sum(
                -B[eq_i][k] / b_jj * var_syms[vars_ordered[k]]['t']
                for k in range(n) if k != j and B[eq_i][k] != 0
            )
            elim_subs_symbolic[var] = subst_expr
            elim_var_names.append(var)

            if verbose:
                print(f"  [Elimination] '{var}' = {subst_expr}  (eq {eq_i})")

            # Substitute var_j_t -> subst_expr in ALL other equations (sym form)
            for i2, eq in enumerate(eqs):
                if i2 != eq_i:
                    eqs[i2] = eq.subs(var_sym_t, subst_expr)

            # Update A, B, C rows for the substitution (propagate into working matrices)
            for i2 in range(n):
                if i2 == eq_i:
                    continue
                coeff = B[i2][j]
                if coeff != 0:
                    for k in range(n):
                        if k != j:
                            B[i2][k] = sympy.simplify(B[i2][k] + coeff * (-B[eq_i][k] / b_jj))
                    B[i2][j] = 0

            # Remove row eq_i and column j from A, B, C, D
            rows_keep = [r for r in range(n) if r != eq_i]
            cols_keep = [c for c in range(n) if c != j]
            A = [[A[r][c] for c in cols_keep] for r in rows_keep]
            B = [[B[r][c] for c in cols_keep] for r in rows_keep]
            C = [[C[r][c] for c in cols_keep] for r in rows_keep]
            D = [D[r] for r in rows_keep]   # D columns (shocks) are unchanged

            # Remove equation and variable from ordered lists
            eqs = [eqs[r] for r in rows_keep]
            vars_ordered = [vars_ordered[c] for c in cols_keep]

            # Rebuild mappings (eq indices shifted after row deletion)
            new_var_to_eq = {}
            for vname, old_eq in var_to_eq.items():
                if vname == var:
                    continue   # removed
                new_eq = old_eq - (1 if old_eq > eq_i else 0)
                new_var_to_eq[vname] = new_eq
            var_to_eq = new_var_to_eq
            eq_to_var = {eq_i2: vn for vn, eq_i2 in var_to_eq.items()}

            changed = True
            break   # Restart loop with updated sizes

    if not elim_var_names:
        # Nothing eliminated - return originals unchanged
        if verbose:
            print("  [Elimination] No eliminable static variables found.")
        return (list(sym_equations), list(initial_vars_ordered),
                [], {}, None)

    n_elim    = len(elim_var_names)
    n_reduced = len(vars_ordered)

    if verbose:
        print(f"  [Elimination] Eliminated {n_elim} variable(s): {elim_var_names}")
        print(f"  [Elimination] Reduced system dimension: {n_reduced}")

    # Build symbolic S matrix: S[i, j] = coeff of reduced_var_j_t in elim_expr_i
    param_sym_list = [param_syms[p] for p in param_names]
    sympy_S = sympy.zeros(n_elim, n_reduced)
    for i, var in enumerate(elim_var_names):
        expr = elim_subs_symbolic[var]
        for j, rvar in enumerate(vars_ordered):
            sympy_S[i, j] = sympy.diff(expr, var_syms[rvar]['t'])

    if sympy_S.free_symbols:
        func_S = robust_lambdify(param_sym_list, sympy_S, modules='jax')
    else:
        # Constant S matrix (no parameter dependence)
        _S_const = jnp.array(sympy_S.tolist(), dtype=jnp.float64)
        func_S = lambda *a: _S_const

    return eqs, vars_ordered, elim_var_names, elim_subs_symbolic, func_S


_SDA_JITTER = 1e-14


def _sda_while_loop(A, B, C, tol, max_iter):
    """SDA forward solve using lax.while_loop (true early stopping).

    Solves AP^2 + BP + C = 0.  Stops as soon as the iterate converges
    (relative change < tol) or numerical validity is lost.
    """
    dtype = A.dtype
    n = A.shape[0]
    I = jnp.eye(n, dtype=dtype)

    # Initial SDA setup (X_guess = 0  =>  Bbar = B)
    Bbar_reg = B + _SDA_JITTER * I
    E0 = jax.scipy.linalg.solve(Bbar_reg, -C, assume_a='gen')
    F0 = jax.scipy.linalg.solve(Bbar_reg, -A, assume_a='gen')
    init_valid = jnp.all(jnp.isfinite(E0)) & jnp.all(jnp.isfinite(F0))

    # State: (Xk, Yk, Ek, Fk, k, converged, is_valid)
    init_state = (E0, F0, E0, F0,
                  jnp.array(0, jnp.int32),
                  jnp.array(False),
                  init_valid)

    def cond_fn(state):
        _, _, _, _, k, converged, is_valid = state
        return is_valid & (~converged) & (k < max_iter)

    def body_fn(state):
        Xk, Yk, Ek, Fk, k, _, is_valid = state

        M1 = I - Yk @ Xk + _SDA_JITTER * I
        M2 = I - Xk @ Yk + _SDA_JITTER * I

        E_new = Ek @ jax.scipy.linalg.solve(M1, Ek, assume_a='gen')
        F_new = Fk @ jax.scipy.linalg.solve(M2, Fk, assume_a='gen')
        X_new = Xk + Fk @ jax.scipy.linalg.solve(M2, Xk @ Ek, assume_a='gen')
        Y_new = Yk + Ek @ jax.scipy.linalg.solve(M1, Yk @ Fk, assume_a='gen')

        rel_diff = (jnp.linalg.norm(X_new - Xk, ord='fro')
                    / jnp.maximum(jnp.linalg.norm(X_new, ord='fro'), 1e-15))
        step_valid = (jnp.all(jnp.isfinite(X_new))
                      & jnp.all(jnp.isfinite(Y_new))
                      & jnp.all(jnp.isfinite(E_new))
                      & jnp.all(jnp.isfinite(F_new))
                      & jnp.isfinite(rel_diff))
        converged = step_valid & (rel_diff < tol)

        return (X_new, Y_new, E_new, F_new, k + 1,
                converged, is_valid & step_valid)

    Xk, _, _, _, iters, converged, is_valid = lax.while_loop(
        cond_fn, body_fn, init_state)
    return Xk, iters, converged & is_valid


# ---- Differentiable wrapper via custom_vjp ----

@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _solve_qme(A, B, C, tol=1e-12, max_iter=50):
    """Differentiable QME solver: AP^2 + BP + C = 0."""
    P, _, _ = _sda_while_loop(A, B, C, tol, max_iter)
    return P


def _solve_qme_fwd(A, B, C, tol, max_iter):
    P = _solve_qme(A, B, C, tol, max_iter)
    return P, (A, B, P)  # residuals saved for backward


def _solve_discrete_sylvester_doubling(A_coeff, B_coeff, C_rhs, n_iters=30):
    """Solve X - AXB = C via doubling iterations.  O(n^3) per step.

    Computes the Neumann series X = sum_{k=0}^{inf} A^k C B^k which
    converges when rho(A)*rho(B) < 1.  Doubling accumulates partial
    sums up to index 2^k - 1, giving quadratic convergence.

    To solve AXB + X = C instead, call with -A (negate first argument).
    """
    def body(_, state):
        Ak, Bk, Ck = state
        Ck = Ak @ Ck @ Bk + Ck
        Ak = Ak @ Ak
        Bk = Bk @ Bk
        return (Ak, Bk, Ck)

    _, _, X = lax.fori_loop(0, n_iters, body, (A_coeff, B_coeff, C_rhs))
    return X


def _solve_qme_bwd(tol, max_iter, res, g_P):
    """Backward pass via the implicit function theorem.

    Given the fixed-point equation  F(P) = AP^2 + BP + C = 0,
    the total differential is:
        dF = A dP P + (AP + B) dP  +  dA P^2 + dB P + dC = 0

    Defining M = AP + B, the adjoint Sylvester equation is:
        M^T W + A^T W P^T = g_P

    Rewritten as discrete Sylvester (dividing by M^T):
        (M^{-T} A^T) W P^T + W = M^{-T} g_P

    Solved via doubling iterations at O(n^3) per step (~15 iterations).

    Then the parameter gradients are:
        dL/dA = -W (P^2)^T,   dL/dB = -W P^T,   dL/dC = -W
    """
    A, B, P = res

    M = A @ P + B
    # Transform to standard discrete Sylvester: A_hat W B_hat + W = C_hat
    A_hat = jax.scipy.linalg.solve(M.T, A.T, assume_a='gen')   # M^{-T} A^T
    C_hat = jax.scipy.linalg.solve(M.T, g_P, assume_a='gen')   # M^{-T} g_P

    W = _solve_discrete_sylvester_doubling(-A_hat, P.T, C_hat)

    g_A = -W @ (P @ P).T
    g_B = -W @ P.T
    g_C = -W
    return g_A, g_B, g_C


_solve_qme.defvjp(_solve_qme_fwd, _solve_qme_bwd)


# ---- Public API (same signature as the old solver) ----

def solve_quadratic_matrix_equation_jax(A, B, C, initial_guess=None,
                                        tol=1e-12, max_iter=500,
                                        verbose=False):
    """Solve AP^2 + BP + C = 0 with implicit differentiation.

    Forward pass uses SDA with ``lax.while_loop`` (true early stopping).
    Backward pass uses the implicit function theorem (adjoint Sylvester eq).

    Args:
        A, B, C: coefficient matrices (n x n JAX arrays).
        initial_guess: ignored (kept for API compatibility).
        tol: convergence tolerance for relative iterate change.
        max_iter: maximum SDA iterations.
        verbose: ignored.

    Returns:
        (P, iters, residual_ratio, converged_flag)
        P carries gradients; diagnostics are computed via stop_gradient.
    """
    P = _solve_qme(A, B, C, tol, max_iter)

    # Diagnostics via stop_gradient (no gradient contribution)
    P_sg = lax.stop_gradient(P)
    A_sg = lax.stop_gradient(A)
    B_sg = lax.stop_gradient(B)
    C_sg = lax.stop_gradient(C)

    residual = A_sg @ (P_sg @ P_sg) + B_sg @ P_sg + C_sg
    res_norm = jnp.linalg.norm(residual, 'fro')
    scale = (jnp.linalg.norm(A_sg @ P_sg @ P_sg, 'fro')
             + jnp.linalg.norm(B_sg @ P_sg, 'fro')
             + jnp.linalg.norm(C_sg, 'fro'))
    residual_ratio = jnp.where(scale > 1e-15, res_norm / scale, jnp.inf)

    converged = jnp.all(jnp.isfinite(P_sg)) & (residual_ratio < 1e-3)
    iters = jnp.array(0, jnp.int32)  # not tracked through the differentiable path

    return P, iters, residual_ratio, converged

def compute_Q_jax(A: ArrayLike, B: ArrayLike, D: ArrayLike, P: ArrayLike,
                  dtype: Optional[jnp.dtype] = None) -> jax.Array:
    """
    Computes Q for y_t = P y_{t-1} + Q e_t using JAX. Solves (A P + B) Q = D.
    Allows NaNs to propagate if the solve fails.
    """
    effective_dtype = dtype if dtype is not None else (A.dtype if hasattr(A, 'dtype') else jnp.float64)
    A_jax = jnp.asarray(A, dtype=effective_dtype); B_jax = jnp.asarray(B, dtype=effective_dtype)
    D_jax = jnp.asarray(D, dtype=effective_dtype); P_jax = jnp.asarray(P, dtype=effective_dtype)
    n = A_jax.shape[0]
    n_shock = D_jax.shape[1] if D_jax.ndim == 2 else 0
    if n_shock == 0: return jnp.zeros((n, 0), dtype=effective_dtype)

    APB = A_jax @ P_jax + B_jax
    # Add jitter for potentially ill-conditioned APB matrix
    APB_reg = APB + _SDA_JITTER * jnp.eye(n, dtype=effective_dtype)
    # Attempt solve, let NaNs propagate if it fails
    Q = jax.scipy.linalg.solve(APB_reg, D_jax, assume_a='gen')
    return Q


# --- Trend/Observation Parsing Functions (Mostly unchanged) ---

def extract_trend_declarations(model_string):
    trend_vars = []; trend_shocks = []
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n'); cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " \n ".join(cleaned_lines)
    trend_model_marker = re.search(r'\btrend_model\b\s*;', processed_content, re.IGNORECASE | re.DOTALL)
    search_area = processed_content[:trend_model_marker.start()] if trend_model_marker else processed_content
    match_tv = re.search(r'(?i)\btrends_vars\b(.*?);', search_area, re.DOTALL | re.IGNORECASE)
    if match_tv: content = match_tv.group(1).replace('\n', ' ').strip(); trend_vars = [v for v in re.split(r'[,\s]+', content) if v and re.fullmatch(r'[a-zA-Z_]\w*', v)]; trend_vars = list(dict.fromkeys(trend_vars).keys())
    match_vt = re.search(r'(?i)\bvarexo_trends\b(.*?);', search_area, re.DOTALL | re.IGNORECASE)
    if match_vt: content = match_vt.group(1).replace('\n', ' ').strip(); trend_shocks = [s for s in re.split(r'[,\s]+', content) if s and re.fullmatch(r'[a-zA-Z_]\w*', s)]; trend_shocks = list(dict.fromkeys(trend_shocks).keys())
    return trend_vars, trend_shocks

def extract_trend_equations(model_string):
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n'); cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " ".join(cleaned_lines)
    match = re.search(r'(?i)\btrend_model\b\s*;(.*?)\bend\b\s*;', processed_content, re.DOTALL | re.IGNORECASE)
    if not match: print("Warning: 'trend_model; ... end;' block not found."); return []
    content = match.group(1); eq_raw = [eq.strip() for eq in content.split(';') if eq.strip()]
    processed_equations = []
    for line in eq_raw:
        if '=' in line:
            parts = line.split('=', 1); lhs, rhs = parts[0].strip(), parts[1].strip()
            lhs_match = re.match(r'([a-zA-Z_]\w*)\s*(\(\s*([+-]?\d+)\s*\))?', lhs)
            if lhs_match:
                base_lhs = lhs_match.group(1)
                if lhs_match.group(2) and lhs_match.group(3) != '0': print(f"Warning: Trend eq '{line}' has non-standard LHS. Assuming it defines '{base_lhs}(t)'."); processed_equations.append(f"({base_lhs}) - ({rhs})")
                else: processed_equations.append(f"({base_lhs}) - ({rhs})")
            else: print(f"Warning: Skipping malformed trend eq line: '{line}' - Cannot parse LHS")
        else: print(f"Warning: Trend eq '{line}' has no '='. Assuming 'expr = 0'."); processed_equations.append(line)
    return processed_equations

def extract_observation_declarations(model_string):
    obs_vars = []
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n'); cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " \n ".join(cleaned_lines)
    meas_eq_marker = re.search(r'\bmeas(?:urement)?_equations\b\s*;', processed_content, re.IGNORECASE | re.DOTALL)
    search_area = processed_content[:meas_eq_marker.start()] if meas_eq_marker else processed_content
    match = re.search(r'(?i)\bvarobs\b(.*?);', search_area, re.DOTALL | re.IGNORECASE)
    if match: content = match.group(1).replace('\n', ' ').strip(); obs_vars = [v for v in re.split(r'[,\s]+', content) if v and re.fullmatch(r'[a-zA-Z_]\w*', v)]; obs_vars = list(dict.fromkeys(obs_vars).keys())
    return obs_vars

def extract_measurement_equations(model_string):
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n'); cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " ".join(cleaned_lines)
    match = re.search(r'(?i)\bmeas(?:urement)?_equations\b\s*;(.*?)\bend\b\s*;', processed_content, re.DOTALL | re.IGNORECASE)
    if not match: print("Warning: 'measurement_equations; ... end;' block not found."); return []
    content = match.group(1); eq_raw = [eq.strip() for eq in content.split(';') if eq.strip()]
    processed_equations = []
    for line in eq_raw:
        if '=' in line:
            parts = line.split('=', 1)
            if len(parts) == 2: lhs, rhs = parts; processed_equations.append(f"({lhs.strip()}) - ({rhs.strip()})")
            else: print(f"Warning: Skipping malformed measurement eq line: '{line}'")
        else: print(f"Warning: Measurement eq '{line}' has no '='. Assuming 'expr = 0'."); processed_equations.append(line)
    return processed_equations

def extract_trend_shock_stderrs(model_string):
    stderrs = {}
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n'); cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " ".join(cleaned_lines)
    match = re.search(r'(?i)\btrend_shocks\b\s*;(.*?)\bend\b\s*;', processed_content, re.DOTALL | re.IGNORECASE)
    if not match: return {}
    content = match.group(1)
    stderr_matches = re.finditer(r'(?i)\bvar\s+([a-zA-Z_]\w*)\s*;\s*stderr\s+([^;]+);', content)
    for m in stderr_matches:
        shock_name = m.group(1); stderr_val_str = m.group(2).strip()
        try: stderr_val = float(stderr_val_str); param_name = f"sigma_{shock_name}"; stderrs[param_name] = stderr_val
        except ValueError: print(f"Warning: Could not parse stderr value '{stderr_val_str}' for trend shock '{shock_name}'.")
    return stderrs

def extract_model_shock_stderrs(model_string):
    stderrs = {}
    processed_content = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    lines = processed_content.split('\n'); cleaned_lines = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed_content = " ".join(cleaned_lines)
    match = re.search(r'(?i)\bshocks\b\s*;(.*?)\bend\b\s*;', processed_content, re.DOTALL | re.IGNORECASE)
    if not match: return {}
    content = match.group(1)
    stderr_matches = re.finditer(r'(?i)\bvar\s+([a-zA-Z_]\w*)\s*;\s*stderr\s+([^;]+);', content)
    for m in stderr_matches:
        shock_name = m.group(1); stderr_val_str = m.group(2).strip()
        try: stderr_val = float(stderr_val_str); param_name = f"sigma_{shock_name}"; stderrs[param_name] = stderr_val
        except ValueError: print(f"Warning: Could not parse stderr value '{stderr_val_str}' for trend shock '{shock_name}'.")
    return stderrs


def _parse_linear_rhs(rhs_str):
    """Parse a linear RHS expression into a list of (coeff, var_name, shift) tuples.

    Uses SymPy to handle arbitrary linear expressions, including global scalar
    multipliers applied to a parenthesised group.  Supports:

    - ``[+/-] [num*] VAR_NAME [(-n)]``  - standard per-term coefficients
    - ``scalar*(linear_expr)``          - e.g. ``4*(A - A(-1) + B(-1))``

    Examples::

        "GDP_GAP - GDP_GAP(-1) + G_TREND(-1)"
        -> [(1.0, 'GDP_GAP', 0), (-1.0, 'GDP_GAP', -1), (1.0, 'G_TREND', -1)]

        "4*(GDP_GAP - GDP_GAP(-1) + LGDP_BAR - LGDP_BAR(-1))"
        -> [(4.0, 'GDP_GAP', 0), (-4.0, 'GDP_GAP', -1),
           (4.0, 'LGDP_BAR', 0), (-4.0, 'LGDP_BAR', -1)]
    """
    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                             implicit_multiplication_application)

    s = re.sub(r'\s+', '', rhs_str)
    if not s:
        return []

    # Step 1: Replace lagged variables VAR(-n) / VAR(n) with unique symbol
    # names so that SymPy sees plain identifiers rather than function calls.
    lag_map = {}  # {unique_sym_name: (var_name, shift)}

    def _replace_lag(m):
        var_name, shift_str = m.group(1), m.group(2)
        shift = int(shift_str)
        tag = f"LAG{-shift}" if shift < 0 else f"LEAD{shift}"
        sym_name = f"{var_name}_{tag}"
        lag_map[sym_name] = (var_name, shift)
        return sym_name

    s_sub = re.sub(r'([A-Za-z_]\w*)\(([+-]?\d+)\)', _replace_lag, s)

    # Step 2: Build a SymPy local_dict covering all identifiers.
    all_idents = re.findall(r'[A-Za-z_]\w*', s_sub)
    local_dict = {name: sympy.Symbol(name) for name in set(all_idents)}

    # Step 3: Parse and expand (distributes scalar prefixes automatically).
    transformations = standard_transformations + (implicit_multiplication_application,)
    try:
        sym_expr = parse_expr(s_sub, local_dict=local_dict,
                              transformations=transformations)
    except Exception as e:
        raise ValueError(
            f"Could not parse reporting equation RHS: {rhs_str!r}\n"
            f"(after lag substitution: {s_sub!r})\nSymPy error: {e}"
        ) from e

    sym_expr = sympy.expand(sym_expr)

    # Step 4: Extract (coeff, var_name, shift) from the expanded polynomial.
    terms = []
    for sym, coeff in sym_expr.as_coefficients_dict().items():
        if sym == sympy.S.One:
            # Constant term - not meaningful in a reporting equation; skip.
            continue
        coeff_f = float(coeff)
        if coeff_f == 0.0:
            continue
        sym_name = str(sym)
        if sym_name in lag_map:
            var_name, shift = lag_map[sym_name]
        else:
            var_name, shift = sym_name, 0
        terms.append((coeff_f, var_name, shift))
    return terms


def extract_reporting_equations(model_string):
    """Parse the ``reporting_equations; ... end;`` block from a .qpm file.

    Each line inside the block must be a linear combination of augmented state
    variables (stationary or trend) with optional time lags::

        reporting_equations;
            // GDP growth: change in gap + trend growth
            DA_LGDP = GDP_GAP - GDP_GAP(-1) + G_TREND(-1);
        end;

    These variables are **not** part of the Kalman filter - they are computed
    purely as post-processing linear combinations of smoothed state
    contributions, enabling direct shock decomposition of derived quantities.

    Returns:
        reporting_vars (list[str]):  ordered list of reporting variable names.
        reporting_eqs  (dict):       {var_name: [(coeff, state_name, shift), ...]}
    """
    reporting_vars = []
    reporting_eqs = {}

    # Strip block comments
    processed = re.sub(r'/\*.*?\*/', '', model_string, flags=re.DOTALL)
    # Strip line comments
    lines = processed.split('\n')
    cleaned = [re.sub(r'(//|%).*$', '', ln).strip() for ln in lines]
    processed = ' '.join(cleaned)

    match = re.search(
        r'(?i)\breporting_equations\b\s*;(.*?)\bend\b\s*;',
        processed, re.DOTALL | re.IGNORECASE)
    if not match:
        return reporting_vars, reporting_eqs

    content = match.group(1)
    eq_raw = [eq.strip() for eq in content.split(';') if eq.strip()]
    for line in eq_raw:
        if '=' not in line:
            print(f"Warning: Reporting equation '{line}' has no '='. Skipping.")
            continue
        lhs, rhs = line.split('=', 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if not re.fullmatch(r'[A-Za-z_]\w*', lhs):
            print(f"Warning: Invalid LHS '{lhs}' in reporting equation. Skipping.")
            continue
        terms = _parse_linear_rhs(rhs)
        if not terms:
            print(f"Warning: Could not parse RHS '{rhs}' for '{lhs}'. Skipping.")
            continue
        if lhs not in reporting_vars:
            reporting_vars.append(lhs)
        reporting_eqs[lhs] = terms

    return reporting_vars, reporting_eqs


# --- State-Space Building Functions (JAX Compatible) ---

def build_trend_matrices(trend_equations, trend_vars, trend_shocks, param_names, param_assignments, verbose=True):
    if verbose: print("\n--- Building Trend State-Space Matrices (P_trends, Q_trends) ---")
    contemporaneous_defs = {}; state_trend_vars = []; defining_equations_for_state_trends = []
    var_time_regex = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]?\d+)\s*\)')
    base_var_regex = re.compile(r'\b([a-zA-Z_]\w*)\b')
    # if verbose: print("Analyzing trend equations for state definition...")
    for eq_str in trend_equations:
        eq_parts = eq_str.split('-', 1);
        if len(eq_parts) != 2: continue
        lhs_str = eq_parts[0].strip().strip('()'); rhs_str = eq_parts[1].strip().strip('()')
        if lhs_str not in trend_vars: continue
        rhs_symbols = set(base_var_regex.findall(rhs_str)); has_contemporaneous_rhs = False
        for sym in rhs_symbols:
            if sym in trend_vars and sym != lhs_str:
                 explicit_lag = f"{sym}(-1)"
                 if sym in rhs_str and explicit_lag not in rhs_str: has_contemporaneous_rhs = True; break
        for match in var_time_regex.finditer(rhs_str):
            base, shift = match.group(1), int(match.group(2))
            if base in trend_vars and shift != -1: has_contemporaneous_rhs = True; break
        if has_contemporaneous_rhs:
            # if verbose: print(f"  Trend '{lhs_str}' defined contemporaneously: {eq_str}.")
            contemporaneous_defs[lhs_str] = rhs_str
        else:
            if lhs_str not in state_trend_vars: state_trend_vars.append(lhs_str)
            defining_equations_for_state_trends.append(eq_str)

    if not state_trend_vars:
        if verbose: print("Warning: No state trend variables identified. Returning empty matrices.")
        # Return lambdas returning empty JAX arrays
        dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        return (lambda *a: jnp.empty((0,0), dtype=dtype),
                lambda *a: jnp.empty((0,0), dtype=dtype),
                lambda *a: jnp.empty((0,), dtype=dtype),
                [], {})

    # if verbose: print(f"Identified state trend variables ({len(state_trend_vars)}): {state_trend_vars}")
    num_state_trends = len(state_trend_vars); num_trend_shocks = len(trend_shocks)
    param_syms = {p: sympy.symbols(p) for p in param_names}; trend_shock_syms = {s: sympy.symbols(s) for s in trend_shocks}
    trend_var_syms = {}; all_syms = set(param_syms.values()) | set(trend_shock_syms.values())
    for var in state_trend_vars: sym_m1 = create_timed_symbol(var, -1); sym_t = create_timed_symbol(var, 0); trend_var_syms[var] = {'m1': sym_m1, 't': sym_t}; all_syms.update([sym_m1, sym_t])
    for var in trend_vars:
         if var not in trend_var_syms: sym_m1 = create_timed_symbol(var, -1); sym_t = create_timed_symbol(var, 0); trend_var_syms[var] = {'m1': sym_m1, 't': sym_t}; all_syms.add(sym_m1); all_syms.add(sym_t)
    local_dict = {str(s): s for s in all_syms}; local_dict.update({'log': sympy.log, 'exp': sympy.exp, 'sqrt': sympy.sqrt, 'abs': sympy.Abs})
    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application, rationalize)
    transformations = (standard_transformations + (implicit_multiplication_application, rationalize))
    sym_trend_equations = []; # if verbose: print("Parsing state trend equations into symbolic form...")
    parsed_eq_for_state_var = {var: None for var in state_trend_vars}; final_sym_equations_ordered = []
    for i, eq_str in enumerate(defining_equations_for_state_trends):
        eq_parts = eq_str.split('-', 1); lhs_var = eq_parts[0].strip().strip('()')
        if lhs_var not in state_trend_vars: continue
        eq_str_sym = eq_str
        def replace_trend_time(match):
            base, shift_str = match.groups(); shift = int(shift_str)
            if base in trend_shock_syms:
                if shift == 0: return str(trend_shock_syms[base])
                else: raise ValueError(f"Trend shock {base}({shift}) invalid.")
            elif base in trend_var_syms:
                 if shift == -1: return str(trend_var_syms[base]['m1'])
                 if shift == 0 and base == lhs_var: return str(trend_var_syms[base]['t'])
                 raise ValueError(f"Unexpected term {base}({shift}) in state trend eq for {lhs_var}.")
            elif base in param_syms: raise ValueError(f"Parameter {base}({shift}) invalid.")
            else: sym = sympy.symbols(f"{base}_t{shift:+d}"); return str(sym) # Use +d for sign
        eq_str_sym = var_time_regex.sub(replace_trend_time, eq_str_sym)
        all_base_names = sorted(list(trend_var_syms.keys()) + param_names + list(trend_shock_syms.keys()), key=len, reverse=True)
        for name in all_base_names:
            pattern = r'\b' + re.escape(name) + r'\b'
            if name == lhs_var: replacement = str(trend_var_syms[name]['t'])
            elif name in trend_var_syms: continue
            elif name in param_syms: replacement = str(param_syms[name])
            elif name in trend_shock_syms: replacement = str(trend_shock_syms[name])
            else: continue
            eq_str_sym = re.sub(pattern, replacement, eq_str_sym)
        try: sym_eq = parse_expr(eq_str_sym, local_dict=local_dict, transformations=transformations); parsed_eq_for_state_var[lhs_var] = sym_eq
        except Exception as e: print(f"Error parsing trend eq: '{eq_str}' -> '{eq_str_sym}'\n{e}"); raise
    for var in state_trend_vars:
         eq = parsed_eq_for_state_var[var]
         if eq is None: raise RuntimeError(f"Missing defining eq for state trend var '{var}'.")
         final_sym_equations_ordered.append(eq)
    # if verbose: print("Symbolic parsing of state trend equations completed.")
    sympy_P_trends = sympy.zeros(num_state_trends, num_state_trends); sympy_Q_trends = sympy.zeros(num_state_trends, num_trend_shocks)
    state_trend_m1_syms = [trend_var_syms[v]['m1'] for v in state_trend_vars]; trend_shock_t_syms = [trend_shock_syms[s] for s in trend_shocks]
    for i, eq in enumerate(final_sym_equations_ordered):
        for j, var_m1 in enumerate(state_trend_m1_syms): sympy_P_trends[i, j] = -sympy.diff(eq, var_m1)
        for k, shock_t in enumerate(trend_shock_t_syms): sympy_Q_trends[i, k] = -sympy.diff(eq, shock_t)
    # if verbose: print("Symbolic P_trends and Q_trends matrices generated.")
    # Intercept: constant term in each trend equation (nonzero for mean-reverting trends)
    # Equations are in residual form: eq = x_t - rhs.  Setting all variables/shocks to 0
    # gives eq -> -c, so c = -eq.subs(all_symbols=0).
    state_trend_t_syms = [trend_var_syms[v]['t'] for v in state_trend_vars]
    subs_zero = {s: 0 for s in state_trend_m1_syms + trend_shock_t_syms + state_trend_t_syms}
    sympy_c_trends = sympy.zeros(num_state_trends, 1)
    for i, eq in enumerate(final_sym_equations_ordered):
        sympy_c_trends[i, 0] = -eq.subs(subs_zero)
    # if verbose: print("Lambdifying trend matrices...")
    param_sym_list = [param_syms[p] for p in param_names]
    func_P_trends = robust_lambdify(param_sym_list, sympy_P_trends)
    func_Q_trends = robust_lambdify(param_sym_list, sympy_Q_trends)
    func_c_trends = robust_lambdify(param_sym_list, sympy_c_trends)
    # if verbose: print("Lambdification successful.")
    return func_P_trends, func_Q_trends, func_c_trends, state_trend_vars, contemporaneous_defs

def build_observation_matrix(measurement_equations, obs_vars, stationary_vars,
                             trend_state_vars, contemporaneous_trend_defs,
                             param_names, param_assignments, verbose=True,
                             eliminated_stat_defs=None):
    """Build the observation matrix Omega (n_obs x n_aug).

    Parameters
    ----------
    eliminated_stat_defs : dict {var_name: sympy_expr} or None
        Back-substitution expressions for eliminated stationary variables.
        When an eliminated variable appears in a measurement equation, it is
        replaced by its expression in terms of the *reduced* stationary state.
        These expressions may contain parameter symbols.
    """
    if eliminated_stat_defs is None:
        eliminated_stat_defs = {}
    if verbose: print("\n--- Building Observation Matrix (Omega) ---")
    num_obs = len(obs_vars); num_stationary = len(stationary_vars); num_trend_state = len(trend_state_vars)
    num_augmented_state = num_stationary + num_trend_state
    if num_obs == 0:
        if verbose: print("No observable variables declared. Returning empty Omega.")
        dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        return lambda *a: jnp.empty((0, num_augmented_state), dtype=dtype), []
    if len(measurement_equations) != num_obs: raise ValueError(f"Num measurement eqs ({len(measurement_equations)}) != num varobs ({num_obs}).")

    param_syms = {p: sympy.symbols(p) for p in param_names}; obs_syms = {v: sympy.symbols(v) for v in obs_vars}
    # Include eliminated stat vars as known symbols on the RHS (they will be substituted out below)
    all_rhs_vars = stationary_vars + trend_state_vars + list(contemporaneous_trend_defs.keys()) + list(eliminated_stat_defs.keys())
    rhs_var_syms = {v: sympy.symbols(v) for v in all_rhs_vars}
    all_syms = set(param_syms.values()) | set(obs_syms.values()) | set(rhs_var_syms.values())
    local_dict = {str(s): s for s in all_syms}; local_dict.update({'log': sympy.log, 'exp': sympy.exp, 'sqrt': sympy.sqrt, 'abs': sympy.Abs})
    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application, rationalize)
    transformations = (standard_transformations + (implicit_multiplication_application, rationalize))
    sym_measurement_equations = []; # if verbose: print("Parsing measurement equations into symbolic form...")
    parsed_eq_for_obs_var = {var: None for var in obs_vars}
    for eq_str in measurement_equations:
        eq_parts = eq_str.split('-', 1);
        if len(eq_parts) != 2: continue
        lhs_var = eq_parts[0].strip().strip('()')
        if lhs_var not in obs_vars: continue
        rhs_str = eq_parts[1].strip().strip('()'); rhs_processed = rhs_str
        # Substitute contemporaneous trend definitions (string-level, as before)
        n_contemp_iters = len(contemporaneous_trend_defs) + len(eliminated_stat_defs) + 1
        for _ in range(n_contemp_iters):
            made_substitution = False
            for contemp_var, contemp_expr_str in contemporaneous_trend_defs.items():
                pattern = r'\b' + re.escape(contemp_var) + r'\b'
                if re.search(pattern, rhs_processed): rhs_processed = re.sub(pattern, f"({contemp_expr_str})", rhs_processed); made_substitution = True
            if not made_substitution: break
        eq_str_subbed = f"{lhs_var} - ({rhs_processed})"
        eq_str_sym = eq_str_subbed
        all_base_names_obs = sorted(list(obs_syms.keys()) + list(rhs_var_syms.keys()) + param_names, key=len, reverse=True)
        for name in all_base_names_obs:
             pattern = r'\b' + re.escape(name) + r'\b'
             if name in obs_syms: replacement = str(obs_syms[name])
             elif name in rhs_var_syms: replacement = str(rhs_var_syms[name])
             elif name in param_syms: replacement = str(param_syms[name])
             else: continue
             eq_str_sym = re.sub(pattern, replacement, eq_str_sym)
        try: sym_eq = parse_expr(eq_str_sym, local_dict=local_dict, transformations=transformations); parsed_eq_for_obs_var[lhs_var] = sym_eq
        except Exception as e: print(f"Error parsing measurement eq: '{eq_str}' -> '{eq_str_sym}'\n{e}"); raise
    final_sym_equations_ordered = []
    for var in obs_vars:
         eq = parsed_eq_for_obs_var[var]
         if eq is None: raise RuntimeError(f"Missing measurement eq for observable '{var}'.")
         # Substitute eliminated stationary variables at the symbolic level.
         # eliminated_stat_defs maps var_name -> sympy_expr (in terms of time-t reduced state).
         # The expression's symbols are the *reduced* stationary vars (already in rhs_var_syms).
         if eliminated_stat_defs:
             for elim_var, elim_expr in eliminated_stat_defs.items():
                 elim_sym = rhs_var_syms.get(elim_var)
                 if elim_sym is not None:
                     eq = eq.subs(elim_sym, elim_expr)
         final_sym_equations_ordered.append(eq)
    # if verbose: print("Symbolic parsing of measurement equations completed.")
    # Omega columns correspond to the *reduced* stationary state + trend state.
    # Eliminated stationary vars have already been substituted out of the equations.
    sympy_Omega = sympy.zeros(num_obs, num_augmented_state)
    augmented_state_syms = [rhs_var_syms[v] for v in stationary_vars] + [rhs_var_syms[v] for v in trend_state_vars]
    for i, eq in enumerate(final_sym_equations_ordered):
        for j, state_sym in enumerate(augmented_state_syms):
            sympy_Omega[i, j] = -sympy.diff(eq, state_sym)
    # if verbose: print("Symbolic Omega matrix generated.")
    # if verbose: print("Lambdifying observation matrix...")
    param_sym_list = [param_syms[p] for p in param_names]
    func_Omega = robust_lambdify(param_sym_list, sympy_Omega)
    # if verbose: print("Lambdification successful.")
    return func_Omega, obs_vars

# --- IRF Calculation Functions (JAX) ---

def irf(P: ArrayLike, R: ArrayLike, shock_index: int, horizon: int = 40) -> jax.Array:
    """
    Compute impulse responses for y_t = P y_{t-1} + R eps_t (eps_t ~ N(0,I)).
    Uses JAX arrays and lax.scan.

    Args:
        P: State transition matrix (n x n).
        R: Shock impact matrix (n x n_shock).
        shock_index: Index of the shock to perturb (0-based).
        horizon: Number of periods for the IRF.

    Returns:
        jax.Array: IRF values (horizon x n).
    """
    P_jax = jnp.asarray(P); R_jax = jnp.asarray(R)
    n = P_jax.shape[0]; n_shock = R_jax.shape[1]
    dtype = P_jax.dtype
    if not (0 <= shock_index < n_shock): raise ValueError(f"shock_index must be 0 <= index < {n_shock}")

    y_resp = jnp.zeros((horizon, n), dtype=dtype)
    # Unit shock impulse at time 0
    impulse = R_jax[:, shock_index] # Select column corresponding to shock

    # y_0 = P*y_{-1} + impulse. Assume y_{-1} = 0.
    y_current = impulse
    y_resp = y_resp.at[0, :].set(y_current)

    def step(y_prev, _):
        y_next = P_jax @ y_prev
        return y_next, y_next

    # Scan for remaining steps
    _, y_resp_scan = jax.lax.scan(step, y_current, xs=None, length=horizon - 1)
    y_resp = y_resp.at[1:, :].set(y_resp_scan)

    # Clean very small values
    y_resp_clean = jnp.where(jnp.abs(y_resp) < 1e-14, 0.0, y_resp)
    return y_resp_clean

def irf_observables(P_aug: ArrayLike, R_aug: ArrayLike, Omega: ArrayLike,
                     shock_index: int, horizon: int = 40) -> jax.Array:
    """
    Compute impulse responses for observable variables.
    obs(t) = Omega * state_aug(t)

    Args:
        P_aug: Augmented state transition matrix (n_aug x n_aug).
        R_aug: Augmented shock impact matrix (n_aug x n_aug_shock).
        Omega: Observation matrix (n_obs x n_aug).
        shock_index: Index of the shock in the *augmented* shock vector.
        horizon: Number of periods.

    Returns:
        jax.Array: Observable IRF values (horizon x n_obs).
    """
    P_aug_jax = jnp.asarray(P_aug); R_aug_jax = jnp.asarray(R_aug); Omega_jax = jnp.asarray(Omega)
    n_aug = P_aug_jax.shape[0]; n_aug_shock = R_aug_jax.shape[1]; n_obs = Omega_jax.shape[0]
    if not (0 <= shock_index < n_aug_shock): raise ValueError(f"Aug shock_index must be 0 <= index < {n_aug_shock}")
    if Omega_jax.shape[1] != n_aug: raise ValueError(f"Omega columns ({Omega_jax.shape[1]}) != P_aug dim ({n_aug}).")

    # Compute state IRF using the JAX function
    state_irf = irf(P_aug_jax, R_aug_jax, shock_index, horizon) # (horizon, n_aug)

    # Map state responses to observables: obs_irf = state_irf @ Omega.T
    obs_irf = state_irf @ Omega_jax.T # (horizon, n_obs)

    # Clean very small values
    obs_irf_clean = jnp.where(jnp.abs(obs_irf) < 1e-14, 0.0, obs_irf)
    return obs_irf_clean

# This function only is use to setup the initial conditions of the simulation
def construct_initial_state(
    n_aug: int,
    n_stat: int,
    aug_state_vars: List[str],
    key_init: jax.random.PRNGKey,
    initial_state_config: Optional[Dict[str, Dict[str, float]]] = None,
    default_trend_std: float = 0.01, # Default std for unspecified trends
    dtype: Optional[jnp.dtype] = None
) -> jax.Array:
    """
    Constructs the initial state vector s0 for simulation or filtering.

    Allows setting specific means and standard deviations for trend variables
    via the initial_state_config dictionary. Stationary variables default to zero mean.

    Args:
        n_aug: Total number of augmented state variables.
        n_stat: Number of stationary variables.
        aug_state_vars: Ordered list of all augmented state variable names.
        key_init: JAX PRNG key for drawing initial random components.
        initial_state_config: Dictionary specifying initial states, e.g.,
            {
                "VAR_NAME_1": {"mean": value1, "std": std_dev1},
                "VAR_NAME_2": {"mean": value2}, # std defaults to default_trend_std
                "VAR_NAME_3": {"std": std_dev3}, # mean defaults to 0.0
                ...
            }. Applies primarily to trend variables.
        default_trend_std: Default standard deviation for trend variables not
                           specified in the config.
        dtype: JAX dtype for the state vector.

    Returns:
        jax.Array: The constructed initial state vector s0 (size n_aug).
    """
    dtype = dtype or get_dtype()
    n_trend = n_aug - n_stat
    s0 = jnp.zeros(n_aug, dtype=dtype) # Start with zeros for all states

    if n_trend > 0:
        # Set defaults for ALL trend variables first
        trend_means = jnp.zeros(n_trend, dtype=dtype)
        trend_stds = jnp.full(n_trend, default_trend_std, dtype=dtype)

        if initial_state_config:
            print("\n--- Applying Initial State Configuration for Trends ---")
            trend_state_vars = aug_state_vars[n_stat:] # Names of trend vars only

            for var_name, config in initial_state_config.items():
                if not isinstance(config, dict):
                    print(f"Warning: Invalid config format for '{var_name}' (expected dict). Skipping.")
                    continue

                if var_name in trend_state_vars:
                    try:
                        # Find index *within the trend block*
                        trend_idx = trend_state_vars.index(var_name)

                        # Get mean and std, falling back to defaults if not provided
                        mean_val = config.get("mean")
                        std_val = config.get("std")

                        if mean_val is not None:
                            val = float(mean_val)
                            trend_means = trend_means.at[trend_idx].set(val)
                            print(f"  Set initial mean for '{var_name}' (trend idx {trend_idx}) to: {val:.4f}")
                        # else: Mean remains default 0.0

                        if std_val is not None:
                            val = jnp.maximum(0.0, float(std_val)) # Ensure non-negative
                            trend_stds = trend_stds.at[trend_idx].set(val)
                            print(f"  Set initial std dev for '{var_name}' (trend idx {trend_idx}) to: {val:.4f}")
                        # else: Std dev remains default_trend_std

                    except ValueError: # Should not happen if var_name in trend_state_vars
                        print(f"Warning: Internal error finding index for trend var '{var_name}'.")
                    except (TypeError, KeyError) as e:
                        print(f"Warning: Invalid format/value in initial state config for '{var_name}': {config}. Error: {e}. Skipping.")
                else:
                    # Optionally warn if config given for a stationary var or unknown var
                    if var_name in aug_state_vars[:n_stat]:
                        print(f"Info: Initial state config provided for stationary var '{var_name}'. Ignoring (mean defaults to 0).")
                    else:
                        print(f"Warning: Variable '{var_name}' in initial_state_config not found in trend state list. Skipping.")


        # Draw initial trends from N(mean, std^2) using the final mean/std arrays
        print("\nDrawing initial trend states from N(mean, std^2)...")
        initial_trends = trend_means + trend_stds * jax.random.normal(key_init, (n_trend,), dtype=dtype)

        # Place the drawn values into the s0 vector
        s0 = s0.at[n_stat:].set(initial_trends)
        print(f"Initial state vector s0 constructed (shape {s0.shape}).")

    else:
        print("No trend variables, initial state s0 defaults to zeros.")


    return s0



# --- Smoke test ---
if __name__ == "__main__":
    import jax
    jax.config.update("jax_enable_x64", True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mod_file_path = os.path.join(script_dir, "qpm_simpl1_with_trends.dyn")
    with open(mod_file_path, 'r') as f:
        model_def = f.read()

    # Parse
    (func_A, func_B, func_C, func_D, ordered_stat_vars, stat_shocks,
     param_names_stat, param_assignments_stat, _, _) = parse_lambdify_and_order_model(model_def, verbose=False)

    trend_vars, trend_shocks = extract_trend_declarations(model_def)
    trend_equations = extract_trend_equations(model_def)
    obs_vars = extract_observation_declarations(model_def)
    measurement_equations = extract_measurement_equations(model_def)
    trend_stderr_params = extract_trend_shock_stderrs(model_def)

    all_param_names = list(dict.fromkeys(
        param_names_stat + list(trend_stderr_params.keys())).keys())
    all_param_assignments = param_assignments_stat.copy()
    all_param_assignments.update(trend_stderr_params)
    all_param_assignments.update({
        'b1': 0.75, 'b4': 0.75, 'a1': 0.5, 'a2': 0.1,
        'g1': 0.7, 'g2': 0.3, 'g3': 0.25,
        'rho_L_GDP_GAP': 0.8, 'rho_DLA_CPI': 0.7,
        'rho_rs': 0.75, 'rho_rs2': 0.01,
    })

    (func_P_trends, func_Q_trends, func_c_trends,
     ordered_trend_state_vars, contemp_trend_defs) = build_trend_matrices(
        trend_equations, trend_vars, trend_shocks,
        all_param_names, all_param_assignments, verbose=False)
    (func_Omega, ordered_obs_vars) = build_observation_matrix(
        measurement_equations, obs_vars, ordered_stat_vars,
        ordered_trend_state_vars, contemp_trend_defs,
        all_param_names, all_param_assignments, verbose=False)

    # Solve stationary block (SDA)
    stat_args = [all_param_assignments.get(p, 1.0) for p in param_names_stat]
    A = jnp.asarray(func_A(*stat_args))
    B = jnp.asarray(func_B(*stat_args))
    C_mat = jnp.asarray(func_C(*stat_args))
    D = jnp.asarray(func_D(*stat_args))

    P_sol, iters, resid, converged = solve_quadratic_matrix_equation_jax(A, B, C_mat)
    assert converged, f"SDA failed: resid={resid:.2e}"
    Q_sol = compute_Q_jax(A, B, D, P_sol)

    # Build augmented system
    all_args = [all_param_assignments.get(p, 1.0) for p in all_param_names]
    P_trend = jnp.asarray(func_P_trends(*all_args))
    Q_trend = jnp.asarray(func_Q_trends(*all_args))
    Omega = jnp.asarray(func_Omega(*all_args))

    n_stat = len(ordered_stat_vars)
    n_trend = len(ordered_trend_state_vars)
    aug_shocks = stat_shocks + trend_shocks

    shock_stds = {s: all_param_assignments.get(f"sigma_{s}", 1.0) for s in aug_shocks}
    R_stat = Q_sol @ jnp.diag(jnp.array([shock_stds[s] for s in stat_shocks]))
    R_trend = Q_trend @ jnp.diag(jnp.array([shock_stds[s] for s in trend_shocks]))

    P_aug = jax.scipy.linalg.block_diag(P_sol, P_trend)
    n_aug = n_stat + n_trend
    R_aug = jnp.zeros((n_aug, len(aug_shocks)), dtype=P_aug.dtype)
    R_aug = R_aug.at[:n_stat, :len(stat_shocks)].set(R_stat)
    R_aug = R_aug.at[n_stat:, len(stat_shocks):].set(R_trend)

    # Compute IRF
    irf_vals = irf_observables(P_aug, R_aug, Omega,
                               shock_index=aug_shocks.index("SHK_RS"), horizon=40)
    assert jnp.all(jnp.isfinite(irf_vals)), "IRF contains non-finite values"

    print(f"Smoke test passed: dtype={get_dtype()}, "
          f"n_stat={n_stat}, n_trend={n_trend}, "
          f"SDA resid={resid:.2e}, IRF shape={irf_vals.shape}")