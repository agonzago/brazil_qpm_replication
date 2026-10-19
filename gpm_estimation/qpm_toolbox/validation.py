"""Model file (.qpm/.dyn) validator.

Checks that a QPM model file is well-formed before parsing:
  - All required sections are present
  - All variables referenced in equations are declared
  - All parameters referenced are declared
  - All shocks referenced are declared
  - No undefined symbols in the model

The extraction logic mirrors parser.py so that the validator and
the actual parser agree on what is declared.

Usage:
    from qpm_toolbox.validation import validate_model_file
    issues = validate_model_file("models/brazil_qpm.qpm")
    if issues:
        for issue in issues:
            print(f"  {issue}")
"""

import re
from typing import List, Optional, Set


# ======================================================================
# Low-level helpers
# ======================================================================

def _preprocess(raw: str) -> str:
    """Strip block and line comments, return cleaned text."""
    text = re.sub(r'/\*.*?\*/', '', raw, flags=re.DOTALL)
    lines = text.split('\n')
    cleaned = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    return " \n ".join(cleaned)


def _names_from_content(content: str) -> List[str]:
    """Split content string into valid identifier names."""
    content = content.replace('\n', ' ')
    raw = re.split(r'[,\s]+', content.strip())
    keywords = {'var', 'varexo', 'varexo_trends', 'trends_vars',
                'parameters', 'model', 'end', 'trend_model',
                'varobs', 'measurement_equations', 'shocks',
                'trend_shocks', 'initval', 'steady', 'check',
                'stoch_simul'}
    return [n for n in raw
            if n and re.fullmatch(r'[a-zA-Z_]\w*', n) and n not in keywords]


def _extract_block(text: str, block_name: str) -> Optional[str]:
    """Extract content of  ``block_name; ... end;``."""
    pattern = rf'\b{block_name}\b\s*;(.*?)\bend\b\s*;'
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else None


def _extract_symbols(equations_text: str) -> Set[str]:
    """Return all identifier-like symbols from equation text.

    Time subscripts like ``(+1)``, ``(-1)`` are removed first.
    """
    cleaned = re.sub(r'\(\s*[+-]?\d+\s*\)', '', equations_text)
    tokens = set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', cleaned))
    reserved = {
        'end', 'model', 'var', 'varexo', 'parameters', 'stderr',
        'trend_model', 'trend_shocks', 'shocks', 'varobs',
        'measurement_equations', 'steady_state_model',
        'trends_vars', 'varexo_trends', 'initval', 'steady', 'check',
        'stoch_simul', 'order', 'irf',
        'log', 'exp', 'sqrt', 'abs', 'sin', 'cos', 'pi',
        'inf', 'nan', 'true', 'false',
    }
    return {t for t in tokens if t not in reserved}


# ======================================================================
# Declaration extraction  (mirrors parser.py logic)
# ======================================================================

def _extract_stationary_declarations(text: str):
    """Extract var, varexo, parameters from the region BEFORE ``model;``.

    Returns (stat_vars, stat_shocks, declared_params, param_assignments).
    """
    model_marker = re.search(r'\bmodel\b\s*;', text, re.IGNORECASE)
    area = text[:model_marker.start()] if model_marker else text

    declarations = {'var': [], 'varexo': [], 'parameters': []}

    block_re = re.compile(
        r'(?i)\b(var|varexo|parameters)\b(.*?)'
        r'(?=\b(?:var|varexo|parameters|model)\b|$)',
        re.DOTALL | re.IGNORECASE,
    )
    for m in block_re.finditer(area):
        kw = m.group(1).lower()
        content = m.group(2).strip()
        # Take only up to first semicolon
        sc = content.find(';')
        if sc >= 0:
            content = content[:sc]
        declarations[kw].extend(_names_from_content(content))

    # De-duplicate while preserving order
    for k in declarations:
        declarations[k] = list(dict.fromkeys(declarations[k]))

    # Parameter assignments
    param_assignments = {}
    param_set = set(declarations['parameters'])
    for m in re.finditer(r'\b([a-zA-Z_]\w*)\s*=\s*([^;]+);', area):
        name = m.group(1)
        if name in param_set:
            try:
                param_assignments[name] = float(m.group(2).strip())
            except ValueError:
                pass

    return (set(declarations['var']),
            set(declarations['varexo']),
            set(declarations['parameters']),
            param_assignments)


def _extract_trend_declarations(text: str):
    """Extract trends_vars and varexo_trends from region BEFORE ``trend_model;``.

    Returns (trend_vars, trend_shocks).
    """
    trend_vars: Set[str] = set()
    trend_shocks: Set[str] = set()

    marker = re.search(r'\btrend_model\b\s*;', text, re.IGNORECASE)
    area = text[:marker.start()] if marker else text

    m_tv = re.search(r'(?i)\btrends_vars\b(.*?);', area, re.DOTALL)
    if m_tv:
        trend_vars = set(_names_from_content(m_tv.group(1)))

    m_vt = re.search(r'(?i)\bvarexo_trends\b(.*?);', area, re.DOTALL)
    if m_vt:
        trend_shocks = set(_names_from_content(m_vt.group(1)))

    return trend_vars, trend_shocks


def _extract_observables(text: str) -> Set[str]:
    """Extract varobs names from region BEFORE ``measurement_equations;``."""
    marker = re.search(r'\bmeas(?:urement)?_equations\b\s*;', text,
                       re.IGNORECASE)
    area = text[:marker.start()] if marker else text
    m = re.search(r'(?i)\bvarobs\b(.*?);', area, re.DOTALL)
    if m:
        return set(_names_from_content(m.group(1)))
    return set()


# ======================================================================
# Main validator
# ======================================================================

def validate_model_file(filepath: str) -> List[str]:
    """Validate a .qpm/.dyn model file for completeness and consistency.

    Checks:
      1. Required sections present (var, model, varobs, measurement_equations)
      2. All symbols in model block declared as var / varexo / parameter
      3. All symbols in trend_model block declared as trends_vars / varexo_trends / parameter
      4. All symbols in measurement_equations declared appropriately
      5. Shock stderr declarations match declared shocks
      6. Observables appear in measurement equations
      7. Equation counts match variable counts

    Returns:
        List of issue strings.  Empty list => file is valid.
    """
    with open(filepath, 'r') as f:
        raw = f.read()

    text = _preprocess(raw)
    issues: List[str] = []

    # ------------------------------------------------------------------
    # 1. Required sections
    # ------------------------------------------------------------------
    required = {
        'var':                    r'\bvar\b\s+\w',
        'model':                  r'\bmodel\s*;',
        'varobs':                 r'\bvarobs\b',
        'measurement_equations':  r'\bmeas(?:urement)?_equations\s*;',
    }
    for name, pat in required.items():
        if not re.search(pat, text, re.IGNORECASE):
            issues.append(f"MISSING SECTION: '{name}' block not found.")

    has_trend_model  = bool(re.search(r'\btrend_model\s*;', text, re.IGNORECASE))
    has_shocks       = bool(re.search(r'\bshocks\s*;', text, re.IGNORECASE))
    has_trend_shocks = bool(re.search(r'\btrend_shocks\s*;', text, re.IGNORECASE))

    # ------------------------------------------------------------------
    # 2. Extract all declarations
    # ------------------------------------------------------------------
    stat_vars, stat_shocks, declared_params, _ = \
        _extract_stationary_declarations(text)

    if has_trend_model:
        trend_vars, trend_shocks = _extract_trend_declarations(text)
    else:
        trend_vars, trend_shocks = set(), set()

    obs_vars = _extract_observables(text)

    # ------------------------------------------------------------------
    # 3. Check symbols in model block
    # ------------------------------------------------------------------
    model_block = _extract_block(text, 'model')
    if model_block:
        syms = _extract_symbols(model_block)
        allowed = stat_vars | stat_shocks | declared_params
        for s in sorted(syms - allowed):
            issues.append(
                f"UNDEFINED in model block: '{s}' is not declared "
                f"as var, varexo, or parameter.")

    # ------------------------------------------------------------------
    # 4. Check symbols in trend_model block
    # ------------------------------------------------------------------
    trend_block = _extract_block(text, 'trend_model')
    if trend_block:
        syms = _extract_symbols(trend_block)
        allowed = trend_vars | trend_shocks | declared_params
        for s in sorted(syms - allowed):
            issues.append(
                f"UNDEFINED in trend_model block: '{s}' is not declared "
                f"as trends_vars, varexo_trends, or parameter.")

    # ------------------------------------------------------------------
    # 5. Check symbols in measurement_equations
    # ------------------------------------------------------------------
    meas_block = _extract_block(text, 'measurement_equations')
    if meas_block:
        syms = _extract_symbols(meas_block)
        allowed = obs_vars | stat_vars | trend_vars | declared_params
        for s in sorted(syms - allowed):
            issues.append(
                f"UNDEFINED in measurement_equations: '{s}' is not declared "
                f"as observable, var, trends_vars, or parameter.")

    # ------------------------------------------------------------------
    # 6. Shock stderr declarations
    # ------------------------------------------------------------------
    if has_shocks:
        sblk = _extract_block(text, 'shocks')
        if sblk:
            stderr_names = set(re.findall(
                r'\bvar\s+(\w+)\s*;\s*stderr', sblk))
            for s in sorted(stderr_names - stat_shocks):
                issues.append(
                    f"SHOCK STDERR: '{s}' has stderr in shocks block "
                    f"but is not declared in varexo.")

    if has_trend_shocks:
        tblk = _extract_block(text, 'trend_shocks')
        if tblk:
            stderr_names = set(re.findall(
                r'\bvar\s+(\w+)\s*;\s*stderr', tblk))
            for s in sorted(stderr_names - trend_shocks):
                issues.append(
                    f"TREND SHOCK STDERR: '{s}' has stderr in trend_shocks "
                    f"but is not declared in varexo_trends.")

    # ------------------------------------------------------------------
    # 7. Observables appear in measurement equations
    # ------------------------------------------------------------------
    if meas_block:
        meas_all = set(re.findall(r'\b([A-Za-z_]\w*)\b', meas_block))
        for obs in sorted(obs_vars):
            if obs not in meas_all:
                issues.append(
                    f"UNUSED OBSERVABLE: '{obs}' is in varobs but does "
                    f"not appear in measurement_equations.")

    # ------------------------------------------------------------------
    # 8. Equation count vs variable count
    # ------------------------------------------------------------------
    if model_block:
        eq_lines = [l.strip() for l in model_block.split(';')
                    if '=' in l and l.strip()]
        n_eq  = len(eq_lines)
        n_var = len(stat_vars)
        if n_eq != n_var:
            issues.append(
                f"EQUATION COUNT: {n_eq} equations in model block vs "
                f"{n_var} declared variables (should be equal).")

    if meas_block:
        meq = [l.strip() for l in meas_block.split(';')
               if '=' in l and l.strip()]
        if len(meq) != len(obs_vars):
            issues.append(
                f"MEASUREMENT COUNT: {len(meq)} measurement equations vs "
                f"{len(obs_vars)} observables in varobs (should be equal).")

    # ------------------------------------------------------------------
    # 9. Trend equation count vs trend variable count
    # ------------------------------------------------------------------
    if trend_block:
        teq = [l.strip() for l in trend_block.split(';')
               if '=' in l and l.strip()]
        n_teq = len(teq)
        n_tv  = len(trend_vars)
        if n_teq != n_tv:
            issues.append(
                f"TREND EQ COUNT: {n_teq} equations in trend_model vs "
                f"{n_tv} declared trends_vars (should be equal).")

    return issues


def validate_and_report(filepath: str) -> bool:
    """Validate a model file and print a report.

    Returns True if the file is valid, False otherwise.
    """
    print(f"\nValidating model file: {filepath}")
    issues = validate_model_file(filepath)

    if not issues:
        print("  Model file is valid. All checks passed.")
        return True

    print(f"  Found {len(issues)} issue(s):")
    for issue in issues:
        print(f"    - {issue}")
    return False
