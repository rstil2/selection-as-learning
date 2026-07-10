"""Utilities for SSD manifold theory: V_A(SSD), N_e*, and sensitivity analysis."""

from __future__ import annotations

import math
from typing import Iterable


def va_ssd(v_af: float, v_am: float, r_mf: float) -> float:
    """Additive variance in the sex-divergence direction (single trait)."""
    return max(v_af + v_am - 2.0 * r_mf * math.sqrt(v_af * v_am), 0.0)


def h2_ssd(v_af: float, v_am: float, r_mf: float, v_p: float) -> float:
    """Heritability of the divergence phenotype I = z_f - z_m."""
    v = va_ssd(v_af, v_am, r_mf)
    return v / v_p if v_p > 0 else 0.0


def ne_star(
    v_a: float,
    v_p: float,
    h2: float,
    epsilon: float,
    delta: float,
    L: int,
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> float:
    """
    Solve Corollary 6 from Stillwell (in prep.):
    N_e* = (2 V_P / (h² ε²)) · [L ln(e N_e*/L) + ln(4/δ)].

    For SSD, pass V_A(SSD) as v_a and use h² = V_A(SSD)/V_P(SSD), or set
    v_p = v_a (h² = 1) when working directly in divergence units.
    """
    if h2 <= 0 or epsilon <= 0 or v_p <= 0:
        return math.inf

    log_term = math.log(4.0 / delta)
    coeff = 2.0 * v_p / (h2 * epsilon**2)

    # Fixed-point iteration: N = coeff · [L ln(eN/L) + ln(4/δ)]
    n = max(coeff * (math.log(L) + log_term), float(L))
    for _ in range(max_iter):
        bracket = L * math.log(math.e * n / L) + log_term
        n_new = coeff * bracket
        if abs(n_new - n) / max(n, 1.0) < tol:
            return n_new
        n = n_new
    return n


def ne_star_ssd(
    v_af: float,
    v_am: float,
    r_mf: float,
    v_p: float,
    epsilon: float,
    delta: float,
    L: int,
) -> float:
    """N_e* for SSD using V_A(SSD) and h²_ssd."""
    v = va_ssd(v_af, v_am, r_mf)
    h2 = h2_ssd(v_af, v_am, r_mf, v_p)
    return ne_star(v, v_p, h2, epsilon, delta, L)


def ne_star_direct_va_ssd(
    v_ssd: float,
    epsilon: float,
    delta: float,
    L: int,
) -> float:
    """N_e* when h² = 1 for the divergence trait (V_P = V_A(SSD))."""
    if v_ssd <= 0:
        return math.inf
    return ne_star(v_ssd, v_ssd, 1.0, epsilon, delta, L)


def sensitivity_r_mf(
    v_af: float,
    v_am: float,
    r_grid: Iterable[float],
    *,
    v_p: float | None = None,
    epsilon: float = 0.05,
    delta: float = 0.05,
    L: int = 50,
) -> list[tuple[float, float, float]]:
    """Return (r_mf, V_A(SSD), N_e*) tuples."""
    if v_p is None:
        v_p = v_af + v_am  # rough default
    out = []
    for r in r_grid:
        v = va_ssd(v_af, v_am, r)
        h2 = v / v_p if v_p > 0 else 0.0
        n = ne_star(v, v_p, h2, epsilon, delta, L) if h2 > 0 else math.inf
        out.append((r, v, n))
    return out
