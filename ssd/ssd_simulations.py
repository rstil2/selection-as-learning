#!/usr/bin/env python3
"""
Simulations for SSD manifold theory manuscript.

Generates:
  - Figure 1: schematic (manifold + escape route + N_e* curve)
  - Figure 2: selection experiment outcomes vs theory
  - Figure S1: N_e* sensitivity to r_mf and L
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ssd_theory_utils import ne_star_ssd, va_ssd

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"


def simulate_ssd_selection(
    *,
    generations: int = 50,
    n_e: float = 5000.0,
    v_af: float = 0.046,
    v_am: float = 0.023,
    r_mf: float = 0.98,
    v_p: float = 0.10,
    selection_intensity: float = 0.5,
    bulmer_k: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recursive breeder's-equation simulation for SSD index I = z_f - z_m.

    Each generation:
      - V_A(SSD) may decay under selection (Bulmer effect)
      - Response: ΔI = (V_A(SSD)/V_P(SSD)) · S + drift
    """
    rng = np.random.default_rng(seed)
    v_ssd = va_ssd(v_af, v_am, r_mf)
    h2 = v_ssd / v_p if v_p > 0 else 0.0

    I = np.zeros(generations + 1)
    v_trace = np.zeros(generations + 1)
    v_trace[0] = v_ssd

    for g in range(generations):
        v = v_trace[g]
        h2_g = v / v_p if v_p > 0 else 0.0
        S = selection_intensity
        delta_I = h2_g * S
        drift = rng.normal(0.0, math.sqrt(1.0 / (2.0 * n_e)))
        I[g + 1] = I[g] + delta_I + drift
        # Bulmer: proportional reduction in V_A(SSD) under directional selection
        v_trace[g + 1] = max(v * (1.0 - bulmer_k * h2_g * S**2), 1e-8)

    t = np.arange(generations + 1)
    return t, I, v_trace


def figure1_schematic() -> None:
    """Three-panel schematic for main text."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    # Panel A: shared manifold, coupled ascent
    ax = axes[0]
    x = np.linspace(-2, 2, 200)
    ax.fill_between(x, 0.3 * x**2 - 0.5, 0.3 * x**2 + 0.5, alpha=0.25, color="gray", label="Shared constraint (high r_mf)")
    ax.plot(x, 0.3 * x**2 - 0.5, "k--", lw=1)
    ax.plot(x, 0.3 * x**2 + 0.5, "k--", lw=1)
    ax.scatter([-0.8, 0.8], [0.5, 0.5], c=["#e74c3c", "#3498db"], s=80, zorder=5)
    ax.annotate("", xy=(-0.3, 0.35), xytext=(-0.7, 0.45), arrowprops=dict(arrowstyle="->", color="#e74c3c"))
    ax.annotate("", xy=(0.3, 0.35), xytext=(0.7, 0.45), arrowprops=dict(arrowstyle="->", color="#3498db"))
    ax.text(-0.8, 0.65, "♀ optimum", ha="center", fontsize=9, color="#e74c3c")
    ax.text(0.8, 0.65, "♂ optimum", ha="center", fontsize=9, color="#3498db")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.2, 1.4)
    ax.set_xlabel("Phenotype axis")
    ax.set_ylabel("Fitness")
    ax.set_title("A. Coupled ascent (r_mf → 1)")
    ax.set_xticks([])

    # Panel B: Y-linked orthogonal escape
    ax = axes[1]
    ax.axvspan(-0.15, 0.15, alpha=0.2, color="gray")
    ax.axvline(0, color="k", ls=":", lw=1)
    ax.scatter([0], [0], c="gold", s=120, edgecolors="k", zorder=5, label="Autosome (blocked)")
    ax.scatter([-1.2, 1.2], [0, 0], c=["#e74c3c", "#3498db"], s=80, zorder=5)
    ax.annotate("", xy=(-1.2, 0.35), xytext=(-1.2, 0.05),
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2))
    ax.annotate("", xy=(1.2, -0.35), xytext=(1.2, -0.05),
                arrowprops=dict(arrowstyle="->", color="#3498db", lw=2))
    ax.text(0, 0.25, "Y-linked\nescape", ha="center", fontsize=9)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("SSD direction")
    ax.set_title("B. Sex-linked escape route")
    ax.set_xticks([])

    # Panel C: N_e* vs r_mf
    ax = axes[2]
    r = np.linspace(0, 0.999, 200)
    v_af, v_am, v_p = 0.046, 0.023, 0.10
    ne_curve = [ne_star_ssd(v_af, v_am, float(rv), v_p, 0.05, 0.05, 50) for rv in r]
    ax.semilogy(r, ne_curve, "k-", lw=2)
    ax.axhline(500, color="#9b59b6", ls="--", label="Drosophila lab N_e ~ 500")
    ax.axhline(5000, color="#e67e22", ls="--", label="S. limbatus field N_e ~ 10³–10⁴")
    ax.scatter([0.98], [ne_star_ssd(v_af, v_am, 0.98, v_p, 0.05, 0.05, 50)],
               c="#e67e22", s=60, zorder=5)
    # C. maculatus autosomal r_mf ≈ 0.93 (Kaufmann et al. 2021); Y-linked escape raises effective V_A(SSD)
    ax.scatter([0.93], [ne_star_ssd(0.05, 0.05, 0.93, 0.10, 0.05, 0.05, 50)],
               c="#9b59b6", s=60, zorder=5, label="C. maculatus (autosomal)")
    ax.set_xlabel("Intersexual genetic correlation (r_mf)")
    ax.set_ylabel("N_e* (log scale)")
    ax.set_title("C. Sample complexity threshold")
    ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(FIG / "Figure1_schematic.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "Figure1_schematic.pdf", bbox_inches="tight")
    plt.close(fig)


def figure2_simulations() -> None:
    """Simulated SSD trajectories across r_mf and N_e."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: r_mf effect at fixed N_e
    ax = axes[0]
    for r, color in [(0.5, "#27ae60"), (0.9, "#f39c12"), (0.98, "#c0392b")]:
        t, I, _ = simulate_ssd_selection(r_mf=r, n_e=10000, generations=40)
        ax.plot(t, I, color=color, lw=2, label=f"r_mf = {r}")
    ax.axhline(0.05, color="k", ls=":", lw=1, label="ε = 0.05 target")
    ax.set_xlabel("Generation")
    ax.set_ylabel("SSD index (I = z̄_f − z̄_m)")
    ax.set_title("A. Effect of r_mf (N_e = 10⁴)")
    ax.legend(fontsize=8)

    # Panel B: N_e effect at high r_mf
    ax = axes[1]
    for n_e, color in [(500, "#9b59b6"), (5000, "#3498db"), (50000, "#27ae60")]:
        t, I, _ = simulate_ssd_selection(r_mf=0.98, n_e=n_e, generations=40)
        ax.plot(t, I, color=color, lw=2, label=f"N_e = {n_e:,}")
    ax.axhline(0.05, color="k", ls=":", lw=1)
    ax.set_xlabel("Generation")
    ax.set_ylabel("SSD index")
    ax.set_title("B. Effect of N_e (r_mf = 0.98)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG / "Figure2_simulations.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "Figure2_simulations.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_s1_sensitivity() -> None:
    """Supplementary sensitivity of N_e* to r_mf and L."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    v_af, v_am, v_p = 0.046, 0.023, 0.10
    r = np.linspace(0, 0.999, 150)

    ax = axes[0]
    for L, color in [(10, "#3498db"), (50, "#2c3e50"), (200, "#e74c3c")]:
        curve = [ne_star_ssd(v_af, v_am, float(rv), v_p, 0.05, 0.05, L) for rv in r]
        ax.semilogy(r, curve, color=color, lw=2, label=f"L = {L}")
    ax.set_xlabel("r_mf")
    ax.set_ylabel("N_e*")
    ax.set_title("Sensitivity to locus number L")
    ax.legend()

    ax = axes[1]
    for eps, color in [(0.02, "#3498db"), (0.05, "#2c3e50"), (0.10, "#e74c3c")]:
        curve = [ne_star_ssd(v_af, v_am, float(rv), v_p, eps, 0.05, 50) for rv in r]
        ax.semilogy(r, curve, color=color, lw=2, label=f"ε = {eps}")
    ax.set_xlabel("r_mf")
    ax.set_ylabel("N_e*")
    ax.set_title("Sensitivity to accuracy target ε")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG / "FigureS1_Ne_star_sensitivity.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "FigureS1_Ne_star_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    figure1_schematic()
    figure2_simulations()
    figure_s1_sensitivity()
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
