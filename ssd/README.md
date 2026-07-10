# SSD manifold theory — code

Simulation and threshold calculations for:

> **Genetic Constraints on Sexual Size Dimorphism as Capacity Limits on a Shared Fisher-Information Manifold**  
> R. Craig Stillwell (*Journal of Evolutionary Biology*, submitted)

Extends the empirical-risk-minimisation framework in the parent repository to sexual size dimorphism (SSD) evolution under Wright–Fisher drift.

## Contents

| File | Description |
|------|-------------|
| `ssd_theory_utils.py` | `V_A(SSD)`, `N_e*` fixed-point solver, sensitivity helpers |
| `ssd_simulations.py` | Breeder's-equation simulations + figure generation |
| `figures/` | Pre-generated Figures 1, 2, and S1 (PDF) |
| `tables/selection_experiments_comparative.md` | Table 1 source (12 published selection experiments) |

## Requirements

From the repository root:

```bash
pip install -r requirements.txt
```

(`numpy`, `matplotlib` — already listed in root `requirements.txt`.)

## Quick start

### Compute N_e* for a given system

```python
from ssd_theory_utils import ne_star_ssd, va_ssd

# Example: S. limbatus–like parameters (§3.3)
v_af, v_am, r_mf, v_p = 0.046, 0.023, 0.98, 0.10
epsilon, delta, L = 0.05, 0.05, 50

print("V_A(SSD) =", va_ssd(v_af, v_am, r_mf))
print("N_e*     =", ne_star_ssd(v_af, v_am, r_mf, v_p, epsilon, delta, L))
```

### Regenerate figures

```bash
cd ssd
python ssd_simulations.py
```

Outputs PNG and PDF to `ssd/figures/`.

## Key functions

- **`va_ssd(v_af, v_am, r_mf)`** — additive variance in the divergence direction  
  \(V_A(\mathrm{SSD}) = V_{A,f} + V_{A,m} - 2 r_{mf}\sqrt{V_{A,f} V_{A,m}}\)

- **`ne_star_ssd(...)`** — effective population size threshold below which SSD divergence cannot reliably accumulate against drift (Theorem 3; self-contained proof in manuscript Supplementary Note S1)

- **`simulate_ssd_selection(...)`** — recursive breeder's-equation trajectory with optional Bulmer effect on `V_A(SSD)`

## Citation

If you use this code, please cite the SSD manuscript (in review) and the parent framework:

> Stillwell, R. C. (in prep.). *Natural selection is empirical risk minimisation.*

## License

Same as parent repository: [CC BY-NC 4.0](../LICENSE).
