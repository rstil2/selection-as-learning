# Selection as Learning

**Selection as Learning: A Formal Unification of Evolutionary Quantitative Genetics and PAC Learning Theory**

R. Craig Stillwell (independent researcher)

[bioRxiv preprint] · [Nature submission (under review)]

---

## Key Results

| Result | Value | What it means |
|--------|-------|---------------|
| Cross-cancer PAC bound validation | **r = 0.96**, p < 0.0001 (9 cancer types) | The theoretical resistance-evolution threshold predicts observed resistance durability from two clinical measurements |
| FTNS convergence-rate validation | **r = 0.999**, p < 0.0001 (20 cancer types) | Fisher's Fundamental Theorem predicts measured tumour selection coefficients with near-perfect accuracy |
| Computable clinical threshold | **N\*_e** from standard WGS | The tumour cell burden below which reliable resistance evolution is mathematically precluded — calculable at diagnosis |

---

## Abstract

Drug resistance is responsible for the majority of cancer deaths, yet its evolutionary dynamics remain poorly predicted at treatment initiation. This paper shows that resistance evolution timing across nine cancer types is predictable (*r* = 0.96, *p* < 0.0001) using a formula parameterised by two quantities already measured in standard clinical sequencing pipelines: intratumour heterogeneity (*h*² proxy) and effective population size (*N*<sub>e</sub>).

This predictive power arises from a formal proof that evolutionary quantitative genetics and PAC learning theory are instances of a single mathematical structure: **empirical risk minimization on a Fisher-information manifold under a capacity constraint**. Fisher's Fundamental Theorem of Natural Selection, the Price Equation, and PAC generalization bounds are facets of this shared geometry. Additive genetic variance *V*<sub>A</sub> is the evolutionary analogue of Rademacher complexity; *N*<sub>e</sub> plays the role of training set size.

Standard-of-care sequential monotherapy inadvertently provides tumours with the iterative training loops that maximise evolutionary learning. Upfront combination therapy maintaining *N*<sub>e</sub> < *N*\*<sub>e</sub> closes the evolutionary search space before resistance can be discovered. **This is not a conceptual reframing: it is a theorem.**

---

## Repository structure

```
.
├── figures/
│   ├── generate_figures.py      # all figure code (run from repo root)
│   ├── msk_pd1_cache.json       # cached cBioPortal data (CC-BY, public)
│   ├── fig1_conceptual.pdf      # Figure 1: conceptual schematic
│   ├── fig2_tracerx421.pdf      # Figure 2: empirical validation
│   ├── fig3_pac_bound.pdf       # Figure 3: cross-cancer PAC bound test
│   └── fig4_VA_capacity.pdf     # Figure 4: V_A as evolutionary capacity
└── arxiv_submission/
    ├── main.tex                 # LaTeX manuscript (original)
    ├── main_v2.tex              # clinical reframing revision
    ├── main_v3.tex              # structural revision (Nature submission)
    ├── main_v3.pdf              # compiled manuscript (current submission)
    ├── cover_letter.tex/.pdf    # Nature cover letter
    ├── refs.bib                 # BibTeX bibliography (58 entries)
    ├── supplementary.tex        # Supplementary Notes (S1–S7)
    └── supplementary.pdf        # compiled supplementary (17 pages)
```

---

## Reproducing the figures

### Requirements

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9. See [requirements.txt](requirements.txt) for exact dependencies.

### Run

From the **repository root**:

```bash
python figures/generate_figures.py
```

This writes `fig1_conceptual.pdf` through `fig4_VA_capacity.pdf` to `figures/`.

**Network access**: Figure 2B fetches live data from the [cBioPortal](https://www.cbioportal.org/) public REST API (study `nsclc_pd1_msk_2018`; Hellmann et al. 2018 *Cancer Cell*). The result is cached to `figures/msk_pd1_cache.json` after the first run, so subsequent runs are fully offline. The pre-fetched cache is included in this repository.

### Reproducibility note

The random seed is fixed (`numpy.random.default_rng(42)`). Figure 2B scatter positions are deterministic given the cached data. All other figures use only closed-form computations over the published parameter tables embedded in the script.

---

## Data sources

| Figure | Data | Source | Licence |
|--------|------|--------|---------|
| Fig 2A | TRACERx 421 ITH tertile adaptation rates | Abbosh et al. 2023 *Nature* Fig. 3b/c | Published aggregate statistics |
| Fig 2B | MSK PD-1 NSCLC TMB + PFS | Hellmann et al. 2018 *Cancer Cell*; cBioPortal `nsclc_pd1_msk_2018` | CC-BY |
| Fig 3A | *N*<sub>e</sub> estimates | Williams et al. 2016 *Nat Genet* Table S1; Dentro et al. 2021 *Science* | CC-BY |
| Fig 3A | *V*<sub>A</sub> (CCF variance) | Dentro et al. 2021 *Science* Extended Data Table 1 | CC-BY |
| Fig 3A | Resistance durability (TTR₂) | Per-cancer sources listed in script header | Published |
| Fig 4  | *V*<sub>A</sub>, *s̄* | Dentro et al. 2021; Williams et al. 2016; Tarabichi et al. 2021 | CC-BY |

Full provenance for each data point is documented in the module-level docstring of `figures/generate_figures.py`.

---

## Compiling the manuscript

Requires a TeX Live installation with `pdflatex` and `bibtex`.

```bash
cd arxiv_submission
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex       # produces main.pdf (23 pages)

pdflatex supplementary.tex
pdflatex supplementary.tex  # produces supplementary.pdf (17 pages)
```

---

## Citation

```bibtex
@article{stillwell2026selection,
  title   = {Selection as Learning: A Formal Unification of Evolutionary
             Quantitative Genetics and {PAC} Learning Theory},
  author  = {Stillwell, R. Craig},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {}
}
```

---

## Licence

Code: [MIT](LICENSE)  
Cached public data (`msk_pd1_cache.json`): [CC-BY](https://creativecommons.org/licenses/by/4.0/) (original licence from cBioPortal / Hellmann et al. 2018)
