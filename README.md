# ML4SCI – HEPSIM GSoC 2026 Evaluation
## Quark vs. Gluon Jet Classification with Lorentz-Boost Analysis  
### + Symbolic Regression Proof-of-Concept for MC-Style Reweighting

**Author:** Shashwat Srivastava  
**Proposal:** Symbolic Regression for Interpretable Event-Level Reweighting (HEPSIM1)  
**Program:** Google Summer of Code 2026 — ML4SCI  

---

## Overview

This notebook is the evaluation task submission for the HEPSIM GSoC 2026 project. It has two parts:

**Part 1 (Evaluation Task):** A complete physics analysis of the Pythia 8 Quark and Gluon Jets dataset — constituent-level data loading, jet observable computation, numerically stable Lorentz boost to the jet rest frame, and a quark/gluon classifier with lab-frame vs. rest-frame comparison.

**Part 2 (PoC Section):** A proof-of-concept for the proposed GSoC project — a CARL-to-symbolic-regression reweighting pipeline run on the same dataset, treating quark jets as source and gluon jets as reference to validate the pipeline infrastructure before the production Z+jets Pythia→Herwig target.

---

## Repository Structure

```
├── fixit.ipynb          # Main submission notebook (all outputs embedded)
├── README.md            # This file
```

---

## Dataset

**Pythia 8 Quark and Gluon Jets** (Komiske, Metodiev & Thaler, Zenodo 2019)  
DOI: [10.5281/zenodo.3164691](https://zenodo.org/records/3164691)

- 5 files × 100k jets = 500k jets total (250k quark, 250k gluon)
- Each jet: variable-length constituent list zero-padded to max multiplicity
- Constituent features: pT, rapidity y, azimuthal φ, PDG ID
- **Requires Internet ON** in Kaggle settings — files are downloaded automatically

---

## Evaluation Task Results

### Part (a) — Data Loading and Exploration

| Observable | Gluon | Quark |
|---|---|---|
| Total real constituents | 13,283,118 | 8,350,222 |
| Mean multiplicity | 53.13 ± 15.67 | 33.40 ± 13.27 |
| Observed ratio g/q | 1.591 | — |
| QCD prediction (C_A/C_F) | 2.250 | — |

The observed multiplicity ratio (1.59) is lower than the perturbative QCD prediction (2.25) due to non-perturbative effects and the finite pT threshold of the dataset.

### Part (b) — Jet Observables

| Observable | Gluon | Quark |
|---|---|---|
| Mean jet mass (GeV) | 46.86 ± 20.73 | 32.14 ± 18.80 |
| Mean jet width | 0.0653 | 0.0390 |
| Mean pT dispersion | 0.2728 ± 0.0722 | 0.3826 ± 0.1130 |

Gluon jets are heavier, broader, and have lower pT dispersion — consistent with softer, more democratic energy sharing driven by the larger color Casimir C_A = 3 vs C_F = 4/3.

### Part (c) — Lorentz Boost

The boost to the jet rest frame uses the numerically stable formula:

```
p' = p + β[(γ−1)/β² (β·p) − γE]
E' = γ(E − β·p)
```

This avoids explicit parallel/perpendicular decomposition and remains stable as β → 0 (where the naive (γ−1)/β² term would lose precision).

**Verification results:**
- Boost residual |p3_total| mean: 2.21 × 10⁻¹² GeV (float64 noise floor)
- Mass invariance δm mean: 1.51 × 10⁻¹⁰ GeV
- Massless condition |E²−p²| mean: 2.93 × 10⁻¹³ GeV²

### Part (d) — Classification

Gradient Boosting classifier trained on 400k jets, evaluated on 100k.

| Frame | AUC | Most Discriminating Feature | Feature Importance |
|---|---|---|---|
| Lab | 0.8650 | multiplicity | 0.854 |
| Rest | 0.8576 | multiplicity | 0.841 |

**ΔAUC (rest − lab) = +0.0075** — negligibly small, confirming that multiplicity dominates regardless of frame. The boost removes trivial lab-frame kinematic variables (jet pT, η) but the primary discriminant (constituent count) is frame-independent.

**Robustness (3 seeds):** Lab 0.8645 ± 0.0001 | Rest 0.8575 ± 0.0002

Multiplicity carries ~85% of the discriminating power — a direct consequence of the C_A/C_F color factor ratio. This is not a coincidence; it is a perturbative QCD prediction that the data confirms.

---

## PoC Section: Symbolic Regression Reweighting Pipeline

This section demonstrates the core pipeline from the proposed GSoC project on a controlled test case.

**Test case:** Reweight quark jets (source) → gluon jets (reference). This is an intentionally hard test: the quark/gluon density ratio requires AUC = 0.87 to approximate with a neural network, meaning the underlying correction function is genuinely high-dimensional. The production target (Z+jets Pythia→Herwig) involves smaller, more structured generator-level differences where compact analytic corrections are more likely to exist.

### Pipeline

```
Raw jets → Feature matrix (n, w, m/pT, ptD)
         → CARL MLP (density ratio estimation)
         → Validation gate (chi-sq/ndf check)
         → PySR symbolic regression on ln(w_CARL)
         → Pareto-front selection
         → Closure test (four-way comparison)
```

### CARL Results

| Method | chi-sq/ndf (multiplicity) | chi-sq/ndf (jet width) |
|---|---|---|
| Unweighted | 26,502.8 | 37,526.1 |
| CARL (AUC = 0.8674) | **5.60** | **19.60** |
| 2D binned ceiling | 9.35 | 88.21 |

CARL achieves 0.60× of the 2D binned ceiling — it draws signal from features beyond the (n, w) pair, confirming the 4-feature set carries real multi-dimensional information. Validation gate: **PASS**.

### PySR Result

Training: 60k quark jets | Runtime: 8.8 minutes | Pareto front: 10 expressions

**Selected expression (complexity = 10):**
```
ln(w) = (sqrt(n) − 6.501) / ((w × 4.558) + 0.623)
→ w_SR = exp((sqrt(n) − 6.501) / ((w × 4.558) + 0.623))
```

**Physics interpretation:**
- `n` in numerator: high multiplicity → more gluon-like, consistent with C_A/C_F
- `w` in denominator: wider jets need less correction since gluon jets are already broad
- `m/pT` and `ptD`: excluded by parsimony penalty — correctly suppressed

### Closure Test

| Method | chi-sq/ndf (n) | chi-sq/ndf (w) | Mean |
|---|---|---|---|
| Unweighted | 26,502.82 | 37,526.07 | 32,014.45 |
| 1D binned (n only) | 6.65 | 1,146.69 | 576.67 |
| CARL | **5.60** | **19.60** | **12.60** |
| Symbolic SR | 58.41 | 792.78 | 425.59 |

SR achieves partial closure: it correctly identifies the leading physics variables and their functional roles, but cannot match CARL's closure on a problem this high-dimensional. This is expected — SR trades closure quality for interpretability. On Z+jets where Pythia/Herwig differences are a few percent (rather than requiring AUC = 0.87 to detect), the same pipeline is expected to find expressions with substantially better closure.

---

## Environment

Runs end-to-end on **Kaggle** with Internet ON.

```
Python      : 3.12.12
NumPy       : 2.0.2
Matplotlib  : 3.10.0
scikit-learn: 1.6.1
SciPy       : 1.16.3
PySR        : 1.5.9 (auto-installed)
```

All random seeds fixed at 42. Reproducible from fresh kernel with Kernel → Restart & Run All.

---

## Key Implementation Notes

**Zero-padding:** A constituent is real iff pT > 0. All padding slots are verified to be exactly zero before any computation.

**φ-wrap safety:** Jet axis φ_J computed via `atan2(Σ pT sin φ, Σ pT cos φ)` — correctly handles jets straddling the φ = ±π boundary where naive averaging fails.

**Massless approximation:** Valid for light hadrons (π±, K±, γ) dominating jet content. Heavy hadrons introduce percent-level errors at low constituent pT; a rigorous treatment would use PDG masses from the stored PDG ID column.

**Boost stability:** The (γ−1)/β² coefficient is evaluated once per jet and applied vectorized — no explicit decomposition into parallel/perpendicular components, avoiding the cancellation errors that appear in the naive formulation at small β.

**CARL safeguards:** Probabilities clipped to [0.001, 0.999] before ratio conversion. Mean weight verified at ~1.00 after normalization. Events with w > 10 clipped before SR training to prevent noise amplification.

---

## References

1. P. Komiske, E. Metodiev, J. Thaler — Pythia8 Quark and Gluon Jets, Zenodo (2019)
2. K. Cranmer, J. Pavez, G. Louppe — Approximating Likelihood Ratios with Calibrated Discriminative Classifiers, arXiv:1506.02169
3. M. Cranmer — PySR: Fast & Parallelized Symbolic Regression in Python, github.com/MilesCranmer/PySR
4. K. Matchev, K. Matcheva, A. Roman — [arXiv:2202.02306](https://www.researchgate.net/publication/369212528_Is_the_machine_smarter_than_the_theorist_Deriving_formulas_for_particle_kinematics_with_symbolic_regression)

