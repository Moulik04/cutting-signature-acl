# ACL Injury Prediction with Calibrated Ensemble Learning
### Quantifying the "Cutting Signature": A Multi-Algorithm Approach

> DS340W Research Project — Penn State University, 2026  
> Moulik Jain

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)](https://xgboost.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

200,000 ACL surgeries occur annually in the US, each requiring 6–12 months of rehabilitation. The gold standard for predicting ACL injury risk, 3D motion capture, costs over $150,000 in equipment, takes 60–90 minutes per athlete, and requires specialist staff. For a 60-athlete collegiate roster, the math doesn't work.

This project asks: **can machine learning replicate that biomechanical signal using only data athletic trainers are already collecting?**

The answer is yes. This pipeline achieves **AUC = 0.8932**, a **41.8% improvement** over the state-of-the-art (Jauhiainen et al., 2022: AUC = 0.63, using full 3D motion capture), using GPS load, fatigue scores, and symmetry measures with no force plates or motion capture equipment.

The central methodological contribution is **probability calibration**: most ensemble models are confident but miscalibrated. A model outputting "65% injury risk" is only useful if 65% of athletes at that score actually get injured. This is the **first ACL prediction pipeline to guarantee calibration** via isotonic regression (PAV algorithm) across the full probability range — a property the paper terms the **Clinical Trust property**.

---

## Key Results

Full six-way technique tournament on 20% stratified hold-out (N = 1,286 athletes, ≈137 injury events):

| ID | Model | AUC | Recall | F1 | Brier ↓ | AvgPrec |
|---|---|---|---|---|---|---|
| T0 | SMOTE Ensemble (control) | ≈0.872 | ≈0.847 | ≈0.550 | ≈0.100 | ≈0.447 |
| T1 | ADASYN Ensemble | ≈0.859 | ≈0.774 | ≈0.568 | ≈0.107 | ≈0.393 |
| T2 | BorderlineSMOTE Ensemble | ≈0.878 | ≈0.869 | ≈0.572 | ≈0.097 | ≈0.459 |
| T3 | Cost-Sensitive Ensemble | ≈0.894 | ≈0.891 | ≈0.577 | ≈0.123 | ≈0.453 |
| T4 | Cost-Sensitive + Platt | ≈0.894 | ≈0.905 | ≈0.577 | ≈0.064 | ≈0.457 |
| **T5** | **Cost-Sensitive + Isotonic ★** | **≈0.894** | **≈0.906** | **≈0.587** | **≈0.060** | **≈0.470** |
| Baseline | Jauhiainen 2022 (3D MoCap) | 0.63 | — | — | — | — |

> ★ = proposed method. Brier Score ↓: lower is better. All other metrics ↑: higher is better.  
> Threshold selected per-model via Youden's J statistic (J = TPR − FPR).

**T5 vs. Jauhiainen 2022**: +41.8% AUC, no lab equipment  
**T5 vs. SMOTE control**: −39.9% Brier Score  
**T5 vs. no-skill prevalence baseline** (BS₀ ≈ 0.0948): −32.4% Brier Score

---

## The Clinical Trust Property

A well-calibrated model is one where **65% means 65%** — if a model outputs 65% injury risk, then 65% of athletes at that score in the training data actually sustained an ACL injury.

Most ensemble models fail this test:

- **T0–T2** (oversampling models): overconfident at low probabilities (0.05–0.25), flat at high probabilities (0.60–0.90)
- **T3** (cost-sensitive): significantly reduces over-confidence in low-risk range but curves at the 65% clinical action point
- **T4** (Platt/sigmoid): fixes mid-range curvature but introduces over-confidence below 20% due to the parametric sigmoid assumption
- **T5** (isotonic): tracks the calibration diagonal cleanly across the full probability range including the 65% clinical decision zone

T5 achieves this via the **Pool-Adjacent-Violators (PAV) algorithm** — a non-parametric step function with no distributional assumption. Each raw score maps to the historical fraction of positives in that score bin.

### Coach Action Protocol (T5 only — empirically grounded)

| Zone | Probability | Action |
|---|---|---|
| 🟢 Green | < 35% | Normal training load |
| 🟡 Amber | 35% – 65% | Modified training, monitor closely |
| 🔴 Red | > 65% | Reduce load, consult athletic trainer |

Under T5, these thresholds are empirically validated. Under any other configuration in this tournament, they are arbitrary.

---

## The Cutting Signature — Lab Biomechanics Without the Lab

ACL injuries have a measurable motor pattern (the "cutting signature"): single-leg deceleration, lateral trunk displacement, and asymmetric ground reaction forces at a near-extended knee. Replicating this in a lab requires $150K of equipment.

Three proxy features are engineered from training-day monitoring data:

| Feature | Formula | What It Approximates |
|---|---|---|
| `fatigue_accumulation` | (0.50 × fatigue) + (0.30 × workload) − (0.20 × recovery), clipped to [0, 1] | 7-day rolling neuromuscular load |
| `lateral_force_proxy` | speed × (1 − symmetry), clipped to [0, 1] | Center of pressure excursion (inspired by Taborri et al., 2021) |
| `symmetry_index` | \|impact − symmetry\| | Limb load asymmetry (confirmed by Guo et al., 2025) |

**SHAP validation (TreeExplainer on T5)** — all three engineered features rank in the top 3 positions:

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | `fatigue_accumulation` | 0.142 |
| 2 | `lateral_force_proxy` | 0.118 |
| 3 | `symmetry_index` | 0.097 |

The signal was always in the training data. The pipeline just needed to extract it correctly.

---

## Dataset

Three publicly available Kaggle datasets fused into one canonical schema via case-insensitive column mapping:

| File | Description |
|---|---|
| `collegiate_athlete_injury_dataset.csv` | Collegiate athlete training and injury records |
| `sports_multimodal_data.csv` | IMU/GPS sensor data across multiple sports |
| `data-3.csv` | University football training load data |

**Combined**: 6,430 athlete-seasons | 15 features | 10.6% injury prevalence | 1:8.4 class ratio

Source datasets are in `data/` — included in the repository.

---

## The Six-Way Tournament

Each model (T0–T5) replaces exactly one component from the model above it, isolating every causal contribution:

```
T0  SMOTE oversampling + stacking ensemble (RF + XGBoost + LightGBM)     ← control
 │
 └─ T1  SMOTE → ADASYN (adaptive synthesis: more samples near decision boundary)
     │
     └─ T2  ADASYN → BorderlineSMOTE (synthesis from DANGER zone only)
         │
         └─ T3  Remove oversampling → cost-sensitive learning (α = N⁻/N⁺ ≈ 8.43)
             │
             └─ T4  Add Platt (sigmoid) calibration layer
                 │
                 └─ T5  Sigmoid → isotonic regression (PAV algorithm)   ← proposed
```

Every improvement is traceable to one decision.

---

## Repository Structure

```
acl-injury-prediction/
│
├── README.md
│
├── data/
│   ├── collegiate_athlete_injury_dataset.csv
│   ├── sports_multimodal_data.csv
│   └── data-3.csv
│
├── scripts/
│   ├── acl_novelty_experiment.py    # Main pipeline — full T0–T5 tournament (run this)
│   ├── ds340w_acl_pipeline.py       # Supporting pipeline utilities
│   └── DS340W code.py               # Early exploratory code
│
├── visualizations/
│   ├── acl_novelty_calibration_plot.png    # Reliability diagrams for T0–T5
│   ├── acl_novelty_improvement_bars.png    # AUC, Recall, F1, Brier comparison bars
│   └── acl_novelty_comparison_table.csv   # Full model results table
│
├── prior_work/
│   ├── 10.1177_03635465221112095.pdf       # Jauhiainen et al. 2022 (state-of-the-art)
│   ├── peerj-20141.pdf                     # Guo et al. 2025
│   └── sensors-21-03141.pdf                # Taborri et al. 2021
│
└── report/
    ├── DS340W_ACL_Presentation.pptx
    └── Research Paper DS340W.pdf
```

---

## Installation

```bash
git clone https://github.com/Moulik04/acl-injury-prediction.git
cd acl-injury-prediction
pip install -r requirements.txt
```

**requirements.txt**
```
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
imbalanced-learn>=0.10.0
shap>=0.41.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

---

## Usage

```bash
# Run the full T0–T5 tournament
python scripts/acl_novelty_experiment.py
```

Outputs: model comparison table, calibration reliability diagrams, SHAP feature importance, Brier score comparison bars.

---

## Visualizations

### Reliability Diagrams — T0 through T5
![Calibration Plot](visualizations/acl_novelty_calibration_plot.png)

T5 (isotonic) tracks the diagonal cleanly. T0–T2 are overconfident at low probabilities. T3 curves at the 65% clinical action zone.

### Progressive Improvement — AUC, Recall, F1, Brier Score
![Improvement Bars](visualizations/acl_novelty_improvement_bars.png)

---

## Limitations

- Source datasets are observational — training loads are GPS-estimated or self-reported, not validated 3D motion capture
- Isotonic calibrator uses cv=3 on the 80% training partition; leave-one-out CV would tighten calibration estimates at boundary thresholds
- Temporal dynamics not modeled — fatigue accumulates over days/weeks, not captured in instantaneous feature values
- Requires external validation on independent cohorts (female soccer, male basketball, gymnastics) before clinical deployment

---

## Future Work

- **Temporal modeling**: LSTM or TCN over 7-day rolling load windows
- **External validation**: refit and validate isotonic calibration on independent cohorts
- **Live deployment**: real-time coach dashboard co-designed with certified athletic trainers
- **Larger calibration holdouts**: leave-one-out CV for the calibration step

---

## References

1. Jauhiainen S, et al. (2022). Predicting ACL Injury Using Machine Learning on Data From an Extensive Screening Test Battery of 880 Female Elite Athletes. *Am J Sports Med*, 50(11):2917-2924. **[state-of-the-art baseline, AUC = 0.63]**
2. Taborri J, et al. (2021). A Machine-Learning Approach to Measure the ACL Injury Risk in Female Basketball Players. *Sensors*, 21(9):3141. **[source of lateral force proxy inspiration]**
3. Guo L, et al. (2025). Prediction of ACL injury incidence in basketball players based on multi-algorithm models. *PeerJ*, 13:e20141. **[AUC = 0.80, male-only, 3D MoCap]**

---

## Citation

```bibtex
@misc{jain2026acl,
  author = {Moulik Jain},
  title  = {Quantifying the "Cutting Signature": A Multi-Algorithm Approach to Predicting ACL Injury Risk in Athletes},
  year   = {2026},
  school = {Pennsylvania State University},
  note   = {DS340W Research Project}
}
```
