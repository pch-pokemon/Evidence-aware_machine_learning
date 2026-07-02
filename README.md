# Evidence-aware machine learning for oxidation kinetics in superalloys

## Overview

This repository contains the main code used to implement an evidence-aware machine-learning workflow for literature-derived oxidation-kinetics data of Ni- and Co-based superalloys.

Rather than treating each machine-learning prediction as an isolated numerical output, the framework links every prediction to its supporting evidence in two complementary spaces:

- **Model evidence space**, quantified by the model evidence distance (**MED**), which evaluates how familiar a test sample is to the trained XGBoost model in terminal-leaf representation space.
- **Physical evidence space**, quantified by the physical evidence distance (**PED**), which evaluates how close a test sample is to experimentally similar training samples in composition, oxidation temperature, and heat-treatment space.

The joint MED-PED distribution is then used to define four evidence-aware knowledge states:

| Knowledge state | MED | PED | Interpretation |
|---|---:|---:|---|
| **Q1** | Low | Low | Evidence-consolidated |
| **Q2** | Low | High | Model-inferred |
| **Q3** | High | Low | Model-decoupled |
| **Q4** | High | High | Evidence-deficient / knowledge-blind |

These knowledge states can be projected into alloy-composition space to construct reliability-informed design landscapes. They can also be compared across publication-year cutoffs to examine how the available evidence and the resulting design landscape evolve over time.

---

## Repository contents

### `data_demo.xlsx`

A representative subset of the processed oxidation-kinetics dataset.

The demo file is provided to illustrate the data structure and support inspection or testing of the workflow. It does **not** contain the complete dataset used to generate all numerical results reported in the manuscript.

Typical fields include:

- alloy composition;
- oxidation temperature;
- heat-treatment temperature and duration;
- heat-treatment presence flags;
- literature-source identifier;
- publication year;
- logarithm of the parabolic oxidation rate constant, `lgkp`.

### `split_dataset.py`

Preprocesses the input dataset and generates:

- a random holdout split;
- a source-aware holdout split;
- random five-fold cross-validation splits;
- source-aware five-fold cross-validation splits.

In the source-aware strategy, all records originating from the same publication are assigned exclusively to either the training subset or the test subset, preventing source overlap.

### `train_xgb_random_kfold_optuna.py`

Performs Optuna-based hyperparameter optimization for XGBoost under random five-fold cross-validation and retrains fold-specific models using the globally optimized hyperparameters.

### `train_xgb_source_kfold_optuna.py`

Performs Optuna-based hyperparameter optimization for XGBoost under source-aware five-fold cross-validation.

This script is used to evaluate generalization across independent literature sources and to train the models used in the evidence-aware analysis.

### `train_xgb_single_fold_timecut.py`

Trains a single XGBoost model using a fixed test set and a training set restricted by publication year.

This script is used for temporal comparisons, such as reconstructing evidence landscapes from data available up to different publication-year cutoffs.

### `compute_gibbs_htgain_weights.py`

Generates the fixed weights used in the physical evidence distance.

The weighting strategy combines three sources of information:

- elemental weights derived from oxidation affinity using the Gibbs free energies of formation of representative oxides;
- a comparatively large prior weight for oxidation temperature;
- weights for heat-treatment temperature and duration derived from the mean normalized XGBoost gain importance across source-aware folds.

The resulting weights are fixed and reused across ordinary-fold and temporal analyses so that PED remains on a consistent physical-distance scale.

### `Evidence_tracing.py`

Calculates MED, PED, local evidence statistics, and knowledge-state assignments.

The script can:

- extract terminal-leaf representations from a trained XGBoost model;
- compute pairwise leaf-space distances;
- calculate MED from the nearest neighbors in model space;
- calculate PED from the nearest neighbors in physical space;
- extract local evidence statistics, including the number of distinct literature sources and the standard deviation of neighboring `lgkp` values;
- assign Q1-Q4 knowledge states;
- export fold-level summaries, quadrant statistics, and case-specific neighboring evidence.

The MED and PED neighborhoods are selected independently.

### `Cr-Almap.py`

Projects MED-PED knowledge states into the Cr-Al compositional space and generates the reliability-informed design landscape.

The script can be adapted to visualize:

- quadrant assignments;
- prediction errors;
- oxidation-temperature ranges;
- local evidence density;
- local evidence consistency;
- temporal transitions between knowledge states.

---

## Recommended workflow

Run the main scripts in the following order:

```text
1. split_dataset.py
2. train_xgb_random_kfold_optuna.py
3. train_xgb_source_kfold_optuna.py
4. compute_gibbs_htgain_weights.py
5. Evidence_tracing.py
6. Cr-Almap.py
```

For temporal analysis:

```text
1. Prepare a fixed source-aware train/test split.
2. Run train_xgb_single_fold_timecut.py for the selected publication-year cutoff.
3. Reuse the same fixed physical-distance weights.
4. Run Evidence_tracing.py using the time-cut model and the fixed test set.
5. Compare the resulting knowledge states with those obtained from the updated dataset.
```

---

## Core definitions

### Model evidence distance

For a test sample $x_i$ and a training sample $x_j$, the terminal-leaf distance is defined as

$$
d_{\mathrm{leaf}}(x_i,x_j)
=
\frac{1}{T}
\sum_{t=1}^{T}
\mathbb{I}\!\left(l_{i,t}\neq l_{j,t}\right),
$$

where:

- $T$ is the number of trees in the XGBoost ensemble;
- $l_{i,t}$ is the terminal leaf assigned to sample $i$ in tree $t$;
- $\mathbb{I}(\cdot)$ is the indicator function, equal to 1 when the condition is true and 0 otherwise.

The model evidence distance of test sample $i$ is

$$
\mathrm{MED}_i
=
\frac{1}{k}
\sum_{j\in \mathcal{N}^{\mathrm{leaf}}_i}
d_{\mathrm{leaf}}(x_i,x_j),
$$

where $\mathcal{N}^{\mathrm{leaf}}_i$ denotes the set of the $k$ nearest training samples in terminal-leaf space.

A low MED indicates that the test sample lies in a model-representation region similar to regions occupied by the training data. A high MED indicates increasing model-space extrapolation.

### Physical evidence distance

All numerical variables are standardized using statistics estimated from the corresponding training set only.

The main composition-temperature distance is

$$
d_{\mathrm{main}}^2(x_i,x_j)=\sum_{m=1}^{M}w_m\left(z_{i,m}-z_{j,m}\right)^2,
$$

where:

- $z_{i,m}$ and $z_{j,m}$ are standardized values of physical variable $m$;
- $w_m$ is the normalized weight assigned to variable $m$;
- $M$ is the number of composition and oxidation-temperature variables.

The heat-treatment distance is denoted by

$$
d_{\mathrm{HT}}(x_i,x_j),
$$

and accounts for differences in heat-treatment temperature, duration, and treatment route.

The combined pairwise physical distance is

$$
d_{\mathrm{phys}}(x_i,x_j)
=
\sqrt{
(1-\lambda_{\mathrm{HT}})
\,d_{\mathrm{main}}^2(x_i,x_j)
+
\lambda_{\mathrm{HT}}
\,d_{\mathrm{HT}}^2(x_i,x_j)
},
$$

where $\lambda_{\mathrm{HT}}$ controls the relative contribution of heat-treatment information.

The physical evidence distance of test sample $i$ is

$$
\mathrm{PED}_i
=
\frac{1}{k}
\sum_{j\in \mathcal{N}^{\mathrm{phys}}_i}
d_{\mathrm{phys}}(x_i,x_j),
$$

where $\mathcal{N}^{\mathrm{phys}}_i$ denotes the set of the $k$ nearest training samples in physical evidence space.

A low PED indicates dense support from compositionally and experimentally similar training samples. A high PED indicates weak physical support or extrapolation beyond the available experimental evidence.

---

## Local evidence statistics

For each test sample, local evidence statistics are extracted from its $k$ nearest physical-space neighbors:

- `n_sources`: number of distinct literature sources among the neighboring samples;
- `lgkp_std`: standard deviation of the neighboring `lgkp` values;
- mean physical distance to the neighboring samples;
- publication-year range of the neighboring evidence, when available.

These statistics characterize local evidence diversity, consistency, and proximity. They are complementary to MED and PED and are useful for case-level provenance analysis.
