# Evidence-aware machine learning for oxidation kinetics in superalloys

## Overview

This repository contains the main code used to implement an evidence-aware machine-learning workflow for literature-derived oxidation-kinetics data of Ni- and Co-based superalloys.

The framework does not treat each machine-learning prediction as an isolated numerical output. Instead, each prediction is linked to its supporting evidence in two complementary spaces:

- **Model evidence space**, quantified by the model evidence distance (MED), which measures how familiar a test sample is to the trained XGBoost model in terminal-leaf space.
- **Physical evidence space**, quantified by the physical evidence distance (PED), which measures the proximity of a test sample to experimentally similar training samples in composition, oxidation temperature, and heat-treatment space.

The joint MED–PED distribution is used to define four knowledge states:

- **Q1:** low MED / low PED — evidence-consolidated
- **Q2:** low MED / high PED — model-inferred
- **Q3:** high MED / low PED — model-decoupled
- **Q4:** high MED / high PED — evidence-deficient or knowledge-blind

These knowledge states can be projected into alloy-composition space to construct reliability-informed design landscapes and can also be compared across publication-year cutoffs to evaluate the temporal evolution of the available evidence.

## Repository contents

### `data_demo.xlsx`

A representative subset of the processed oxidation-kinetics dataset.

The demo file is provided to illustrate the data structure and enable users to inspect or test the workflow. It does not contain the complete dataset used to generate all numerical results reported in the manuscript.

Typical fields include:

- alloy composition
- oxidation temperature
- heat-treatment temperature and duration
- heat-treatment presence flags
- literature-source identifier
- publication year
- logarithm of the parabolic oxidation rate constant, `lgkp`

### `split_dataset.py`

Preprocesses the dataset and generates:

- random holdout split
- source-aware holdout split
- random five-fold cross-validation splits
- source-aware five-fold cross-validation splits

The source-aware split assigns all records originating from the same publication exclusively to either the training or test subset.

### `train_xgb_random_kfold_optuna.py`

Performs Optuna-based hyperparameter optimization for XGBoost under random five-fold cross-validation and retrains the fold-specific models using the globally optimized hyperparameters.

### `train_xgb_source_kfold_optuna.py`

Performs Optuna-based hyperparameter optimization for XGBoost under source-aware five-fold cross-validation.

This script is used to evaluate model generalization across independent literature sources and to train the models used in the evidence-aware analysis.

### `train_xgb_single_fold_timecut.py`

Trains a single XGBoost model using a fixed test set and a training set restricted by publication year.

This script is used for temporal comparisons, such as constructing evidence landscapes from data available up to different cutoff years.

### `compute_gibbs_htgain_weights.py`

Generates the fixed weights used in the physical evidence distance.

The weighting strategy is:

- elemental weights are derived from oxidation affinity based on the Gibbs free energies of formation of representative oxides;
- oxidation temperature is assigned the largest individual prior weight;
- heat-treatment temperature and duration variables are weighted using the mean normalized XGBoost gain importance across the source-aware folds.

The resulting weights are fixed and should be reused across ordinary-fold and temporal analyses to maintain a consistent physical-distance scale.

### `Evidence_tracing.py`

Calculates MED, PED, local evidence statistics, and knowledge-state assignments.

The script:

- extracts terminal-leaf representations from the trained XGBoost model;
- computes pairwise leaf-space distances;
- calculates MED as the mean distance to the `k` nearest training samples in model space;
- calculates PED as the mean distance to the `k` nearest training samples in physical space;
- extracts local evidence statistics, including the number of distinct literature sources and the standard deviation of neighboring `lgkp` values;
- assigns Q1–Q4 knowledge states;
- exports fold-level summaries, quadrant statistics, and case-specific neighboring evidence.

MED and PED neighborhoods are selected independently.

### `Cr-Almap.py`

Projects the MED–PED knowledge states into the Cr–Al compositional space and generates the reliability-informed design landscape.

The script can be adapted to visualize:

- quadrant assignments
- prediction errors
- temperature ranges
- local evidence density
- local evidence consistency
- temporal transitions between knowledge states

## Recommended workflow

Run the scripts in the following order:

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
1. prepare the fixed source-aware split
2. run train_xgb_single_fold_timecut.py for the selected publication-year cutoff
3. reuse the same fixed physical weights
4. run Evidence_tracing.py using the time-cut model and fixed test set
5. compare the resulting knowledge states with those from the updated dataset
```

## Core definitions

### Model evidence distance

For a test sample \(x_i\) and a training sample \(x_j\), the leaf-space distance is

\[
d_{\mathrm{leaf}}(x_i,x_j)
=
\frac{1}{T}
\sum_{t=1}^{T}
\mathbb{I}(l_{i,t}\neq l_{j,t}),
\]

where \(T\) is the number of trees and \(l_{i,t}\) is the terminal leaf assigned to sample \(i\) in tree \(t\).

The model evidence distance is

\[
\mathrm{MED}_i
=
\frac{1}{k}
\sum_{j\in N_i^{\mathrm{leaf}}}
d_{\mathrm{leaf}}(x_i,x_j).
\]

### Physical evidence distance

The pairwise physical distance is

\[
d_{\mathrm{phys}}(x_i,x_j)
=
\sqrt{
(1-\lambda_{\mathrm{HT}})d_{\mathrm{main}}^2
+
\lambda_{\mathrm{HT}}d_{\mathrm{HT}}^2
}.
\]

The physical evidence distance is

\[
\mathrm{PED}_i
=
\frac{1}{k}
\sum_{j\in N_i^{\mathrm{phys}}}
d_{\mathrm{phys}}(x_i,x_j).
\]

All numerical variables are standardized using statistics estimated from the corresponding training set only.