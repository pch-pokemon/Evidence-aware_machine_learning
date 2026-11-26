# README｜Project Code Overview

**项目说明**：本项目包含高温合金氧化动力学预测的文献来源感知建模、证据距离计算、时间演化与前瞻性筛选流程。公开资料包括模型特征定义、文献来源与验证划分，以及论文图表使用的数值结果。

**Project description**: This repository contains the data-processing, nested validation, evidence quantification, temporal analysis, and prospective screening code for evidence-aware prediction of superalloy oxidation kinetics. The released data include literature provenance, fold assignments, and numeric results used in the paper.

---

## File-by-File Function Overview

### 1. [prepare_validation_splits.py](code/preprocessing/prepare_validation_splits.py)

**功能**：按样本标识恢复论文使用的外折与全量部署参考集，保留每折原始行序。

**Description**: Restores the paper's ordered folds and deployment reference from the author-supplied modeling table.

### 2. [train_source_aware_xgboost.py](code/nested_cv/train_source_aware_xgboost.py)

**功能**：按文献来源分组进行 Nested CV，各外折内独立使用四折 Optuna 优化。

**Description**: Fits nested source-aware XGBoost with independent four-fold Optuna tuning inside each outer training fold.

### 3. [train_random_xgboost.py](code/nested_cv/train_random_xgboost.py)

**功能**：在记录级随机划分上执行匹配的 Nested CV，用于比较验证口径。

**Description**: Fits the corresponding record-level random nested XGBoost comparison.

### 4. [benchmark_regression_models.py](code/nested_cv/benchmark_regression_models.py)

**功能**：用相同外折比较 XGBoost 与 LR、KNN、SVR、RF 和 MLP。

**Description**: Compares pooled OOF performance of XGBoost and five regression baselines on the same folds.

### 5. [evidence_core.py](code/evidence_core.py)

**功能**：封装 MED、PED、参考集归一化、近邻统计和 Q1–Q4 判定。由分析脚本导入。

**Description**: Implements shared distance, reference normalization, neighborhood and evidence-state functions. Imported by the analysis scripts.

### 6. [compute_oof_evidence.py](code/nested_cv/compute_oof_evidence.py)

**功能**：用各外折模型和训练参考分别计算 MED、PED、P40–P60 阈值及 OOF 状态。

**Description**: Computes outer-fold evidence scores and training-reference P40–P60 thresholds for held-out samples.

### 7. [plot_evidence_error_distributions.py](code/figures/plot_evidence_error_distributions.py)

**功能**：从受控 OOF 输出绘制 Fig. 3(d) 的绝对误差分布。

**Description**: Plots Fig. 3(d) from the controlled OOF residuals.

### 8. [evaluate_evidence_weights.py](code/sensitivity_and_ablation/evaluate_evidence_weights.py)

**功能**：评估 MED 混合系数与 PED 协议空间权重的敏感性。

**Description**: Evaluates sensitivity to MED mixing and the PED protocol-space contribution.

### 9. [evaluate_neighborhood_ablation.py](code/sensitivity_and_ablation/evaluate_neighborhood_ablation.py)

**功能**：比较不同邻居数，以及 MED、PED、联合证据与欧氏近邻的表现。

**Description**: Evaluates neighborhood size and compares MED, PED, joint evidence, and Euclidean nearest neighbors.

### 10. [fit_cross_fitted_med_reference.py](code/sensitivity_and_ablation/fit_cross_fitted_med_reference.py)

**功能**：用来源分离的辅助模型估计外折训练子集的 MED 参考秩。

**Description**: Fits source-disjoint auxiliary models for the MED reference rank analysis.

### 11. [evaluate_med_calibration.py](code/sensitivity_and_ablation/evaluate_med_calibration.py)

**功能**：将交叉拟合参考秩映射到外折模型的留一分布，复算状态及来源聚类区间。

**Description**: Maps cross-fitted ranks to the fitted outer model's leave-one-out reference and evaluates state and error-gap sensitivity.

### 12. [summarize_q4_coverage.py](code/sensitivity_and_ablation/summarize_q4_coverage.py)

**功能**：汇总 Q4 的 Ni/Co 合金类别、Cr–Al 区间与文献覆盖。

**Description**: Summarizes Q4 by alloy family, Cr–Al bin, and literature coverage.

### 13. [summarize_fitting_metadata.py](code/sensitivity_and_ablation/summarize_fitting_metadata.py)

**功能**：统计受控数据的拟合点数、拟合 (R^2) 和可用于子集分析的记录数。

**Description**: Summarizes fitting metadata and candidate evaluation subsets.

### 14. [evaluate_fitting_quality.py](code/sensitivity_and_ablation/evaluate_fitting_quality.py)

**功能**：在原 Nested OOF 预测上按拟合质量划定评价子集。

**Description**: Evaluates fixed nested OOF predictions in subsets defined by fitting metadata.

### 15. [fit_historical_fold_models.py](code/temporal_evolution/fit_historical_fold_models.py)

**功能**：沿用相应当前外折的参数，只用截至 2022 年的训练记录拟合配对历史模型。

**Description**: Fits paired historical models on pre-2023 outer training records with the corresponding contemporary hyperparameters.

### 16. [compute_temporal_evidence.py](code/temporal_evolution/compute_temporal_evidence.py)

**功能**：在同一批外折测试记录上比较历史和当前证据状态。

**Description**: Compares historical and contemporary evidence states on the same held-out records.

### 17. [summarize_temporal_transitions.py](code/temporal_evolution/summarize_temporal_transitions.py)

**功能**：汇总状态转移、来源聚类区间和 Fig. 5(d) 两个案例的近邻。

**Description**: Summarizes transitions, source-cluster intervals, and physical neighborhoods for the Fig. 5(d) cases.

### 18. [plot_spatial_and_temporal_evidence.py](code/temporal_evolution/plot_spatial_and_temporal_evidence.py)

**功能**：绘制 Cr–Al 证据图、局部 Q1 比例图和状态转移矩阵。

**Description**: Plots Cr–Al evidence maps, local Q1 fractions and the temporal transition matrix.

### 19. [plot_temporal_cases.py](code/temporal_evolution/plot_temporal_cases.py)

**功能**：绘制 Fig. 5(d) 的两个案例，导出 Table S10。

**Description**: Plots the Fig. 5(d) cases and exports Table S10.

### 20. [validate_chronological_holdout.py](code/chronological_validation/validate_chronological_holdout.py)

**功能**：执行独立年代切分、历史模型训练和后续文献验证。

**Description**: Implements the source-disjoint chronological split, historical training, and future evaluation.

### 21. [analyze_chronological_error.py](code/chronological_validation/analyze_chronological_error.py)

**功能**：比较后续文献中的证据状态误差，并进行分层与逐来源分析。

**Description**: Analyzes future prediction errors by evidence state, prediction strata, and leave-one-source-out stability.

### 22. [evaluate_screening_budget_sensitivity.py](code/chronological_validation/evaluate_screening_budget_sensitivity.py)

**功能**：比较配对预算与预测容差对筛选结果的影响。

**Description**: Evaluates matched-budget screening across selection budgets and prediction tolerances.

### 23. [fit_deployment_model.py](code/prospective_screening/fit_deployment_model.py)

**功能**：用原五折最优超参数在 613 条记录上拟合单一 XGBoost。

**Description**: Refits a single deployment XGBoost model on all 613 records using the original five-fold optimum.

### 24. [define_candidate_space.py](code/prospective_screening/define_candidate_space.py)

**功能**：定义 651 个虚拟合金的组成网格、工艺条件和配对规则。由筛选脚本导入。

**Description**: Defines the 651-candidate grid, fixed conditions and match rules. Imported by the screening workflow.

### 25. [compute_candidate_evidence.py](code/prospective_screening/compute_candidate_evidence.py)

**功能**：用全数据模型及其归一化 gain 重新校准并计算候选预测、MED、PED 和真实近邻。

**Description**: Calibrates the deployment reference using the full-data model's gain and scores virtual candidates and literature neighbors.

### 26. [candidate_result_tables.py](code/prospective_screening/candidate_result_tables.py)

**功能**：从候选分数生成 Table 3 与 Fig. 6 所需数据。由筛选和复核脚本导入。

**Description**: Builds Table 3 and the Fig. 6 data from scored virtual candidates. Imported by the screening and verification scripts.

### 27. [plot_prospective_screening.py](code/figures/plot_prospective_screening.py)

**功能**：从公开虚拟筛选数值重绘 Fig. 6(a–d)。

**Description**: Plots Fig. 6(a–d) from the released virtual screening data.

### 28. [verify_published_evidence.py](code/verify_published_evidence.py)

**功能**：核对全文公开证据结果，包括文献划分、Nested 阈值、时序状态与虚拟配对。

**Description**: Checks public provenance, nested thresholds, temporal transitions, chronological holdout and candidate pairing.

### 29. [run_analysis_workflow.py](code/run_analysis_workflow.py)

**功能**：在独立工作目录运行指定的受控数据分析阶段。

**Description**: Runs selected controlled analyses in a separate working directory.

---

## Evidence Distance and Calibration

The model evidence distance uses normalized mean leaf-space distance and normalized inverse leaf support.

$$
\mathrm{MED}(x)=\alpha D_{\mathrm{leaf}}(x)+(1-\alpha)D_{\mathrm{support}}(x),
\qquad \alpha=0.50.
$$

For the physical distance, the main space covers composition and oxidation temperature. The protocol space covers treatment conditions and exposure duration. Treatment variables are gated by route and a fixed penalty is applied when stage-presence flags differ.

$$
d_{\mathrm{phys}}(x,z)=\sqrt{(1-\lambda)d_{\mathrm{main}}^2(x,z)+\lambda d_{\mathrm{protocol}}^2(x,z)},
\qquad \lambda=0.15,
\qquad
\mathrm{PED}(x)=\frac{1}{k}\sum_{z\in\mathcal N_k(x)}d_{\mathrm{phys}}(x,z),\quad k=10.
$$

Evidence states are assigned using MED and \(\ln(1+\mathrm{PED})\) relative to their respective P50 thresholds. Reference P50 thresholds are obtained from leave-one-out calculations within the appropriate training reference. Table 3 and Fig. 6(d) use signed differences from the deployment P50 thresholds.

| Analysis | Prediction model | PED weights and scale |
| --- | --- | --- |
| Nested source-aware OOF | Corresponding nested outer-fold model | Outer model gain and its training reference |
| Historical-to-current OOF comparison | Historical and contemporary models for the same outer fold | Shared current-fold protocol weights, standardization, and PED thresholds |
| Virtual screening | Single model fitted on 613 records using the original five-fold optimum | Fitted deployment model gain and deployment reference |
| Independent chronological validation | Historical model tuned and fitted within the historical period | Historical model gain and historical reference |

The original source-aware Optuna search selected one parameter set by mean five-fold validation \(R^2\). Trial 93 and the deployment settings are recorded in [configs/deployment](configs/deployment/).

## Directory and Execution

| Path | Contents |
| --- | --- |
| [code](code/) | Formal analysis and plotting scripts |
| [configs](configs/) | Model settings, physical priors and evidence calibration |
| [data](data/) | Source identifiers and validation fold assignments |
| [results](results/) | Numeric source data for paper figures, tables, and analyses |
| [docs/paper_results_index.csv](docs/paper_results_index.csv) | Paper figures and tables mapped to their data and scripts |
| [notebooks/analysis_workflow.ipynb](notebooks/analysis_workflow.ipynb) | Step-by-step workflow with archived training outputs and one explanation per code cell |

Use Python 3.12 and install the dependencies from the package root:

    python -m pip install -r requirements.txt
    python code/verify_published_evidence.py
    python code/figures/plot_prospective_screening.py --output-dir ../generated_evidence_figures

The notebook runs the public steps directly. For controlled work, set EVIDENCE_WORK_DIR to a separate working directory and, when preparing inputs, set EVIDENCE_MODELING_CSV to the author-supplied processed CSV. Existing fitted models can be used for downstream computation. EVIDENCE_RUN_TRAINING=1 enables the training cells. Set EVIDENCE_RUN_FITTING_QUALITY=1 only with the fitting metadata intended for that analysis.

The complete 613-record feature–target dataset is available from the corresponding author upon reasonable request. See the [feature dictionary](docs/feature_dictionary.csv) and [preprocessing notes](docs/preprocessing_and_quality_control.md). Refitting the models, rebuilding training reference distances, and recalculating individual prediction errors require the controlled inputs. The CSVs for Tables S10 and S11 provide sample identifiers, source DOI keys, ranks and distances, and all 32 modeled predictors for the listed neighbors. These predictors cover alloy composition, recorded heat-treatment temperatures and durations, stage-presence flags, oxidation temperature, and exposure duration. The public package provides numeric data and scripts. Plots generated during a workflow run are written to the chosen working directory.

