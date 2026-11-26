# README｜Project Code Overview

**Project Description**: This repository contains the complete data processing, model training, evidence quantification, temporal evolution analysis, and prospective screening workflows for oxidation kinetics prediction and evidence\-aware model interpretation based on literature\-derived machine learning\.

---

## File\-by\-File Function Overview

### 1\. split\_dataset\.py

**功能**：数据预处理与数据集划分。对原始氧化动力学数据进行清洗、特征构建、目标变量构建，同时生成**随机五折交叉验证**与**基于文献来源的 Source\-Aware 五折交叉验证**数据集。全文所有模型训练均基于本脚本的划分结果。

**Description**: Performs data preprocessing, feature engineering, and target variable construction for raw oxidation kinetics data\. Generates two types of five\-fold splits: conventional random split and literature\-source\-aware split\. All subsequent model training follows the data partition defined herein\.

### 2\. train\_xgb\_source\_kfold\_optuna\.py

**功能**：Source\-aware XGBoost 建模与超参数优化。基于文献来源分组的五折数据，采用 Optuna 进行全局超参数寻优，使用最优统一参数训练五折 Source\-Aware XGBoost 模型，输出各折模型权重、预测性能指标与模型元数据。

**Description**: Conducts Optuna\-based hyperparameter optimization for source\-aware cross\-validation\. Trains five fold\-specific XGBoost models with unified optimal hyperparameters, and outputs model weights, predictive performance metrics, and metadata\.

### 3\. train\_xgb\_random\_kfold\_optuna\.py

**功能**：随机五折 XGBoost 基线模型。在样本级随机五折划分上执行与 Source\-Aware 模型完全一致的优化与训练流程，作为对照组，用于量化分析来源感知划分的性能优势。

**Description**: Builds a random five\-fold XGBoost baseline with identical optimization and training pipelines\. Serves as a baseline benchmark for comparing conventional random validation and source\-aware validation\.

### 4\. compute\_evidence\_weights\.py

**功能**：证据距离权重构建。结合氧化领域物理先验知识与 Source\-Aware XGBoost 模型的特征贡献，构建 PED 所需的主空间权重与实验协议空间权重，输出统一权重配置文件，供后续全部证据分析调用。

**Description**: Constructs fixed evidence distance weights for MED and PED calculations based on physical priors and XGBoost feature attributions\. Exports unified weight configurations for all downstream evidence analyses\.

### 5\. evidence\_core\.py

**功能**：证据计算核心公共模块。集中封装 MED、PED、物理距离计算、训练集归一化、近邻统计、Q1–Q4 证据状态判定等全部底层核心函数。本文件为依赖模块，无需独立运行，由所有分析脚本自动调用。

**Description**: Centralized core module for evidence quantification\. Implements MED/PED calculation, physical distance measurement, training reference normalization, neighborhood statistics, and Q1–Q4 evidence state assignment\. This module is imported by other scripts and requires no standalone execution\.

### 6\. build\_pooled\_oof\_evidence\.py

**功能**：源感知 OOF 集成证据分析。对五折 Source\-Aware 测试集分别计算 MED、PED 与训练参考分布，汇总得到完整的 Pooled OOF 证据数据集。同时完成训练阈值构建、Q1–Q4 证据状态划分、状态误差统计与 P40–P60 阈值敏感性分析。

**Description**: Computes fold\-wise MED, PED, and training reference distributions for source\-aware out\-of\-fold samples and integrates a unified pooled OOF evidence dataset\. It also generates training\-derived thresholds, Q1–Q4 evidence states, state error statistics, and P40–P60 threshold sensitivity results\.

### 7\. train\_xgb\_temporal\_source\_kfold\.py

**功能**：时间截断对照模型训练。固定原有 Source\-Aware 测试集不变，仅将训练集限制为**2022 年及以前发表的文献**，复用最优超参数构建历史模型，用于隔离文献数据积累对模型证据状态的影响。

**Description**: Implements a temporal\-cut training strategy\. With held\-out folds unchanged, all training data is restricted to literature published before 2023\. Reuses optimized hyperparameters to isolate the impact of literature accumulation on evidence state evolution\.

### 8\. build\_temporal\_oof\_evidence\.py

**功能**：证据状态时序演化分析。在相同测试样本上对比「2022 历史证据状态」与「当前全量数据证据状态」，输出历史 OOF 结果、Q1–Q4 状态转移矩阵、状态保持/转变统计，以及证据固化与证据修正典型案例数据（用于图表绘制）。

**Description**: Compares historical \(pre\-2023\) and current evidence states on identical held\-out samples\. Generates temporal OOF results, Q1–Q4 transition matrices, state persistence/transition statistics, and datasets for evidence consolidation and revision case analysis\.

### 9\. train\_xgb\_full\_data\.py

**功能**：全量数据部署模型训练。使用全部 613 条样本训练最终部署版 XGBoost 模型，直接复用交叉验证最优超参数，不重复寻优。该模型不用于性能评估，专用于后续前瞻性虚拟候选筛选。

**Description**: Refits the final deployment\-oriented XGBoost model on the full 613\-sample dataset with fixed optimized hyperparameters\. No additional hyperparameter tuning is performed\. The model is dedicated to prospective candidate screening rather than retrospective evaluation\.

### 10\. calibrate\_full\_data\_evidence\.py

**功能**：全量模型证据校准。对部署模型执行留一交叉验证 MED/PED 校准，建立专属训练参考分布与 P40–P60 证据阈值，为前瞻性候选的证据状态判定提供统一标准。

**Description**: Performs leave\-one\-out MED/PED calibration for the full\-data deployment model\. Constructs model\-specific reference distributions and P40–P60 evidence thresholds to standardize evidence\-state assignment for prospective candidates\.

### 11\. prospective\_evidence\_screening\.py

**功能**：前瞻性虚拟合金证据筛选。基于部署模型对大规模虚拟候选空间进行氧化动力学预测，结合全量文献参考池计算 MED、PED 与 Q1–Q4 证据状态。筛选预测相似但证据状态差异的配对样本，输出图表分析与结果数据（对应 Figure 6）。

**Description**: Conducts evidence\-aware prospective screening over predefined virtual alloy candidates\. Predicts oxidation kinetics and computes MED, PED, and Q1–Q4 evidence states against the full literature pool\. Identifies matched candidate pairs with similar predictions but distinct evidence states, supporting Figure 6 and associated analyses\.

---
