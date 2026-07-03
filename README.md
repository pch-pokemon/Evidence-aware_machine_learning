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
d_{\mathrm{leaf}}(x_i,x_j)=\frac{1}{T}\sum_{t=1}^{T}\mathbb{I}\!\left(l_{i,t}\neq l_{j,t}\right),
$$

where:

- $T$ is the number of trees in the XGBoost ensemble;
- $l_{i,t}$ is the terminal leaf assigned to sample $i$ in tree $t$;
- $\mathbb{I}(\cdot)$ is the indicator function, equal to 1 when the condition is true and 0 otherwise.

The model evidence distance of test sample $i$ is

$$
\mathrm{MED}_i=\frac{1}{k}\sum_{j\in \mathcal{N}^{\mathrm{leaf}}_i}d_{\mathrm{leaf}}(x_i,x_j),
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
d_{\mathrm{phys}}(x_i,x_j)=\sqrt{(1-\lambda_{\mathrm{HT}})\,d_{\mathrm{main}}^2(x_i,x_j)+\lambda_{\mathrm{HT}}\,d_{\mathrm{HT}}^2(x_i,x_j)},
$$

where $\lambda_{\mathrm{HT}}$ controls the relative contribution of heat-treatment information.

The physical evidence distance of test sample $i$ is

$$
\mathrm{PED}_i=\frac{1}{k}\sum_{j\in \mathcal{N}^{\mathrm{phys}}_i}d_{\mathrm{phys}}(x_i,x_j),
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
——————————————————————————————————————————
# 高温合金氧化动力学的证据感知机器学习

## 项目概述

本仓库包含用于实现文献来源的Ni基和Co基高温合金氧化动力学数据证据感知机器学习工作流的主要代码。

该框架不再将每个机器学习预测视为孤立的数值输出，而是将每个预测与两个互补空间中的支持证据相关联：

- **模型证据空间**，通过模型证据距离（**MED**）进行量化，用于评估测试样本在XGBoost终端叶节点表征空间中对于已训练模型而言的熟悉程度。
- **物理证据空间**，通过物理证据距离（**PED**）进行量化，用于评估测试样本与训练集中在合金成分、氧化温度和热处理条件方面相似的实验样本之间的接近程度。

随后，利用MED-PED联合分布定义四种证据感知知识状态：

| 知识状态 | MED | PED | 解释 |
|---|---:|---:|---|
| **Q1** | 低 | 低 | 证据巩固 |
| **Q2** | 低 | 高 | 模型推断 |
| **Q3** | 高 | 低 | 模型解耦 |
| **Q4** | 高 | 高 | 证据缺失 / 知识盲区 |

这些知识状态可以投影到合金成分空间中，从而构建可靠性导向的设计景观。还可以在不同文献发表年份截断条件下对知识状态进行比较，以分析可用证据及其对应设计景观随时间的演化。

---

## 仓库内容

### `data_demo.xlsx`

经过处理的氧化动力学数据集的代表性子集。

典型字段包括：

- 合金成分；
- 氧化温度；
- 热处理温度和时间；
- 热处理步骤存在性标志；
- 文献来源标识符；
- 发表年份；
- 抛物线氧化速率常数的对数值`lgkp`。

### `split_dataset.py`

对输入数据集进行预处理，并生成：

- 随机留出划分；
- 来源感知留出划分；
- 随机五折交叉验证划分；
- 来源感知五折交叉验证划分。

在来源感知划分策略中，同一文献来源的全部记录仅分配到训练子集或测试子集中的一方，从而避免来源重叠。

### `train_xgb_random_kfold_optuna.py`

在随机五折交叉验证条件下，使用Optuna对XGBoost进行超参数优化，并利用全局最优超参数重新训练各折模型。

### `train_xgb_source_kfold_optuna.py`

在来源感知五折交叉验证条件下，使用Optuna对XGBoost进行超参数优化。

该脚本用于评估模型在独立文献来源之间的泛化能力，并训练证据感知分析所使用的模型。

### `train_xgb_single_fold_timecut.py`

使用固定测试集和按照发表年份截断的训练集训练单个XGBoost模型。

该脚本用于时间演化比较，例如根据截至不同发表年份可获得的数据重建证据景观。

### `compute_gibbs_htgain_weights.py`

生成物理证据距离计算中使用的固定权重。

该权重策略结合了三类信息：

- 根据代表性氧化物形成Gibbs自由能所反映的氧化亲和力构建元素权重；
- 为氧化温度设置相对较大的先验权重；
- 根据来源感知五折模型中XGBoost归一化增益重要性的平均值，确定热处理温度和时间的权重。

最终得到的权重在普通交叉验证分析和时间演化分析中保持固定并重复使用，从而确保PED具有一致的物理距离尺度。

### `Evidence_tracing.py`

计算MED、PED、局部证据统计量以及知识状态划分结果。

该脚本可以：

- 从已训练的XGBoost模型中提取终端叶节点表征；
- 计算成对叶空间距离；
- 根据模型空间中的最近邻计算MED；
- 根据物理空间中的最近邻计算PED；
- 提取局部证据统计量，包括不同文献来源的数量以及邻近样本`lgkp`值的标准差；
- 划分Q1-Q4知识状态；
- 导出各折汇总结果、象限统计结果以及特定案例的邻近证据。

MED和PED所使用的邻域相互独立选择。

### `Cr-Almap.py`

将MED-PED知识状态投影到Cr-Al成分空间中，并生成可靠性导向的设计景观。

该脚本可以调整用于可视化：

- 象限划分结果；
- 预测误差；
- 氧化温度区间；
- 局部证据密度；
- 局部证据一致性；
- 不同时间阶段之间的知识状态转变。

---

## 推荐工作流

按照以下顺序运行主要脚本：

```text
1. split_dataset.py
2. train_xgb_random_kfold_optuna.py
3. train_xgb_source_kfold_optuna.py
4. compute_gibbs_htgain_weights.py
5. Evidence_tracing.py
6. Cr-Almap.py
```

对于时间演化分析：

```text
1. 准备固定的来源感知训练集和测试集划分。
2. 针对选定的文献发表年份截断条件运行train_xgb_single_fold_timecut.py。
3. 重复使用相同的固定物理距离权重。
4. 使用时间截断模型和固定测试集运行Evidence_tracing.py。
5. 将得到的知识状态与更新后数据集对应的知识状态进行比较。
```

---

## 核心定义

### 模型证据距离

对于测试样本$x_i$和训练样本$x_j$，终端叶节点距离定义为：

$$
d_{\mathrm{leaf}}(x_i,x_j)=\frac{1}{T}\sum_{t=1}^{T}\mathbb{I}\!\left(l_{i,t}\neq l_{j,t}\right),
$$

其中：

- $T$为XGBoost集成模型中的决策树数量；
- $l_{i,t}$表示样本$i$在第$t$棵树中被分配到的终端叶节点；
- $\mathbb{I}(\cdot)$为指示函数，当条件成立时取值为1，否则取值为0。

测试样本$i$的模型证据距离定义为：

$$
\mathrm{MED}_i=\frac{1}{k}\sum_{j\in \mathcal{N}^{\mathrm{leaf}}_i}d_{\mathrm{leaf}}(x_i,x_j),
$$

其中，$\mathcal{N}^{\mathrm{leaf}}_i$表示终端叶节点空间中距离测试样本$i$最近的$k$个训练样本集合。

较低的MED表示测试样本位于与训练数据所占据区域相似的模型表征区域中。较高的MED表示模型空间外推程度逐渐增加。

### 物理证据距离

所有数值变量均使用对应训练集统计量进行标准化。

主成分-温度距离定义为：

$$
d_{\mathrm{main}}^2(x_i,x_j)=\sum_{m=1}^{M}w_m\left(z_{i,m}-z_{j,m}\right)^2,
$$

其中：

- $z_{i,m}$和$z_{j,m}$分别为样本$i$和样本$j$在物理变量$m$上的标准化数值；
- $w_m$为分配给物理变量$m$的归一化权重；
- $M$为合金成分变量和氧化温度变量的总数。

热处理距离表示为：

$$
d_{\mathrm{HT}}(x_i,x_j),
$$

用于描述热处理温度、热处理时间和热处理路径之间的差异。

组合后的成对物理距离定义为：

$$
d_{\mathrm{phys}}(x_i,x_j)=\sqrt{(1-\lambda_{\mathrm{HT}})\,d_{\mathrm{main}}^2(x_i,x_j)+\lambda_{\mathrm{HT}}\,d_{\mathrm{HT}}^2(x_i,x_j)},
$$

其中，$\lambda_{\mathrm{HT}}$控制热处理信息在总物理距离中的相对贡献。

测试样本$i$的物理证据距离定义为：

$$
\mathrm{PED}_i=\frac{1}{k}\sum_{j\in \mathcal{N}^{\mathrm{phys}}_i}d_{\mathrm{phys}}(x_i,x_j),
$$

其中，$\mathcal{N}^{\mathrm{phys}}_i$表示物理证据空间中距离测试样本$i$最近的$k$个训练样本集合。

较低的PED表示存在较密集的、在成分和实验条件方面相似的训练样本支持。较高的PED表示物理证据较弱，或者测试样本超出了现有实验数据所覆盖的范围。

---

## 局部证据统计量

对于每个测试样本，从其物理空间中最近的$k$个训练样本中提取局部证据统计量：

- `n_sources`：邻近样本中不同文献来源的数量；
- `lgkp_std`：邻近样本`lgkp`值的标准差；
- 与邻近样本之间的平均物理距离；
- 在具备发表年份信息时，邻近证据的发表年份范围。

这些统计量用于表征局部证据的多样性、一致性和接近程度。它们与MED和PED相互补充，可用于案例层面的证据来源追溯分析。
