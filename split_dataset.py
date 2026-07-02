"""
split_dataset.py

Functions:
1. Read sheet1 and sheet2 from data.xlsx
2. Perform data preprocessing (filter records where n_points >= 4, handle NaN values from heat treatment, process fields including has_solu, has_middle and has_aging)
3. Construct the feature matrix X and target vector y
4. Execute the following dataset splitting methods:
   - One-time random split at the ratio of 8:2
   - One-time split by source_key at the ratio of 8:2
   - 5-fold random KFold cross-validation
   - 5-fold GroupKFold cross-validation grouped by source_key
5. Save all splitting results as CSV files to the splits/ directory

Run in Jupyter:
    %run split_dataset.py
——————————————————————————
功能：
1. 从 data.xlsx 中读取 sheet1 / sheet2
2. 预处理（n_points >= 4、热处理 NaN、has_solu/has_middle/has_aging 等）
3. 构造特征矩阵 X 和目标 y
4. 进行：
   - 随机一次性 8:2 划分
   - 按 source_key 的一次性 8:2 划分
   - 随机 5 折 KFold
   - 按 source_key 的 5 折 GroupKFold
5. 将各划分结果保存为 csv 文件到 splits/ 目录

在 Jupyter 中运行：
    %run split_dataset.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
    KFold,
    GroupKFold,
)


# ========= 工具函数 =========

def save_split(df_train: pd.DataFrame,
               df_test: pd.DataFrame,
               out_dir: str,
               prefix: str = ""):
    """
    将当前划分的 train/test DataFrame 保存为 csv 文件。

    out_dir: 输出目录，例如 "splits/source_holdout/"
    prefix : 文件名前缀，例如 "fold1_"
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_path = out_path / f"{prefix}train.csv"
    test_path = out_path / f"{prefix}test.csv"

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"  Saved: {train_path}")
    print(f"  Saved: {test_path}")


def summarize_split(name: str,
                    df_train: pd.DataFrame,
                    df_test: pd.DataFrame):
    print(f"\n=== {name} ===")
    print("Train samples :", len(df_train))
    print("Train sources :", df_train['source_key'].nunique())
    print("Test  samples :", len(df_test))
    print("Test  sources :", df_test['source_key'].nunique())


# ========= 核心预处理函数 =========

def load_and_preprocess(data_path: str = "data.xlsx"):
    """
    读入 data.xlsx, 预处理后返回：
        df           : 预处理后的 DataFrame
        X, y         : numpy 数组
        groups       : source_key 数组（用于 GroupSplit）
        feature_cols : 特征列名
        target_col   : 目标列名（lgkp）
    """

    data_path = Path(data_path)
    print("File exists:", data_path.exists())
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found!")

    # ==== 1. 读入两个 sheet ====
    df1 = pd.read_excel(data_path, sheet_name="sheet1")
    df2 = pd.read_excel(data_path, sheet_name="sheet2")

    print("sheet1 shape:", df1.shape)
    print("sheet2 shape:", df2.shape)

    # ==== 2. 对 sheet2 聚合得到 n_points 和 t_max ====

    # 去掉列名中的空格
    df2.columns = [c.strip() for c in df2.columns]

    # 显式 forward fill sample_id（防止合并单元格导致 NaN）
    print("Before ffill, sample_id NaN count:", df2["sample_id"].isna().sum())
    df2["sample_id"] = df2["sample_id"].ffill()
    print("After ffill, sample_id NaN count:", df2["sample_id"].isna().sum())

    # 按 sample_id 聚合 Time → n_points, t_max, t_min
    agg = (
        df2
        .groupby("sample_id", as_index=False)
        .agg(
            n_points=("Time", "count"),
            t_max=("Time", "max"),
            t_min=("Time", "min"),
        )
    )

    print("agg shape:", agg.shape)

    # ==== 3. merge 回 sheet1 ====
    df = df1.merge(agg, on="sample_id", how="left")
    print("merged df shape:", df.shape)

    # ==== 4. 按 n_points 进行筛选 ====
    print("\n[Filter by n_points]")
    print("Before filter, df shape:", df.shape)
    print("n_points describe:")
    print(df["n_points"].describe())
    print("n_points value counts:")
    print(df["n_points"].value_counts().sort_index())

    df = df[df["n_points"] >= 0].copy()  # 决定ponit阈值进入数据集
    print("After filter (n_points >= 0), df shape:", df.shape)

    # ==== 5. 处理热处理 NaN，并增加 has_* 标志 ====
    ht_cols = [
        "solu_temp", "solu_time",
        "middle_temp", "middle_time",
        "aging_temp", "aging_time",
    ]

    # 是否存在对应热处理步骤：两个字段都 NaN → 0，否则 1
    df["has_solu"] = (~df[["solu_temp", "solu_time"]].isna().all(axis=1)).astype(int)
    df["has_middle"] = (~df[["middle_temp", "middle_time"]].isna().all(axis=1)).astype(int)
    df["has_aging"] = (~df[["aging_temp", "aging_time"]].isna().all(axis=1)).astype(int)

    # 热处理数值列 NaN → 0
    df[ht_cols] = df[ht_cols].fillna(0.0)

    # ==== 6. 成分列和其它数值列的 NaN 处理 ====
    comp_cols = [
        "x_Co", "x_Al", "x_W", "x_Ni", "x_Cr", "x_Mo", "x_Fe", "x_Nb",
        "x_C", "x_Hf", "x_Si", "x_Ta", "x_Ti", "x_Y", "x_V", "x_B",
        "x_Zr", "x_Mn", "x_Sc", "x_La", "x_Re",
    ]

    other_feature_cols = ["Temperature"]

    numeric_cols = ht_cols + comp_cols + other_feature_cols + ["t_max", "n_points"]

    # 只对这些列做 fillna(0)，避免 Year、R2 等乱掉
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    print("\n[Numeric NaN check]")
    print("Any NaN left in numeric_cols?",
          df[numeric_cols].isna().any().any())

    # ==== 7. 定义特征列和目标列 ====
    feature_cols = ht_cols + comp_cols + ["Temperature", "has_solu", "has_middle", "has_aging"]
    target_col = "lgkp"

    missing_feats = [c for c in feature_cols if c not in df.columns]
    print("Missing feature columns:", missing_feats)
    if missing_feats:
        raise ValueError(f"Some feature columns are missing: {missing_feats}")

    # 丢弃 lgkp 为 NaN 的行
    df = df[~df[target_col].isna()].copy()

    # 构造 X, y
    X = df[feature_cols].astype(float).values
    y = df[target_col].astype(float).values

    print("\n[Final shapes]")
    print("df shape:", df.shape)
    print("X shape :", X.shape)
    print("y shape :", y.shape)

    assert len(df) == X.shape[0] == y.shape[0], "Row number mismatch!"

    if "source_key" not in df.columns:
        raise ValueError("df 中缺少 source_key 列，用于按文献分组划分！")

    groups = df["source_key"].values

    return df, X, y, groups, feature_cols, target_col


# ========= 主流程 =========

def main():
    # 1. 预处理 & 构造 X/y/groups
    df, X, y, groups, feature_cols, target_col = load_and_preprocess("data.xlsx")

    # 2. 随机一次性 8:2 划分（baseline）
    print("\n===== Random 8:2 split (baseline) =====")
    X_train_rand, X_test_rand, y_train_rand, y_test_rand, df_train_rand, df_test_rand = train_test_split(
        X,
        y,
        df,
        test_size=0.2,
        random_state=40,
        shuffle=True,
    )

    summarize_split("Random 8:2 split", df_train_rand, df_test_rand)

    overlap_rand = set(df_train_rand["source_key"]) & set(df_test_rand["source_key"])
    print("Overlapping source_key count (Random):", len(overlap_rand))

    # 保存 random holdout
    save_split(
        df_train_rand,
        df_test_rand,
        out_dir="splits/random_holdout/",
        prefix=""
    )

    # 3. 按 source_key 的一次性 8:2 划分（source-aware）
    print("\n===== Source-aware 8:2 split (GroupShuffleSplit) =====")
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=30,
    )
    train_idx_src, test_idx_src = next(gss.split(X, y, groups=groups))

    df_train_src = df.iloc[train_idx_src].copy()
    df_test_src = df.iloc[test_idx_src].copy()

    summarize_split("Source-aware 8:2 split", df_train_src, df_test_src)

    overlap_src = set(df_train_src["source_key"]) & set(df_test_src["source_key"])
    print("Overlapping source_key count (Source-aware):", len(overlap_src))

    # 保存 source-aware holdout
    save_split(
        df_train_src,
        df_test_src,
        out_dir="splits/source_holdout/",
        prefix=""
    )

    # 4. 随机 5 折 KFold（不分 source）
    print("\n===== Random 5-fold KFold =====")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X, y), 1):
        df_tr = df.iloc[tr_idx].copy()
        df_te = df.iloc[te_idx].copy()

        print(f"\n[Random KFold] Fold {fold}")
        summarize_split(f"Random KFold Fold {fold}", df_tr, df_te)

        # 保存该 fold 的 train/test
        save_split(
            df_tr,
            df_te,
            out_dir="splits/random_kfold/",
            prefix=f"fold{fold}_"
        )

    # 5. 按 source_key 的 5 折 GroupKFold（严格文献分组）
    print("\n===== Source-aware 5-fold GroupKFold =====")
    gkf = GroupKFold(n_splits=5)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        df_tr = df.iloc[tr_idx].copy()
        df_te = df.iloc[te_idx].copy()

        overlap = set(df_tr["source_key"]) & set(df_te["source_key"])

        print(f"\n[GroupKFold] Fold {fold}")
        summarize_split(f"Source-aware KFold Fold {fold}", df_tr, df_te)
        print("Overlap sources in this fold:", len(overlap))

        # 保存该 fold 的 train/test
        save_split(
            df_tr,
            df_te,
            out_dir="splits/source_kfold/",
            prefix=f"fold{fold}_"
        )

    print("\nAll splits finished. CSV files saved in 'splits/' directory.")


if __name__ == "__main__":
    main()
