"""
train_xgb_single_fold_timecut.py

Functions:
1. Read fold4_train.csv and fold4_test.csv from the specified directory.
2. Optional: Truncate the training set by Year to build the "old model".
3. Train a single XGBoost model.
4. Make predictions and perform evaluation on the fixed test set.
5. Save the model, metadata and prediction result table.
6. Additionally save the "truncated training set + fixed test set" as fold6_train/fold6_test for direct reading by subsequent provenance / physical-model-space tracing code.

Notes:
- The test set remains fixed.
- Use YEAR_CUTOFF to decide whether to remove newer records from the training set.
- Feature columns are consistent with those in split_dataset.py.
- Outputs are named using OUTPUT_FOLD_ID, e.g. 6
——————————————————————————————
功能：
1. 从指定目录读取 fold4_train.csv 和 fold4_test.csv
2. 可选：按 Year 截断训练集，构建“旧模型”
3. 训练单个 XGBoost 模型
4. 在固定测试集上预测并评估
5. 保存模型、元信息、预测结果表
6. 额外将“删减后的训练集 + 固定测试集”另存为 fold6_train/fold6_test，
   供后续 provenance / physical-model-space tracing 代码直接读取

说明：
- 测试集固定不变
- 训练集可通过 YEAR_CUTOFF 控制是否删去较新文献
- 特征列与 split_dataset.py 保持一致
- 输出采用 OUTPUT_FOLD_ID 命名，例如 6
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump

import xgboost as xgb


# =========================
# 配置区域
# =========================

# 数据所在目录（例如 splits/source_kfold 或 splits/random_kfold）
DATA_DIR = Path("splits/random_kfold")

# 输出目录（模型/预测表/meta 等）
MODEL_DIR = Path("models/xgb_random_kfold")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 原始读取的 fold（固定从这个 fold 派生）
FOLD_ID = 4

# 输出时伪装成哪个 fold
OUTPUT_FOLD_ID = 6

# 是否按年份裁剪训练集
USE_YEAR_CUTOFF = True

# 例如设为 2022，则训练集仅保留 Year <= 2022 的样本
# 如果 USE_YEAR_CUTOFF = False，则忽略该参数
YEAR_CUTOFF = 2022

# 模型参数（可以自行更改）
MODEL_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "booster": "gbtree",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 8,

    "max_depth": 48,
    "learning_rate": 0.009842734250719654,
    "n_estimators": 999,
    "min_child_weight": 7.217180498867476,
    "subsample": 0.6148180877939232,
    "colsample_bytree": 0.7351058572714095,
    "gamma": 0.10821616636681253,
    "reg_alpha": 0.10089508731721672,
    "reg_lambda": 0.6367100755779387,
}


# 特征列（与 split_dataset.py 保持一致）
ht_cols = [
    "solu_temp", "solu_time",
    "middle_temp", "middle_time",
    "aging_temp", "aging_time",
]

comp_cols = [
    "x_Co", "x_Al", "x_W", "x_Ni", "x_Cr", "x_Mo", "x_Fe", "x_Nb",
    "x_C", "x_Hf", "x_Si", "x_Ta", "x_Ti", "x_Y", "x_V", "x_B",
    "x_Zr", "x_Mn", "x_Sc", "x_La", "x_Re",
]

feature_cols = ht_cols + comp_cols + [
    "Temperature",
    "has_solu", "has_middle", "has_aging",
]

target_col = "lgkp"


# =========================
# 工具函数
# =========================

def load_single_fold_data(data_dir: Path, fold_id: int):
    """
    加载单个 fold 的 train/test 数据
    """
    train_path = data_dir / f"fold{fold_id}_train.csv"
    test_path = data_dir / f"fold{fold_id}_test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    return df_train, df_test, train_path, test_path


def check_required_columns(df: pd.DataFrame, required_cols, df_name="DataFrame"):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} 缺少列: {missing}")


def apply_year_cutoff(df_train: pd.DataFrame, year_cutoff: int):
    """
    仅保留 Year <= year_cutoff 的训练样本
    """
    if "Year" not in df_train.columns:
        raise ValueError("训练集中缺少 Year 列，无法按年份裁剪。")

    before_n = len(df_train)
    df_old = df_train[df_train["Year"] <= year_cutoff].copy()
    after_n = len(df_old)

    print(f"\n[Year cutoff]")
    print(f"Before cutoff: {before_n} samples")
    print(f"After  cutoff: {after_n} samples (Year <= {year_cutoff})")

    if after_n == 0:
        raise ValueError(f"Year <= {year_cutoff} 后训练集为空，请调整 YEAR_CUTOFF。")

    return df_old


def build_xy(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    从 train/test DataFrame 中提取 X/y
    """
    check_required_columns(df_train, feature_cols + [target_col], df_name="df_train")
    check_required_columns(df_test, feature_cols + [target_col], df_name="df_test")

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values

    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values

    return X_train, y_train, X_test, y_test


def calc_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# =========================
# 主流程
# =========================

def main():
    print("=== Load single fold data ===")
    df_train, df_test, train_path, test_path = load_single_fold_data(DATA_DIR, FOLD_ID)

    print(f"Original train path: {train_path}")
    print(f"Fixed test path    : {test_path}")
    print(f"Original train size: {len(df_train)}")
    print(f"Fixed test size    : {len(df_test)}")

    if "source_key" in df_train.columns and "source_key" in df_test.columns:
        print(f"Train sources: {df_train['source_key'].nunique()}")
        print(f"Test  sources: {df_test['source_key'].nunique()}")

    # 可选：按年份裁剪训练集
    if USE_YEAR_CUTOFF:
        df_train_used = apply_year_cutoff(df_train, YEAR_CUTOFF)
    else:
        df_train_used = df_train.copy()

    # 提取 X / y
    X_train, y_train, X_test, y_test = build_xy(df_train_used, df_test)

    print("\n=== Final dataset used for training/testing ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test  shape: {X_test.shape}")
    print(f"y_test  shape: {y_test.shape}")

    if "Year" in df_train_used.columns:
        print(f"Train year range: {df_train_used['Year'].min()} - {df_train_used['Year'].max()}")
    if "Year" in df_test.columns:
        print(f"Test  year range: {df_test['Year'].min()} - {df_test['Year'].max()}")

    # =========================
    # 训练模型
    # =========================
    print("\n=== Train model ===")
    model = xgb.XGBRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    # =========================
    # 预测与评估
    # =========================
    print("\n=== Evaluate on fixed test set ===")
    y_pred = model.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = calc_rmse(y_test, y_pred)

    print(f"Output fold {OUTPUT_FOLD_ID} | R2 = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")

    # =========================
    # 真实 vs 预测 散点图
    # =========================
    plt.figure(figsize=(6, 6))
    plt.scatter(
        y_test,
        y_pred,
        s=60,
        alpha=0.75,
        edgecolor="k",
        linewidth=0.4
    )

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=1.5
    )

    plt.xlabel("True lg(kp)", fontsize=14, fontweight="bold")
    plt.ylabel("Predicted lg(kp)", fontsize=14, fontweight="bold")
    plt.title(
        f"Fold {OUTPUT_FOLD_ID} (time-cut)\n"
        f"$R^2$={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}",
        fontsize=13
    )
    plt.grid(alpha=0.2)
    plt.tight_layout()

    fig_path = MODEL_DIR / f"fold{OUTPUT_FOLD_ID}_scatter.png"
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.show()
    print(f"Saved scatter plot to: {fig_path}")

    # =========================
    # 保存模型
    # =========================
    model_path = MODEL_DIR / f"xgb_fold{OUTPUT_FOLD_ID}.joblib"
    dump(model, model_path)
    print(f"Saved model to: {model_path}")

    # =========================
    # 保存预测表
    # =========================
    df_pred = df_test.copy()
    df_pred["y_true"] = y_test
    df_pred["y_pred"] = y_pred
    df_pred["error"] = df_pred["y_pred"] - df_pred["y_true"]
    df_pred["abs_error"] = np.abs(df_pred["error"])

    pred_path = MODEL_DIR / f"fold{OUTPUT_FOLD_ID}_pred_vs_true.csv"
    df_pred.to_csv(pred_path, index=False)
    print(f"Saved predictions to: {pred_path}")

    # =========================
    # 保存训练集快照（模型目录内）
    # =========================
    train_used_path = MODEL_DIR / f"fold{OUTPUT_FOLD_ID}_train_used.csv"
    df_train_used.to_csv(train_used_path, index=False)
    print(f"Saved used training set to: {train_used_path}")

    # =========================
    # 额外保存为 fold6_train / fold6_test（供追溯代码直接读取）
    # =========================
    split_train_path = DATA_DIR / f"fold{OUTPUT_FOLD_ID}_train.csv"
    split_test_path = DATA_DIR / f"fold{OUTPUT_FOLD_ID}_test.csv"

    df_train_used.to_csv(split_train_path, index=False)
    df_test.to_csv(split_test_path, index=False)

    print(f"Saved split train to: {split_train_path}")
    print(f"Saved split test  to: {split_test_path}")

    # =========================
    # 保存元信息
    # =========================
    meta = {
        "fold": OUTPUT_FOLD_ID,
        "derived_from_fold": FOLD_ID,
        "train_path_original": str(train_path),
        "test_path_fixed_original": str(test_path),
        "train_path_saved_for_tracing": str(split_train_path),
        "test_path_saved_for_tracing": str(split_test_path),
        "model_path": str(model_path),
        "predictions_path": str(pred_path),
        "train_used_path": str(train_used_path),
        "feature_cols": feature_cols,
        "target_col": target_col,
        "use_year_cutoff": USE_YEAR_CUTOFF,
        "year_cutoff": YEAR_CUTOFF if USE_YEAR_CUTOFF else None,
        "model_params": MODEL_PARAMS,
        "n_train_original": int(len(df_train)),
        "n_train_used": int(len(df_train_used)),
        "n_test": int(len(df_test)),
        "n_train_sources_original": int(df_train["source_key"].nunique()) if "source_key" in df_train.columns else None,
        "n_train_sources_used": int(df_train_used["source_key"].nunique()) if "source_key" in df_train_used.columns else None,
        "n_test_sources": int(df_test["source_key"].nunique()) if "source_key" in df_test.columns else None,
        "train_year_min": int(df_train_used["Year"].min()) if "Year" in df_train_used.columns else None,
        "train_year_max": int(df_train_used["Year"].max()) if "Year" in df_train_used.columns else None,
        "test_year_min": int(df_test["Year"].min()) if "Year" in df_test.columns else None,
        "test_year_max": int(df_test["Year"].max()) if "Year" in df_test.columns else None,
        "metrics": {
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
        },
    }

    meta_path = MODEL_DIR / f"xgb_fold{OUTPUT_FOLD_ID}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to: {meta_path}")

    # =========================
    # 汇总一行结果
    # =========================
    df_result = pd.DataFrame([{
        "output_fold": OUTPUT_FOLD_ID,
        "derived_from_fold": FOLD_ID,
        "use_year_cutoff": USE_YEAR_CUTOFF,
        "year_cutoff": YEAR_CUTOFF if USE_YEAR_CUTOFF else None,
        "n_train_original": int(len(df_train)),
        "n_train_used": int(len(df_train_used)),
        "n_test": int(len(df_test)),
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
    }])

    result_path = MODEL_DIR / f"fold{OUTPUT_FOLD_ID}_result_summary.csv"
    df_result.to_csv(result_path, index=False)
    print(f"Saved summary to: {result_path}")

    print("\n=== Done ===")
    print(f"Now you can set fold = {OUTPUT_FOLD_ID} in the provenance / tracing script.")


if __name__ == "__main__":
    main()