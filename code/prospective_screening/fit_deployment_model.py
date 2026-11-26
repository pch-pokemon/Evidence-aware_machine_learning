"""Refit the single full-data deployment booster using the saved global optimum."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBRegressor

PACKAGE = Path(__file__).resolve().parents[2]
CONFIG = PACKAGE / "configs/deployment/full_data_xgboost_parameters.json"


def read_reference(filename):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    reference = pd.read_csv(filename, float_precision="round_trip")
    membership = pd.read_csv(PACKAGE / "data/fold_assignments/outer_folds.csv")
    expected = membership.query("scheme == 'source_kfold' and role == 'test'").sort_values(["outer_fold", "row_position"])
    actual = reference[["sample_id", "source_key"]]
    pd.testing.assert_frame_equal(actual, expected[actual.columns].reset_index(drop=True))
    return reference, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Ordered full_reference.csv")
    parser.add_argument("--model", required=True, type=Path, help="Destination for the fitted joblib model")
    args = parser.parse_args()
    if args.model.resolve().is_relative_to(PACKAGE):
        raise ValueError("Save the fitted model outside the public package.")
    reference, config = read_reference(args.reference)
    inputs = reference[config["feature_cols"]].to_numpy(dtype=float)
    model = XGBRegressor(**config["model_params"])
    model.fit(inputs, reference[config["target_col"]].to_numpy(dtype=float))
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    print(f"Fitted the deployment model on {len(reference)} records. Saved {args.model}.")


if __name__ == "__main__":
    main()

