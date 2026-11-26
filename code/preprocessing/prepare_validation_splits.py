"""Validate author-supplied modeling inputs and restore the paper's ordered folds."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[2]
FOLDS = PACKAGE / "data/fold_assignments/outer_folds.csv"
PARAMETERS = PACKAGE / "configs/deployment/full_data_xgboost_parameters.json"
HEAT_STAGES = ("solu", "middle", "aging")


def read_modeling_records(filename):
    config = json.loads(PARAMETERS.read_text(encoding="utf-8"))
    frame = pd.read_csv(filename, float_precision="round_trip")
    required = {"sample_id", "source_key", "Year", config["target_col"]}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Modeling table is missing {sorted(missing)}")
    if frame["sample_id"].isna().any() or not frame["sample_id"].is_unique:
        raise ValueError("The modeling table must contain unique, nonempty sample_id values.")

    for stage in HEAT_STAGES:
        columns = [f"{stage}_temp", f"{stage}_time"]
        flag = f"has_{stage}"
        if flag not in frame:
            frame[flag] = frame[columns].notna().any(axis=1).astype(int)
        frame[columns] = frame[columns].fillna(0.0)
    features = config["feature_cols"]
    composition = [name for name in features if name.startswith("x_")]
    frame[composition] = frame[composition].fillna(0.0)
    for flag in (f"has_{stage}" for stage in HEAT_STAGES):
        if not frame[flag].isin([0, 1]).all():
            raise ValueError(f"{flag} must contain 0/1 values.")
    numeric_columns = [*features, config["target_col"], "Year"]
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("All modeling inputs, targets and publication years must be finite.")
    return frame.set_index("sample_id", drop=False)


def ordered_records(records, membership):
    row_ids = membership.sort_values("row_position")["sample_id"]
    part = records.loc[row_ids].reset_index(drop=True)
    expected = membership.sort_values("row_position")["source_key"].to_numpy()
    if not np.array_equal(part["source_key"].to_numpy(), expected):
        raise ValueError("A source key differs from the published fold assignment.")
    return part


def prepare(filename, work_dir):
    records = read_modeling_records(filename)
    membership = pd.read_csv(FOLDS)
    if set(records.index) != set(membership.sample_id):
        raise ValueError("The modeling table does not match the released 613-record manifest.")

    for (scheme, fold, role), group in membership.groupby(["scheme", "outer_fold", "role"]):
        destination = work_dir / "splits" / scheme / f"fold{fold}_{role}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        ordered_records(records, group).to_csv(destination, index=False)
    held_out = membership.query("scheme == 'source_kfold' and role == 'test'")
    reference = pd.concat(
        [ordered_records(records, fold).assign(source_oof_fold=number)
         for number, fold in held_out.groupby("outer_fold")],
        ignore_index=True,
    )
    destination = work_dir / "full_reference.csv"
    reference.to_csv(destination, index=False)
    print(f"Prepared ordered folds and {len(reference)} reference rows in {work_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-data", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    if work_dir.is_relative_to(PACKAGE):
        raise ValueError("Use a separate working directory for the controlled modeling files.")
    prepare(args.modeling_data, work_dir)


if __name__ == "__main__":
    main()

