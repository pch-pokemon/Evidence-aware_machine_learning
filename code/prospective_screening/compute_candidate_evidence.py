"""Recompute deployment MED, PED and literature neighbors from controlled inputs."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
PACKAGE = CODE.parent
sys.path[:0] = [str(CODE), str(CODE / "nested_cv")]

from compute_oof_evidence import CFG, PERCENTILES, model_inputs, protocol_gain
from define_candidate_space import build_candidate_space
from evidence_core import (
    build_med_from_training_reference,
    compute_ped_external,
    compute_ped_training_loo,
    derive_training_thresholds,
    load_main_weights,
    local_evidence_statistics,
    transform_ped_for_threshold,
)
from fit_deployment_model import read_reference
from candidate_result_tables import screening_tables


def evaluate_candidates(reference, candidates, model, features, target):
    ref_inputs = model_inputs(model, reference, features)
    virtual_inputs = model_inputs(model, candidates, features)
    ref_med, virtual_med, model_indices, model_distances, normalization = (
        build_med_from_training_reference(model.apply(ref_inputs), model.apply(virtual_inputs), CFG)
    )
    main = load_main_weights(PACKAGE / "configs/physical_priors.json")
    protocol = protocol_gain(model, features)
    ref_ped = compute_ped_training_loo(reference, main, protocol, CFG)
    raw_ped, physical_indices, physical_distances, _ = compute_ped_external(
        reference, candidates, main, protocol, CFG
    )
    thresholds = derive_training_thresholds(
        ref_med.MED, transform_ped_for_threshold(ref_ped, CFG), PERCENTILES
    )
    scores = candidates.join(virtual_med).join(local_evidence_statistics(reference, physical_indices, target))
    scores = scores.assign(
        predicted_lgkp=model.predict(virtual_inputs).astype(float),
        PED_raw=raw_ped,
        PED_log1p=transform_ped_for_threshold(raw_ped, CFG),
    )
    calibration = {
        "evidence_config": CFG.to_dict(),
        "main_weights": main,
        "protocol_weights": protocol,
        "MED_normalization": normalization,
        "deployment_thresholds": thresholds.to_dict("records"),
    }
    return scores, calibration, (model_indices, model_distances), (physical_indices, physical_distances)


def extract_neighbors(reference, scores, pair, neighborhoods, features, *, physical):
    indices, distances = neighborhoods
    displayed = ["x_Al", "x_Cr", "Temperature", "lgkp", "Year", "source_key"]
    additional = [feature for feature in features if feature not in displayed]
    distance_column = "ln1p_physical_distance" if physical else "leaf_distance"
    columns = [
        "candidate", "state", "rank", distance_column, *displayed,
        "candidate_id", "sample_id", *additional,
    ]
    pieces = []
    for label in ("A", "B"):
        candidate_id = pair[f"candidate_{label}"]
        row, = scores.index[scores.candidate_id.eq(candidate_id)]
        panel = reference.iloc[indices[row]][
            [*displayed, "sample_id", *additional]
        ].reset_index(drop=True)
        panel[distance_column] = np.log1p(distances[row]) if physical else distances[row]
        panel.insert(0, "rank", np.arange(1, len(panel) + 1))
        panel.insert(0, "state", pair[f"state_{label}"])
        panel.insert(0, "candidate", label)
        panel["candidate_id"] = candidate_id
        pieces.append(panel[columns])
    return pd.concat(pieces, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Ordered full_reference.csv")
    parser.add_argument("--model", required=True, type=Path, help="Trusted fitted deployment joblib artifact")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.is_relative_to(PACKAGE):
        raise ValueError("Write controlled results outside the public package.")

    reference, config = read_reference(args.reference)
    model = joblib.load(args.model)
    features = config["feature_cols"]
    candidates, varied = build_candidate_space(features)
    scores, calibration, model_neighborhood, physical_neighborhood = evaluate_candidates(
        reference, candidates, model, features, config["target_col"]
    )
    p50 = next(row for row in calibration["deployment_thresholds"] if row["percentile"] == 50)
    scores, tables = screening_tables(scores, p50["MED_threshold"], p50["PED_threshold"], varied)
    pair = tables["Fig_6d_representative_pair.csv"].iloc[0]
    tables["Table_S11a_model_space.csv"] = extract_neighbors(
        reference, scores, pair, model_neighborhood, features, physical=False
    )
    tables["Table_S11b_physical_space.csv"] = extract_neighbors(
        reference, scores, pair, physical_neighborhood, features, physical=True
    )
    tables["virtual_candidates.csv"] = scores
    output.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        table.to_csv(output / filename, index=False, encoding="utf-8-sig")
    (output / "deployment_calibration.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote the Section 3.5 results to {output}")


if __name__ == "__main__":
    main()

