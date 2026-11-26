"""Build the paper's prospective tables from already evaluated virtual candidates."""

import numpy as np
import pandas as pd

from define_candidate_space import (
    MAX_DELTA_PREDICTED_LGKP,
    build_representative_pair_pool,
    candidate_state_summary,
    matched_pair_tolerance_summary,
    normalized_boundary_margin,
    search_pairs_for_contrast,
)
from evidence_core import assign_quadrants

MAP_COLUMNS = ["candidate_id", "x_Al", "x_Cr", "x_Co"]


def screening_tables(candidates, med_p50, ped_p50, varied=("x_Al", "x_Cr")):
    scores = candidates.assign(
        PED_for_threshold=candidates.PED_log1p,
        MED_threshold_P50=med_p50,
        PED_threshold_P50=ped_p50,
        state=assign_quadrants(candidates.MED, candidates.PED_log1p, med_p50, ped_p50),
        normalized_boundary_margin=normalized_boundary_margin(
            candidates.MED, candidates.PED_log1p, med_p50, ped_p50
        ),
    )
    table = candidate_state_summary(scores)
    table["MED_minus_P50_mean"] = table["MED_mean"] - med_p50
    table["ln1p_PED_minus_P50_mean"] = table["PED_log1p_mean"] - ped_p50
    table = table[["state", "count", "fraction", "predicted_lgkp_mean", "MED_minus_P50_mean", "ln1p_PED_minus_P50_mean"]]

    pairs = search_pairs_for_contrast(scores, "Q1", "Q4", varied, MAX_DELTA_PREDICTED_LGKP)
    eligible = build_representative_pair_pool(pairs)
    if eligible.empty:
        raise ValueError("No eligible representative Q1–Q4 pair.")
    representative = eligible.iloc[[0]].copy()
    for label in ("A", "B"):
        representative[f"MED_minus_P50_{label}"] = representative[f"MED_{label}"] - med_p50
        representative[f"ln1p_PED_minus_P50_{label}"] = representative[f"PED_{label}"] - ped_p50

    return scores, {
        "Table_3.csv": table,
        "Fig_6a_evidence_state_map.csv": scores[MAP_COLUMNS + ["state", "MED", "PED_log1p"]],
        "Fig_6b_predicted_lgkp_map.csv": scores[MAP_COLUMNS + ["predicted_lgkp"]],
        "Fig_6c_matched_pairs.csv": pairs.assign(cumulative_count=np.arange(1, len(pairs) + 1)),
        "Fig_6c_tolerances.csv": matched_pair_tolerance_summary(pairs),
        "Fig_6d_representative_pair.csv": representative,
    }

