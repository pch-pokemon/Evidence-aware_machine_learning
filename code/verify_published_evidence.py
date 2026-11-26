"""Verify and optionally regenerate paper results from the public CSV files."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
PACKAGE = CODE.parent
sys.path.insert(0, str(CODE / "prospective_screening"))

from evidence_core import assign_quadrants
from candidate_result_tables import screening_tables

STATES = ["Q1", "Q2", "Q3", "Q4"]
RESULTS = PACKAGE / "results"


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def verify_provenance():
    records = read_csv(PACKAGE / "data/provenance/record_source_map.csv")
    sources = read_csv(PACKAGE / "data/provenance/literature_sources.csv")
    folds = read_csv(PACKAGE / "data/fold_assignments/outer_folds.csv")
    inner = read_csv(PACKAGE / "data/fold_assignments/nested_inner_folds.csv")
    if len(records) != records.sample_id.nunique() or len(sources) != records.source_key.nunique():
        raise ValueError("Record or publication manifest contains duplicate or missing IDs.")
    expected = records.set_index("sample_id").source_key
    for name, scheme in folds.groupby("scheme"):
        held_out = scheme.loc[scheme.role.eq("test")]
        if len(held_out) != len(records) or set(held_out.sample_id) != set(expected.index):
            raise ValueError(f"{name}: held-out rows must cover the manifest exactly once.")
        for fold, membership in scheme.groupby("outer_fold"):
            test = membership[membership.role.eq("test")].set_index("sample_id").source_key
            train = membership[membership.role.eq("train")].set_index("sample_id").source_key
            pd.testing.assert_series_equal(pd.concat([test, train]).sort_index(), expected.sort_index())
            split = inner[(inner.scheme == name) & (inner.outer_fold == fold)].set_index("sample_id")
            pd.testing.assert_series_equal(split.source_key.sort_index(), train.sort_index())
            if name == "source_kfold":
                if set(test) & set(train):
                    raise ValueError(f"{name}, outer fold {fold}: overlapping test and train sources.")
                if split.groupby("source_key").inner_validation_fold.nunique().max() != 1:
                    raise ValueError(f"{name}, outer fold {fold}: a source occurs in multiple inner folds.")
    return records


def verify_nested_states(records):
    nested = RESULTS / "nested_cv"
    data = read_csv(nested / "source/OOF_evidence.csv")
    recorded = read_csv(nested / "threshold_sensitivity_quadrant_stats.csv")
    thresholds = read_csv(nested / "fold_training_reference_thresholds.csv")
    if not data.sample_id.is_unique or set(data.sample_id) != set(records.sample_id):
        raise ValueError("OOF evidence does not cover the released sample IDs.")
    rows = []
    for percentile, fold_summary in thresholds.groupby("percentile"):
        states = np.empty(len(data), dtype="<U2")
        for t in fold_summary.itertuples(index=False):
            mask = data.fold.eq(t.fold)
            states[mask] = assign_quadrants(
                data.loc[mask, "MED"], data.loc[mask, "PED_log1p"], t.MED_threshold, t.PED_threshold
            )
        np.testing.assert_array_equal(states, data[f"quadrant_P{percentile}"])
        counts = pd.Series(states).value_counts().reindex(STATES, fill_value=0)
        expected = recorded.loc[recorded.percentile.eq(percentile)].set_index("quadrant").loc[STATES, "count"]
        np.testing.assert_array_equal(counts, expected)
        rows.extend({"percentile": percentile, "quadrant": label, "count": int(count)}
                    for label, count in counts.items())
    return pd.DataFrame(rows)


def verify_temporal():
    root = RESULTS / "temporal_evolution"
    data = read_csv(root / "record_state_transitions.csv")
    counts = pd.crosstab(data.quadrant_2022, data.quadrant_current).reindex(index=STATES, columns=STATES)
    expected = read_csv(root / "transition_matrix_counts.csv").set_index("historical")
    np.testing.assert_array_equal(counts, expected)
    return counts.rename_axis(index="historical", columns=None).reset_index()


def verify_chronological(records):
    manifest = read_csv(PACKAGE / "data/provenance/source_manifest.csv")
    future = read_csv(RESULTS / "chronological_validation/future_evidence.csv")
    joined = records.merge(manifest[["source_key", "split"]], on="source_key", validate="many_to_one")
    eligible_ids = set(joined.loc[joined.split.eq("future"), "sample_id"])
    if set(future.sample_id) != eligible_ids:
        raise ValueError("Future evidence does not match the chronological source manifest.")
    expected = read_csv(RESULTS / "chronological_validation/state_error_summary.csv")
    counts = future.quadrant.value_counts().reindex(expected.quadrant)
    np.testing.assert_array_equal(expected.records, counts)


def verify_neighbor_profiles(records):
    config = json.loads((PACKAGE / "configs/deployment/full_data_xgboost_parameters.json").read_text(encoding="utf-8"))
    features = config["feature_cols"]
    temporal = read_csv(RESULTS / "temporal_evolution/Table_S10.csv").rename(
        columns={"Neighbor sample": "sample_id", "Literature source": "source_key"}
    )
    if len(temporal) != 40 or temporal.groupby(["Case", "Period"]).ngroups != 4:
        raise ValueError("Table S10 must contain four ten-neighbor lists.")
    for _, group in temporal.groupby(["Case", "Period"]):
        np.testing.assert_array_equal(group["Rank"], np.arange(1, 11))
    if (temporal.loc[temporal.Period.eq("Historical (<=2022)"), "Year"] > 2022).any():
        raise ValueError("Table S10 contains a post-2022 historical neighbor.")

    profiles = [temporal]
    for suffix in ("a_model_space", "b_physical_space"):
        profiles.append(read_csv(RESULTS / "prospective_screening" / f"Table_S11{suffix}.csv"))
    fields = ["Year", "lgkp", *features]
    columns = ["sample_id", "source_key", *fields]
    combined = pd.concat([table[columns] for table in profiles], ignore_index=True)
    if not np.isfinite(combined[fields].to_numpy(dtype=float)).all():
        raise ValueError("A released neighbor profile has missing or nonfinite values.")
    expected = combined.merge(records[["sample_id", "source_key", "Year"]],
                              on="sample_id", how="left", validate="many_to_one", suffixes=("", "_manifest"))
    if not (expected.source_key.eq(expected.source_key_manifest) & expected.Year.eq(expected.Year_manifest)).all():
        raise ValueError("A neighbor's DOI or year differs from the literature manifest.")
    profile_span = combined.groupby("sample_id")[fields].agg(["min", "max"])
    np.testing.assert_allclose(profile_span.xs("min", axis=1, level=1),
                               profile_span.xs("max", axis=1, level=1), rtol=0, atol=1e-10)
    return len(combined)


def verify_prospective():
    root = RESULTS / "prospective_screening"
    scores = read_csv(root / "virtual_candidates.csv")
    calibration = json.loads((PACKAGE / "configs/deployment/deployment_evidence_calibration.json").read_text(encoding="utf-8"))
    p50 = next(row for row in calibration["deployment_thresholds"] if row["percentile"] == 50)
    revised, tables = screening_tables(scores, p50["MED_threshold"], p50["PED_threshold"])
    np.testing.assert_array_equal(revised.state, scores.state)
    for name, table in tables.items():
        published = read_csv(root / ("Fig_6" if name.startswith("Fig_6") else "") / name)
        pd.testing.assert_frame_equal(
            table, published[table.columns], check_dtype=False, rtol=0, atol=1e-13
        )
    representative = tables["Fig_6d_representative_pair.csv"].iloc[0]
    for suffix in ("a_model_space", "b_physical_space"):
        neighbors = read_csv(root / f"Table_S11{suffix}.csv")
        for label in ("A", "B"):
            rows = neighbors[neighbors.candidate.eq(label)]
            np.testing.assert_array_equal(rows["rank"], np.arange(1, 11))
            if set(rows.candidate_id) != {representative[f"candidate_{label}"]}:
                raise ValueError("Table S11 candidate IDs differ from the representative pair.")
    return revised, tables


def reproduce(output_dir=None):
    records = verify_provenance()
    nested = verify_nested_states(records)
    temporal = verify_temporal()
    verify_chronological(records)
    neighbor_rows = verify_neighbor_profiles(records)
    scores, tables = verify_prospective()
    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        if output_dir.is_relative_to(PACKAGE):
            raise ValueError("Use a separate directory for generated results.")
        output_dir.mkdir(parents=True, exist_ok=True)
        tables["nested_threshold_state_counts.csv"] = nested
        tables["temporal_transition_matrix.csv"] = temporal
        for name, table in tables.items():
            table.to_csv(output_dir / name, index=False, encoding="utf-8-sig")
        print(f"Regenerated {len(tables)} CSVs in {output_dir}")
    print(f"Verified {len(records)} literature IDs, {neighbor_rows} neighbor entries, {len(scores)} virtual candidates and "
          f"{len(tables['Fig_6c_matched_pairs.csv'])} Q1–Q4 pairs using public files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    reproduce(parser.parse_args().output_dir)

