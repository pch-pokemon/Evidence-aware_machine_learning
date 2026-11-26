# -*- coding: utf-8 -*-
"""Analyze Q4 alloy categories and literature coverage from the formal nested OOF file.

Usage (run in the same folder as pooled_source_oof_evidence.csv):
    python analyze_q4_alloy_coverage.py
    # Jupyter: !python analyze_q4_alloy_coverage.py

Input (no editing required):
    pooled_source_oof_evidence.csv

Outputs (q4_coverage_tables/):
    Table_A_state_overview.csv
    Table_B_alloy_families.csv
    Table_C_CrAl_bins.csv
    Table_D_source_coverage.csv
    Table_E_Q4_composition.csv

This is a DESCRIPTIVE analysis of frozen OOF data. It does not refit models,
redefine Q1-Q4, or demonstrate why a literature evidence gap arose.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# -------------------------- CONFIGURATION --------------------------
INPUT_CSV = Path("models/nested_evidence_final/pooled_source_oof_evidence.csv")
OUTPUT_DIR = Path("q4_coverage_tables")
CR_BIN_WIDTH = 3.0        # match existing Fig. 4(b)
AL_BIN_WIDTH = 2.0        # match existing Fig. 4(b)
MIN_BIN_N = 5             # match existing Fig. 4(b)
TOP_N_PRINT = 12
STATES = ("Q1", "Q2", "Q3", "Q4")
# -------------------------------------------------------------------


def require_columns(frame, columns):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}\n"
                         f"Available columns: {list(frame.columns)}")


def load_data():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Cannot find {INPUT_CSV.resolve()}\n"
            "Place this script in the same folder as pooled_source_oof_evidence.csv "
            "and run it from that folder."
        )
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    require_columns(df, ["quadrant", "source_key", "x_Ni", "x_Co", "x_Cr", "x_Al"])
    df["quadrant"] = df["quadrant"].astype("string").str.strip().str.upper()
    bad = df.loc[~df["quadrant"].isin(STATES), "quadrant"]
    if len(bad):
        raise ValueError(f"Invalid quadrant values: {bad.unique().tolist()}")
    if "sample_id" in df.columns:
        if df["sample_id"].isna().any() or df["sample_id"].duplicated().any():
            raise ValueError("sample_id missing or duplicated: check pooled OOF input")
    if df["source_key"].isna().any():
        raise ValueError("source_key contains missing values")
    for c in ["x_Ni", "x_Co", "x_Cr", "x_Al"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any() or not np.isfinite(df[c].to_numpy()).all():
            raise ValueError(f"{c} has missing/non-numeric/infinite entries")
    for c in ["x_Fe", "Year", "n_sources", "lgkp_std", "abs_error", "y_true", "y_pred"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "quadrant_P50" in df.columns:
        alt = df["quadrant_P50"].astype("string").str.strip().str.upper()
        if not df["quadrant"].eq(alt).all():
            raise ValueError("quadrant differs from quadrant_P50; check input version")
    if "abs_error" not in df and {"y_true", "y_pred"} <= set(df.columns):
        df["abs_error"] = (df["y_true"] - df["y_pred"]).abs()
    if len(df) != 613:
        print(f"[NOTE] Data has {len(df)} rows (expected 613 in the manuscript).")
    return df


def classify_family(df):
    """Operational categories, not claims about formal commercial alloy families.

    If x_Fe is available and exceeds BOTH Ni and Co, tag Fe-dominant.
    Otherwise classify by Ni vs Co concentration.
    """
    ni = df["x_Ni"].to_numpy(float)
    co = df["x_Co"].to_numpy(float)
    labels = np.where(ni > co, "Ni-dominant", np.where(co > ni, "Co-dominant", "Ni-Co tie"))
    if "x_Fe" in df.columns:
        fe = df["x_Fe"].to_numpy(float)
        labels = np.where(np.isfinite(fe) & (fe > np.maximum(ni, co)), "Fe-dominant", labels)
    labels = np.where((ni == 0) & (co == 0), "Neither Ni nor Co", labels)
    return pd.Series(labels, index=df.index)


def median_or_nan(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.median()) if len(s) else np.nan


def overall_states(df):
    rows = []
    total = len(df)
    for q in STATES:
        s = df.loc[df.quadrant.eq(q)]
        row = {
            "State": q,
            "N_records": len(s),
            "Fraction_all_pct": round(100 * len(s) / total, 2),
            "N_unique_literature_sources": s.source_key.nunique(),
        }
        if "abs_error" in df:
            row["MAE"] = round(float(s["abs_error"].mean()), 4) if len(s) else np.nan
        if "n_sources" in df:
            row["Median_distinct_sources_in_top10_physical_neighbors"] = round(median_or_nan(s["n_sources"]), 3)
            row["Pct_with_le_2_sources_in_top10"] = round(100 * s["n_sources"].le(2).mean(), 2) if len(s) else np.nan
        if "lgkp_std" in df:
            row["Median_top10_neighbor_lgkp_std"] = round(median_or_nan(s["lgkp_std"]), 4)
        if "Year" in df:
            row["Median_publication_year"] = median_or_nan(s["Year"])
            row["Pct_published_2023_or_later"] = round(100 * s["Year"].ge(2023).mean(), 2) if len(s) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def alloy_family_table(df):
    q4_total = int(df.quadrant.eq("Q4").sum())
    rows = []
    for fam, sub in df.groupby("Alloy_category", dropna=False, sort=True):
        q4 = sub.loc[sub.quadrant.eq("Q4")]
        row = {
            "Alloy_category": fam,
            "N_total": len(sub),
            "N_Q1": int(sub.quadrant.eq("Q1").sum()),
            "N_Q2": int(sub.quadrant.eq("Q2").sum()),
            "N_Q3": int(sub.quadrant.eq("Q3").sum()),
            "N_Q4": len(q4),
            "Q4_fraction_within_family_pct": round(100 * len(q4) / len(sub), 2),
            "Share_of_all_Q4_pct": round(100 * len(q4) / q4_total, 2) if q4_total else np.nan,
            "N_sources_total": sub.source_key.nunique(),
            "N_sources_Q4": q4.source_key.nunique(),
            "Median_Cr_Q4_atpct": round(median_or_nan(q4.x_Cr), 2),
            "Median_Al_Q4_atpct": round(median_or_nan(q4.x_Al), 2),
        }
        if "n_sources" in df:
            row["Median_distinct_sources_in_top10_Q4"] = median_or_nan(q4.n_sources)
        if "Year" in df:
            row["Median_publication_year_Q4"] = median_or_nan(q4.Year)
            row["Q4_published_2023plus_pct"] = round(100 * q4.Year.ge(2023).mean(), 2) if len(q4) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("N_Q4", ascending=False).reset_index(drop=True)


def aligned_edges(values, step):
    """Match the 3-at.% Cr / 2-at.% Al bin-edge convention in Fig. 4(b)."""
    mn, mx = float(np.min(values)), float(np.max(values))
    start, stop = np.floor(mn / step) * step, np.ceil(mx / step) * step
    edges = np.arange(start, stop + step * 1.001, step, dtype=float)
    if edges[0] > mn:
        edges = np.insert(edges, 0, edges[0] - step)
    if edges[-1] <= mx:
        edges = np.append(edges, edges[-1] + step)
    return edges


def cr_al_bins(df):
    cr_edges = aligned_edges(df.x_Cr.to_numpy(), CR_BIN_WIDTH)
    al_edges = aligned_edges(df.x_Al.to_numpy(), AL_BIN_WIDTH)
    tmp = df.copy()
    tmp["_Cr_bin"] = pd.cut(tmp.x_Cr, bins=cr_edges, right=False, include_lowest=True)
    tmp["_Al_bin"] = pd.cut(tmp.x_Al, bins=al_edges, right=False, include_lowest=True)
    if tmp[["_Cr_bin", "_Al_bin"]].isna().any().any():
        raise ValueError("One or more alloy records fell outside Cr-Al bin edges")
    all_q4 = int(tmp.quadrant.eq("Q4").sum())
    rows = []
    for (al_bin, cr_bin), grp in tmp.groupby(["_Al_bin", "_Cr_bin"], observed=True):
        q4 = grp.loc[grp.quadrant.eq("Q4")]
        row = {
            "Cr_from_atpct_inclusive": float(cr_bin.left),
            "Cr_to_atpct_exclusive": float(cr_bin.right),
            "Al_from_atpct_inclusive": float(al_bin.left),
            "Al_to_atpct_exclusive": float(al_bin.right),
            "N_bin": len(grp),
            "N_Q4": len(q4),
            "f_Q4_pct": round(100 * len(q4) / len(grp), 2),
            "Share_of_all_Q4_pct": round(100 * len(q4) / all_q4, 2) if all_q4 else np.nan,
            "N_sources_bin": grp.source_key.nunique(),
            "N_sources_Q4_in_bin": q4.source_key.nunique(),
            "N_Ni_dominant_in_bin": int(grp.Alloy_category.eq("Ni-dominant").sum()),
            "N_Co_dominant_in_bin": int(grp.Alloy_category.eq("Co-dominant").sum()),
            "Keep_for_discussion_Nbin_ge5": bool(len(grp) >= MIN_BIN_N),
        }
        if "n_sources" in df:
            row["Median_top10_sources_Q4"] = median_or_nan(q4.n_sources)
        rows.append(row)
    full = pd.DataFrame(rows)
    full = full.sort_values(["f_Q4_pct", "N_Q4", "N_bin"], ascending=[False, False, False]).reset_index(drop=True)
    return full


def source_coverage(df):
    """By SOURCE (source_key), not by local top-10 neighbor metadata."""
    records = []
    for source, grp in df.groupby("source_key", dropna=False):
        n4 = int(grp.quadrant.eq("Q4").sum())
        if not n4:
            continue
        records.append({
            "source_key": source,
            "N_records_from_source": len(grp),
            "N_Q4_from_source": n4,
            "Q4_fraction_within_source_pct": round(100 * n4 / len(grp), 2),
            "Dominant_Ni_or_Co_category": grp.Alloy_category.value_counts().index[0],
            "Median_Cr_Q4_atpct": median_or_nan(grp.loc[grp.quadrant.eq("Q4"), "x_Cr"]),
            "Median_Al_Q4_atpct": median_or_nan(grp.loc[grp.quadrant.eq("Q4"), "x_Al"]),
        })
    return pd.DataFrame(records).sort_values("N_Q4_from_source", ascending=False).reset_index(drop=True)


def composition_profile(df):
    """No mechanism labels: report descriptive Q4 composition quantiles only."""
    q4 = df.loc[df.quadrant.eq("Q4")]
    q1 = df.loc[df.quadrant.eq("Q1")]
    candidate_cols = ["x_Ni", "x_Co", "x_Cr", "x_Al", "x_Ti", "x_W", "x_Fe"]
    rows = []
    for feature in candidate_cols:
        if feature not in df:
            continue
        a, b = q4[feature].dropna(), q1[feature].dropna()
        rows.append({
            "Composition_feature": feature,
            "Q4_median_atpct": round(median_or_nan(a), 3),
            "Q4_p25_atpct": round(float(a.quantile(.25)), 3) if len(a) else np.nan,
            "Q4_p75_atpct": round(float(a.quantile(.75)), 3) if len(a) else np.nan,
            "Q4_nonzero_pct": round(100 * a.gt(0).mean(), 2) if len(a) else np.nan,
            "Q1_median_atpct": round(median_or_nan(b), 3),
            "Q1_nonzero_pct": round(100 * b.gt(0).mean(), 2) if len(b) else np.nan,
        })
    return pd.DataFrame(rows)


def show_table(title, df, n=None):
    print("\n" + "=" * 105)
    print(title)
    print("=" * 105)
    if df.empty:
        print("[No rows]")
    else:
        print(df.head(n).to_string(index=False, float_format=lambda v: f"{v:.2f}", na_rep="NA"))


def main():
    df = load_data()
    df["Alloy_category"] = classify_family(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overall = overall_states(df)
    family = alloy_family_table(df)
    bins = cr_al_bins(df)
    sources = source_coverage(df)
    comp = composition_profile(df)

    tables = {
        "Table_A_state_overview.csv": overall,
        "Table_B_alloy_families.csv": family,
        "Table_C_CrAl_bins.csv": bins,
        "Table_D_source_coverage.csv": sources,
        "Table_E_Q4_composition.csv": comp,
    }
    for filename, table in tables.items():
        table.to_csv(OUTPUT_DIR / filename, encoding="utf-8-sig", index=False)

    q4total = int(df.quadrant.eq("Q4").sum())
    print(f"INPUT: {INPUT_CSV} | N={len(df)} | N_sources={df.source_key.nunique()}")
    print(f"Q4: {q4total}/{len(df)} = {100*q4total/len(df):.2f}%")
    print("Ni/Co classification: largest of Ni vs Co (Fe-dominant checked only if x_Fe exists).")
    print("Cr-Al bins are left-inclusive/right-exclusive; N_bin >= 5 for main discussion.")
    show_table("TABLE A | Evidence state overview", overall)
    show_table("TABLE B | Alloy categories (share of Q4 vs within-family Q4 fraction)", family)
    show_table("TABLE C1 | Cr-Al bins with >=5 records, highest Q4 FRACTION", 
               bins.loc[bins.Keep_for_discussion_Nbin_ge5], TOP_N_PRINT)
    show_table("TABLE C2 | Cr-Al bins with >=5 records, highest Q4 COUNT", 
               bins.loc[bins.Keep_for_discussion_Nbin_ge5].sort_values("N_Q4", ascending=False), TOP_N_PRINT)
    show_table("TABLE E | Composition profile: Q4 vs Q1", comp)
    if not sources.empty:
        show_table("TABLE D | Top source contributions to Q4 (descriptive, NOT a cause of scarcity)", 
                   sources, 10)
    print(f"\n[SAVED] {len(tables)} CSV files -> {OUTPUT_DIR.resolve()}")
    print("IMPORTANT: Q4 is a relative evidence label, not proof of testing difficulty or an emerging alloy category.")


if __name__ == "__main__":
    main()
