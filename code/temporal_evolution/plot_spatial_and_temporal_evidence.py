# -*- coding: utf-8 -*-
"""
plot_spatial_and_temporal_evidence.py
==================================

Unified plotting script for:
- Fig. 4(a): current pooled-OOF Q1-Q4 states in Cr-Al space
- Fig. 4(b): current local Q1-fraction heatmap
- Fig. 4(a-b): combined figure

- Fig. 5(a): historical (<=2022) Q1-Q4 states in Cr-Al space
- Fig. 5(b): spatial map of temporal state changes:
              Unchanged / Change to Q1 / Change to Q2 / Change to Q3 / Change to Q4
- Fig. 5(c): 2022 -> current Q-state transition matrix
- Fig. 5(a-c): combined figure

- Supplementary: historical (<=2022) local Q1-fraction heatmap,
                 directly comparable with Fig. 4(b)

Required FROZEN Nested CSVs (relative to project root):
    models/nested_evidence_final/pooled_source_oof_evidence.csv
    models/nested_temporal_evidence_2022_vs_current/temporal_comparison.csv
    models/nested_temporal_evidence_2022_vs_current/transition_matrix_counts.csv

NOTE: the exported historical_2022_pooled_oof.csv is a compact statistical table
      WITHOUT x_Cr/x_Al. It cannot be used alone for spatial maps. The paired
      temporal_comparison.csv contains the SAME 613 held-out composition coordinates
      and their historical <=2022 and current evidence states.
NOTE: raw historical/current MED values are model-relative and are NOT subtracted.
      Only PED, which uses the same fixed physical distance scale, is subtracted.

Key design rules
----------------
1) Fig. 4(a) and Fig. 4(b) use EXACTLY the same x/y limits.
2) Fig. 5(a) defines the composition-space limits for Fig. 5(b);
   Fig. 5(b) is forced to use those exact limits.
3) Current and historical Q1-fraction heatmaps use the SAME bin edges,
   so the supplementary historical heatmap can be compared directly with Fig. 4(b).
4) Point sizes are controlled in one clearly labeled section.
5) All font sizes are controlled in one clearly labeled section.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator


# =============================================================================
# 1. FILES / OUTPUTS
# =============================================================================

# Run from the project root, or edit PROJECT_ROOT below.
PROJECT_ROOT = Path.cwd()
CURRENT_CSV = PROJECT_ROOT / "models/nested_evidence_final/pooled_source_oof_evidence.csv" # models/nested_evidence_final/pooled_source_oof_evidence.csv
HISTORICAL_CSV = PROJECT_ROOT / "models/nested_temporal_evidence_2022_vs_current/temporal_comparison.csv" # models/nested_temporal_evidence_2022_vs_current/temporal_comparison.csv
REFERENCE_TRANSITION_MATRIX_CSV = (
    PROJECT_ROOT / "models/nested_temporal_evidence_2022_vs_current/transition_matrix_counts.csv" # models/nested_temporal_evidence_2022_vs_current/transition_matrix_counts.csv
)
EXPECTED_RECORDS = 613

# Keep original published/previous figures untouched.
OUT_ROOT = PROJECT_ROOT / "figures"
FIG4_DIR = OUT_ROOT / "Fig4_CrAl_nested_final"
FIG5_DIR = OUT_ROOT / "Fig5_temporal_nested_final"

FIG4_DIR.mkdir(parents=True, exist_ok=True)
FIG5_DIR.mkdir(parents=True, exist_ok=True)

DPI = 600
FONT_FAMILY = "Times New Roman"


# =============================================================================
# 2. COLUMN NAMES
# =============================================================================

CR_COL = "x_Cr"
AL_COL = "x_Al"
SOURCE_COL = "source_key"

CUR_STATE_COL = "quadrant"
HIS_STATE_COL = "quadrant_2022"

CUR_MED_COL = "MED"
HIS_MED_COL = "MED_2022"

CUR_PED_COL = "PED_log1p"
HIS_PED_COL = "PED_log1p_2022"

CUR_NSRC_COL = "n_sources"
HIS_NSRC_COL = "n_sources_2022"

CUR_STD_COL = "lgkp_std"
HIS_STD_COL = "lgkp_std_2022"

CUR_ABS_ERR_COL = "abs_error"
HIS_ABS_ERR_COL = "abs_error_2022"


# =============================================================================
# 3. COLOR / MARKER CONTROL ZONE
# =============================================================================

STATE_COLORS = {
    "Q1": "#4DAF4A",
    "Q2": "#5DA5DA",
    "Q3": "#F0AD4E",
    "Q4": "#D95F5F",
}

PLOT_ORDER = ["Q4", "Q2", "Q3", "Q1"]

STATE_ALPHA = {
    "Q4": 0.40,
    "Q2": 0.76,
    "Q3": 0.80,
    "Q1": 0.96,
}

# -------------------------------------------------------------------------
# >>> MARKER SIZE CONTROL <<<
# 调整 Fig.4 / Fig.5(a) Q1-Q4 点大小，主要改这里
# -------------------------------------------------------------------------
STATE_SIZE = {
    "Q4": 38,
    "Q2": 34,
    "Q3": 34,
    "Q1": 46,   # 保持 Q1 > Q2/Q3
}

STATE_ZORDER = {
    "Q4": 1,
    "Q2": 2,
    "Q3": 3,
    "Q1": 4,
}

# Fig.5(b): temporal change categories
CHANGE_COLORS = {
    "Unchanged": "#B7B7B7",
    "Change to Q1": STATE_COLORS["Q1"],
    "Change to Q2": STATE_COLORS["Q2"],
    "Change to Q3": STATE_COLORS["Q3"],
    "Change to Q4": STATE_COLORS["Q4"],
}

CHANGE_ORDER = [
    "Unchanged",
    "Change to Q2",
    "Change to Q3",
    "Change to Q4",
    "Change to Q1",
]

# -------------------------------------------------------------------------
# >>> MARKER SIZE CONTROL FOR FIG.5(b) <<<
# -------------------------------------------------------------------------
CHANGE_SIZE = {
    "Unchanged": 26,
    "Change to Q1": 46,
    "Change to Q2": 38,
    "Change to Q3": 38,
    "Change to Q4": 42,
}

CHANGE_ALPHA = {
    "Unchanged": 0.42,
    "Change to Q1": 0.96,
    "Change to Q2": 0.90,
    "Change to Q3": 0.90,
    "Change to Q4": 0.94,
}


# =============================================================================
# 4. BINNING / AXIS PADDING CONTROL
# =============================================================================

CR_BIN_WIDTH = 3.0
AL_BIN_WIDTH = 2.0
MIN_BIN_N = 5
ANNOTATE_COUNTS = False

# -------------------------------------------------------------------------
# >>> AXIS NEGATIVE PADDING CONTROL <<<
# 点变大后稍微多留一点呼吸空间
# -------------------------------------------------------------------------
X_AXIS_MIN_FLOOR = -1.3
Y_AXIS_MIN_FLOOR = -0.9

# Additional upper padding derived from data span
X_PAD_MIN = 1.0
Y_PAD_MIN = 0.7

# Note position for heatmaps
NOTE_X = 0.97
NOTE_Y = 0.96
NOTE_HA = "right"
NOTE_VA = "top"


# =============================================================================
# 5. FONT / TEXT CONTROL ZONE
# =============================================================================

BASE_FONT_SIZE = 12

AXIS_LABEL_SIZE = 18
AXIS_LABEL_WEIGHT = "bold"

TICK_LABEL_SIZE = 18
TICK_LABEL_WEIGHT = "bold"

PANEL_LABEL_SIZE = 10
PANEL_LABEL_WEIGHT = "bold"

LEGEND_FONT_SIZE = 11
LEGEND_TITLE_SIZE = 11
LEGEND_TITLE_WEIGHT = "bold"

INPANEL_TITLE_SIZE = 13
INPANEL_TITLE_WEIGHT = "bold"

NOTE_FONT_SIZE = 12
NOTE_FONT_WEIGHT = "bold"

COLORBAR_LABEL_SIZE = 15
COLORBAR_LABEL_WEIGHT = "bold"

COLORBAR_TICK_SIZE = 13
COLORBAR_TICK_WEIGHT = "bold"

MATRIX_AXIS_LABEL_SIZE = 15
MATRIX_AXIS_LABEL_WEIGHT = "bold"

MATRIX_ANNOT_SIZE = 12
MATRIX_ANNOT_WEIGHT = "bold"

COUNT_ANNOT_SIZE = 10


# =============================================================================
# 6. STYLE HELPERS
# =============================================================================

def set_publication_style():
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.labelweight": AXIS_LABEL_WEIGHT,
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.35,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 0.9,
        "ytick.minor.width": 0.9,
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "xtick.minor.size": 2.8,
        "ytick.minor.size": 2.8,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def style_composition_axis(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.35)

    ax.tick_params(top=False, right=False)

    ax.set_xlabel(
        "Cr content (at.%)",
        fontsize=AXIS_LABEL_SIZE,
        fontweight=AXIS_LABEL_WEIGHT,
    )
    ax.set_ylabel(
        "Al content (at.%)",
        fontsize=AXIS_LABEL_SIZE,
        fontweight=AXIS_LABEL_WEIGHT,
    )

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(TICK_LABEL_SIZE)
        lbl.set_fontweight(TICK_LABEL_WEIGHT)

    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(TICK_LABEL_SIZE)
        lbl.set_fontweight(TICK_LABEL_WEIGHT)


def panel_label(ax, text):
    ax.text(
        -0.13, 1.04, text,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight=PANEL_LABEL_WEIGHT,
        va="bottom",
        ha="left",
        clip_on=False,
    )


def inpanel_title(ax, text):
    ax.text(
        0.03, 0.97, text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=INPANEL_TITLE_SIZE,
        fontweight=INPANEL_TITLE_WEIGHT,
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor="white",
            edgecolor="none",
            alpha=0.82,
        ),
        zorder=10,
    )


def save_figure(fig, stem):
    stem = Path(stem)
    for ext in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{ext}")
        if ext == "png":
            fig.savefig(path, dpi=DPI)
        else:
            fig.savefig(path)
        print(f"Saved: {path}")


# =============================================================================
# 7. LOAD / VALIDATE DATA
# =============================================================================

def load_csv(path, name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")

    df = pd.read_csv(path)
    print(f"Loaded {name}: {path} | rows={len(df)}")
    return df


def validate_current(df):
    required = [
        "fold", "sample_id", CR_COL, AL_COL,
        CUR_STATE_COL, CUR_MED_COL, CUR_PED_COL,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Current table missing columns: {missing}")

    out = df.copy()
    out[CR_COL] = pd.to_numeric(out[CR_COL], errors="coerce")
    out[AL_COL] = pd.to_numeric(out[AL_COL], errors="coerce")
    out[CUR_STATE_COL] = out[CUR_STATE_COL].astype(str).str.strip()

    valid = set(STATE_COLORS)
    bad = (
        out[CR_COL].isna()
        | out[AL_COL].isna()
        | ~out[CUR_STATE_COL].isin(valid)
    )
    if bad.any():
        raise ValueError(f"Current pooled OOF has {int(bad.sum())} invalid coordinates/states")
    if out.sample_id.isna().any() or out.sample_id.astype(str).duplicated().any():
        raise ValueError("Current pooled OOF has missing or duplicate sample_id")
    if EXPECTED_RECORDS and len(out) != EXPECTED_RECORDS:
        raise ValueError(f"Expected {EXPECTED_RECORDS} current OOF records; found {len(out)}")
    return out


def validate_historical(df):
    required = [
        "fold", "sample_id", CR_COL, AL_COL,
        HIS_STATE_COL, HIS_MED_COL, HIS_PED_COL,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Historical table missing columns: {missing}")

    out = df.copy()
    out[CR_COL] = pd.to_numeric(out[CR_COL], errors="coerce")
    out[AL_COL] = pd.to_numeric(out[AL_COL], errors="coerce")
    out[HIS_STATE_COL] = out[HIS_STATE_COL].astype(str).str.strip()

    valid = set(STATE_COLORS)
    bad = (
        out[CR_COL].isna()
        | out[AL_COL].isna()
        | ~out[HIS_STATE_COL].isin(valid)
    )
    if bad.any():
        raise ValueError(f"Historical temporal OOF has {int(bad.sum())} invalid coordinates/states")
    if out.sample_id.isna().any() or out.sample_id.astype(str).duplicated().any():
        raise ValueError("Historical temporal OOF has missing or duplicate sample_id")
    if EXPECTED_RECORDS and len(out) != EXPECTED_RECORDS:
        raise ValueError(f"Expected {EXPECTED_RECORDS} historical OOF records; found {len(out)}")
    return out


def build_temporal_merge(current_df, historical_df):
    # Both periods assess EXACTLY the same outer-held-out queries. Verify this
    # before plotting; do not allow an inner join to silently drop mismatches.
    if len(current_df) != len(historical_df):
        raise RuntimeError("Current/historical sample counts differ")
    if set(current_df.sample_id.astype(str)) != set(historical_df.sample_id.astype(str)):
        raise RuntimeError("Current/historical held-out sample IDs differ")
    required_hist = ["source_key", "y_true", "quadrant_current", "MED_current", "PED_log1p_current"]
    missing = [x for x in required_hist if x not in historical_df.columns]
    if missing:
        raise RuntimeError(f"Expected the complete temporal_comparison.csv, missing: {missing}")
    paired = current_df[["sample_id", "fold", "source_key", CR_COL, AL_COL,
                         "y_true", "quadrant", "MED", "PED_log1p"]].merge(
        historical_df[["sample_id", "fold", "source_key", CR_COL, AL_COL,
                       "y_true", "quadrant_current", "MED_current", "PED_log1p_current"]],
        on="sample_id", validate="one_to_one", suffixes=("_cur", "_temp"), how="inner"
    )
    for label in ("fold", "source_key", "quadrant"):
        right = "quadrant_current" if label == "quadrant" else label + "_temp"
        left = label + "_cur" if label != "quadrant" else "quadrant"
        if not np.array_equal(paired[left].astype(str).to_numpy(), paired[right].astype(str).to_numpy()):
            raise RuntimeError(f"Historical/current pairing mismatch for {label}")
    for col in (CR_COL, AL_COL, "y_true", "MED", "PED_log1p"):
        right = ("MED_current" if col == "MED" else "PED_log1p_current"
                 if col == "PED_log1p" else col + "_temp")
        left = col if col in ("MED", "PED_log1p") else col + "_cur"
        if not np.allclose(paired[left].to_numpy(float), paired[right].to_numpy(float),
                           rtol=0, atol=2e-7):
            raise RuntimeError(f"Historical/current pairing mismatch for {col}")

    cur_keep = [
        "fold", "sample_id", "fold_test_idx",
        "source_key", "Ref_source_num", "Year",
        "Temperature", "E_Time",
        CR_COL, AL_COL, "x_Co", "x_Ni", "x_Ti", "x_W",
        "y_true", "y_pred", "error", CUR_ABS_ERR_COL,
        CUR_MED_COL, CUR_PED_COL, CUR_NSRC_COL, CUR_STD_COL,
        CUR_STATE_COL,
    ]
    his_keep = [
        "fold", "sample_id",
        "y_pred_2022", "error_2022", HIS_ABS_ERR_COL,
        "D_LEAF_2022", "D_SUPPORT_2022",
        HIS_MED_COL, HIS_PED_COL,
        HIS_NSRC_COL, HIS_STD_COL,
        HIS_STATE_COL,
    ]

    cur_keep = [c for c in cur_keep if c in current_df.columns]
    his_keep = [c for c in his_keep if c in historical_df.columns]

    merged = historical_df[his_keep].merge(
        current_df[cur_keep],
        on=["fold", "sample_id"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(current_df):
        raise RuntimeError(
            f"Paired merge lost records: merged={len(merged)}, current={len(current_df)}"
        )

    merged["stable_state"] = (
        merged[HIS_STATE_COL] == merged[CUR_STATE_COL]
    )

    merged["transition"] = (
        merged[HIS_STATE_COL].astype(str)
        + "→"
        + merged[CUR_STATE_COL].astype(str)
    )

    merged["change_category"] = np.select(
        [
            merged["stable_state"],
            merged[CUR_STATE_COL] == "Q1",
            merged[CUR_STATE_COL] == "Q2",
            merged[CUR_STATE_COL] == "Q3",
            merged[CUR_STATE_COL] == "Q4",
        ],
        [
            "Unchanged",
            "Change to Q1",
            "Change to Q2",
            "Change to Q3",
            "Change to Q4",
        ],
        default="Unchanged",
    )

    merged["into_Q1"] = (
        (merged[HIS_STATE_COL] != "Q1")
        & (merged[CUR_STATE_COL] == "Q1")
    )
    merged["into_Q4"] = (
        (merged[HIS_STATE_COL] != "Q4")
        & (merged[CUR_STATE_COL] == "Q4")
    )
    merged["Q4_to_Q1"] = (
        (merged[HIS_STATE_COL] == "Q4")
        & (merged[CUR_STATE_COL] == "Q1")
    )
    merged["Q1_to_Q4"] = (
        (merged[HIS_STATE_COL] == "Q1")
        & (merged[CUR_STATE_COL] == "Q4")
    )

    # DO NOT subtract raw historical/current MED: different fitted trees and
    # independently normalized MED reference distributions.
    merged["delta_PED_current_minus_2022"] = (
        merged[CUR_PED_COL] - merged[HIS_PED_COL]
    )

    if CUR_NSRC_COL in merged.columns and HIS_NSRC_COL in merged.columns:
        merged["delta_n_sources_current_minus_2022"] = (
            merged[CUR_NSRC_COL] - merged[HIS_NSRC_COL]
        )

    if CUR_STD_COL in merged.columns and HIS_STD_COL in merged.columns:
        merged["delta_lgkp_std_current_minus_2022"] = (
            merged[CUR_STD_COL] - merged[HIS_STD_COL]
        )

    if CUR_ABS_ERR_COL in merged.columns and HIS_ABS_ERR_COL in merged.columns:
        merged["delta_abs_error_current_minus_2022"] = (
            merged[CUR_ABS_ERR_COL] - merged[HIS_ABS_ERR_COL]
        )

    merged.to_csv(
        FIG5_DIR / "merged_temporal_current_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return merged


# =============================================================================
# 8. AXIS LIMITS
# =============================================================================

def axis_limits_from_df(df):
    """
    Build composition-space limits from one reference dataframe.

    For Fig. 4:
        derive once from current data -> use for BOTH panels (a,b)

    For Fig. 5:
        derive once from historical data -> use for BOTH panels (a,b)
    """
    cr = df[CR_COL]
    al = df[AL_COL]

    pad_x = max(X_PAD_MIN, 0.03 * (cr.max() - cr.min()))
    pad_y = max(Y_PAD_MIN, 0.04 * (al.max() - al.min()))

    x_lower = min(cr.min() - pad_x, X_AXIS_MIN_FLOOR)
    y_lower = min(al.min() - pad_y, Y_AXIS_MIN_FLOOR)

    x_upper = cr.max() + pad_x
    y_upper = al.max() + pad_y

    return (x_lower, x_upper), (y_lower, y_upper)


# =============================================================================
# 9. COMMON STATE LANDSCAPE
# =============================================================================

def plot_state_landscape(
    ax,
    df,
    state_col,
    xlim,
    ylim,
    *,
    legend=False,
    legend_title="Evidence state",
    title_text=None,
):
    total = len(df)

    for state in PLOT_ORDER:
        sub = df[df[state_col] == state]
        if len(sub) == 0:
            continue

        edgecolor = "black" if state == "Q1" else "white"
        linewidth = 0.78 if state == "Q1" else 0.34

        ax.scatter(
            sub[CR_COL],
            sub[AL_COL],
            s=STATE_SIZE[state],
            c=STATE_COLORS[state],
            alpha=STATE_ALPHA[state],
            edgecolors=edgecolor,
            linewidths=linewidth,
            zorder=STATE_ZORDER[state],
            label=f"{state} (n={len(sub)}, {100*len(sub)/total:.1f}%)",
            rasterized=True,
        )

    ax.grid(
        which="major",
        linestyle="--",
        linewidth=0.45,
        alpha=0.22,
        zorder=0,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    style_composition_axis(ax)

    if title_text:
        inpanel_title(ax, title_text)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        mapping = {
            lab.split()[0]: (h, lab)
            for h, lab in zip(handles, labels)
        }
        ordered = [
            mapping[q]
            for q in ["Q1", "Q2", "Q3", "Q4"]
            if q in mapping
        ]

        lg = ax.legend(
            [x[0] for x in ordered],
            [x[1] for x in ordered],
            title=legend_title,
            fontsize=LEGEND_FONT_SIZE,
            loc="upper right",
            handletextpad=0.5,
            borderaxespad=0.5,
        )

        lg.get_title().set_fontsize(LEGEND_TITLE_SIZE)
        lg.get_title().set_fontweight(LEGEND_TITLE_WEIGHT)


# =============================================================================
# 10. SHARED BIN EDGES + Q1 FRACTION HEATMAP
# =============================================================================

def aligned_edges(values, width):
    values = np.asarray(values, dtype=float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    start = np.floor(vmin / width) * width
    stop = np.ceil(vmax / width) * width

    edges = np.arange(
        start,
        stop + width * 1.001,
        width,
        dtype=float,
    )

    if edges[0] > vmin:
        edges = np.insert(edges, 0, edges[0] - width)

    if edges[-1] <= vmax:
        edges = np.append(edges, edges[-1] + width)

    return edges


def build_master_bin_edges(current_df, historical_df):
    """
    Use both tables only to guarantee one identical binning grid.
    Since the held-out samples should be the same, this should normally
    produce the same result as using either table alone.
    """
    cr_all = pd.concat(
        [current_df[CR_COL], historical_df[CR_COL]],
        ignore_index=True,
    )
    al_all = pd.concat(
        [current_df[AL_COL], historical_df[AL_COL]],
        ignore_index=True,
    )

    cr_edges = aligned_edges(cr_all, CR_BIN_WIDTH)
    al_edges = aligned_edges(al_all, AL_BIN_WIDTH)

    return cr_edges, al_edges


def compute_q1_fraction(
    df,
    state_col,
    cr_edges,
    al_edges,
):
    work = df.copy()

    work["_cr_bin"] = pd.cut(
        work[CR_COL],
        bins=cr_edges,
        right=False,
        include_lowest=True,
    )
    work["_al_bin"] = pd.cut(
        work[AL_COL],
        bins=al_edges,
        right=False,
        include_lowest=True,
    )
    work["_is_q1"] = (work[state_col] == "Q1").astype(int)

    agg = {
        "N_bin": (state_col, "size"),
        "N_Q1": ("_is_q1", "sum"),
    }

    if SOURCE_COL in work.columns:
        agg["N_sources_bin"] = (SOURCE_COL, "nunique")

    stats = (
        work
        .groupby(
            ["_al_bin", "_cr_bin"],
            observed=False,
        )
        .agg(**agg)
        .reset_index()
    )

    stats["f_Q1"] = np.where(
        stats["N_bin"] > 0,
        stats["N_Q1"] / stats["N_bin"],
        np.nan,
    )

    stats["Cr_left"] = stats["_cr_bin"].apply(
        lambda x: x.left if pd.notna(x) else np.nan
    ).astype(float)
    stats["Cr_right"] = stats["_cr_bin"].apply(
        lambda x: x.right if pd.notna(x) else np.nan
    ).astype(float)

    stats["Al_left"] = stats["_al_bin"].apply(
        lambda x: x.left if pd.notna(x) else np.nan
    ).astype(float)
    stats["Al_right"] = stats["_al_bin"].apply(
        lambda x: x.right if pd.notna(x) else np.nan
    ).astype(float)

    stats["Cr_center"] = (
        stats["Cr_left"] + stats["Cr_right"]
    ) / 2
    stats["Al_center"] = (
        stats["Al_left"] + stats["Al_right"]
    ) / 2

    stats["shown_main"] = stats["N_bin"] >= MIN_BIN_N

    n_al = len(al_edges) - 1
    n_cr = len(cr_edges) - 1

    n_matrix = np.zeros((n_al, n_cr), dtype=float)
    q1_matrix = np.zeros((n_al, n_cr), dtype=float)

    al_codes = work["_al_bin"].cat.codes.to_numpy()
    cr_codes = work["_cr_bin"].cat.codes.to_numpy()
    is_q1 = work["_is_q1"].to_numpy(dtype=int)

    for ai, ci, q1 in zip(al_codes, cr_codes, is_q1):
        if ai < 0 or ci < 0:
            continue

        n_matrix[ai, ci] += 1
        q1_matrix[ai, ci] += q1

    with np.errstate(
        invalid="ignore",
        divide="ignore",
    ):
        frac_matrix = q1_matrix / n_matrix

    frac_matrix[n_matrix < MIN_BIN_N] = np.nan

    return stats, n_matrix, frac_matrix


def plot_q1_fraction_heatmap(
    ax,
    fig,
    n_matrix,
    frac_matrix,
    cr_edges,
    al_edges,
    xlim,
    ylim,
    *,
    colorbar_label,
    note=True,
):
    low_n_mask = (
        (n_matrix > 0)
        & (n_matrix < MIN_BIN_N)
    ).astype(float)

    low_n_mask[low_n_mask == 0] = np.nan

    sparse_cmap = ListedColormap(["#E7E7E7"])

    ax.pcolormesh(
        cr_edges,
        al_edges,
        low_n_mask,
        cmap=sparse_cmap,
        shading="flat",
        vmin=0,
        vmax=1,
        zorder=0,
    )

    cmap = plt.get_cmap("YlGn").copy()
    cmap.set_bad((1, 1, 1, 0))

    im = ax.pcolormesh(
        cr_edges,
        al_edges,
        np.ma.masked_invalid(frac_matrix),
        cmap=cmap,
        shading="flat",
        vmin=0.0,
        vmax=1.0,
        edgecolors="white",
        linewidth=0.55,
        zorder=1,
    )

    if ANNOTATE_COUNTS:
        for ai in range(n_matrix.shape[0]):
            for ci in range(n_matrix.shape[1]):
                n = int(n_matrix[ai, ci])

                if n < MIN_BIN_N:
                    continue

                x = (
                    cr_edges[ci]
                    + cr_edges[ci + 1]
                ) / 2
                y = (
                    al_edges[ai]
                    + al_edges[ai + 1]
                ) / 2

                f = frac_matrix[ai, ci]

                text_color = (
                    "white"
                    if np.isfinite(f) and f >= 0.62
                    else "#333333"
                )

                ax.text(
                    x,
                    y,
                    f"{n}",
                    ha="center",
                    va="center",
                    fontsize=COUNT_ANNOT_SIZE,
                    color=text_color,
                    zorder=2,
                )

    style_composition_axis(ax)

    # IMPORTANT:
    # force the heatmap to use exactly the same range as its scatter panel
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    cbar = fig.colorbar(
        im,
        ax=ax,
        pad=0.025,
        fraction=0.050,
    )

    cbar.set_label(
        colorbar_label,
        fontsize=COLORBAR_LABEL_SIZE,
        fontweight=COLORBAR_LABEL_WEIGHT,
    )

    cbar.set_ticks(np.linspace(0, 1, 6))

    cbar.ax.tick_params(
        labelsize=COLORBAR_TICK_SIZE,
        width=0.95,
        length=3.2,
    )

    cbar.outline.set_linewidth(1.0)

    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontsize(COLORBAR_TICK_SIZE)
        lbl.set_fontweight(COLORBAR_TICK_WEIGHT)

    if note:
        ax.text(
            NOTE_X,
            NOTE_Y,
            rf"Colored bins: $N_{{\rm bin}}\geq {MIN_BIN_N}$",
            transform=ax.transAxes,
            ha=NOTE_HA,
            va=NOTE_VA,
            fontsize=NOTE_FONT_SIZE,
            fontweight=NOTE_FONT_WEIGHT,
            color="#444444",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="none",
                alpha=0.78,
            ),
            zorder=5,
        )

    return im


# =============================================================================
# 11. FIG. 4
# =============================================================================

def make_fig4(
    current_df,
    cr_edges,
    al_edges,
):
    # one canonical range derived from Fig.4(a)
    fig4_xlim, fig4_ylim = axis_limits_from_df(current_df)

    # current Q1-fraction data
    cur_stats, cur_n, cur_frac = compute_q1_fraction(
        current_df,
        CUR_STATE_COL,
        cr_edges,
        al_edges,
    )

    cur_stats.to_csv(
        FIG4_DIR / "Fig4b_bin_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    qualifying = cur_stats.loc[cur_stats["shown_main"] & cur_stats["N_Q1"].gt(0)].copy()
    if not qualifying.empty:
        print("\nCurrent Q1-positive composition bins (N_bin>=5), sorted by f_Q1:")
        cols = ["Cr_left", "Cr_right", "Al_left", "Al_right", "N_bin", "N_Q1", "f_Q1"]
        print(qualifying.sort_values(["f_Q1", "N_bin"], ascending=False)[cols].head(20).to_string(index=False))

    # -------------------------
    # Fig. 4(a)
    # -------------------------
    fig, ax = plt.subplots(
        figsize=(5.65, 4.85)
    )

    plot_state_landscape(
        ax,
        current_df,
        CUR_STATE_COL,
        fig4_xlim,
        fig4_ylim,
        legend=True,
        legend_title="Evidence state",
    )

    panel_label(ax, "(a)")

    fig.tight_layout()

    save_figure(
        fig,
        FIG4_DIR / "Fig4a_current_evidence_states",
    )

    plt.close(fig)

    # -------------------------
    # Fig. 4(b)
    # -------------------------
    fig, ax = plt.subplots(
        figsize=(5.65, 4.85)
    )

    plot_q1_fraction_heatmap(
        ax,
        fig,
        cur_n,
        cur_frac,
        cr_edges,
        al_edges,
        fig4_xlim,
        fig4_ylim,
        colorbar_label=r"Local Q1 fraction, $f_{\mathrm{Q1}}$",
        note=True,
    )

    panel_label(ax, "(b)")

    fig.tight_layout()

    save_figure(
        fig,
        FIG4_DIR / "Fig4b_current_Q1_fraction",
    )

    plt.close(fig)

    # -------------------------
    # Combined Fig. 4(a-b)
    # -------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.15, 4.95),
        gridspec_kw={"wspace": 0.34},
    )

    plot_state_landscape(
        axes[0],
        current_df,
        CUR_STATE_COL,
        fig4_xlim,
        fig4_ylim,
        legend=True,
        legend_title="Evidence state",
    )
    panel_label(axes[0], "(a)")

    plot_q1_fraction_heatmap(
        axes[1],
        fig,
        cur_n,
        cur_frac,
        cr_edges,
        al_edges,
        fig4_xlim,
        fig4_ylim,
        colorbar_label=r"Local Q1 fraction, $f_{\mathrm{Q1}}$",
        note=True,
    )
    panel_label(axes[1], "(b)")

    fig.subplots_adjust(
        left=0.075,
        right=0.955,
        bottom=0.14,
        top=0.96,
        wspace=0.34,
    )

    save_figure(
        fig,
        FIG4_DIR / "Fig4ab_current_CrAl_evidence",
    )

    plt.close(fig)

    return fig4_xlim, fig4_ylim


# =============================================================================
# 12. FIG. 5(b): CHANGE MAP
# =============================================================================

def plot_change_landscape(
    ax,
    merged,
    xlim,
    ylim,
    *,
    legend=False,
):
    total = len(merged)

    for cat in CHANGE_ORDER:
        sub = merged[
            merged["change_category"] == cat
        ]

        if len(sub) == 0:
            continue

        edgecolor = (
            "black"
            if cat == "Change to Q1"
            else "white"
        )

        linewidth = (
            0.78
            if cat == "Change to Q1"
            else 0.34
        )

        zorder = (
            5
            if cat == "Change to Q1"
            else 1 if cat == "Unchanged"
            else 3
        )

        ax.scatter(
            sub[CR_COL],
            sub[AL_COL],
            s=CHANGE_SIZE[cat],
            c=CHANGE_COLORS[cat],
            alpha=CHANGE_ALPHA[cat],
            edgecolors=edgecolor,
            linewidths=linewidth,
            zorder=zorder,
            label=f"{cat} (n={len(sub)}, {100*len(sub)/total:.1f}%)",
            rasterized=True,
        )

    ax.grid(
        which="major",
        linestyle="--",
        linewidth=0.45,
        alpha=0.22,
        zorder=0,
    )

    # IMPORTANT:
    # use EXACT same axis range as historical Fig.5(a)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    style_composition_axis(ax)

    if legend:
        desired = [
            "Unchanged",
            "Change to Q1",
            "Change to Q2",
            "Change to Q3",
            "Change to Q4",
        ]

        handles, labels = ax.get_legend_handles_labels()

        mapping = {
            lab.split(" (n=")[0]: (h, lab)
            for h, lab in zip(handles, labels)
        }

        ordered = [
            mapping[x]
            for x in desired
            if x in mapping
        ]

        lg = ax.legend(
            [x[0] for x in ordered],
            [x[1] for x in ordered],
            title="Temporal change",
            fontsize=LEGEND_FONT_SIZE,
            loc="upper right",
            handletextpad=0.5,
            borderaxespad=0.5,
        )

        lg.get_title().set_fontsize(
            LEGEND_TITLE_SIZE
        )
        lg.get_title().set_fontweight(
            LEGEND_TITLE_WEIGHT
        )


# =============================================================================
# 13. FIG. 5(c): TRANSITION MATRIX
# =============================================================================

def build_transition_matrices(merged):
    order = ["Q1", "Q2", "Q3", "Q4"]

    counts = pd.crosstab(
        pd.Categorical(
            merged[HIS_STATE_COL],
            categories=order,
            ordered=True,
        ),
        pd.Categorical(
            merged[CUR_STATE_COL],
            categories=order,
            ordered=True,
        ),
        dropna=False,
    )

    counts.index.name = "state_2022"
    counts.columns.name = "state_current"

    if not REFERENCE_TRANSITION_MATRIX_CSV.is_file():
        raise FileNotFoundError(f"Frozen transition matrix not found: {REFERENCE_TRANSITION_MATRIX_CSV}")
    frozen_counts = pd.read_csv(REFERENCE_TRANSITION_MATRIX_CSV, index_col=0)
    frozen_counts = frozen_counts.reindex(index=order, columns=order, fill_value=0)
    if not np.array_equal(counts.to_numpy(dtype=int), frozen_counts.to_numpy(dtype=int)):
        raise RuntimeError("Plotted transition matrix does not reproduce official Nested temporal matrix")
    print("Frozen Nested temporal transition matrix: STRICT PASS")

    row_frac = counts.div(
        counts.sum(axis=1).replace(0, np.nan),
        axis=0,
    )

    counts.to_csv(
        FIG5_DIR / "transition_matrix_counts.csv",
        encoding="utf-8-sig",
    )

    row_frac.to_csv(
        FIG5_DIR / "transition_matrix_row_fraction.csv",
        encoding="utf-8-sig",
    )

    return counts, row_frac


def plot_transition_matrix(
    ax,
    fig,
    counts,
    row_frac,
):
    mat = row_frac.to_numpy(dtype=float)

    im = ax.imshow(
        mat,
        vmin=0,
        vmax=1,
        cmap="Blues",
        aspect="equal",
    )

    order = list(counts.columns)

    ax.set_xticks(
        np.arange(len(order))
    )
    ax.set_yticks(
        np.arange(len(order))
    )

    ax.set_xticklabels(
        order,
        fontsize=TICK_LABEL_SIZE,
        fontweight=TICK_LABEL_WEIGHT,
    )
    ax.set_yticklabels(
        order,
        fontsize=TICK_LABEL_SIZE,
        fontweight=TICK_LABEL_WEIGHT,
    )

    ax.set_xlabel(
        "Current state",
        fontsize=MATRIX_AXIS_LABEL_SIZE,
        fontweight=MATRIX_AXIS_LABEL_WEIGHT,
    )
    ax.set_ylabel(
        "Historical state (≤2022 reference)",
        fontsize=MATRIX_AXIS_LABEL_SIZE,
        fontweight=MATRIX_AXIS_LABEL_WEIGHT,
    )

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            count = int(
                counts.iloc[i, j]
            )
            frac = mat[i, j]

            if np.isnan(frac):
                text = f"{count}\n(NA)"
            else:
                text = (
                    f"{count}\n"
                    f"({100*frac:.1f}%)"
                )

            text_color = (
                "white"
                if np.isfinite(frac) and frac >= 0.45
                else "black"
            )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=MATRIX_ANNOT_SIZE,
                fontweight=MATRIX_ANNOT_WEIGHT,
                color=text_color,
            )

    ax.set_xticks(
        np.arange(-0.5, len(order), 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, len(order), 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=1.6,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.045,
        pad=0.03,
    )

    cbar.set_label(
        "Row-normalized fraction",
        fontsize=COLORBAR_LABEL_SIZE,
        fontweight=COLORBAR_LABEL_WEIGHT,
    )

    cbar.ax.tick_params(
        labelsize=COLORBAR_TICK_SIZE,
        width=0.95,
        length=3.2,
    )

    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontsize(COLORBAR_TICK_SIZE)
        lbl.set_fontweight(COLORBAR_TICK_WEIGHT)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    return im


# =============================================================================
# 14. SUMMARY EXPORT
# =============================================================================

def export_temporal_summary(merged):
    n = len(merged)
    rows = []

    if {
        "y_true",
        "y_pred_2022",
    }.issubset(merged.columns):

        y = merged["y_true"].to_numpy(dtype=float)
        yp = merged["y_pred_2022"].to_numpy(dtype=float)

        ss_res = np.sum((y - yp) ** 2)
        ss_tot = np.sum(
            (y - np.mean(y)) ** 2
        )

        r2 = (
            1 - ss_res / ss_tot
            if ss_tot > 0
            else np.nan
        )
        mae = np.mean(
            np.abs(y - yp)
        )
        rmse = np.sqrt(
            np.mean((y - yp) ** 2)
        )

        rows.extend([
            ["Historical pooled OOF R2", r2],
            ["Historical pooled OOF MAE", mae],
            ["Historical pooled OOF RMSE", rmse],
        ])

    if {
        "y_true",
        "y_pred",
    }.issubset(merged.columns):

        y = merged["y_true"].to_numpy(dtype=float)
        yp = merged["y_pred"].to_numpy(dtype=float)

        ss_res = np.sum((y - yp) ** 2)
        ss_tot = np.sum(
            (y - np.mean(y)) ** 2
        )

        r2 = (
            1 - ss_res / ss_tot
            if ss_tot > 0
            else np.nan
        )
        mae = np.mean(
            np.abs(y - yp)
        )
        rmse = np.sqrt(
            np.mean((y - yp) ** 2)
        )

        rows.extend([
            ["Current pooled OOF R2", r2],
            ["Current pooled OOF MAE", mae],
            ["Current pooled OOF RMSE", rmse],
        ])

    stable = int(
        merged["stable_state"].sum()
    )

    changed = int(
        (~merged["stable_state"]).sum()
    )

    into_q1 = int(
        merged["into_Q1"].sum()
    )

    into_q4 = int(
        merged["into_Q4"].sum()
    )

    q4_q1 = int(
        merged["Q4_to_Q1"].sum()
    )

    q1_q4 = int(
        merged["Q1_to_Q4"].sum()
    )

    counts = [
        ("Stable state", stable),
        ("Changed state", changed),
        ("Into Q1", into_q1),
        ("Into Q4", into_q4),
        ("Q4 to Q1", q4_q1),
        ("Q1 to Q4", q1_q4),
    ]

    for name, count in counts:
        rows.append(
            [name, count]
        )
        rows.append(
            [f"{name} fraction", count / n]
        )

    summary = pd.DataFrame(
        rows,
        columns=["item", "value"],
    )

    summary.to_csv(
        FIG5_DIR / "temporal_transition_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return summary


# =============================================================================
# 15. FIG. 5 + SUPPLEMENTARY HISTORICAL HEATMAP
# =============================================================================

def make_fig5(
    historical_df,
    merged,
    cr_edges,
    al_edges,
):
    # Fig.5(a) is the reference.
    # Fig.5(b) MUST use exactly this range.
    fig5_xlim, fig5_ylim = axis_limits_from_df(
        historical_df
    )

    counts, row_frac = build_transition_matrices(
        merged
    )

    # -------------------------
    # Fig. 5(a)
    # -------------------------
    fig, ax = plt.subplots(
        figsize=(5.65, 4.85)
    )

    plot_state_landscape(
        ax,
        historical_df,
        HIS_STATE_COL,
        fig5_xlim,
        fig5_ylim,
        legend=True,
        legend_title="Historical state",
        title_text="Historical evidence state (≤2022)",
    )

    panel_label(ax, "(a)")

    fig.tight_layout()

    save_figure(
        fig,
        FIG5_DIR / "Fig5a_historical_evidence_states",
    )

    plt.close(fig)

    # -------------------------
    # Fig. 5(b)
    # -------------------------
    fig, ax = plt.subplots(
        figsize=(5.65, 4.85)
    )

    plot_change_landscape(
        ax,
        merged,
        fig5_xlim,
        fig5_ylim,
        legend=True,
    )

    inpanel_title(
        ax,
        "Evidence-state changes to current",
    )

    panel_label(ax, "(b)")

    fig.tight_layout()

    save_figure(
        fig,
        FIG5_DIR / "Fig5b_temporal_change_map",
    )

    plt.close(fig)

    # -------------------------
    # Fig. 5(c)
    # -------------------------
    fig, ax = plt.subplots(
        figsize=(5.35, 4.85)
    )

    plot_transition_matrix(
        ax,
        fig,
        counts,
        row_frac,
    )

    panel_label(ax, "(c)")

    fig.tight_layout()

    save_figure(
        fig,
        FIG5_DIR / "Fig5c_transition_matrix",
    )

    plt.close(fig)

    # -------------------------
    # Combined Fig. 5(a-c)
    # -------------------------
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.45, 5.0),
        gridspec_kw={"wspace": 0.36},
    )

    plot_state_landscape(
        axes[0],
        historical_df,
        HIS_STATE_COL,
        fig5_xlim,
        fig5_ylim,
        legend=False,
        title_text="Historical evidence state (≤2022)",
    )
    panel_label(
        axes[0],
        "(a)",
    )

    plot_change_landscape(
        axes[1],
        merged,
        fig5_xlim,
        fig5_ylim,
        legend=True,
    )
    inpanel_title(
        axes[1],
        "Evidence-state changes to current",
    )
    panel_label(
        axes[1],
        "(b)",
    )

    plot_transition_matrix(
        axes[2],
        fig,
        counts,
        row_frac,
    )
    panel_label(
        axes[2],
        "(c)",
    )

    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.14,
        top=0.95,
        wspace=0.34,
    )

    save_figure(
        fig,
        FIG5_DIR / "Fig5abc_temporal_evolution",
    )

    plt.close(fig)

    # -------------------------
    # Supplementary:
    # historical Q1 fraction
    # same bin grid as current Fig.4(b)
    # -------------------------
    hist_stats, hist_n, hist_frac = compute_q1_fraction(
        historical_df,
        HIS_STATE_COL,
        cr_edges,
        al_edges,
    )

    hist_stats.to_csv(
        FIG5_DIR / "SFig_historical_Q1_fraction_bin_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(
        figsize=(5.65, 4.85)
    )

    plot_q1_fraction_heatmap(
        ax,
        fig,
        hist_n,
        hist_frac,
        cr_edges,
        al_edges,
        fig5_xlim,
        fig5_ylim,
        colorbar_label=r"Historical local Q1 fraction, $f_{\mathrm{Q1}}$",
        note=True,
    )

    fig.tight_layout()

    save_figure(
        fig,
        FIG5_DIR / "SFig_historical_Q1_fraction_heatmap",
    )

    plt.close(fig)

    return fig5_xlim, fig5_ylim


# =============================================================================
# 16. MAIN
# =============================================================================

def main():
    set_publication_style()

    current = validate_current(
        load_csv(
            CURRENT_CSV,
            "current pooled OOF",
        )
    )

    historical = validate_historical(
        load_csv(
            HISTORICAL_CSV,
            "historical pooled OOF",
        )
    )

    merged = build_temporal_merge(
        current,
        historical,
    )
    print("Paired Nested records: STRICT PASS", len(merged))
    print("Current P50 states:", current[CUR_STATE_COL].value_counts().reindex(STATE_COLORS, fill_value=0).to_dict())
    print("Historical P50 states:", historical[HIS_STATE_COL].value_counts().reindex(STATE_COLORS, fill_value=0).to_dict())

    # One shared binning grid for:
    # - Fig.4(b) current Q1 fraction
    # - supplementary historical Q1 fraction
    cr_edges, al_edges = build_master_bin_edges(
        current,
        historical,
    )

    # Fig.4
    fig4_xlim, fig4_ylim = make_fig4(
        current,
        cr_edges,
        al_edges,
    )

    # Fig.5
    fig5_xlim, fig5_ylim = make_fig5(
        historical,
        merged,
        cr_edges,
        al_edges,
    )

    export_temporal_summary(
        merged
    )

    print("\n" + "=" * 80)
    print("Finished")
    print("=" * 80)

    print("\nFig.4 axis range:")
    print(f"  x = {fig4_xlim}")
    print(f"  y = {fig4_ylim}")
    print("  Fig.4(a) and Fig.4(b) use EXACTLY this same range.")

    print("\nFig.5 axis range:")
    print(f"  x = {fig5_xlim}")
    print(f"  y = {fig5_ylim}")
    print("  Fig.5(a) and Fig.5(b) use EXACTLY this same range.")

    print("\nKey controls:")
    print("  Point size Fig.4/Fig.5a : STATE_SIZE")
    print("  Point size Fig.5b       : CHANGE_SIZE")
    print("  Negative axis padding   : X_AXIS_MIN_FLOOR / Y_AXIS_MIN_FLOOR")
    print("  Heatmap note position   : NOTE_X / NOTE_Y / NOTE_HA / NOTE_VA")

    print(f"\nFig.4 outputs: {FIG4_DIR.resolve()}")
    print(f"Fig.5 outputs: {FIG5_DIR.resolve()}")


if __name__ == "__main__":
    main()
