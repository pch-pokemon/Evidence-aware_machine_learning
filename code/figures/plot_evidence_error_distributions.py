# -*- coding: utf-8 -*-
"""
Further optimized plot for Q1-Q4 absolute prediction-error distributions.

Input
-----
Controlled nested OOF CSV with quadrant and abs_error

Output
------
Fig_error_distribution_Q1_Q4_v3.png
Fig_error_distribution_Q1_Q4_v3.svg
Fig_error_distribution_Q1_Q4_v3.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. Paths and style
# =========================
INPUT_CSV = Path("models/nested_evidence_final/pooled_source_oof_evidence.csv")
OUT_STEM = Path("Fig_error_distribution_Q1_Q4_v3")

COLORS = ["#4C956C", "#5B8EBC", "#E6A84A", "#D66A6A"]
LABELS = ["Q1", "Q2", "Q3", "Q4"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 15,
    "axes.labelsize": 20,
    "axes.labelweight": "bold",
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.linewidth": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================
# 2. Read data
# =========================
df = pd.read_csv(INPUT_CSV, usecols=["quadrant", "abs_error"])
data = []
for label in LABELS:
    s = pd.to_numeric(df.loc[df.quadrant.eq(label), "abs_error"], errors="coerce").dropna()
    s = s[np.isfinite(s)]
    if len(s) == 0:
        raise ValueError(f"{c} contains no valid numeric values.")
    data.append(s.to_numpy(dtype=float))

summary = []
for label, arr in zip(LABELS, data):
    q1 = np.quantile(arr, 0.25)
    q3 = np.quantile(arr, 0.75)
    summary.append({
        "state": label,
        "n": len(arr),
        "mean_abs_error_MAE": np.mean(arr),
        "median": np.median(arr),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    })

summary_df = pd.DataFrame(summary)
print("\nSummary statistics")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# =========================
# 3. Plot
# =========================
fig, ax = plt.subplots(figsize=(5.7, 4.9))
positions = np.arange(1, 5)

# violin
vp = ax.violinplot(
    data,
    positions=positions,
    widths=0.78,
    showmeans=False,
    showmedians=False,
    showextrema=False,
    bw_method="scott",
)

for body, color in zip(vp["bodies"], COLORS):
    body.set_facecolor(color)
    body.set_edgecolor(color)
    body.set_alpha(0.30)
    body.set_linewidth(1.1)

# jitter points
rng = np.random.default_rng(42)
for i, (arr, color) in enumerate(zip(data, COLORS), start=1):
    n = len(arr)
    alpha = 0.42 if n <= 120 else 0.20
    size = 16 if n <= 120 else 11

    jitter = rng.normal(loc=0.0, scale=0.055, size=n)
    jitter = np.clip(jitter, -0.18, 0.18)

    ax.scatter(
        np.full(n, i) + jitter,
        arr,
        s=size,
        alpha=alpha,
        facecolor=color,
        edgecolor="none",
        zorder=2,
        rasterized=True,
    )

# boxplot
bp = ax.boxplot(
    data,
    positions=positions,
    widths=0.20,
    patch_artist=True,
    showfliers=False,
    whis=1.5,
    zorder=4,
)

for patch, color in zip(bp["boxes"], COLORS):
    patch.set_facecolor("white")
    patch.set_edgecolor(color)
    patch.set_alpha(0.95)
    patch.set_linewidth(1.2)

for median in bp["medians"]:
    median.set_color("black")
    median.set_linewidth(1.6)

for whisker in bp["whiskers"]:
    whisker.set_color("#3A3A3A")
    whisker.set_linewidth(1.0)

for cap in bp["caps"]:
    cap.set_color("#3A3A3A")
    cap.set_linewidth(1.0)

# mean marker
means = [np.mean(arr) for arr in data]
ax.scatter(
    positions,
    means,
    marker="D",
    s=36,
    facecolor="black",
    edgecolor="white",
    linewidth=0.7,
    zorder=5,
    label="Mean (MAE)",
)

# =========================
# 4. Axes / labels / annotations
# =========================
# Use mathtext bold for n so that it is visibly bold.
xticklabels = []
for label, row in zip(LABELS, summary):
    xticklabels.append(
        f"{label}\n" + rf"$\mathbf{{n}}$={row['n']}" + f"\nMAE={row['mean_abs_error_MAE']:.3f}"
    )

ax.set_xticks(positions)
ax.set_xticklabels(xticklabels, fontweight="bold")

ax.set_ylabel(r"Absolute prediction error, $|\Delta \lg k_p|$", fontsize=15, fontweight="bold")
ax.set_xlabel("Evidence state", fontsize=15, fontweight="bold")

for tick in ax.get_yticklabels():
    tick.set_fontweight("bold")

for side in ["left", "bottom", "top", "right"]:
    ax.spines[side].set_visible(True)
    ax.spines[side].set_linewidth(1.2)

ax.tick_params(direction="in", length=4.5, width=1.0)

ax.axhline(0, color="#777777", linewidth=0.7, linestyle="-", alpha=0.5, zorder=0)

ymax = max(np.max(arr) for arr in data)
for x, row in zip(positions, summary):
    ax.text(
        x,
        row["q3"] + 0.05 * ymax,
        f"Med={row['median']:.3f}",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
    )

ax.set_ylim(bottom=0, top=ymax * 1.16 if ymax > 0 else 1.0)

legend = ax.legend(
    frameon=False,
    loc="upper left",
    fontsize=12.5,
    handletextpad=0.6,
    borderpad=0.2,
    labelspacing=0.4,
)
for text in legend.get_texts():
    text.set_fontweight("bold")

fig.tight_layout()

# =========================
# 5. Save
# =========================
fig.savefig(f"{OUT_STEM}.png", dpi=600, bbox_inches="tight")
fig.savefig(f"{OUT_STEM}.svg", bbox_inches="tight")
fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight")

print("\nSaved:")
print(f"  {OUT_STEM}.png")
print(f"  {OUT_STEM}.svg")
print(f"  {OUT_STEM}.pdf")

plt.show()
