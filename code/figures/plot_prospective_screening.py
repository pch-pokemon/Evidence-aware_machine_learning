"""Render the four prospective screening panels from the released candidate data."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from verify_published_evidence import verify_prospective


def render(output):
    candidates, tables = verify_prospective()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    states, prediction, matches, cards = axes.ravel()
    palette = {"Q1": "#489266", "Q2": "#397bac", "Q3": "#e6ae42", "Q4": "#c75854"}
    for state, color in palette.items():
        subset = candidates[candidates.state.eq(state)]
        states.scatter(subset.x_Al, subset.x_Cr, s=11, c=color, label=state)
    states.legend(ncol=4, loc="upper center", frameon=False)
    surface = prediction.scatter(candidates.x_Al, candidates.x_Cr, s=11,
                                 c=candidates.predicted_lgkp, cmap="viridis")
    fig.colorbar(surface, ax=prediction, label=r"Predicted $\lg k_p$")
    for ax in (states, prediction):
        ax.set(xlabel="Al (at.%)", ylabel="Cr (at.%)")

    pairs = tables["Fig_6c_matched_pairs.csv"]
    matches.plot(pairs.delta_predicted_lgkp, pairs.cumulative_count, color="#215d87")
    for row in tables["Fig_6c_tolerances.csv"].itertuples(index=False):
        matches.scatter(row.delta_lgkp_tolerance, row.pair_count, c="#215d87", zorder=3)
        matches.annotate(f"{row.pair_count:,}", (row.delta_lgkp_tolerance, row.pair_count),
                         xytext=(-5, 7), textcoords="offset points", ha="right")
    matches.set(xlabel=r"$|\Delta \widehat{\lg k_p}|$", ylabel="Cumulative Q1–Q4 pairs",
                xlim=(0, .104), ylim=(0, 4250))

    rep = tables["Fig_6d_representative_pair.csv"].iloc[0]
    cards.set_axis_off()
    for x, label in ((.05, "A"), (.54, "B")):
        state = rep[f"state_{label}"]
        lines = [
            f"Candidate {label} · {state}",
            rep[f"candidate_{label}"],
            f"Al {rep[f'x_Al_{label}']:.1f}, Cr {rep[f'x_Cr_{label}']:.1f} at.%",
            f"Predicted lgkp  {rep[f'predicted_lgkp_{label}']:.5f}",
            f"MED − P50  {rep[f'MED_minus_P50_{label}']:+.4f}",
            f"ln(1+PED) − P50  {rep[f'ln1p_PED_minus_P50_{label}']:+.4f}",
            f"Neighbor sources  {rep[f'n_sources_{label}']:.0f}",
            f"Neighbor lgkp SD  {rep[f'lgkp_std_{label}']:.3f}",
        ]
        cards.text(x, .84, "\n".join(lines), transform=cards.transAxes, va="top",
                   linespacing=1.5, bbox={"boxstyle": "round,pad=.65", "fc": palette[state], "alpha": .16})
    cards.text(.5, .12, f"Absolute prediction difference  {rep.delta_predicted_lgkp:.6f}",
               ha="center", transform=cards.transAxes)
    for label, axis in zip("abcd", axes.ravel()):
        axis.set_title(f"({label})", loc="left", fontweight="bold")

    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "Fig_6.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    render(parser.parse_args().output_dir)

