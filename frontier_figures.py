"""Publication figures for the quality-vs-debiasing frontier.

Reads analysis-output/sweep_points.json (produced by frontier_analysis.py) and
writes vector PDFs. Palette is Okabe-Ito (colourblind-safe); every method also has
a distinct marker so the figures survive greyscale printing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent / "analysis-output"
FIG_DIR = OUT_DIR / "figures"

# Okabe-Ito, colourblind-safe
STYLE: Dict[str, tuple] = {
    "CAD projection": ("#0072B2", "o"),
    "CAD mean-diff":  ("#009E73", "s"),
    "CAD probe":      ("#56B4E9", "D"),
    "CAA":            ("#D55E00", "^"),
    "FairSteer":      ("#E69F00", "v"),
    "Angular":        ("#CC79A7", "P"),
    "SADI":           ("#000000", "X"),
}
CAD_METHODS = {"CAD projection", "CAD mean-diff", "CAD probe"}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def load() -> dict:
    return json.loads((OUT_DIR / "sweep_points.json").read_text())


def clean_pareto(points: List[dict]) -> List[dict]:
    """Non-dominated among points that are NOT degenerate and not past neutral."""
    pool = [p for p in points if not p["degenerate"] and p["med"] >= 0]
    front = [p for p in pool if not any(
        q is not p and q["med"] <= p["med"] and q["judge_mean"] >= p["judge_mean"]
        and (q["med"] < p["med"] or q["judge_mean"] > p["judge_mean"]) for q in pool)]
    return sorted(front, key=lambda p: -p["med"])


def figure_main(data: dict) -> None:
    pts = data["points"]
    base_j, base_m = data["baseline_judge_mean"], data["baseline_med_mean"]
    base_j_sd = data["baseline_judge_sd"]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    # overshoot region: pushed past neutral, information withheld
    ax.axvspan(-0.40, 0.0, color="#D55E00", alpha=0.055, zorder=0)
    ax.text(-0.20, 1.15, "overshoot\n(past neutral)", ha="center", va="bottom",
            fontsize=7.5, color="#8a3800", style="italic")
    ax.axvline(0.0, color="#555555", lw=0.9, ls="--", zorder=1)
    ax.text(0.012, 9.05, "neutral", fontsize=7.5, color="#555555", rotation=90, va="top")

    # baseline reference band (unsteered quality, 10 no-steer points)
    ax.axhspan(base_j - 2 * base_j_sd, base_j + 2 * base_j_sd,
               color="#999999", alpha=0.18, zorder=0)
    ax.axhline(base_j, color="#666666", lw=0.9, ls=":", zorder=1)
    ax.text(-0.375, base_j + 0.10, f"unsteered baseline ({base_j:.2f})",
            fontsize=7.5, color="#444444")

    # clean Pareto front line, drawn first so markers sit on top
    front = clean_pareto(pts)
    ax.plot([p["med"] for p in front], [p["judge_mean"] for p in front],
            color="#0072B2", lw=1.3, ls="-", alpha=0.45, zorder=2)

    for method, (colour, marker) in STYLE.items():
        sub = [p for p in pts if p["method"] == method]
        if not sub:
            continue
        for degen, kwargs in [
            (False, dict(facecolors=colour, edgecolors="white", linewidths=0.5, alpha=0.95)),
            (True,  dict(facecolors="none", edgecolors=colour, linewidths=1.0, alpha=0.75)),
        ]:
            grp = [p for p in sub if p["degenerate"] is degen]
            if not grp:
                continue
            ax.scatter([p["med"] for p in grp], [p["judge_mean"] for p in grp],
                       marker=marker, s=46 if not degen else 42, zorder=4, **kwargs)
            if not degen:   # CI only on the points we make claims about
                ax.errorbar([p["med"] for p in grp], [p["judge_mean"] for p in grp],
                            yerr=[[p["judge_mean"] - p["judge_ci_lo"] for p in grp],
                                  [p["judge_ci_hi"] - p["judge_mean"] for p in grp]],
                            fmt="none", ecolor=colour, elinewidth=0.7, alpha=0.5, zorder=3)

    # annotate the published operating points that carry the argument.
    # Text sits in empty regions; the x-axis is inverted, so larger x is further left.
    for label, med, j, tx, ty, ha in [
        ("CAD proj. (published)", 0.199, 7.59, 0.44, 8.78, "left"),
        ("CAA (published)",       0.146, 7.29, 0.44, 8.32, "left"),
        ("Angular (published)",  -0.222, 6.06, -0.30, 7.25, "center"),
        ("SADI (published)",      0.119, 3.19, 0.52, 3.95, "left"),
    ]:
        ax.annotate(label, xy=(med, j), xytext=(tx, ty), ha=ha, fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#444444",
                                    shrinkA=2, shrinkB=3))

    ax.set_xlabel("Medicalization score  (lower = more bias removed)")
    ax.set_ylabel("Judged response quality (0–10)")
    ax.invert_xaxis()
    ax.set_xlim(0.78, -0.40)
    ax.set_ylim(1.0, 9.3)
    ax.grid(alpha=0.22, lw=0.5)

    handles = [Line2D([], [], marker=m, color=c, ls="none", ms=5.5,
                      markeredgecolor="white", markeredgewidth=0.4, label=k)
               for k, (c, m) in STYLE.items()]
    handles += [
        Line2D([], [], marker="o", color="#444444", ls="none", ms=5.5, label="intact output"),
        Line2D([], [], marker="o", color="#444444", ls="none", ms=5.5,
               markerfacecolor="none", label="degenerate output"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, ncol=2,
              handletextpad=0.4, columnspacing=1.0, borderaxespad=0.3)

    fig.savefig(FIG_DIR / "figure-01-quality-debiasing-frontier.pdf")
    fig.savefig(FIG_DIR / "figure-01-quality-debiasing-frontier.png", dpi=300)
    plt.close(fig)
    logger.info("figure 1 written (clean front: %d points)", len(front))


# Canonical configuration family per method, so each dose-response curve varies
# only the steering strength. Angular is excluded: its parameter is an ANGLE
# (0-330 deg), which is circular, so 270 deg is not "stronger" than 150 deg.
DOSE_FAMILIES = [
    ("CAA",            "caa_results_L14.json",       "accesseval/layers/L14"),
    ("FairSteer",      "fairsteer_results_L29.json", "accesseval/layers/L29/thresh_0.5"),
    ("SADI",           "sadi_results.json",          "accesseval/strengths"),
    ("CAD projection", "proj_",                      None),
    ("CAD mean-diff",  "meandiff_",                  None),
    ("CAD probe",      "probe_",                     None),
]

PUBLISHED = [
    ("SADI",            "strength_10.0"),
    ("Angular",         "max_sim_L23/mode_0/angle_150"),
    ("FairSteer",       "L29/thresh_0.5/alpha_2.0"),
    ("CAA",             "L14/alpha_1.0"),
    ("CAD projection",  "alpha1.0"),
]


def figure_support(data: dict, raw: dict) -> None:
    pts = data["points"]
    base_j = data["baseline_judge_mean"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.5))

    # ---- (a) true dose-response against the actual steering strength ----
    from collections import defaultdict
    grouped = defaultdict(list)
    for rec in raw["records"]:
        grouped[(rec["file"], rec["path"])].append(rec["judge_score"])

    for method, fkey, pathkey in DOSE_FAMILIES:
        colour, marker = STYLE[method]
        series = []
        for (fname, path), scores in grouped.items():
            if pathkey is None:
                if not fname.startswith(fkey):
                    continue
                strength = float(fname.split("alpha")[1].split("_")[0])
            else:
                if fname != fkey or path.rsplit("/", 1)[0] != pathkey:
                    continue
                strength = float(path.rsplit("_", 1)[1])
            if strength <= 0:          # no-steer point is the baseline line
                continue
            series.append((strength, float(np.mean(scores))))
        series.sort()
        ax1.plot([x for x, _ in series], [y for _, y in series],
                 color=colour, marker=marker, ms=3.8, lw=1.3, alpha=0.9, label=method)

    ax1.axhline(base_j, color="#666666", lw=0.9, ls=":")
    ax1.text(0.11, base_j + 0.18, "unsteered baseline", fontsize=6.8, color="#444444")
    ax1.set_xscale("log")
    ax1.set_xlabel("Steering strength (method's own units, log scale)")
    ax1.set_ylabel("Judged quality (0\u201310)")
    ax1.set_title("(a) Every method collapses past a threshold", loc="left", fontsize=9)
    ax1.set_ylim(0, 9.4)
    ax1.grid(alpha=0.22, lw=0.5)
    ax1.legend(frameon=False, fontsize=6.8, loc="lower left", ncol=2,
               handletextpad=0.3, columnspacing=0.8)

    # ---- (b) published operating points: quality cost, with 95% CI ----
    index = {(p["method"], p["config"]): p for p in pts}
    rows = [index[k] for k in PUBLISHED if k in index]
    ypos = np.arange(len(rows))
    drops = [p["judge_mean"] - base_j for p in rows]
    err_lo = [p["judge_mean"] - p["judge_ci_lo"] for p in rows]
    err_hi = [p["judge_ci_hi"] - p["judge_mean"] for p in rows]
    colours = [STYLE[p["method"]][0] for p in rows]

    ax2.barh(ypos, drops, color=colours, alpha=0.85, height=0.62)
    ax2.errorbar(drops, ypos, xerr=[err_lo, err_hi], fmt="none",
                 ecolor="#333333", elinewidth=0.9, capsize=2.5)
    ax2.axvline(0, color="#444444", lw=0.9)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([p["method"] for p in rows], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Change in judged quality vs baseline", fontsize=9)
    ax2.set_title("(b) Cost at each method's published setting", loc="left", fontsize=9)
    ax2.grid(alpha=0.22, lw=0.5, axis="x")
    for y, d, e in zip(ypos, drops, err_lo):
        ax2.text(d - e - 0.16, y, f"{d:+.2f}", va="center", ha="right",
                 fontsize=7.2, color="#222222")
    ax2.set_xlim(min(drops) - 1.15, 0.35)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure-02-collapse-and-cost.pdf")
    fig.savefig(FIG_DIR / "figure-02-collapse-and-cost.png", dpi=300)
    plt.close(fig)
    logger.info("figure 2 written (dose families: %d, published points: %d)",
                len(DOSE_FAMILIES), len(rows))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load()
    raw = json.loads((Path(__file__).parent / "results" / "judge_scores.json").read_text())
    figure_main(data)
    figure_support(data, raw)
    logger.info("figures in %s", FIG_DIR)


if __name__ == "__main__":
    main()
