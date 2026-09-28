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
FIG_DIR = Path(__file__).parent / "paper_arr2026" / "figures"

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

# Drawn at PRINTED size (ACL column 3.03 in, text width 6.3 in), so these are real point sizes.
plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "legend.fontsize": 6.3, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
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


PUBLISHED_PTS = [("SADI", "strength_10.0"), ("Angular", "max_sim_L23/mode_0/angle_150"),
                 ("FairSteer", "L29/thresh_0.5/alpha_2.0"), ("CAA", "L14/alpha_1.0"),
                 ("CAD projection", "alpha1.0")]


def figure_main(data: dict) -> None:
    """Single-column frontier. No per-point error bars (median half-width 0.18, listed in the
    appendix sweep table); published settings are ringed, and only the two far-off ones labelled."""
    pts = data["points"]
    base_j = data["baseline_judge_mean"]
    base_j_sd = data["baseline_judge_sd"]

    fig, ax = plt.subplots(figsize=(3.03, 2.3))

    ax.axvspan(-0.40, 0.0, color="#D55E00", alpha=0.06, zorder=0)
    ax.text(-0.21, 1.25, "past neutral", ha="center", va="bottom", fontsize=6.2,
            color="#8a3800", style="italic")
    ax.axvline(0.0, color="#555555", lw=0.7, ls="--", zorder=1)
    ax.axhspan(base_j - 2 * base_j_sd, base_j + 2 * base_j_sd, color="#999999", alpha=0.2, zorder=0)
    ax.text(0.77, base_j + 0.22, "unsteered", fontsize=6.2, color="#444444", ha="left")

    front = clean_pareto(pts)
    ax.plot([p["med"] for p in front], [p["judge_mean"] for p in front],
            color="#0072B2", lw=1.1, alpha=0.55, zorder=2)

    for method, (colour, marker) in STYLE.items():
        sub = [p for p in pts if p["method"] == method]
        for degen, kw in [(False, dict(facecolors=colour, edgecolors="white", linewidths=0.3)),
                          (True, dict(facecolors="none", edgecolors=colour, linewidths=0.8))]:
            grp = [p for p in sub if p["degenerate"] is degen]
            if grp:
                ax.scatter([p["med"] for p in grp], [p["judge_mean"] for p in grp],
                           marker=marker, s=16, zorder=4, **kw)

    index = {(p["method"], p["config"]): p for p in pts}
    pub = [index[k] for k in PUBLISHED_PTS if k in index]
    ax.scatter([p["med"] for p in pub], [p["judge_mean"] for p in pub], s=70, facecolors="none",
               edgecolors="black", linewidths=0.8, zorder=5)
    for p, (dx, dy) in zip(pub, [(0.10, 0.9), (0.02, 1.35)]):   # SADI, Angular only
        ax.annotate(f"{p['method']} (published)", xy=(p["med"], p["judge_mean"]),
                    xytext=(p["med"] + dx, p["judge_mean"] + dy), fontsize=6.2, ha="center",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#444444", shrinkB=4))

    ax.set_xlabel("Medicalization (lower = more bias removed)")
    ax.set_ylabel("Judged quality (0\u201310)")
    ax.set_xlim(0.78, -0.40)
    ax.set_ylim(1.0, 9.0)
    ax.grid(alpha=0.2, lw=0.4)

    handles = [Line2D([], [], marker=m, color=c, ls="none", ms=3.6, markeredgecolor="white",
                      markeredgewidth=0.3, label=k) for k, (c, m) in STYLE.items()]
    handles += [Line2D([], [], marker="o", color="#444444", ls="none", ms=3.6,
                       markerfacecolor="none", label="degenerate"),
                Line2D([], [], marker="o", color="black", ls="none", ms=6, markerfacecolor="none",
                       label="published setting")]
    # two compact rows above the axes: inside, it covered the points near neutral
    short = {"CAD projection": "CAD proj.", "CAD mean-diff": "CAD mean-diff.",
             "published setting": "published"}
    for h in handles:
        h.set_label(short.get(h.get_label(), h.get_label()))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5,
              frameon=False, fontsize=5.9, handletextpad=0.1, columnspacing=0.55,
              borderaxespad=0.2, labelspacing=0.25)

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.5))

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
    ax1.text(22, base_j + 0.25, "unsteered", fontsize=6.3, color="#444444")
    ax1.set_xscale("log")
    ax1.set_xlabel("Steering strength (method's own units, log scale)")
    ax1.set_ylabel("Judged quality (0\u201310)")
    ax1.set_title("(a) Every method collapses past a threshold", loc="left")
    ax1.set_ylim(0, 9.4)
    ax1.grid(alpha=0.22, lw=0.5)
    ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3,
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
    ax2.set_yticklabels([p["method"] for p in rows])
    ax2.invert_yaxis()
    ax2.set_xlabel("Change in judged quality vs. unsteered")
    ax2.set_title("(b) Cost at each method's published setting", loc="left")
    ax2.grid(alpha=0.22, lw=0.5, axis="x")
    for y, d, e in zip(ypos, drops, err_lo):
        ax2.text(d - e - 0.16, y, f"{d:+.2f}", va="center", ha="right",
                 fontsize=6.5, color="#222222")
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
