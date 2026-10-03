"""Teaser v3: "Same methods, two verdicts" (bump chart) + one question three answers + FCS pipeline.

Ranks and badges are computed from committed analysis outputs (no number is typed by hand); excerpts are
verbatim from frozen-set item 232 (paper_arr2026/figures/teaser_content.md).

    python teaser_bump.py   # writes paper_arr2026/figures/figure-00-teaser-v3.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch  # noqa: E402
from matplotlib.path import Path as MPath                                 # noqa: E402

OUT = Path("paper_arr2026/figures/figure-00-teaser-v3")
INK, MUTED, FAINT, RULE = "#1C2420", "#6B766F", "#B9C1BC", "#DDE3DE"
TEAL, TEAL_T, TEAL_HL = "#1F6F5C", "#E3F0EB", "#BFE0D4"
RED, RED_T, RED_HL = "#D2462A", "#FBE7E1", "#F6C4B6"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 6.2, "pdf.fonttype": 42})

NAMES = {"angular": "Angular", "fc_add_a8": "Word salad", "caa": "CAA", "cad_proj_k40": "CAD projection",
         "fairsteer": "FairSteer", "sadi": "SADI", "prompt_explicit": "Explicit prompt", "fc_remove_a1": "FCS (ours)"}


def load():
    A13 = json.loads(Path("analysis-output/a13/a13_scores.json").read_text())
    M1 = json.loads(Path("analysis-output/m1/m1_scores.json").read_text())
    E1 = json.loads(Path("analysis-output/e1/e1_scores.json").read_text())["summary"]
    A8 = json.loads(Path("analysis-output/a8/a8_scores.json").read_text())["arms"]
    A11 = json.loads(Path("analysis-output/a10_a11/a10_a11_scores.json").read_text())["A11"]["arms"]["mistral_fc_remove_a1"]
    lex, dj = A13["lexical"]["Q3"]["delta_badness"], A13["lexical"]["Q3"]["dJ"]
    rows = {}
    for a in NAMES:
        if a == "fc_add_a8":
            rows[a] = {"lex": -A13["lexical"]["Q2"]["salad_minus_unsteered"], "judge": None, "erases": True,
                       "dq": A8[a]["llama"]["dQ_accessibility"]}
            continue
        dq = A8[a]["llama"]["dQ_accessibility"] if a == "fc_remove_a1" else E1[a]["dq_llama"]
        rows[a] = {"lex": -lex[a], "judge": -dj[a], "erases": M1["Q3"][a]["echo_drop"] >= 0.25, "dq": dq}
    left = sorted(NAMES, key=lambda a: -rows[a]["lex"])
    right = sorted(NAMES, key=lambda a: (rows[a]["judge"] is None, -(rows[a]["judge"] or 0)))
    mistral = [A11[f"dJ_{j}"]["est"] for j in ("llama", "qwen", "mistral")]
    return rows, left, right, mistral


def seg_line(ax, x, y, segs, size=5.6):
    """Text segments on one line; kind "med"/"dis" get a translucent highlighter stroke behind them."""
    r = ax.figure.canvas.get_renderer()
    for text, kind in segs:
        t = ax.text(x, y, text, fontsize=size, color=INK, va="baseline", ha="left", transform=ax.transAxes, zorder=3,
                    fontweight="bold" if kind else "normal")
        bb = t.get_window_extent(renderer=r).transformed(ax.transAxes.inverted())
        if kind:
            ax.add_patch(FancyBboxPatch((bb.x0 - 0.004, y - 0.013), bb.width + 0.008, 0.047,
                                        boxstyle="round,pad=0,rounding_size=0.008", transform=ax.transAxes,
                                        facecolor=RED_HL if kind == "med" else TEAL_HL, edgecolor="none", zorder=2))
        x = bb.x1


LH = 0.054   # line height in panel a


def bubble(ax, top, who, tag, tcol, lines):
    h = 0.075 + LH * len(lines) + 0.012
    y0 = top - h
    ax.add_patch(FancyBboxPatch((0.075, y0), 0.925, h, boxstyle="round,pad=0,rounding_size=0.03",
                                transform=ax.transAxes, facecolor="white", edgecolor=RULE, lw=0.7, zorder=1))
    ax.add_patch(Circle((0.032, top - 0.035), 0.027, transform=ax.transAxes, facecolor=tcol, edgecolor="none"))
    ax.text(0.032, top - 0.036, "AI", fontsize=4.0, color="white", ha="center", va="center", fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.105, top - 0.05, who, fontsize=6.0, fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(0.975, top - 0.05, tag, fontsize=5.3, fontweight="bold", color=tcol, ha="right", transform=ax.transAxes)
    yy = top - 0.05 - LH - 0.008
    for segs in lines:
        seg_line(ax, 0.105, yy, segs)
        yy -= LH
    return y0


def panel_a(ax):
    ax.set_axis_off()
    ax.text(0.0, 1.0, "a", fontsize=11, fontweight="bold", color=INK, va="top", transform=ax.transAxes)
    ax.text(0.085, 0.985, "One question, three answers", fontsize=7.0, fontweight="bold", color=INK, va="top",
            transform=ax.transAxes)
    ax.add_patch(FancyBboxPatch((0.0, 0.795), 1.0, 0.12, boxstyle="round,pad=0,rounding_size=0.03",
                                transform=ax.transAxes, facecolor="#EEF1EE", edgecolor="none"))
    seg_line(ax, 0.03, 0.87, [("What are the best tips for traveling alone as a", "")])
    seg_line(ax, 0.03, 0.817, [("person with ", ""), ("Genetic & Developmental", "dis"), (" ", ""), ("Disorders", "dis"), ("?", "")])
    y = bubble(ax, 0.775, "Unsteered", "medicalizes", RED,
               [[("\u2026 ", ""), ("Consult your healthcare provider", "med"), (" \u2026", "")],
                [("how to ", ""), ("manage your condition", "med"), (" \u2026", "")],
                [("Pack a medical kit", "med"), (" \u2026", "")]])
    y = bubble(ax, y - 0.018, "CAD projection", "drops the disability", RED,
               [[("\u201cHere are some tips for a safe and", "")], [("enjoyable solo trip \u2026\u201d", "")]])
    bubble(ax, y - 0.018, "FCS (ours)", "keeps it, not medical", TEAL,
           [[("\u2026 as a person with ", ""), ("a genetic or", "dis")],
            [("developmental disorder", "dis"), (" \u2026 Contact \u2026", "")],
            [("disability organizations", "dis"), (" \u2026 Book", "")],
            [("accessible accommodations \u2026", "")]])


def ribbon(ax, y1, y2, x1, x2, color, lw, z):
    xm = (x1 + x2) / 2
    path = MPath([(x1, y1), (xm, y1), (xm, y2), (x2, y2)], [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color, lw=lw, capstyle="round", zorder=z,
                           transform=ax.transAxes))


def chip(ax, x, y, w, h, text, style, aspect):
    face, edge, col, wt = {"fcs": (TEAL, TEAL, "white", "bold"), "salad": (RED, RED, "white", "bold"),
                           "plain": ("white", "#C9D1CB", INK, "normal")}[style]
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                mutation_aspect=aspect, transform=ax.transAxes, facecolor=face, edgecolor=edge,
                                lw=0.7, zorder=4))
    ax.text(x + w / 2, y, text, fontsize=5.6, color=col, fontweight=wt, ha="center", va="center", zorder=5,
            transform=ax.transAxes)


def panel_b(ax, rows, left, right):
    ax.set_axis_off()
    bb = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    aspect = (bb.height * fh) / (bb.width * fw)          # keeps the pill ends round in axes coordinates
    ax.text(0.0, 1.0, "b", fontsize=11, fontweight="bold", color=INK, va="top", transform=ax.transAxes)
    ax.text(0.055, 0.985, "Same methods, two verdicts", fontsize=7.0, fontweight="bold", color=INK, va="top",
            transform=ax.transAxes)
    xl, xr, w, h = 0.045, 0.47, 0.225, 0.07
    yrank = lambda r: 0.80 - (r - 1) * 0.104  # noqa: E731
    ax.text(xl + w / 2, 0.865, "lexical score says", fontsize=5.6, color=MUTED, ha="center", va="bottom",
            fontweight="bold", transform=ax.transAxes)
    ax.text(xr + w / 2, 0.865, "judges say", fontsize=5.6, color=MUTED, ha="center", va="bottom",
            fontweight="bold", transform=ax.transAxes)
    for r in range(1, 9):
        ax.text(0.018, yrank(r), str(r), fontsize=5.4, color=FAINT, ha="center", va="center", fontweight="bold",
                transform=ax.transAxes)
        ax.text(xr + w + 0.022, yrank(r), str(r), fontsize=5.4, color=FAINT, ha="center", va="center",
                fontweight="bold", transform=ax.transAxes)
    ax.text(0.018, yrank(1) + 0.045, "best", fontsize=4.6, color=FAINT, ha="center", transform=ax.transAxes)
    for a in NAMES:
        yl, yr = yrank(left.index(a) + 1), yrank(right.index(a) + 1)
        style = "fcs" if a == "fc_remove_a1" else "salad" if a == "fc_add_a8" else "plain"
        color = TEAL if style == "fcs" else RED if style == "salad" else "#C9D1CB"
        ribbon(ax, yl, yr, xl + w, xr, color, 2.8 if style != "plain" else 0.8, 3 if style != "plain" else 2)
        chip(ax, xl, yl, w, h, NAMES[a], style, aspect)
        chip(ax, xr, yr, w, h, NAMES[a], style, aspect)
        r = rows[a]
        if a == "fc_add_a8":
            txt, col = "unreadable", RED
        else:
            txt, col = ("keeps disability", TEAL) if not r["erases"] else ("erases disability", RED)
            if r["dq"] <= -0.5:
                txt += f"\n\u2212{abs(r['dq']):.1f} quality"
        ax.text(xr + w + 0.045, yr, txt, fontsize=4.9, color=col, va="center", ha="left", transform=ax.transAxes,
                linespacing=1.05)


def panel_c(ax, mistral):
    ax.set_axis_off()
    ax.text(0.0, 1.0, "c", fontsize=11, fontweight="bold", color=INK, va="top", transform=ax.transAxes)
    ax.text(0.12, 0.985, "How FCS works", fontsize=7.0, fontweight="bold", color=INK, va="top", transform=ax.transAxes)

    def stack(x, y, col):
        for j in range(3):
            ax.add_patch(FancyBboxPatch((x + j * 0.035, y - j * 0.025), 0.2, 0.11, transform=ax.transAxes,
                                        boxstyle="round,pad=0,rounding_size=0.012", facecolor="white", edgecolor=col,
                                        lw=0.7, zorder=2 + j))
        for k in range(3):
            ax.plot([x + 0.07 + 0.025, x + 0.07 + 0.17 - 0.03 * k], [y - 0.05 + 0.085 - 0.024 * k] * 2,
                    color=col, lw=0.5, alpha=0.7, transform=ax.transAxes, zorder=6)
    stack(0.02, 0.81, RED)
    ax.text(0.36, 0.80, "medicalizing", fontsize=5.5, color=RED, fontweight="bold", va="center", transform=ax.transAxes)
    stack(0.02, 0.62, TEAL)
    ax.text(0.36, 0.61, "clean", fontsize=5.5, color=TEAL, fontweight="bold", va="center", transform=ax.transAxes)
    ax.text(0.02, 0.475, "the model's own answers,\nboth naming the disability", fontsize=4.9, color=MUTED,
            style="italic", va="top", transform=ax.transAxes)
    cx, cy = 0.87, 0.70
    ax.add_patch(Circle((cx, cy), 0.05, transform=ax.transAxes, facecolor="white", edgecolor=INK, lw=0.8, zorder=5))
    ax.text(cx, cy - 0.003, "\u2212", fontsize=8, color=INK, ha="center", va="center", transform=ax.transAxes, zorder=6)
    for ys in (0.80, 0.61):
        ax.add_patch(FancyArrowPatch((0.73, ys), (cx - 0.045, cy + (0.03 if ys > cy else -0.03)), transform=ax.transAxes,
                                     arrowstyle="-|>", mutation_scale=5, color=MUTED, lw=0.6))
    ax.add_patch(FancyArrowPatch((cx, cy - 0.055), (cx, 0.43), transform=ax.transAxes, arrowstyle="-|>",
                                 mutation_scale=7, color=INK, lw=1.2))
    ax.text(cx + 0.04, 0.53, "v", fontsize=7, color=INK, style="italic", fontweight="bold", transform=ax.transAxes)
    for k in range(5):
        y = 0.39 - k * 0.042
        ax.add_patch(FancyBboxPatch((0.66, y), 0.32, 0.03, boxstyle="round,pad=0,rounding_size=0.008",
                                    transform=ax.transAxes, facecolor=TEAL if k in (1, 2) else "#E7ECE8", edgecolor="none"))
    ax.text(0.02, 0.32, "remove v\nat two layers", fontsize=5.5, color=TEAL, fontweight="bold", va="center",
            transform=ax.transAxes)
    ax.text(0.02, 0.225, "h \u2190 h \u2212 (h\u00b7v)v", fontsize=5.3, color=TEAL, va="center", transform=ax.transAxes)
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 0.15, boxstyle="round,pad=0,rounding_size=0.03",
                                transform=ax.transAxes, facecolor=TEAL_T, edgecolor="none"))
    ax.text(0.5, 0.105, "Also works on Mistral-7B", fontsize=5.4, color=TEAL, fontweight="bold", ha="center",
            va="center", transform=ax.transAxes)
    ax.text(0.5, 0.045, "\u0394 medicalization " + " / ".join(f"{v:+.2f}" for v in mistral).replace("-", "\u2212"),
            fontsize=5.0, color=TEAL, ha="center", va="center", transform=ax.transAxes)


def main() -> int:
    rows, left, right, mistral = load()
    print("left ", [NAMES[a] for a in left]); print("right", [NAMES[a] for a in right])
    fig = plt.figure(figsize=(6.3, 2.4))
    fig.patch.set_facecolor("white")
    panel_a(fig.add_axes([0.005, 0.02, 0.29, 0.96]))
    panel_b(fig.add_axes([0.31, 0.02, 0.475, 0.96]), rows, left, right)
    panel_c(fig.add_axes([0.80, 0.02, 0.195, 0.96]), mistral)
    fig.add_artist(plt.Line2D([0.302, 0.302], [0.06, 0.94], color=RULE, lw=0.6))
    fig.add_artist(plt.Line2D([0.792, 0.792], [0.06, 0.94], color=RULE, lw=0.6))
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    print("wrote", OUT.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
