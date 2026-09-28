"""Score P12 (PREREG_E1 amendment A7): each baseline at its authors' setting vs its E1 setting.

Reads analysis-output/e1/e1_scores.json (text flags, accessibility-quality change) and
analysis-output/m1/m1_scores.json (M1 U-rates). Run e1_analysis.py and m1_analysis.py first.

  (a) intact at the authors' setting: no text flag, M1 U-rate <= 10% on both judges, and
      accessibility-quality loss smaller than at the E1 setting on both judges
  (b) broken at the authors' setting too: a text flag, or M1 U-rate > 10% on either judge
  (c) neither

    python a7_p12.py
"""
from __future__ import annotations

import json
from pathlib import Path

PAIRS = {"Angular": ("angular_adaptive", "angular"), "CAA": ("caa_m2", "caa"),
         "FairSteer": ("fairsteer_a1", "fairsteer"), "SADI": ("sadi_s5", "sadi")}


def main() -> int:
    E = json.loads(Path("analysis-output/e1/e1_scores.json").read_text())["summary"]
    M = json.loads(Path("analysis-output/m1/m1_scores.json").read_text())
    out = {}
    for method, (a7, e1) in PAIRS.items():
        u = {j: M["Q2"][a7][j]["u_rate"] for j in ("llama", "qwen")}
        flags = E[a7]["degenerate"]
        less_loss = all(E[a7][f"dq_{j}"] > E[e1][f"dq_{j}"] for j in ("llama", "qwen"))
        broken = bool(flags) or any(v > 0.10 for v in u.values())
        intact = not flags and all(v <= 0.10 for v in u.values()) and less_loss
        out[method] = {
            "authors_arm": a7, "e1_arm": e1, "text_flags": flags, "m1_u_rate": u,
            "dq_authors": {j: E[a7][f"dq_{j}"] for j in ("llama", "qwen")},
            "dq_e1": {j: E[e1][f"dq_{j}"] for j in ("llama", "qwen")},
            "d_authors": E[a7]["d"], "d_e1": E[e1]["d"],
            "dJ_authors": {j: M["Q2"][a7][j]["dJ"] for j in ("llama", "qwen")},
            "dJ_authors_ci": {j: M["Q2"][a7][j]["ci"] for j in ("llama", "qwen")},
            "m1_call_authors": M["Q2"][a7]["call"], "echo_drop_authors": M["Q3"][a7]["echo_drop"],
            "outcome": "(a) intact at the authors' setting" if intact else
                       "(b) broken at the authors' setting too" if broken else "(c) neither"}
    Path("analysis-output/a7").mkdir(parents=True, exist_ok=True)
    Path("analysis-output/a7/p12.json").write_text(json.dumps(out, indent=1, default=float))
    for m, r in out.items():
        print(f"{m:<10} {r['outcome']:<40} flags={r['text_flags'] or '-'} U={r['m1_u_rate']} "
              f"dQ authors {r['dq_authors']} vs E1 {r['dq_e1']} | lexical d {r['d_authors']:.2f} (E1 {r['d_e1']:.2f}) "
              f"| M1 {r['m1_call_authors']} dJ {r['dJ_authors']} echo {r['echo_drop_authors']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
