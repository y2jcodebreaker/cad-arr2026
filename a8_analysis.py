"""Score A8 against its pre-registration (claims/PREREG_A8_framing_contrast.md, section 6).

CPU only. Success for an arm needs ALL of: dJ < 0 with a Bonferroni-adjusted (6 arms) question-level
interval excluding 0 on both M1 judges; echo drop < 0.25; accessibility-quality change >= -1.0 on
both accessibility judges; M1 U-rate <= 10% on both judges.

    python a8_analysis.py                  # results/e1 after the A8 outputs are committed
    python a8_analysis.py --base <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.argv = sys.argv[:1] + sys.argv[1:]
import frozen_eval as fe            # noqa: E402
import judge_sweep_outputs as js    # noqa: E402
import m1_arms                      # noqa: E402
from m1_analysis import boot_sets, echo, load_judge, numeric, paired_mean  # noqa: E402

A8 = list(m1_arms.A8_ARMS)
REFS = ["prompt_explicit", "caa", "cad_proj_k40", "prompt_explicit_cad_k40"]
N_ARMS = len(A8)
LO, HI = 100 * 0.05 / N_ARMS / 2, 100 * (1 - 0.05 / N_ARMS / 2)     # Bonferroni: 0.417, 99.583
ECHO_MAX, DQ_MIN, U_MAX = 0.25, -1.0, 0.10


def pct(x, lo=2.5, hi=97.5):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))] if x else [None, None]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/e1")
    ap.add_argument("--out", default="analysis-output/a8")
    args = ap.parse_args(argv)
    base = Path(args.base)
    items = fe.eval_pairs(fe.load())
    arms = m1_arms.load(base, required=False)
    missing = [a for a in A8 + REFS + ["unsteered_s42"] if a not in arms]
    if missing:
        raise SystemExit(f"missing arms: {missing}")
    J = {j: load_judge(base / f"m1_judge_{j}.json") for j in ("llama", "qwen")}
    for j in J:
        unjudged = [a for a in A8 if a not in J[j]["by_arm"]]
        if unjudged:
            raise SystemExit(f"{j} M1 judge has not scored {unjudged}: resume judge_medicalization.py first")
    acc = {j: json.loads((base / f).read_text())["scores_by_hash"]
           for j, f in (("llama", "judge_llama.json"), ("qwen", "judge_second.json"))}

    nonhc = [i for i, it in enumerate(items) if it["domain"] != "Healthcare"]
    qids = np.array([items[i]["question_id"] for i in nonhc])
    boots = [np.array(nonhc)[b] for b in boot_sets(qids)]
    num = {j: {a: [numeric(J[j]["by_arm"][a].get(i)) for i in range(250)] for a in A8 + REFS + ["unsteered_s42"]}
           for j in J}
    ref = "unsteered_s42"
    e_ref = np.array([echo(t, it) for t, it in zip(arms[ref]["texts"], items)], float)

    def quality(j, a):
        s = [acc[j].get(js.text_key(t)) for t in arms[a]["texts"]]
        return np.array([np.nan if v is None else v for v in s], float)

    q_ref = {j: quality(j, ref) for j in J}
    rows = {}
    for a in A8 + REFS:
        r = {}
        e_a = np.array([echo(t, it) for t, it in zip(arms[a]["texts"], items)], float)
        idx = np.array(nonhc)
        r["echo_drop"] = float(1 - e_a[idx].mean() / e_ref[idx].mean())
        for j in J:
            dist = [paired_mean(num[j][a], num[j][ref], b) for b in boots]
            q = quality(j, a)
            r[j] = {"dJ": paired_mean(num[j][a], num[j][ref], nonhc),
                    "ci_bonferroni": pct(dist, LO, HI), "ci95": pct(dist),
                    "u_rate": float(np.mean([J[j]["by_arm"][a].get(i) == "U" for i in nonhc])),
                    "dQ_accessibility": float(np.nanmean(q - q_ref[j])),
                    "unscored_accessibility": int(np.isnan(q).sum())}
        reduces = all(r[j]["ci_bonferroni"][1] is not None and r[j]["ci_bonferroni"][1] < 0 for j in J)
        r["reduces_both_judges"] = bool(reduces)
        r["success"] = bool(reduces and r["echo_drop"] < ECHO_MAX
                            and all(r[j]["dQ_accessibility"] >= DQ_MIN and r[j]["u_rate"] <= U_MAX for j in J))
        rows[a] = r

    succ = [a for a in A8 if rows[a]["success"]]
    erase = [a for a in A8 if rows[a]["reduces_both_judges"] and rows[a]["echo_drop"] >= ECHO_MAX]
    outcome = ("(a) framing-contrast steering reduces medicalization without erasing the disability" if succ else
               "(b) reduces medicalization but erases the disability" if erase else
               "(c) no arm reduces medicalization on both judges")

    # secondary comparisons (95% intervals)
    def pair_diff(a, b):
        out = {}
        for j in J:
            out[j] = {"diff": paired_mean(num[j][a], num[j][b], nonhc),
                      "ci95": pct([paired_mean(num[j][a], num[j][b], s) for s in boots])}
        return out
    secondary = {}
    if succ:
        best = min(succ, key=lambda a: np.mean([rows[a][j]["dJ"] for j in J]))
        secondary["best_arm"] = best
        secondary["best_vs_prompt_explicit"] = pair_diff(best, "prompt_explicit")
        secondary["best_vs_caa"] = pair_diff(best, "caa")
    secondary["selected_vs_L21L25"] = pair_diff("fc_remove_a1", "fc_remove_a1_L21L25")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"bonferroni_interval_percentiles": [LO, HI], "arms": rows, "succeeding_arms": succ,
              "erasing_arms": erase, "outcome": outcome, "secondary": secondary}
    (out / "a8_scores.json").write_text(json.dumps(result, indent=1, default=float))
    print(f"{'arm':<24}{'dJ llama [Bonf CI]':>26}{'dJ qwen [Bonf CI]':>26}{'echo':>7}{'dQ L':>7}{'dQ Q':>7}{'U L/Q':>11}  result")
    for a, r in rows.items():
        f = lambda j: f"{r[j]['dJ']:+.2f} [{r[j]['ci_bonferroni'][0]:+.2f},{r[j]['ci_bonferroni'][1]:+.2f}]"  # noqa: E731
        tag = "SUCCESS" if r["success"] else ("reduces" if r["reduces_both_judges"] else "")
        print(f"{a:<24}{f('llama'):>26}{f('qwen'):>26}{r['echo_drop']:>7.2f}{r['llama']['dQ_accessibility']:>7.2f}"
              f"{r['qwen']['dQ_accessibility']:>7.2f}{r['llama']['u_rate']:>6.2f}/{r['qwen']['u_rate']:.2f}  {tag}")
    print(f"\nOUTCOME: {outcome}\nsecondary: {json.dumps(secondary, default=float)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
