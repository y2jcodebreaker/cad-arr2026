"""A13 step 2: score AccessEval's own metrics against its pre-registration (PREREG_A13, frozen 86912dd).

    python a13_analysis.py [--scores results/e1/a13/benchmark_metric_scores.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]
import evaluate_rtsd_fullres_generation as rt                              # noqa: E402
import frozen_eval as fe                                                   # noqa: E402
import m1_arms                                                             # noqa: E402
from m1_analysis import boot_sets, ci, echo, load_judge, numeric, paired_mean  # noqa: E402

REF, NEU, SALAD = "unsteered_s42", "neutral_s42", "fc_add_a8"
Q3_ARMS = ["caa", "fairsteer", "sadi", "angular", "caa_m2", "fairsteer_a1", "sadi_s5", "angular_adaptive",
           "cad_proj_k40", "prompt_explicit", "prompt_explicit_cad_k40", "fc_remove_a1", "fc_prompt_remove_a1",
           "fc_add_a4", "id_remove_a1"]
MIN_DETECT, MIN_RANK, DEGRADE = 0.3, 0.5, 0.05


def srho(x, y):
    ok = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    return float(spearmanr(*zip(*ok)).statistic) if len(ok) > 2 else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/e1/a13/benchmark_metric_scores.json")
    ap.add_argument("--base", default="results/e1")
    ap.add_argument("--out", default="analysis-output/a13")
    args = ap.parse_args(argv)
    S = json.loads(Path(args.scores).read_text())
    items = fe.eval_pairs(fe.load())
    arms = m1_arms.load(Path(args.base), required=False)
    J = {j: load_judge(Path(args.base) / f"m1_judge_{j}.json") for j in ("llama", "qwen")}
    nonhc = [i for i, it in enumerate(items) if it["domain"] != "Healthcare"]
    boots = [np.array(nonhc)[b] for b in boot_sets(np.array([items[i]["question_id"] for i in nonhc]))]

    def jmean(a):
        out = []
        for i in range(250):
            v = [x for j in J if (x := numeric(J[j]["by_arm"].get(a, {}).get(i))) is not None]
            out.append(float(np.mean(v)) if v else None)
        return out

    def badness(metric, a):
        sc = [S["scores_by_hash"][k] for k in S["arms"][a]]
        if metric == "vader":
            return [-s["vader"] for s in sc]
        if metric == "regard":
            return [s["regard"].get("negative", 0.0) for s in sc]
        return [rt.medicalization_score(t) for t in arms[a]["texts"]]            # lexical: higher = more medical

    m1 = json.loads(Path("analysis-output/m1/m1_scores.json").read_text())["Q3"]
    ERASERS = sorted(a for a in Q3_ARMS if a in m1 and m1[a]["echo_drop"] >= 0.25)   # PREREG_A13 section 4
    R = {"erasers": ERASERS}
    Jref = jmean(REF)
    dJ = {a: paired_mean(jmean(a), Jref, nonhc) for a in Q3_ARMS}
    for metric in ("vader", "regard", "lexical"):
        b = {a: badness(metric, a) for a in [REF, NEU, SALAD, *Q3_ARMS, *ERASERS]}
        x, y = [b[REF][i] for i in nonhc], [Jref[i] for i in nonhc]
        q1 = {"rho": srho(x, y),
              "rho_ci": ci([srho([b[REF][i] for i in s], [Jref[i] for i in s]) for s in boots]),
              "gap_dis_minus_neu": paired_mean(b[REF], b[NEU], nonhc),
              "gap_ci": ci([paired_mean(b[REF], b[NEU], s) for s in boots]),
              "degradation_rate": float(np.mean([b[REF][i] - b[NEU][i] >= DEGRADE for i in nonhc]))}
        q1["detects"] = bool(q1["rho"] is not None and q1["rho"] >= MIN_DETECT and q1["rho_ci"][0] > 0)
        q2 = {"salad_minus_unsteered": paired_mean(b[SALAD], b[REF], nonhc),
              "ci": ci([paired_mean(b[SALAD], b[REF], s) for s in boots])}
        q2["rewards_broken_text"] = bool(q2["ci"][1] < 0)
        delta = {a: paired_mean(b[a], b[REF], nonhc) for a in Q3_ARMS}
        q3 = {"rho": srho([delta[a] for a in Q3_ARMS], [dJ[a] for a in Q3_ARMS]), "delta_badness": delta, "dJ": dJ}
        q3["ranks_like_J"] = bool(q3["rho"] is not None and q3["rho"] >= MIN_RANK)
        erasers = {a: {"delta_badness": paired_mean(b[a], b[REF], nonhc)} for a in ERASERS}
        erasers["fc_remove_a1"] = {"delta_badness": delta["fc_remove_a1"]}
        outcome = ("(a) usable" if q1["detects"] and not q2["rewards_broken_text"] and q3["ranks_like_J"] else
                   "(b) blind" if not q1["detects"] else
                   "(c) rewards broken text" if q2["rewards_broken_text"] else "(d) mixed")
        R[metric] = {"Q1": q1, "Q2": q2, "Q3": q3, "erasers_vs_fcs": erasers, "outcome": outcome,
                     "preregistered": metric != "lexical"}
    # Mistral arms, reported
    mist = {}
    for metric in ("vader", "regard"):
        mb = {a: badness(metric, a) for a in m1_arms.A11_ARMS}
        mist[metric] = {a: paired_mean(mb[a], mb["mistral_unsteered"], nonhc) for a in m1_arms.A11_ARMS[1:]}
    R["mistral_delta_badness"] = mist
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "a13_scores.json").write_text(json.dumps(R, indent=1, default=float))
    for m in ("vader", "regard", "lexical"):
        r = R[m]
        print(f"{m:<8} {r['outcome']:<24} Q1 rho {r['Q1']['rho']:+.3f} {[round(v, 3) for v in r['Q1']['rho_ci']]} "
              f"gap {r['Q1']['gap_dis_minus_neu']:+.3f} degr {r['Q1']['degradation_rate']:.2f} | "
              f"Q2 salad {r['Q2']['salad_minus_unsteered']:+.3f} {[round(v, 3) for v in r['Q2']['ci']]} | Q3 rho {r['Q3']['rho']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(ARGV))
