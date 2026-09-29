"""Score A10 (PREREG_A10_metric_gaming.md) and A11 (PREREG_A11_mistral.md), both frozen 7da220b.

CPU only. Lexical d as E1 (all 250 items, E1's question bootstrap); judge changes as M1 (non-Healthcare
items, 95% question-level intervals).

    python a10_a11_analysis.py                  # results/e1 after the outputs are committed
    python a10_a11_analysis.py --base <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ARGV = sys.argv[1:]                                # captured first: importing e1_analysis clears sys.argv
import e1_analysis as e1a                          # noqa: E402  (boot_indices: E1's lexical bootstrap)
import evaluate_rtsd_fullres_generation as rt      # noqa: E402
import frozen_eval as fe                           # noqa: E402
import judge_sweep_outputs as js                   # noqa: E402
import m1_arms                                     # noqa: E402
from m1_analysis import boot_sets, ci, echo, load_judge, numeric, paired_mean  # noqa: E402

JUDGES = ("llama", "qwen", "mistral")
M1J = ("llama", "qwen")
BEST_BASELINE_D = 0.741           # PREREG_A10 section 5: Angular, E1 setting
BAND = 0.10
ECHO_MAX, DQ_MIN, U_MAX = 0.25, -1.0, 0.10


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/e1")
    ap.add_argument("--out", default="analysis-output/a10_a11")
    args = ap.parse_args(argv)
    base = Path(args.base)
    print(f"scoring from {base}")
    items = fe.eval_pairs(fe.load())
    arms = m1_arms.load(base, required=False)
    J = {j: load_judge(base / f"m1_judge_{j}.json") for j in JUDGES}
    acc = {j: json.loads((base / f).read_text())["scores_by_hash"]
           for j, f in (("llama", "judge_llama.json"), ("qwen", "judge_second.json"))}
    nonhc = [i for i, it in enumerate(items) if it["domain"] != "Healthcare"]
    boots = [np.array(nonhc)[b] for b in boot_sets(np.array([items[i]["question_id"] for i in nonhc]))]
    lboots = e1a.boot_indices(np.array([it["question_id"] for it in items]))
    num = {j: {a: [numeric(J[j]["by_arm"][a].get(i)) for i in range(250)] for a in J[j]["by_arm"]} for j in J}
    med = {a: np.array([rt.medicalization_score(t) for t in arms[a]["texts"]]) for a in arms}

    def dJ(j, a, ref):
        return {"est": paired_mean(num[j][a], num[j][ref], nonhc),
                "ci": ci([paired_mean(num[j][a], num[j][ref], b) for b in boots])}

    def lex_d(a, ref):
        return {"est": rt.cohens_d(med[ref], med[a]), "ci": e1a.ci([rt.cohens_d(med[ref][b], med[a][b]) for b in lboots])}

    def echo_drop(a, ref):
        e = lambda x: np.array([echo(t, it) for t, it in zip(arms[x]["texts"], items)], float)[nonhc]  # noqa: E731
        return float(1 - e(a).mean() / e(ref).mean())

    def dq(j, a, ref):
        q = lambda x: np.array([np.nan if (v := acc[j].get(js.text_key(t))) is None else v for t in arms[x]["texts"]])  # noqa: E731
        return float(np.nanmean(q(a) - q(ref)))

    def u_rate(j, a):
        return float(np.mean([J[j]["by_arm"][a].get(i) == "U" for i in nonhc]))

    R: dict = {}
    # ------------------------------------------------------------------ A10
    ref = "unsteered_s42"
    a10 = {}
    for a in m1_arms.A10_ARMS:
        if a not in arms:
            continue
        r = {"lexical_d": lex_d(a, ref), "echo_drop": echo_drop(a, ref),
             **{f"dJ_{j}": dJ(j, a, ref) for j in JUDGES},
             **{f"dQ_{j}": dq(j, a, ref) for j in M1J}, **{f"U_{j}": u_rate(j, a) for j in JUDGES}}
        r["reaches_best_baseline"] = r["lexical_d"]["est"] >= BEST_BASELINE_D
        r["judges_in_band"] = all(r[f"dJ_{j}"]["ci"][0] is not None and -BAND <= r[f"dJ_{j}"]["ci"][0]
                                  and r[f"dJ_{j}"]["ci"][1] <= BAND for j in JUDGES)
        r["games"] = bool(r["reaches_best_baseline"] and r["judges_in_band"])
        a10[a] = r
    a10["outcome"] = ("(a) the target metric can be beaten by bias-blind edits that leave medicalization unchanged"
                      if any(r["games"] for r in a10.values()) else
                      "(b) an arm reaches the best baseline but a judge's dJ leaves the band"
                      if any(r["reaches_best_baseline"] for r in a10.values()) else
                      "(c) no arm reaches the best published lexical d")
    R["A10"] = a10

    # ------------------------------------------------------------------ A11
    a11: dict = {}
    if "mistral_unsteered" in arms and "mistral_neutral" in arms:
        mu, mn = "mistral_unsteered", "mistral_neutral"
        a11["S0"] = {**{f"gap_{j}": dJ(j, mu, mn) for j in JUDGES},
                     "lexical_gap": {"est": paired_mean(list(med[mu]), list(med[mn]), nonhc),
                                     "ci": ci([paired_mean(list(med[mu]), list(med[mn]), b) for b in boots])},
                     "llama_lexical_gap": paired_mean(list(med["unsteered_s42"]), list(med["neutral_s42"]), nonhc),
                     "mean_J_unsteered": {j: float(np.mean([v for i in nonhc if (v := num[j][mu][i]) is not None]))
                                          for j in JUDGES}}
        rep_p = base / "a11/a11_fit_report.json"
        rep = json.loads(rep_p.read_text()) if rep_p.exists() else {}
        a11["gates"] = {k: rep.get(k) for k in ("n_medicalizing", "n_clean", "GA11_1_pass", "best_cv_acc",
                                                "GA11_2_pass", "selected_layers")}
        arms2 = [a for a in m1_arms.A11_ARMS[2:] if a in arms]
        rows = {}
        for a in arms2:
            r = {"lexical_d": lex_d(a, mu), "echo_drop": echo_drop(a, mu),
                 **{f"dJ_{j}": dJ(j, a, mu) for j in JUDGES},
                 **{f"dQ_{j}": dq(j, a, mu) for j in M1J}, **{f"U_{j}": u_rate(j, a) for j in M1J}}
            r["dJ_holds"] = all(r[f"dJ_{j}"]["ci"][1] is not None and r[f"dJ_{j}"]["ci"][1] < 0 for j in M1J)
            r["others"] = {"echo": r["echo_drop"] < ECHO_MAX, "quality": all(r[f"dQ_{j}"] >= DQ_MIN for j in M1J),
                           "u_rate": all(r[f"U_{j}"] <= U_MAX for j in M1J)}
            rows[a] = r
        if "mistral_fc_shuffle_s0" in arms and "mistral_fc_remove_a1" in arms:
            rows["fc_minus_shuffle"] = {j: dJ(j, "mistral_fc_remove_a1", "mistral_fc_shuffle_s0") for j in JUDGES}
        a11["arms"] = rows
        c = rows.get("mistral_fc_remove_a1")
        a11["outcome"] = ("(d) a fit gate failed; S0 is the result" if not (rep.get("GA11_1_pass") and rep.get("GA11_2_pass"))
                          else "not scored: S2 arms missing" if c is None
                          else "(a) framing-contrast steering replicates on Mistral" if c["dJ_holds"] and all(c["others"].values())
                          else "(b) dJ holds; fails: " + ", ".join(k for k, v in c["others"].items() if not v) if c["dJ_holds"]
                          else "(c) dJ does not hold on both M1 judges")
    R["A11"] = a11

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "a10_a11_scores.json").write_text(json.dumps(R, indent=1, default=float))
    f = lambda b: f"{b['est']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]"  # noqa: E731
    for a, r in a10.items():
        if a != "outcome":
            print(f"A10 {a:<12} lexical d {f(r['lexical_d'])}  " + "  ".join(f"dJ {j} {f(r[f'dJ_{j}'])}" for j in JUDGES)
                  + f"  echo {r['echo_drop']:+.2f}  {'GAMES' if r['games'] else ''}")
    print("A10:", a10["outcome"])
    if a11:
        print("A11 S0:", {k: (f(v) if isinstance(v, dict) and "est" in v else v) for k, v in a11["S0"].items()})
        print("A11 gates:", a11["gates"])
        print("A11:", a11["outcome"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(ARGV))
