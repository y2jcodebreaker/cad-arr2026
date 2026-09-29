"""Post-hoc audit of the A8 arms. NOT PRE-REGISTERED: descriptive checks run after a8_analysis.py.

Degeneracy (unique answers, empty answers, length), the lexical score E1 used, the held-out lexicon
rate L (M1 section 3), accessibility-judge parse failures, fit/eval question overlap, per-item
transitions on both M1 judges, and exploratory paired comparisons for the steering-only arm.

    python a8_audit.py        # after e1_analysis.py, m1_analysis.py and a8_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.argv = sys.argv[:1]
import evaluate_rtsd_fullres_generation as rt  # noqa: E402
import frozen_eval as fe                       # noqa: E402
import judge_sweep_outputs as js               # noqa: E402
import m1_arms                                 # noqa: E402
from m1_analysis import boot_sets, l_rate, load_judge, numeric, paired_mean, tokens  # noqa: E402

BASE = Path("results/e1")
REF = "unsteered_s42"
SHOW = [REF, *m1_arms.A8_ARMS, "prompt_explicit", "caa", "cad_proj_k40", "prompt_explicit_cad_k40"]


def pct(x):
    x = [v for v in x if v is not None]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main() -> int:
    items = fe.eval_pairs(fe.load())
    arms = m1_arms.load(BASE, required=False)
    kept = set(json.loads(Path("analysis-output/m1/m1_scores.json").read_text())["lexicon"]["kept"])
    acc = {j: json.loads((BASE / f).read_text())["parsed_by_hash"]
           for j, f in (("llama", "judge_llama.json"), ("qwen", "judge_second.json"))}
    J = {j: load_judge(BASE / f"m1_judge_{j}.json") for j in ("llama", "qwen")}
    nonhc = [i for i, it in enumerate(items) if it["domain"] != "Healthcare"]
    num = {j: {a: [numeric(J[j]["by_arm"][a].get(i)) for i in range(250)] for a in SHOW} for j in J}
    med = {a: np.array([rt.medicalization_score(t) for t in arms[a]["texts"]]) for a in SHOW}

    rows = {}
    for a in SHOW:
        T = arms[a]["texts"]
        keys = [js.text_key(t) for t in T]
        rows[a] = {
            "lexical_mean": float(med[a].mean()), "lexical_d": float(rt.cohens_d(med[REF], med[a])),
            "unique_frac": len(set(T)) / len(T), "empty": sum(not t.strip() for t in T),
            "mean_words": float(np.mean([len(tokens(t)) for t in T])),
            "L_rate_per_1k_nonhc": float(np.mean([l_rate(T[i], kept) for i in nonhc])),
            "acc_parse_fail": {j: float(np.mean([not acc[j].get(k, True) for k in keys])) for j in acc},
            "mean_J_nonhc": {j: float(np.mean([v for i in nonhc if (v := num[j][a][i]) is not None])) for j in J}}
        if a != REF:
            both = [i for i in nonhc if all(num[j][a][i] is not None and num[j][REF][i] is not None for j in J)]
            rows[a]["to_clean_both"] = sum(all(num[j][REF][i] >= 1 and num[j][a][i] == 0 for j in J) for i in both)
            rows[a]["to_med_both"] = sum(all(num[j][REF][i] == 0 and num[j][a][i] >= 1 for j in J) for i in both)

    fit = json.loads((BASE / "a8/fit_items.json").read_text())
    fit = fit if isinstance(fit, list) else fit["items"]
    overlap = {"fit_answers": len(fit), "fit_question_ids": len({x["question_id"] for x in fit}),
               "eval_question_ids": len({it["question_id"] for it in items}),
               "shared_question_ids": len({x["question_id"] for x in fit} & {it["question_id"] for it in items})}

    qids = np.array([items[i]["question_id"] for i in nonhc])
    boots = [np.array(nonhc)[b] for b in boot_sets(qids)]
    explor = {}
    for other in ("prompt_explicit", "caa"):
        explor[f"fc_remove_a1_vs_{other}"] = {
            j: {"diff": paired_mean(num[j]["fc_remove_a1"], num[j][other], nonhc),
                "ci95": pct([paired_mean(num[j]["fc_remove_a1"], num[j][other], s) for s in boots])} for j in J}

    out = {"note": "NOT PRE-REGISTERED: post-hoc descriptive audit", "arms": rows,
           "fit_eval_overlap": overlap, "exploratory_paired": explor}
    Path("analysis-output/a8").mkdir(parents=True, exist_ok=True)
    Path("analysis-output/a8/a8_audit_NOT_PREREGISTERED.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"{'arm':<24}{'lex':>7}{'d':>6}{'uniq':>6}{'empty':>6}{'words':>7}{'L/1k':>6}{'J L':>6}{'J Q':>6}{'->0':>5}{'->1+':>5}")
    for a, r in rows.items():
        print(f"{a:<24}{r['lexical_mean']:>7.3f}{r['lexical_d']:>6.2f}{r['unique_frac']:>6.2f}{r['empty']:>6}"
              f"{r['mean_words']:>7.0f}{r['L_rate_per_1k_nonhc']:>6.2f}{r['mean_J_nonhc']['llama']:>6.2f}"
              f"{r['mean_J_nonhc']['qwen']:>6.2f}{r.get('to_clean_both', ''):>5}{r.get('to_med_both', ''):>5}")
    print("overlap:", overlap)
    print("exploratory:", json.dumps(explor, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
