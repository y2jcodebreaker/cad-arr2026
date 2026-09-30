"""Score A9 against its pre-registration (claims/PREREG_A9_robustness.md, frozen 281249f; A9-A1, A9-A2).

CPU only. Question-level bootstrap (2,000 resamples, seed of m1_analysis), non-Healthcare items,
95% intervals; one confirmatory arm per question.

    python a9_analysis.py                       # results/e1 after the A9 outputs are committed
    python a9_analysis.py --base <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.argv = sys.argv[:1] + sys.argv[1:]
import frozen_eval as fe            # noqa: E402
import judge_sweep_outputs as js    # noqa: E402
import m1_arms                      # noqa: E402
from m1_analysis import boot_sets, ci, echo, load_judge, numeric, paired_mean  # noqa: E402

M1J = ("llama", "qwen")
SHUF = ("fc_shuffle_s0", "fc_shuffle_s1", "fc_shuffle_s2")
GATE_MISS, GATE_U, GATE_RHO = 0.10, 0.10, 0.5
ECHO_MAX, DQ_MIN, U_MAX = 0.25, -1.0, 0.10
SIDE_BAND, PPL_MAX = 0.10, 0.05
DRIFT_MIN = 0.95          # A9-A5: identical-text share needed to keep the registered references
# A9-A1 item 3: the J3 arm set A9 is scored on, pinned (later experiments add arms to m1_arms.J3_ARMS)
A9_J3 = tuple(a for a in m1_arms.J3_ARMS if a not in m1_arms.A10_ARMS + m1_arms.A11_ARMS)
RERUN = {"unsteered_s42": "unsteered_rerun", "neutral_s42": "neutral_rerun", "fc_remove_a1": "fc_remove_a1_rerun"}


def bootstat(f, idx, boots):
    return {"est": f(idx), "ci": ci([f(b) for b in boots])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/e1")
    ap.add_argument("--out", default="analysis-output/a9")
    args = ap.parse_args(argv)
    base = Path(args.base)
    R: dict = {}

    # ---------------------------------------------------------------- frozen set
    items = fe.eval_pairs(fe.load())
    arms = m1_arms.load(base, required=False)
    need = ["unsteered_s42", "neutral_s42", "fc_remove_a1", *SHUF, "id_remove_a1", "fc_remove_a1_neutral", *RERUN.values()]
    if [a for a in need if a not in arms]:
        raise SystemExit(f"missing arms: {[a for a in need if a not in arms]}")
    J = {j: load_judge(base / f"m1_judge_{j}.json") for j in M1J}
    J["mistral"] = load_judge(base / "m1_judge_mistral.json")
    nonhc = [i for i, it in enumerate(items) if it["domain"] != "Healthcare"]
    qids = np.array([items[i]["question_id"] for i in nonhc])
    boots = [np.array(nonhc)[b] for b in boot_sets(qids)]
    num = {j: {a: [numeric(J[j]["by_arm"][a].get(i)) for i in range(250)] for a in J[j]["by_arm"]} for j in J}
    rep = json.loads((base / "a9/a9_fit_report.json").read_text())

    # ---------------------------------------------------------------- fresh set
    fitems, ftexts = m1_arms.load_fresh(base)
    FJ = {j: load_judge(base / f"m1_judge_{j}_fresh.json") for j in (*M1J, "mistral")}
    fnonhc = [i for i, it in enumerate(fitems) if it["domain"] != "Healthcare"]
    fboots = [np.array(fnonhc)[b] for b in boot_sets(np.array([fitems[i]["question_id"] for i in fnonhc]))]
    fnum = {j: {a: [numeric(FJ[j]["by_arm"][a].get(i)) for i in range(len(fitems))] for a in ftexts} for j in FJ}

    # ---------------------------------------------------------------- A9-A5 environment drift
    ident = {o: float(np.mean([x == y for x, y in zip(arms[o]["texts"], arms[n]["texts"])])) for o, n in RERUN.items()}
    same_env = all(v >= DRIFT_MIN for v in ident.values())
    REF, NREF, FC = (("unsteered_s42", "neutral_s42", "fc_remove_a1") if same_env
                     else ("unsteered_rerun", "neutral_rerun", "fc_remove_a1_rerun"))
    R["A9_A5_drift"] = {"identical_share": ident, "registered_references_used": same_env,
                        "references": {"disability": REF, "neutral": NREF, "framing_arm": FC},
                        "dJ_rerun_minus_committed": {j: bootstat(
                            lambda idx: paired_mean(num[j]["unsteered_rerun"], num[j]["unsteered_s42"], idx), nonhc, boots)
                            for j in M1J}}
    R["GA9_0"] = {"cos": rep["ga9_0_cos"], "pass": rep["GA9_0_pass"]}

    # ---------------------------------------------------------------- gates on J3
    u42 = [J["mistral"]["by_arm"]["unsteered_s42"].get(i) for i in range(250)]
    g1 = {"missing_rate": float(np.mean([v is None for v in u42])), "u_rate": float(np.mean([v == "U" for v in u42]))}
    g1["pass"] = g1["missing_rate"] <= GATE_MISS and g1["u_rate"] <= GATE_U
    R["GA9_1"] = g1
    keys3 = {r["key"] for r in load_records(base / "m1_judge_mistral.json") if r["arm"] in A9_J3} | \
            {r["key"] for r in load_records(base / "m1_judge_mistral_fresh.json")}
    g2 = {}
    for j in M1J:
        rk = {**J[j]["rating_by_key"], **FJ[j]["rating_by_key"]}
        r3 = {**J["mistral"]["rating_by_key"], **FJ["mistral"]["rating_by_key"]}
        pairs = [(numeric(r3.get(k)), numeric(rk.get(k))) for k in keys3]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        g2[j] = {"rho": float(spearmanr(*zip(*pairs)).statistic), "n": len(pairs)}
    g2["pass"] = all(g2[j]["rho"] >= GATE_RHO for j in M1J)
    R["GA9_2"] = g2
    R["gates_pass"] = bool(R["GA9_0"]["pass"] and g1["pass"] and g2["pass"])

    # ---------------------------------------------------------------- Q-J3
    d3 = lambda a, ref="unsteered_s42": (lambda idx: paired_mean(num["mistral"][a], num["mistral"][ref], idx))  # noqa: E731
    qj3 = {"fc_remove_a1": bootstat(d3("fc_remove_a1"), nonhc, boots)}
    e, lo_hi = qj3["fc_remove_a1"]["est"], qj3["fc_remove_a1"]["ci"]
    qj3["outcome"] = ("not scored: a J3 gate (GA9-1 / GA9-2) failed" if not (g1["pass"] and g2["pass"]) else
                      "(a) the reduction holds on a judge never used for fitting" if e < 0 and lo_hi[1] < 0 else
                      "(b) not confirmed" if e < 0 else "(c) contradicted")
    qj3["all_arms"] = {a: bootstat(d3(a), nonhc, boots) for a in A9_J3
                       if a in num["mistral"] and arms.get(a, {}).get("side") == "disability" and a != "unsteered_s42"}
    qj3["Q1_gap_disability_minus_neutral"] = bootstat(d3("unsteered_s42", "neutral_s42"), nonhc, boots)
    qj3["fresh"] = {a: bootstat(lambda idx, a=a: paired_mean(fnum["mistral"][a], fnum["mistral"]["fresh_unsteered"], idx),
                                fnonhc, fboots) for a in ftexts if a != "fresh_unsteered"}
    R["Q_J3"] = qj3

    # ---------------------------------------------------------------- Q-SPEC
    def shuf_mean(j):
        out = []
        for i in range(250):
            v = [num[j][a][i] for a in SHUF if num[j][a][i] is not None]
            out.append(float(np.mean(v)) if v else None)
        return out
    spec = {}
    for j in M1J:
        sm = shuf_mean(j)
        spec[j] = {"fc_minus_shuffle_mean": bootstat(lambda idx: paired_mean(num[j][FC], sm, idx), nonhc, boots),
                   "shuffle_mean_dJ": bootstat(lambda idx: paired_mean(sm, num[j][REF], idx), nonhc, boots),
                   "per_shuffle_dJ": {a: bootstat(lambda idx, a=a: paired_mean(num[j][a], num[j][REF], idx),
                                                  nonhc, boots) for a in SHUF}}
    a_ok = all(spec[j]["fc_minus_shuffle_mean"]["ci"][1] < 0 for j in M1J)
    b_ok = any(spec[j]["fc_minus_shuffle_mean"]["ci"][1] >= 0 and spec[j]["shuffle_mean_dJ"]["ci"][1] < 0 for j in M1J)
    spec["outcome"] = "(a) specific" if a_ok else "(b) not specific" if b_ok else "(c) neither"
    R["Q_SPEC"] = spec

    # ---------------------------------------------------------------- Q-MECH
    e_ref = np.array([echo(t, it) for t, it in zip(arms[REF]["texts"], items)], float)[nonhc]

    def echo_drop(texts, its, idx, ref):
        v = np.array([echo(t, it) for t, it in zip(texts, its)], float)[idx]
        return float(1 - v.mean() / ref.mean())
    mech = {"echo_drop": echo_drop(arms["id_remove_a1"]["texts"], items, nonhc, e_ref),
            "cos_identity_framing": rep["cos_identity_framing"]}
    for j in M1J:
        mech[j] = bootstat(lambda idx: paired_mean(num[j]["id_remove_a1"], num[j][REF], idx), nonhc, boots)
    reduces = all(mech[j]["ci"][1] < 0 for j in M1J)
    mech["outcome"] = ("(a) erasure" if mech["echo_drop"] >= ECHO_MAX else
                       "(b) reduction without erasure: mechanism claim withdrawn" if reduces else "(c) neither")
    R["Q_MECH"] = mech

    # ---------------------------------------------------------------- Q-SIDE
    ppl = json.loads((base / "a9/ppl.json").read_text())
    side = {"ppl": ppl, "ppl_rel_change": {a: ppl[a]["ppl"] / ppl["unsteered"]["ppl"] - 1 for a in ("fc_remove_a1", "fc_remove_a2")}}
    nt, rt_ = arms["fc_remove_a1_neutral"]["texts"], arms[NREF]["texts"]
    side["identical_share"] = float(np.mean([a == b for a, b in zip(nt, rt_)]))
    side["mean_chars"] = {"steered": float(np.mean([len(t) for t in nt])), "unsteered": float(np.mean([len(t) for t in rt_]))}
    for j in M1J:
        side[j] = bootstat(lambda idx: paired_mean(num[j]["fc_remove_a1_neutral"], num[j][NREF], idx), nonhc, boots)
    in_band = all(-SIDE_BAND <= side[j]["ci"][0] and side[j]["ci"][1] <= SIDE_BAND for j in M1J)
    side["outcome"] = ("(a) negligible" if in_band and side["ppl_rel_change"]["fc_remove_a1"] <= PPL_MAX
                       else "(b) side effect reported")
    R["Q_SIDE"] = side

    # ---------------------------------------------------------------- Q-FRESH
    acc = {j: json.loads((base / f).read_text())["scores_by_hash"]
           for j, f in (("llama", "judge_llama.json"), ("qwen", "judge_second.json"))}
    qual = lambda j, a: np.array([np.nan if (v := acc[j].get(js.text_key(t))) is None else v for t in ftexts[a]], float)  # noqa: E731
    fe_ref = np.array([echo(t, it) for t, it in zip(ftexts["fresh_unsteered"], fitems)], float)[fnonhc]
    fresh = {}
    for a in ftexts:
        if a == "fresh_unsteered":
            continue
        row = {"echo_drop": echo_drop(ftexts[a], fitems, fnonhc, fe_ref)}
        for j in M1J:
            row[j] = {**bootstat(lambda idx: paired_mean(fnum[j][a], fnum[j]["fresh_unsteered"], idx), fnonhc, fboots),
                      "u_rate": float(np.mean([FJ[j]["by_arm"][a].get(i) == "U" for i in fnonhc])),
                      "dQ_accessibility": float(np.nanmean(qual(j, a) - qual(j, "fresh_unsteered"))),
                      "unscored_accessibility": int(np.isnan(qual(j, a)).sum())}
        row["dJ_holds"] = all(row[j]["ci"][1] < 0 for j in M1J)
        row["others_hold"] = {"echo": row["echo_drop"] < ECHO_MAX,
                              "quality": all(row[j]["dQ_accessibility"] >= DQ_MIN for j in M1J),
                              "u_rate": all(row[j]["u_rate"] <= U_MAX for j in M1J)}
        fresh[a] = row
    c = fresh["fresh_fc_remove_a1"]
    base_J = {j: {"fresh": float(np.mean([v for i in fnonhc if (v := fnum[j]["fresh_unsteered"][i]) is not None])),
                  "frozen": float(np.mean([v for i in nonhc if (v := num[j]["unsteered_s42"][i]) is not None]))} for j in M1J}
    under = any(base_J[j]["fresh"] < 0.5 * base_J[j]["frozen"] for j in M1J)
    fresh["unsteered_mean_J"] = base_J
    fresh["outcome"] = ("(a) the A8 result carries to unseen questions" if c["dJ_holds"] and all(c["others_hold"].values())
                        else "(b) dJ holds; fails: " + ", ".join(k for k, v in c["others_hold"].items() if not v)
                        if c["dJ_holds"] else
                        "(c) not carried over, under-powered (low unsteered J on the fresh set)" if under
                        else "(c) not carried over")
    R["Q_FRESH"] = fresh

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "a9_scores.json").write_text(json.dumps(R, indent=1, default=float))
    print(f"A9-A5 identical shares {ident} -> references {R['A9_A5_drift']['references']}")
    print(f"gates: GA9-0 {R['GA9_0']}  GA9-1 {g1}  GA9-2 {g2}")
    for q in ("Q_J3", "Q_SPEC", "Q_MECH", "Q_SIDE", "Q_FRESH"):
        print(f"{q}: {R[q]['outcome']}")
    return 0


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text())["records"]


if __name__ == "__main__":
    raise SystemExit(main())
