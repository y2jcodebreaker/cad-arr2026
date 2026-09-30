"""Score A14 (claims/PREREG_A14_human_validation_v2.md, frozen a46b55f).

Inputs: human_eval/a14/ratings_R1.json and ratings_R2.json (the page exports, saved unedited),
human_eval/a14/key.json, the judge files. Same criteria as A12 on the 60 study items; attention checks
reported separately. Nothing is excluded.

    python a14_analysis.py [--ratings_dir human_eval/a14]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]
from a12_analysis import boot, rho            # noqa: E402  (same bootstrap and Spearman as A12)
from m1_analysis import load_judge, numeric   # noqa: E402

KEY = Path("human_eval/a14/key.json")
ITEMS = Path("human_eval/a14/items.json")
JUDGES = {"llama": "m1_judge_llama.json", "qwen": "m1_judge_qwen.json", "mistral": "m1_judge_mistral.json"}
MIN_RHO = 0.5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings_dir", default="human_eval/a14")
    ap.add_argument("--base", default="results/e1")
    ap.add_argument("--out", default="analysis-output/a14")
    args = ap.parse_args(argv)
    import hashlib
    sha = hashlib.sha256(ITEMS.read_bytes()).hexdigest()[:12]
    key = json.loads(KEY.read_text())
    page = json.loads(ITEMS.read_text())
    R, T = {}, {}
    for r in ("R1", "R2"):
        d = json.loads((Path(args.ratings_dir) / f"ratings_{r}.json").read_text())
        if d.get("study") != "A14" or d.get("items") != sha or d.get("rater") != r:
            raise SystemExit(f"ratings_{r}.json is not an A14 export for rater {r} on items {sha}")
        R[r], T[r] = d["ratings"], d.get("times", {})
    J = {j: load_judge(Path(args.base) / f) for j, f in JUDGES.items()}
    ids = sorted(i for i, k in key.items() if k["stratum"] != "attention")
    qids = [key[i]["question_id"] for i in ids]
    h = {r: [numeric(R[r].get(i)) for i in ids] for r in R}
    mean_h = [None if not (v := [h[r][k] for r in R if h[r][k] is not None]) else float(np.mean(v)) for k in range(len(ids))]
    judge = {j: [numeric(J[j]["by_arm"].get(key[i]["arm"], {}).get(key[i]["item"])) for i in ids] for j in J}
    sub = lambda xs, ix: [xs[i] for i in ix]  # noqa: E731
    both = [k for k in range(len(ids)) if h["R1"][k] is not None and h["R2"][k] is not None]
    out = {"n_items": len(ids), "coverage": {r: sum(i in R[r] for i in ids) for r in R}}
    out["human_human"] = {"rho": rho(h["R1"], h["R2"]), "rho_ci": boot(lambda ix: rho(sub(h["R1"], ix), sub(h["R2"], ix)), qids),
                          "kappa_quadratic": float(cohen_kappa_score([h["R1"][k] for k in both], [h["R2"][k] for k in both], weights="quadratic")),
                          "n": len(both)}
    out["human_judge"] = {j: {"rho": rho(mean_h, judge[j]), "rho_ci": boot(lambda ix, j=j: rho(sub(mean_h, ix), sub(judge[j], ix)), qids),
                              "per_rater": {r: rho(h[r], judge[j]) for r in R}} for j in J}
    hh, hj = out["human_human"]["rho"], [out["human_judge"][j]["rho"] for j in ("llama", "qwen")]
    out["outcome"] = ("(a) measure supported" if hh is not None and hh >= MIN_RHO and all(x is not None and x >= MIN_RHO for x in hj)
                      else "(b) humans agree with each other but not with a judge" if hh is not None and hh >= MIN_RHO
                      else "(c) humans do not agree with each other")
    strata = [key[i]["stratum"] for i in ids]
    out["strata"] = {r: {s: float(np.mean([h[r][k] >= 1 for k in range(len(ids)) if strata[k] == s and h[r][k] is not None]))
                         for s in ("high", "low")} for r in R}
    ok = [k for k in range(len(ids)) if mean_h[k] is not None]
    out["stratum_auc_mean_human"] = float(roc_auc_score([strata[k] == "high" for k in ok], [mean_h[k] for k in ok]))
    out["attention"] = {r: {i: {"given": R[r].get(i), "expected": k["expected"]} for i, k in key.items() if k["stratum"] == "attention"} for r in R}
    order = page["orders"]
    secs = {}
    for r in R:
        t = [dt.datetime.fromisoformat(T[r][i].replace("Z", "+00:00")) for i in order[r] if i in T[r]]
        g = sorted((b - a).total_seconds() for a, b in zip(t, t[1:]) if (b - a).total_seconds() < 600)
        secs[r] = g[len(g) // 2] if g else None
    out["median_seconds_per_item"] = secs
    out["U_count"] = {r: sum(R[r].get(i) == "U" for i in ids) for r in R}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "a14_scores.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("coverage", "human_human", "outcome", "strata", "stratum_auc_mean_human",
                                          "attention", "median_seconds_per_item", "U_count")}, default=float))
    for j in J:
        print(j, round(out["human_judge"][j]["rho"], 3), out["human_judge"][j]["rho_ci"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(ARGV))
