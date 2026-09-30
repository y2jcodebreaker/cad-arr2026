"""Score A12 human validation (claims/PREREG_A12_human_validation.md, frozen 42f40d8).

Inputs: the two raters' exported JSON (copied from the rating page) saved as
human_eval/ratings_R1.json and human_eval/ratings_R2.json; human_eval/key.json; the judge files.
Numeric ratings only (U as missing). "Mean human" for an item = the mean of the raters' numeric ratings
available for it (one or two). 95% intervals: 2,000 bootstrap resamples of question ids (seed 0).

    python a12_analysis.py [--ratings_dir human_eval] [--base results/e1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]
from m1_analysis import load_judge, numeric   # noqa: E402

ITEMS_SHA = "3ec7655fea4b"
JUDGES = {"llama": "m1_judge_llama.json", "qwen": "m1_judge_qwen.json", "mistral": "m1_judge_mistral.json"}
MIN_RHO = 0.5
N_BOOT = 2000


def rho(x, y):
    ok = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    return float(spearmanr(*zip(*ok)).statistic) if len(ok) > 2 else None


def boot(fn, qids, n=N_BOOT):
    rng = np.random.default_rng(0)
    uq = sorted(set(qids))
    by = {q: [i for i, x in enumerate(qids) if x == q] for q in uq}
    vals = []
    for _ in range(n):
        idx = [i for q in rng.choice(uq, len(uq)) for i in by[q]]
        v = fn(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings_dir", default="human_eval")
    ap.add_argument("--base", default="results/e1")
    ap.add_argument("--out", default="analysis-output/a12")
    args = ap.parse_args(argv)
    key = json.loads(Path("human_eval/key.json").read_text())
    R = {}
    for r in ("R1", "R2"):
        d = json.loads((Path(args.ratings_dir) / f"ratings_{r}.json").read_text())
        if d.get("study") != "A12" or d.get("items") != ITEMS_SHA or d.get("rater") != r:
            raise SystemExit(f"ratings_{r}.json is not an A12 export for rater {r} on items {ITEMS_SHA}")
        R[r] = d["ratings"]
    J = {j: load_judge(Path(args.base) / f) for j, f in JUDGES.items()}
    ids = sorted(key)
    qids = [key[i]["question_id"] for i in ids]
    h = {r: [numeric(R[r].get(i)) for i in ids] for r in R}
    mean_h = [None if not (v := [h[r][k] for r in R if h[r][k] is not None]) else float(np.mean(v)) for k in range(len(ids))]
    judge = {j: [numeric(J[j]["by_arm"].get(key[i]["arm"], {}).get(key[i]["item"])) for i in ids] for j in J}
    out = {"n_items": len(ids), "coverage": {r: sum(i in R[r] for i in ids) for r in R}}

    sub = lambda xs, idx: [xs[i] for i in idx]  # noqa: E731
    both = [k for k in range(len(ids)) if h["R1"][k] is not None and h["R2"][k] is not None]
    out["human_human"] = {
        "rho": rho(h["R1"], h["R2"]), "rho_ci": boot(lambda ix: rho(sub(h["R1"], ix), sub(h["R2"], ix)), qids),
        "kappa_quadratic": float(cohen_kappa_score([h["R1"][k] for k in both], [h["R2"][k] for k in both], weights="quadratic")),
        "n": len(both), "exact_agreement": float(np.mean([h["R1"][k] == h["R2"][k] for k in both]))}
    out["human_judge"] = {j: {"rho": rho(mean_h, judge[j]),
                              "rho_ci": boot(lambda ix, j=j: rho(sub(mean_h, ix), sub(judge[j], ix)), qids),
                              "per_rater": {r: rho(h[r], judge[j]) for r in R}} for j in J}
    out["judge_judge"] = {"llama_qwen": rho(judge["llama"], judge["qwen"])}
    hh = out["human_human"]["rho"]
    hj = [out["human_judge"][j]["rho"] for j in ("llama", "qwen")]
    out["outcome"] = ("(a) measure supported" if hh is not None and hh >= MIN_RHO and all(x is not None and x >= MIN_RHO for x in hj)
                      else "(b) humans agree with each other but not with a judge" if hh is not None and hh >= MIN_RHO
                      else "(c) humans do not agree with each other")

    # secondary: paired direction, U on broken text, distributions
    def paired(block, a, b):
        pairs = {}
        for i in ids:
            k = key[i]
            if k["block"] == block:
                pairs.setdefault(k["item"], {})[k["arm"]] = mean_h[ids.index(i)]
        items = [p for p in pairs.values() if p.get(a) is not None and p.get(b) is not None]
        diffs = [p[a] - p[b] for p in items]
        qs = [key[next(i for i in ids if key[i]["block"] == block and key[i]["item"] == it)]["question_id"]
              for it, p in pairs.items() if p.get(a) is not None and p.get(b) is not None]
        return {"diff": float(np.mean(diffs)) if diffs else None, "n_pairs": len(diffs),
                "ci": boot(lambda ix: float(np.mean([diffs[i] for i in ix])), qs) if diffs else [None, None]}
    out["secondary"] = {
        "llama_fcs_minus_unsteered": paired("llama_pair", "fc_remove_a1", "unsteered_s42"),
        "mistral_fcs_minus_unsteered": paired("mistral_pair", "mistral_fc_remove_a1", "mistral_unsteered"),
        "salad_U_share": {r: float(np.mean([R[r].get(i) == "U" for i in ids if key[i]["arm"] == "fc_add_a8"])) for r in R},
        "distribution": {r: {v: sum(R[r].get(i) == v for i in ids) for v in ("0", "1", "2", "3", "U")} for r in R}}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "a12_scores.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("coverage", "human_human", "outcome")}, default=float))
    for j in J:
        print(j, round(out["human_judge"][j]["rho"], 3), out["human_judge"][j]["rho_ci"])
    print("secondary", json.dumps(out["secondary"], default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(ARGV))
