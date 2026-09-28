"""Validity checks on the medicalization score (claim audit, 2026-09-28). CPU only.

Inputs (primary evidence, committed):
  results/phase1_vocab/pity_vocabulary-2.json      log-odds analysis of 300 paired responses
  results/phase1_vocab/accesseval_responses-2.json the 300 neutral / disability response pairs
  results/e1/cad_heldout/baseline_seed42_responses.json  E1 unsteered reference

Checks:
  1. z-score of every scorer word in the original log-odds analysis
  2. the score on answers to NEUTRAL questions (is zero "unbiased"?)
  3. how much of the score is the echoed disability name "impairments"

    python audit_metric_validity.py
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.argv = sys.argv[:1]
from evaluate_rtsd_fullres_generation import MEDICAL_WORDS, PROFESSIONAL_WORDS  # the pinned scorer

P = Path("results/phase1_vocab")


def counts(text: str) -> collections.Counter:
    return collections.Counter(re.findall(r"[a-z]+", text.lower()))


def score(c: collections.Counter, med=MEDICAL_WORDS) -> float:
    return float(np.log((sum(c[w] for w in med) + 1) / (sum(c[w] for w in PROFESSIONAL_WORDS) + 1)))


def main() -> int:
    out = {}
    vocab = json.loads((P / "pity_vocabulary-2.json").read_text())
    allw = vocab["all_words"]
    z = {e["word"]: e["z_score"] for e in allw} if isinstance(allw, list) else {k: v["z_score"] for k, v in allw.items()}
    out["z_medical"] = {w: z.get(w) for w in MEDICAL_WORDS}
    out["z_professional"] = {w: z.get(w) for w in PROFESSIONAL_WORDS}
    out["top20_disability_words"] = [e["word"] for e in vocab["top_20_pity_words"]]

    pairs = json.loads((P / "accesseval_responses-2.json").read_text())
    n = np.array([score(counts(p["neutral_response"])) for p in pairs])
    d = np.array([score(counts(p["disability_response"])) for p in pairs])
    by = collections.defaultdict(list)
    for p, s in zip(pairs, n):
        by[p["domain"]].append(s)
    out["neutral_answers"] = dict(n=len(n), mean=float(n.mean()), sd=float(n.std(ddof=1)),
                                  frac_below0=float((n < 0).mean()), frac_eq0=float((n == 0).mean()),
                                  by_domain={k: float(np.mean(v)) for k, v in sorted(by.items())})
    out["disability_answers"] = dict(n=len(d), mean=float(d.mean()), sd=float(d.std(ddof=1)))

    base = json.loads(Path("results/e1/cad_heldout/baseline_seed42_responses.json").read_text())["responses"]
    C = [counts(t) for t in base]
    med_tot = {w: int(sum(c[w] for c in C)) for w in MEDICAL_WORDS}
    no_imp = [w for w in MEDICAL_WORDS if w != "impairments"]
    out["e1_unsteered"] = dict(medical_counts=med_tot,
                               share_impairments=med_tot["impairments"] / sum(med_tot.values()),
                               mean_score=float(np.mean([score(c) for c in C])),
                               mean_score_without_impairments=float(np.mean([score(c, no_imp) for c in C])))
    Path("analysis-output/metric_validity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
