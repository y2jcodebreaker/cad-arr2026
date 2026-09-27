"""Build the frozen E1 evaluation set (pre-registration amendment 2026-09-27).

AccessEval lists every question TWICE (row i == row i+234), so the unit here is the unique
QUESTION (base_query_id % 234) and the unique PAIR (its text), never the row.

Eval:  5 disability variants from each of 50 questions = 250 unique items, questions
       allocated across the 6 domains in proportion to the pool, variants chosen to balance
       the 9 disability categories.
Fit:   every unique loudness-filtered pair whose QUESTION is not in the eval set. Whole
       questions are held out, because the variants of one question are near-duplicates.

Runs on CPU, needs no model. Writes frozen_eval_v1.json and refuses to overwrite it:
a frozen set that can be silently rebuilt is not frozen.

    python build_frozen_eval.py            # builds, inspects, writes
    python build_frozen_eval.py --check    # rebuilds in memory, asserts identical to disk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUT = Path(__file__).parent / "frozen_eval_v1.json"
LOUDNESS = Path(__file__).parent / "angular_checkpoints" / "accesseval_loudness.npz"
SEED = 42
N_QUESTIONS = 50
VARIANTS_PER_QUESTION = 5
N_UNIQUE_QUESTIONS = 234

# Positive controls: the pool must reproduce what day1 measured, or nothing is built.
EXPECT_PAIRS, EXPECT_POOL, EXPECT_BQ, EXPECT_POOL_BQ = 4164, 2083, 468, 292
# ...and, once the duplicated rows are collapsed (verified 2026-09-27):
EXPECT_UNIQUE_PAIRS, EXPECT_UNIQUE_POOL, EXPECT_POOL_QUESTIONS = 2082, 1044, 146


from frozen_eval import pair_key  # single definition, shared with every runner


def build() -> dict:
    sys.argv = sys.argv[:1]  # the imported modules parse no args at import, but be safe
    import day1_extract_activations as d1
    import evaluate_rtsd_fullres_generation as rt

    pairs, meta = d1.build_pairs_with_metadata()
    ref = rt.build_accesseval_pairs()
    assert [(p["clean_text"], p["corrupted_text"]) for p in pairs] == \
           [(r["clean_text"], r["corrupted_text"]) for r in ref], \
        "day1 pairs differ from the pipeline's build_accesseval_pairs()"

    norms = np.load(LOUDNESS)["norms"]
    threshold = float(np.percentile(norms, 100 - rt.SIGNAL_PERCENTILE))
    keep = norms >= threshold
    bq = np.array([m["base_query_id"] for m in meta])
    for name, got, want in [("expanded pairs", len(pairs), EXPECT_PAIRS),
                            ("filtered pool", int(keep.sum()), EXPECT_POOL),
                            ("base queries", len(np.unique(bq)), EXPECT_BQ),
                            ("pool base queries", len(np.unique(bq[keep])), EXPECT_POOL_BQ)]:
        assert got == want, f"POSITIVE CONTROL FAILED: {name} = {got}, expected {want}"

    # collapse duplicated rows: one entry per unique (clean, corrupted) text, first occurrence
    seen: dict = {}
    for i in np.flatnonzero(keep):
        k = pair_key(pairs[i]["clean_text"], pairs[i]["corrupted_text"])
        if k not in seen:
            seen[k] = dict(pool_index=int(i), clean_text=pairs[i]["clean_text"],
                           corrupted_text=pairs[i]["corrupted_text"],
                           question_id=int(meta[i]["base_query_id"]) % N_UNIQUE_QUESTIONS,
                           category=str(meta[i]["category"]), domain=str(meta[i]["domain"]))
    pool = list(seen.values())
    n_unique_all = len({pair_key(p["clean_text"], p["corrupted_text"]) for p in pairs})
    for name, got, want in [("unique expanded pairs", n_unique_all, EXPECT_UNIQUE_PAIRS),
                            ("unique filtered pairs", len(pool), EXPECT_UNIQUE_POOL),
                            ("questions in pool", len({p["question_id"] for p in pool}), EXPECT_POOL_QUESTIONS)]:
        assert got == want, f"POSITIVE CONTROL FAILED: {name} = {got}, expected {want}"
    by_bq: dict[int, list[dict]] = defaultdict(list)
    for p in pool:
        by_bq[p["question_id"]].append(p)

    # --- allocate 125 base queries across domains, proportionally ---------------
    rng = random.Random(SEED)
    eligible = {b: v for b, v in by_bq.items() if len(v) >= VARIANTS_PER_QUESTION}
    dom_of = {b: v[0]["domain"] for b, v in eligible.items()}
    dom_count = Counter(dom_of.values())
    raw = {d: N_QUESTIONS * c / len(eligible) for d, c in dom_count.items()}
    alloc = {d: int(r) for d, r in raw.items()}
    for d in sorted(raw, key=lambda d: raw[d] - alloc[d], reverse=True)[:N_QUESTIONS - sum(alloc.values())]:
        alloc[d] += 1

    chosen: list[int] = []
    for d in sorted(alloc):
        cands = sorted(b for b in eligible if dom_of[b] == d)
        rng.shuffle(cands)
        chosen += cands[:alloc[d]]
    assert len(chosen) == N_QUESTIONS

    # --- two variants per base query, greedily balancing categories -------------
    cat_count: Counter = Counter()
    items: list[dict] = []
    for b in sorted(chosen):
        variants = sorted(by_bq[b], key=lambda p: (cat_count[p["category"]], rng.random()))
        for p in variants[:VARIANTS_PER_QUESTION]:
            cat_count[p["category"]] += 1
            items.append(p)
    rng.shuffle(items)

    eval_bq = set(chosen)
    fit = [p for p in pool if p["question_id"] not in eval_bq]
    held_out = [p for p in pool if p["question_id"] in eval_bq]

    body = {
        "version": "frozen_eval_v1",
        "design": "5 variants x 50 unique questions; whole questions held out of fitting; duplicated dataset rows collapsed",
        "dataset": "Srikant86/AccessEval (HF, train split, 468 rows)",
        "seed": SEED,
        "loudness_npz_sha256": hashlib.sha256(LOUDNESS.read_bytes()).hexdigest(),
        "loudness_threshold": threshold,
        "pool_size": len(pool),
        "n_eval": len(items),
        "eval_question_ids": sorted(eval_bq),
        "expected_fit_pool_size": len(fit),
        "expected_full_pool_size": len(pool),
        "full_pool_note": "unique pairs incl. the eval questions: the leak-control arms fit on this",
        "fit_pair_keys": sorted(pair_key(p["clean_text"], p["corrupted_text"]) for p in fit),
        "eval": [{k: p[k] for k in ("pool_index", "question_id", "category", "domain",
                                    "clean_text", "corrupted_text")} for p in items],
    }
    canon = json.dumps(body, sort_keys=True, ensure_ascii=False)
    body["sha256"] = hashlib.sha256(canon.encode()).hexdigest()
    body["_held_out_pairs"] = len(held_out)
    return body


def inspect(fz: dict) -> None:
    ev = fz["eval"]
    logger.info(f"\nfrozen set {fz['sha256'][:16]}  ({fz['n_eval']} eval items)")
    logger.info(f"  eval questions         : {len(fz['eval_question_ids'])}")
    logger.info(f"  fit pool               : {fz['expected_fit_pool_size']} pairs "
                f"(held out: {fz['_held_out_pairs']} unique pairs of the eval questions)")
    logger.info(f"  duplicate eval items   : {len(ev) - len({(e['clean_text'], e['corrupted_text']) for e in ev})}")
    logger.info("  per category           : " + ", ".join(
        f"{c.split()[0]} {n}" for c, n in sorted(Counter(e["category"] for e in ev).items())))
    logger.info("  per domain (items)     : " + ", ".join(
        f"{d} {n}" for d, n in sorted(Counter(e["domain"] for e in ev).items())))
    per_bq = Counter(e["question_id"] for e in ev)
    logger.info(f"  items per question     : {sorted(Counter(per_bq.values()).items())}")
    fitkeys = set(fz["fit_pair_keys"])
    overlap = sum(pair_key(e["clean_text"], e["corrupted_text"]) in fitkeys for e in ev)
    logger.info(f"  eval items in fit pool : {overlap}  (must be 0)")
    for e in ev[:3]:
        logger.info(f"    [{e['category'].split()[0]:<8}|{e['domain']:<11}] {e['corrupted_text'][:88]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="rebuild in memory and assert identical to the file on disk")
    args = ap.parse_args()

    fz = build()
    inspect(fz)
    assert sum(pair_key(e["clean_text"], e["corrupted_text"]) in set(fz["fit_pair_keys"])
               for e in fz["eval"]) == 0, "LEAK: an eval item is in the fit pool"

    if args.check:
        on_disk = json.loads(OUT.read_text())
        assert on_disk["sha256"] == fz["sha256"], \
            f"REBUILD DIFFERS: disk {on_disk['sha256'][:16]} vs rebuilt {fz['sha256'][:16]}"
        logger.info(f"\nCHECK PASSED: rebuild is identical to {OUT.name}")
        return 0
    if OUT.exists():
        logger.error(f"\nREFUSING to overwrite {OUT.name}: it is frozen. Use --check to verify it.")
        return 1
    fz.pop("_held_out_pairs")
    OUT.write_text(json.dumps(fz, indent=1, ensure_ascii=False) + "\n")
    logger.info(f"\nwrote {OUT.name}  sha256 {fz['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
