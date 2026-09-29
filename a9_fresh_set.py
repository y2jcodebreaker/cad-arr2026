"""Build the A9 fresh question set (claims/PREREG_A9_robustness.md section 4, amendment A9-A1 item 5).

The AccessEval questions used by neither the frozen eval set nor the E1/A8 fit pool: the ones the
E1 loudness filter dropped. Five categories per question, sampled with one random.Random(0) in
ascending question id, each question's unique pairs sorted by category first. Items with no
disability category (3 generic questions; the frozen set has none) are dropped (amendment A9-A3). Run once on the
laptop; the output is committed before any A9 generation, like frozen_eval_v1.json.

    python a9_fresh_set.py            # writes fresh_eval_a9.json and prints its sha256
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import sys
from pathlib import Path

sys_argv, sys.argv = sys.argv, sys.argv[:1]
import evaluate_rtsd_fullres_generation as rt     # noqa: E402
import frozen_eval as fe                          # noqa: E402
sys.argv = sys_argv

OUT = Path("fresh_eval_a9.json")
PER_QUESTION = 5
N_QUESTIONS = 234


def build() -> list[dict]:
    import day1_extract_activations as d1
    raw = rt.build_accesseval_pairs()
    filtered = rt.apply_loudness_filter(raw)
    pairs, meta = d1.build_pairs_with_metadata()
    by_key = {}
    for p, m in zip(pairs, meta):
        by_key.setdefault(fe.pair_key(p["clean_text"], p["corrupted_text"]), m)
    missing = [p for p in raw if fe.pair_key(p["clean_text"], p["corrupted_text"]) not in by_key]
    if missing:
        raise SystemExit(f"{len(missing)} raw pairs have no day-1 metadata")

    def qid(p):
        return int(by_key[fe.pair_key(p["clean_text"], p["corrupted_text"])]["base_query_id"]) % N_QUESTIONS

    used = {qid(p) for p in filtered}
    frozen_q = {it["question_id"] for it in fe.eval_pairs(fe.load())}
    fit_q = {x["question_id"] for x in json.loads(Path("results/e1/a8/fit_items.json").read_text())}
    if not (frozen_q | fit_q) <= used:
        raise SystemExit("frozen or fit questions fall outside the loudness-filtered pool")

    uniq: dict[int, dict[str, dict]] = collections.defaultdict(dict)
    for p in raw:
        q = qid(p)
        if q in used:
            continue
        uniq[q].setdefault(fe.pair_key(p["clean_text"], p["corrupted_text"]), p)
    rng = random.Random(0)
    items = []
    for q in sorted(uniq):
        ps = sorted(uniq[q].values(),
                    key=lambda p: str(by_key[fe.pair_key(p["clean_text"], p["corrupted_text"])]["category"]))
        for p in rng.sample(ps, min(PER_QUESTION, len(ps))):
            m = by_key[fe.pair_key(p["clean_text"], p["corrupted_text"])]
            if not str(m["category"]):     # A9-A3: generic items with no disability category (none in the frozen set)
                continue
            items.append({"item": len(items), "question": p["corrupted_text"], "clean_text": p["clean_text"],
                          "corrupted_text": p["corrupted_text"], "domain": str(m["domain"]),
                          "category": str(m["category"]), "question_id": q})
    assert not {it["question_id"] for it in items} & (frozen_q | fit_q)
    return items


def main() -> int:
    items = build()
    text = json.dumps(items, indent=1)
    OUT.write_text(text)
    doms = collections.Counter(it["domain"] for it in items)
    print(f"{len(items)} items from {len({it['question_id'] for it in items})} questions; domains {dict(doms)}")
    print(f"sha256 {hashlib.sha256(text.encode()).hexdigest()}  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
