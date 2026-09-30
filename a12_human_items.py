"""A12: select the human-validation items (claims/PREREG_A12_human_validation.md, frozen 42f40d8).

Writes two files:
  human_eval/items.json  what the rating page shows: opaque id, question, capped answer; rater orders
  human_eval/key.json    id -> (block, arm, frozen-set item index, question id); never on the page

    python a12_human_items.py
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

sys.argv = sys.argv[:1]
import frozen_eval as fe                 # noqa: E402
import judge_medicalization as jm        # noqa: E402  (cap: the 2,000-character text the judges saw)
import m1_arms                           # noqa: E402

BASE = Path("results/e1")
OUT = Path("human_eval")
BLOCKS = [("llama_pair", ("unsteered_s42", "fc_remove_a1"), 30),
          ("mistral_pair", ("mistral_unsteered", "mistral_fc_remove_a1"), 15),
          ("range_caa", ("caa",), 10), ("range_append", ("game_append",), 5), ("range_salad", ("fc_add_a8",), 5)]


def main() -> int:
    items = fe.eval_pairs(fe.load())
    arms = m1_arms.load(BASE, required=False)
    rng = random.Random(0)
    pool = [i for i, it in enumerate(items) if it["domain"] != "Healthcare"]
    rng.shuffle(pool)
    shown, key, pos = [], {}, 0
    for block, block_arms, n in BLOCKS:
        chosen = pool[pos:pos + n]
        pos += n
        for i in chosen:
            for a in block_arms:
                uid = hashlib.sha256(f"a12|{a}|{i}".encode()).hexdigest()[:8]
                shown.append({"id": uid, "question": items[i]["corrupted_text"], "answer": jm.cap(arms[a]["texts"][i])})
                key[uid] = {"block": block, "arm": a, "item": i, "question_id": items[i]["question_id"],
                            "category": items[i]["category"], "domain": items[i]["domain"]}
    assert len(shown) == 110 and len(set(key)) == 110 and pos == 65
    fit = [x for x in json.loads((BASE / "a8/fit_items.json").read_text()) if x["domain"] != "Healthcare"]
    practice = [{"id": f"p{k + 1}", "question": x["question"], "answer": jm.cap(x["answer"])}
                for k, x in enumerate(random.Random(3).sample(fit, 3))]
    ids = [s["id"] for s in shown]
    orders = {}
    for rater, seed in (("R1", 1), ("R2", 2)):
        o = ids[:]
        random.Random(seed).shuffle(o)
        orders[rater] = o
    rng.shuffle(shown)                      # file order carries no block information
    OUT.mkdir(exist_ok=True)
    page = {"practice": practice, "items": shown, "orders": orders}
    (OUT / "items.json").write_text(json.dumps(page, indent=1))
    (OUT / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(shown)} items, {len(practice)} practice, orders for {list(orders)}")
    print("items.json sha256", hashlib.sha256((OUT / 'items.json').read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
