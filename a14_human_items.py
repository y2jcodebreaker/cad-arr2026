"""A14: select the calibrated human-validation items (claims/PREREG_A14_human_validation_v2.md, frozen a46b55f).

    python a14_human_items.py      # writes human_eval/a14/items.json (page) and human_eval/a14/key.json
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

sys.argv = sys.argv[:1]
import frozen_eval as fe                 # noqa: E402
import judge_medicalization as jm        # noqa: E402
import m1_arms                           # noqa: E402
from m1_analysis import load_judge, numeric   # noqa: E402

BASE = Path("results/e1")
OUT = Path("human_eval/a14")
ARMS = ("unsteered_s42", "fc_remove_a1", "caa", "prompt_explicit", "mistral_unsteered", "mistral_fc_remove_a1")
N_HIGH, N_LOW = 20, 40


def main() -> int:
    items = fe.eval_pairs(fe.load())
    arms = m1_arms.load(BASE, required=False)
    J = {j: load_judge(BASE / f"m1_judge_{j}.json") for j in ("llama", "qwen")}
    used = {v["item"] for v in json.loads(Path("human_eval/key.json").read_text()).values()}
    pool = [i for i, it in enumerate(items) if it["domain"] != "Healthcare" and i not in used]
    cand = [(a, i) for i in pool for a in ARMS]
    rng = random.Random(14)
    rng.shuffle(cand)
    high, low, taken = [], [], set()
    for a, i in cand:
        if i in taken:
            continue
        r = [numeric(J[j]["by_arm"][a].get(i)) for j in J]
        if None in r:
            continue
        if all(v >= 1 for v in r) and len(high) < N_HIGH:
            high.append((a, i)); taken.add(i)
        elif all(v == 0 for v in r) and len(low) < N_LOW:
            low.append((a, i)); taken.add(i)
        if len(high) == N_HIGH and len(low) == N_LOW:
            break
    if len(high) < N_HIGH or len(low) < N_LOW:
        raise SystemExit(f"pool too small: {len(high)} high, {len(low)} low")
    cal = json.loads((OUT / "calibration.json").read_text())
    shown, key = [], {}
    for stratum, chosen in (("high", high), ("low", low)):
        for a, i in chosen:
            uid = hashlib.sha256(f"a14|{a}|{i}".encode()).hexdigest()[:8]
            shown.append({"id": uid, "question": items[i]["corrupted_text"], "answer": jm.cap(arms[a]["texts"][i])})
            key[uid] = {"stratum": stratum, "arm": a, "item": i, "question_id": items[i]["question_id"],
                        "category": items[i]["category"], "domain": items[i]["domain"]}
    for ac in cal["attention_checks"]:
        shown.append({"id": ac["id"], "question": ac["question"], "answer": ac["answer"]})
        key[ac["id"]] = {"stratum": "attention", "expected": ac["expected"]}
    ids = [s["id"] for s in shown]
    orders = {r: (lambda o: (random.Random(seed).shuffle(o), o)[1])(ids[:]) for r, seed in (("R1", 141), ("R2", 142))}
    rng.shuffle(shown)
    practice = [{k: p[k] for k in ("id", "question", "answer", "target", "why")} for p in cal["practice"]]
    (OUT / "items.json").write_text(json.dumps({"practice": practice, "items": shown, "orders": orders}, indent=1))
    (OUT / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(shown)} items ({N_HIGH} high, {N_LOW} low, 2 attention), {len(practice)} practice; "
          f"arms {sorted({v['arm'] for v in key.values() if 'arm' in v})}")
    print("items.json sha256", hashlib.sha256((OUT / "items.json").read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
