"""A9 exploratory comparisons. NOT PRE-REGISTERED: run after a9_analysis.py, reported as exploratory.

Fresh set, non-Healthcare items, question-level bootstrap, 95% intervals: does adding the A8
projection to the explicit prompt lower J beyond the prompt alone, on each of the three judges?

    python a9_exploratory.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.argv = sys.argv[:1]
import m1_arms                                                          # noqa: E402
from m1_analysis import boot_sets, ci, load_judge, numeric, paired_mean  # noqa: E402

BASE = Path("results/e1")
PAIRS = [("fresh_fc_prompt_remove_a1", "fresh_prompt_explicit")]


def main() -> int:
    items, texts = m1_arms.load_fresh(BASE)
    idx = [i for i, it in enumerate(items) if it["domain"] != "Healthcare"]
    boots = [np.array(idx)[b] for b in boot_sets(np.array([items[i]["question_id"] for i in idx]))]
    out = {"note": "NOT PRE-REGISTERED: exploratory", "n_items": len(idx),
           "n_questions": len({items[i]["question_id"] for i in idx}), "comparisons": {}}
    for j in ("llama", "qwen", "mistral"):
        J = load_judge(BASE / f"m1_judge_{j}_fresh.json")
        num = {a: [numeric(J["by_arm"][a].get(i)) for i in range(len(items))] for a in texts}
        for a, b in PAIRS:
            out["comparisons"].setdefault(f"{a}_minus_{b}", {})[j] = {
                "diff": paired_mean(num[a], num[b], idx), "ci95": ci([paired_mean(num[a], num[b], s) for s in boots])}
    Path("analysis-output/a9").mkdir(parents=True, exist_ok=True)
    Path("analysis-output/a9/a9_exploratory_NOT_PREREGISTERED.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
