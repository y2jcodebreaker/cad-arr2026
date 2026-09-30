"""A13 step 1: VADER and Regard scores for the frozen-set arms (claims/PREREG_A13_benchmark_metrics.md).

CPU is enough. Writes results/e1/a13/benchmark_metric_scores.json (evidence; commit before a13_analysis.py).

    python a13_score_texts.py                  # uses the GPU if there is one (the pod: ~2 min)
    python a13_score_texts.py --threads 2      # laptop fallback: fewer CPU threads, slower, cooler
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]
import m1_arms   # noqa: E402

BASE = Path("results/e1")
OUT = BASE / "a13" / "benchmark_metric_scores.json"
REGARD = "sasha/regardv3"
REGARD_REV = "9232a1d0729b2cfe298fbc61cad9d616fa029c04"
ARMS = ["unsteered_s42", "neutral_s42", "caa", "fairsteer", "sadi", "angular", "caa_m2", "fairsteer_a1", "sadi_s5",
        "angular_adaptive", "cad_proj_k40", "prompt_explicit", "prompt_explicit_cad_k40", "fc_remove_a1",
        "fc_prompt_remove_a1", "fc_add_a4", "fc_add_a8", "id_remove_a1", "game_append", "game_sub", "game_ban",
        *m1_arms.A11_ARMS]


def h(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def main() -> int:
    import argparse
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=0, help="limit CPU threads (0 = torch default)")
    args = ap.parse_args(ARGV)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = 0 if torch.cuda.is_available() else -1
    import transformers
    import vaderSentiment
    from transformers import pipeline
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    arms = m1_arms.load(BASE, required=False)
    missing = [a for a in ARMS if a not in arms]
    if missing:
        raise SystemExit(f"arms missing: {missing}")
    texts = {h(t): t for a in ARMS for t in arms[a]["texts"]}
    print(f"{len(ARMS)} arms, {len(texts)} unique texts")
    va = SentimentIntensityAnalyzer()
    vader = {k: va.polarity_scores(t)["compound"] for k, t in texts.items()}
    clf = pipeline("text-classification", model=REGARD, revision=REGARD_REV, top_k=None, truncation=True,
                   max_length=512, device=device)
    keys = list(texts)
    regard, t0 = {}, time.time()
    for i in range(0, len(keys), 32):
        batch = keys[i:i + 32]
        for k, res in zip(batch, clf([texts[k] for k in batch], batch_size=32)):
            regard[k] = {r["label"]: float(r["score"]) for r in res}
        if (i // 32) % 20 == 0:
            print(f"  regard {i + len(batch)}/{len(keys)}  {(i + len(batch)) / (time.time() - t0):.1f} texts/s", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "regard_model": REGARD, "regard_revision": REGARD_REV, "regard_max_tokens": 512, "vader": "compound, full text",
        "env": {"device": "cuda" if device == 0 else f"cpu x{torch.get_num_threads()}", "torch": torch.__version__, "transformers": transformers.__version__,
                "vaderSentiment": getattr(vaderSentiment, "__version__", "unknown")},
        "arms": {a: [h(t) for t in arms[a]["texts"]] for a in ARMS},
        "scores_by_hash": {k: {"vader": vader[k], "regard": regard[k]} for k in keys}}))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
