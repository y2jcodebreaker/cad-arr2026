#!/usr/bin/env python3
"""
day1_extract_activations.py

Day 1 of the ARR-October revision plan: extract and PERSIST the AccessEval
contrastive activations so that every downstream control is pure CPU numpy.

Why this script exists
----------------------
The reviewer round raised three objections that all require the raw activation
matrices, at several layers, with cluster identity:

  (1) SAMPLE-SIZE CONFOUND. EVR_1 is perfectly rank-correlated with the number
      of SVD pairs (n=13 -> 67.9%, n=30 -> 21.6%, n=2082 -> 10.2%). Testing
      whether that is an artifact needs the full pool, subsampled repeatedly.

  (2) LAYER-DEPTH CONFOUND. The "concentrated" profiles were measured at L0,
      where low rank is a tokenisation fact rather than a bias-geometry fact.
      Testing this needs the same pairs at L0 and at the circuit layers.

  (3) CLUSTERING. build_accesseval_pairs() emits up to 9 disability variants
      per base query, so the pairs are NOT independent. The published bootstrap
      resampled pairs, which understates uncertainty on k@80%. A cluster
      bootstrap needs a base-query id per pair, which the pipeline never
      recorded. This script reconstructs it.

The previous run recomputed the SVD from scratch because no activation
checkpoint existed and the pod was then destroyed. This script saves the
activations so that never happens again.

Outputs (one .npz, ~550 MB):
  clean_L{l}, corrupted_L{l}   float32 (N, D) for each layer
  base_query_id               int32  (N,)  dataset row index = cluster id
  category, domain            unicode (N,)
  templated                   bool   (N,)
  loudness_norms              float32 (N,)
  keep_mask                   bool   (N,)  the published >= median filter
  layers, model_name, ...     metadata

Usage (on a GPU box, from this directory):
    python day1_extract_activations.py
    python day1_extract_activations.py --layers 0 14 21 25
    python day1_extract_activations.py --batch-size 64     # if OOM

Re-running is a no-op unless --force is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset

from evaluate_rtsd_fullres_generation import (
    DATASET_NAME,
    DISABILITY_CATEGORIES,
    LOUDNESS_CKPT,
    MODEL_NAME,
    RANDOM_SEED,
    SIGNAL_PERCENTILE,
    TARGET_DOMAINS,
    TRIGGER_LAYER,
    AMPLIFIER_LAYER,
    build_accesseval_pairs,
    extract_full_decoder,
    format_prompt,
    load_model,
    replace_placeholder,
)

# Layers to persist. 0 = embedding-adjacent (the layer-depth confound control),
# 14 = CAA's chosen layer, 21/25 = the paper's circuit layers.
DEFAULT_LAYERS = [0, 14, TRIGGER_LAYER, AMPLIFIER_LAYER]

OUT_DIR = Path("controls_checkpoints")
OUT_FILE = OUT_DIR / "accesseval_activations_multilayer.npz"
META_FILE = OUT_DIR / "accesseval_activations_meta.json"


# ---------------------------------------------------------------------------
# Pair construction WITH cluster identity
# ---------------------------------------------------------------------------

def build_pairs_with_metadata() -> Tuple[List[Dict], List[Dict]]:
    """Mirror build_accesseval_pairs() exactly, recording base-query identity.

    The emission order MUST match build_accesseval_pairs() byte for byte,
    because the cached loudness norms are positional. verify_pair_alignment()
    asserts this before any GPU work happens.
    """
    dataset = load_dataset(DATASET_NAME, split="train")
    domain_col = next((c for c in dataset.column_names if "domain" in c.lower()), None)

    pairs: List[Dict] = []
    meta: List[Dict] = []

    for row_idx, ex in enumerate(dataset):
        domain = ex.get(domain_col, "") if domain_col else ""
        if domain not in TARGET_DOMAINS and domain != "":
            continue
        neutral = ex.get("Neutral Query", "")
        disability = ex.get("Disability-Specific Query", "")
        if not neutral or not disability:
            continue

        # Identical placeholder test to the original (both cases checked).
        if "{disability}" in disability.lower() or "{Disability}" in disability:
            for cat in DISABILITY_CATEGORIES:
                pairs.append({
                    "clean_text": replace_placeholder(neutral, cat),
                    "corrupted_text": replace_placeholder(disability, cat),
                })
                meta.append({
                    "base_query_id": row_idx,
                    "category": cat,
                    "domain": domain,
                    "templated": True,
                })
        else:
            pairs.append({"clean_text": neutral, "corrupted_text": disability})
            meta.append({
                "base_query_id": row_idx,
                "category": "",
                "domain": domain,
                "templated": False,
            })

    return pairs, meta


def verify_pair_alignment(pairs: List[Dict]) -> None:
    """Fail loudly if our reconstruction diverges from the pipeline's."""
    print("  Verifying pair reconstruction against build_accesseval_pairs() ...")
    reference = build_accesseval_pairs()
    if len(reference) != len(pairs):
        raise RuntimeError(
            f"Pair count mismatch: reconstruction={len(pairs)}, "
            f"pipeline={len(reference)}. Cached loudness norms would misalign."
        )
    for i, (a, b) in enumerate(zip(pairs, reference)):
        if a["clean_text"] != b["clean_text"] or a["corrupted_text"] != b["corrupted_text"]:
            raise RuntimeError(
                f"Pair {i} differs from the pipeline's ordering. "
                "Downstream loudness/SVD alignment cannot be trusted."
            )
    print(f"  OK: {len(pairs)} pairs match the pipeline exactly.")


# ---------------------------------------------------------------------------
# Loudness (reuse the published norms when available)
# ---------------------------------------------------------------------------

def get_loudness_norms(pairs, model, tokenizer) -> np.ndarray:
    """Return the per-pair L2 norm of the L21 difference.

    Reuses the published checkpoint when present so the filter reproduces the
    paper exactly; otherwise recomputes with the same procedure.
    """
    if LOUDNESS_CKPT.exists():
        norms = np.load(LOUDNESS_CKPT)["norms"]
        if len(norms) != len(pairs):
            raise ValueError(
                f"Cached loudness norms ({len(norms)}) != pairs ({len(pairs)}). "
                f"Delete {LOUDNESS_CKPT} to recompute."
            )
        print(f"  [CKPT] Reusing published loudness norms from {LOUDNESS_CKPT.name}")
        return norms.astype(np.float32)

    print(f"  Computing loudness norms at L{TRIGGER_LAYER} for {len(pairs)} pairs ...")
    clean_texts = [format_prompt(p["clean_text"]) for p in pairs]
    corrupted_texts = [format_prompt(p["corrupted_text"]) for p in pairs]
    h_clean = extract_full_decoder(model, tokenizer, clean_texts, [TRIGGER_LAYER])
    h_corr = extract_full_decoder(model, tokenizer, corrupted_texts, [TRIGGER_LAYER])
    delta = (h_corr[TRIGGER_LAYER] - h_clean[TRIGGER_LAYER]).float()
    norms = delta.norm(dim=1).numpy().astype(np.float32)
    LOUDNESS_CKPT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(LOUDNESS_CKPT, norms=norms)
    print(f"  Saved loudness norms -> {LOUDNESS_CKPT}")
    return norms


# ---------------------------------------------------------------------------
# Spectrum preview (sanity + an immediate read on the confound)
# ---------------------------------------------------------------------------

def spectrum(clean: np.ndarray, corrupted: np.ndarray,
             var_threshold: float = 0.80) -> Tuple[float, int]:
    """EVR_1 and k@threshold of the mean-centred difference matrix.

    Matches compute_svd_and_probe(): delta = corrupted - clean, mean-centred,
    then SVD; k is the first index whose cumulative variance reaches threshold.
    """
    delta = (corrupted - clean).astype(np.float64)
    delta -= delta.mean(axis=0)
    s = np.linalg.svd(delta, full_matrices=False, compute_uv=False)
    total = np.sum(s ** 2)
    if total <= 0:
        return float("nan"), -1
    evr = (s ** 2) / total
    cumvar = np.cumsum(evr)
    k = int(np.searchsorted(cumvar, var_threshold) + 1)
    return float(evr[0]), k


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
                    help=f"decoder layers to persist (default: {DEFAULT_LAYERS})")
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if the output file already exists")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override EXTRACTION_BATCH_SIZE (lower this on OOM)")
    args = ap.parse_args()

    layers = sorted(set(args.layers))
    if any(l < 0 for l in layers):
        print("ERROR: layer indices must be >= 0", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists() and not args.force:
        print(f"{OUT_FILE} already exists. Re-run with --force to overwrite.")
        return 0

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if args.batch_size is not None:
        import evaluate_rtsd_fullres_generation as pipeline
        pipeline.EXTRACTION_BATCH_SIZE = args.batch_size
        print(f"  EXTRACTION_BATCH_SIZE overridden -> {args.batch_size}")

    print("=" * 72)
    print("DAY 1: PERSIST ACCESSEVAL ACTIVATIONS (multi-layer, cluster-aware)")
    print("=" * 72)

    # --- pairs + cluster ids (CPU only; verify BEFORE loading the model) -----
    print("\n== Building contrastive pairs with base-query ids ==")
    pairs, meta = build_pairs_with_metadata()
    verify_pair_alignment(pairs)

    base_query_id = np.array([m["base_query_id"] for m in meta], dtype=np.int32)
    category = np.array([m["category"] for m in meta])
    domain = np.array([m["domain"] for m in meta])
    templated = np.array([m["templated"] for m in meta], dtype=bool)

    n_pairs = len(pairs)
    n_clusters = len(np.unique(base_query_id))
    print(f"  Pairs: {n_pairs}   Base queries (clusters): {n_clusters}")
    print(f"  Mean variants per base query: {n_pairs / max(n_clusters, 1):.2f}")

    # --- model ---------------------------------------------------------------
    print("\n== Loading model ==")
    model, tokenizer = load_model()

    # extract_full_decoder indexes hidden_states[l+1], so l must be < n_layers.
    # Check now rather than failing mid-extraction.
    n_layers = getattr(model.config, "num_hidden_layers", None)
    if n_layers is not None and max(layers) >= n_layers:
        raise ValueError(
            f"Requested layer {max(layers)} but the model has {n_layers} "
            f"decoder layers (valid: 0..{n_layers - 1})."
        )

    # --- loudness + published filter mask ------------------------------------
    print("\n== Loudness norms ==")
    norms = get_loudness_norms(pairs, model, tokenizer)
    threshold = float(np.percentile(norms, 100 - SIGNAL_PERCENTILE))
    keep_mask = norms >= threshold
    print(f"  Filter: {n_pairs} -> {int(keep_mask.sum())} pairs "
          f"(threshold={threshold:.4f})")
    print(f"  Clusters surviving filter: {len(np.unique(base_query_id[keep_mask]))}")

    # --- activations ---------------------------------------------------------
    print(f"\n== Extracting activations at layers {layers} ==")
    clean_texts = [format_prompt(p["clean_text"]) for p in pairs]
    corrupted_texts = [format_prompt(p["corrupted_text"]) for p in pairs]

    print("  clean ...")
    h_clean = extract_full_decoder(model, tokenizer, clean_texts, layers)
    print("  corrupted ...")
    h_corr = extract_full_decoder(model, tokenizer, corrupted_texts, layers)

    payload: Dict[str, np.ndarray] = {}
    for l in layers:
        c = h_clean[l].numpy().astype(np.float32)
        x = h_corr[l].numpy().astype(np.float32)
        if c.shape[0] != n_pairs or x.shape[0] != n_pairs:
            raise RuntimeError(
                f"Layer {l}: extracted {c.shape[0]}/{x.shape[0]} rows, expected {n_pairs}"
            )
        payload[f"clean_L{l}"] = c
        payload[f"corrupted_L{l}"] = x

    payload.update({
        "base_query_id": base_query_id,
        "category": category,
        "domain": domain,
        "templated": templated,
        "loudness_norms": norms,
        "keep_mask": keep_mask,
        "layers": np.array(layers, dtype=np.int32),
    })

    print(f"\n== Saving -> {OUT_FILE} ==")
    np.savez(OUT_FILE, **payload)
    size_mb = OUT_FILE.stat().st_size / 1e6
    print(f"  Wrote {size_mb:.1f} MB")

    meta_blob = {
        "model_name": MODEL_NAME,
        "dataset": DATASET_NAME,
        "layers": layers,
        "layer_convention": "hidden_states[l+1]: residual stream after decoder layer l",
        "n_pairs": int(n_pairs),
        "n_clusters": int(n_clusters),
        "hidden_dim": int(payload[f"clean_L{layers[0]}"].shape[1]),
        "loudness_threshold": threshold,
        "n_kept": int(keep_mask.sum()),
        "signal_percentile": SIGNAL_PERCENTILE,
        "random_seed": RANDOM_SEED,
        "reused_published_loudness": bool(LOUDNESS_CKPT.exists()),
    }
    META_FILE.write_text(json.dumps(meta_blob, indent=2))
    print(f"  Wrote {META_FILE}")

    # --- preview: spectrum per layer on the published filtered pool ----------
    print("\n" + "=" * 72)
    print("PREVIEW: spectrum on the filtered pool (paper's SVD set)")
    print("=" * 72)
    print(f"  {'layer':>6}  {'EVR_1':>8}  {'k@80%':>6}")
    for l in layers:
        evr1, k = spectrum(payload[f"clean_L{l}"][keep_mask],
                           payload[f"corrupted_L{l}"][keep_mask])
        print(f"  {('L' + str(l)):>6}  {evr1:>7.1%}  {k:>6d}")
    print(f"\n  Paper reports EVR_1=10.2%, k@80%=40 at L{TRIGGER_LAYER}.")
    print("  A close match here confirms the extraction reproduces the pipeline.")
    print("\nNext: day3_subsample_spectrum.py (pure CPU, reads this .npz).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
