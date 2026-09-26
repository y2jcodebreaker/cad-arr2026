#!/usr/bin/env python3
"""
EXP 3 (AAAI-27): Multi-rank (k=40) LEACE vs rank-1 LEACE vs CAD
==============================================================

Reviewer concern this defuses
-----------------------------
"Your LEACE comparison conflates rank-1-vs-rank-40 with
whitened-erasure-vs-unwhitened-projection. A k=40 LEACE row isolates the rank."

What it does
------------
Rank-1 LEACE (the existing baseline) erases ONE concept direction with
concept-aware whitening. This script erases the top-k CONTRASTIVE SVD subspace
(the same V_k that CAD projects out) but WITH LEACE's whitening. Comparing:

    CAD (project out V_k, no whitening)   vs   rank-k LEACE (erase V_k, whitened)

isolates the effect of whitening from the effect of rank. Both share the
identical k=40 subspace at L21+L25, the same 2082-pair training pool, and the
same n=250 / n=197 evaluation as every other method.

Run
---
    pip install concept-erasure
    python exp3_leace_rank_k.py --k 40
Outputs: exp3_leace_rank_k_results.json     (~2-3 h on A100)

Depends on evaluate_leace_baseline.py (same directory) for the shared pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Reuse the EXACT pipeline the rank-1 LEACE baseline uses, so every number is
# comparable bit-for-bit (same pairs, same scoring, same generation).
from evaluate_leace_baseline import (
    SCRIPT_DIR, MODEL_NAME, TRIGGER_LAYER, AMPLIFIER_LAYER, N_EVAL, RANDOM_SEED,
    build_accesseval_pairs, load_model, load_loudness_filter, select_eval_subset,
    extract_layer_output, generate_with_leace, medicalization_score,
    cohens_d, bootstrap_d_ci, is_degenerate, n_empty, format_prompt, hash_prompts,
    load_responses, save_responses,
)

OUT = SCRIPT_DIR / "exp3_leace_rank_k_results.json"
CKPT_DIR = SCRIPT_DIR / "exp3_ckpt"
CKPT_DIR.mkdir(exist_ok=True)


def top_k_svd_basis(h_clean: np.ndarray, h_corr: np.ndarray, k: int) -> np.ndarray:
    """Top-k right singular vectors of the centered contrastive differences.
    Returns V_k of shape (d, k) — the SAME subspace CAD projects out."""
    M = (h_corr - h_clean).astype(np.float64)
    M = M - M.mean(axis=0, keepdims=True)
    # economy SVD; Vt is (min(n,d), d)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[:k].T  # (d, k)


def fit_rank_k_leace(activations: np.ndarray, V_k: np.ndarray):
    """Fit a LEACE eraser whose concept Z is the coordinates of each activation
    in the top-k contrastive subspace V_k. This erases the k-dim subspace with
    LEACE's covariance-aware whitening (rank-k generalization of the rank-1
    baseline). Returns a LeaceEraser object usable by generate_with_leace().
    """
    from concept_erasure import LeaceFitter
    X = torch.from_numpy(activations).to(torch.float32)          # (n, d)
    Vk = torch.from_numpy(V_k).to(torch.float32)                 # (d, k)
    Xc = X - X.mean(dim=0, keepdim=True)
    Z = Xc @ Vk                                                  # (n, k) concept coords
    print(f"    rank-k LEACE: X={tuple(X.shape)}, Z={tuple(Z.shape)} (k={Vk.shape[1]})")
    fitter = LeaceFitter(X.shape[1], Z.shape[1])
    fitter.update(X, Z)
    return fitter.eraser


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=40, help="erasure subspace rank")
    args = ap.parse_args()
    k = args.k

    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    print("=" * 72)
    print(f"EXP 3: rank-{k} LEACE at L21+L25 vs CAD (AccessEval, Llama-3.1-8B)")
    print("=" * 72)

    pairs = build_accesseval_pairs()
    model, tok = load_model()
    filtered = load_loudness_filter(pairs, model, tok)
    eval_pairs = select_eval_subset(filtered, n=N_EVAL)
    steer_prompts = [format_prompt(p["corrupted_text"]) for p in eval_pairs]
    phash = hash_prompts(steer_prompts)

    # --- Activations on the full filtered pool at L21 and L25 (clean+corrupted)
    act_ckpt = CKPT_DIR / "acts_L21_L25.pt"
    if act_ckpt.exists():
        d = torch.load(act_ckpt, map_location="cpu", weights_only=False)
        h21c, h21x, h25c, h25x = d["h21c"], d["h21x"], d["h25c"], d["h25x"]
    else:
        clean_txt = [format_prompt(p["clean_text"]) for p in filtered]
        corr_txt = [format_prompt(p["corrupted_text"]) for p in filtered]
        hc = extract_layer_output(model, tok, clean_txt, [TRIGGER_LAYER, AMPLIFIER_LAYER])
        hx = extract_layer_output(model, tok, corr_txt, [TRIGGER_LAYER, AMPLIFIER_LAYER])
        h21c, h21x = hc[TRIGGER_LAYER], hx[TRIGGER_LAYER]
        h25c, h25x = hc[AMPLIFIER_LAYER], hx[AMPLIFIER_LAYER]
        torch.save({"h21c": h21c, "h21x": h21x, "h25c": h25c, "h25x": h25x}, act_ckpt)

    # --- Fit rank-k LEACE at each layer on the k-dim contrastive subspace
    V21 = top_k_svd_basis(h21c, h21x, k)
    V25 = top_k_svd_basis(h25c, h25x, k)
    X21 = np.concatenate([h21c, h21x], axis=0)
    X25 = np.concatenate([h25c, h25x], axis=0)
    print(f"  Fitting rank-{k} LEACE @ L21..."); er21 = fit_rank_k_leace(X21, V21)
    print(f"  Fitting rank-{k} LEACE @ L25..."); er25 = fit_rank_k_leace(X25, V25)

    # --- Generate baseline + rank-k-LEACE (dual layer, matches CAD pincer)
    base = load_responses(CKPT_DIR / "base.json", phash)
    if base is None:
        base = generate_with_leace(model, tok, steer_prompts, leace_erasers={})
        save_responses(CKPT_DIR / "base.json", base, phash)
    dual = load_responses(CKPT_DIR / f"dual_k{k}.json", phash)
    if dual is None:
        dual = generate_with_leace(model, tok, steer_prompts,
                                   leace_erasers={TRIGGER_LAYER: er21, AMPLIFIER_LAYER: er25})
        save_responses(CKPT_DIR / f"dual_k{k}.json", dual, phash)

    bs = [medicalization_score(r) for r in base]
    ds = [medicalization_score(r) for r in dual]
    d = cohens_d(bs, ds); ci = bootstrap_d_ci(bs, ds)
    bm, sm = float(np.mean(bs)), float(np.mean(ds))
    red = 100.0 * (bm - sm) / abs(bm) if bm else float("nan")
    print("\n" + "=" * 72)
    print(f"rank-{k} LEACE (L21+L25):  d = {d:+.4f}  CI [{ci[0]:+.4f},{ci[1]:+.4f}]")
    print(f"  Delta Med% = {red:+.1f}%   n_empty={n_empty(dual)}   degenerate={is_degenerate(dual)}")
    print(f"  Compare: rank-1 LEACE d=-0.03 (NS); CAD projection d=0.518 (n=197).")

    json.dump({
        "k": k, "model": MODEL_NAME, "n_eval": N_EVAL, "n_pool": len(filtered),
        "cohens_d": d, "ci95": list(ci), "delta_med_pct": red,
        "baseline_mean": bm, "steered_mean": sm,
        "n_empty": n_empty(dual), "degenerate": is_degenerate(dual),
        "interpretation": (
            "rank-k LEACE erases the same top-k contrastive subspace as CAD but "
            "with covariance whitening; comparing to CAD's unwhitened projection "
            "isolates rank from whitening."),
    }, open(OUT, "w"), indent=2)
    print(f"[SAVED] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
