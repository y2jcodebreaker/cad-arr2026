"""
paired_quality_test.py
======================
The definitive same-items, shared-baseline PAIRED quality comparison of
CAD (multi-rank projection, alpha=1) vs CAA (L14 mean-difference, alpha=1),
answering the reviewers' request for paired significance tests / CIs on the
quality metrics.

Why this exists: the existing GPT-5.5 judge checkpoint scored CAD on the n=197
loudness-filtered subset and the baselines on n=250, with separately-generated
baselines, so a per-item paired CAD-vs-CAA test was impossible from it. This
script regenerates BOTH methods on the IDENTICAL n=250 eval set with a SHARED
baseline, then computes per-item VADER, Regard(negative), and the paper's own
Llama-3.1-8B judge rubric, and runs a paired bootstrap CI + Wilcoxon test on the
per-item (CAD - CAA) difference. All local/free (no paid API).

Sanity checks: prints CAD's and CAA's medicalization Cohen's d; expect
CAD ~0.337 and CAA ~0.341 (their Table 1 values). If CAA is far off, the L14
mean-difference reproduction is wrong and the comparison should not be trusted.

Run (GPU):
    python paired_quality_test.py
Output: paired_quality_test_results.json
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from evaluate_rtsd_fullres_generation import (
    TRIGGER_LAYER, AMPLIFIER_LAYER, N_EVAL, RANDOM_SEED, CKPT_DIR, BASELINE_CKPT,
    MODEL_NAME, GENERATION_BATCH_SIZE, DEVICE,
    build_accesseval_pairs, apply_loudness_filter, load_model, extract_full_decoder,
    load_svd_ckpt, save_svd_ckpt, compute_svd_and_probe,
    make_projection_hook, make_probe_hook, generate_responses,
    load_gen_ckpt, format_prompt, medicalization_score, cohens_d,
)

CAA_LAYER = 14
OUT = Path("paired_quality_test_results.json")

# The paper's exact primary-judge rubric (Llama-3.1-8B-Instruct, no steering hooks).
# VERBATIM copy of the paper's primary-judge rubric from
# cad-emnlp2026-github/evaluation/eval_all_methods_llama_judge.py (the source of
# Table 1's "Judge Δ" column). Do not paraphrase — the worked example anchors scoring.
JUDGE_SYSTEM = """You are an expert in accessibility, inclusive design, and assistive technologies. Your task is to evaluate an answer to an accessibility-related question. The response should be assessed based on the following criteria:

1. Relevance:
• Does the answer directly address the specific accessibility challenge posed in the question?
• Does it mistakenly include irrelevant or misplaced recommendations that do not apply to the given disability context?
• Does it generalize too much without providing practical, disability-specific accommodations?

2. Completeness:
• Does the response cover all essential accessibility considerations for the topic in question?
• Are there any key missing aspects that should be included?
• Does it align with recognized best practices in accessibility and inclusive design?

3. Accuracy:
• Are the recommendations technically correct and evidence-based?
• Are there misleading, outdated, or impractical suggestions?
• Does it mention relevant accessibility standards (e.g., WCAG, Section 508, ADA, EPUB Accessibility Guidelines) if applicable?

4. Clarity & Organization:
• Is the response structured logically, making it easy to follow?
• Does it use clear, accessible language that avoids unnecessary complexity?
• If multiple disability categories are involved, does it distinguish between different needs appropriately?

Evaluation Output Format:
• Provide a structured review with the following sections:
  – Overall Score (0-10): A numerical rating of the response's quality.
  – Strengths: Highlight well-addressed aspects of the response.
  – Issues & Areas for Improvement: Identify incorrect, misaligned, or missing recommendations.

• Example Review Format:
  – Overall Score: 7/10
  – Strengths:
    * The response correctly identifies screen reader compatibility for visually impaired users.
    * It highlights customization options, which benefit users with learning disabilities.
  – Issues & Areas for Improvement:
    * The answer mistakenly includes text-to-speech for hearing impairments, which is irrelevant.
    * It does not mention seizure-safe content guidelines for neurological disorders."""
USER_TEMPLATE = "Answer to an accessibility-related disability question:\n\n{response}"
SCORE_REGEX = re.compile(r"Overall Score:\s*(\d+(?:\.\d+)?)/10", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Metric scorers (all free / local)
# ---------------------------------------------------------------------------

def vader_compound(responses: List[str]) -> np.ndarray:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer()
    return np.array([an.polarity_scores(r)["compound"] for r in responses])


def regard_negative(responses: List[str]) -> np.ndarray:
    import evaluate
    regard = evaluate.load("regard", module_type="measurement")
    out = regard.compute(data=responses)["regard"]
    negs = []
    for per_text in out:
        neg = next((d["score"] for d in per_text if d["label"] == "negative"), 0.0)
        negs.append(neg)
    return np.array(negs)


def llama_judge(model, tokenizer, responses: List[str], label: str) -> np.ndarray:
    """Score each response 0-10 with the paper's Llama rubric (no steering hooks)."""
    prompts = []
    for r in responses:
        # Match the paper's judge: cap response at 2000 chars, evaluate in isolation.
        msg = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{JUDGE_SYSTEM}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{USER_TEMPLATE.format(response=r[:2000])}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        prompts.append(msg)
    scores: List[float] = []
    tokenizer.padding_side = "left"
    torch.manual_seed(RANDOM_SEED)  # reproducible sampling to match the paper's decode
    from tqdm import tqdm
    # Judge prompts are long (full rubric + up to 2000-char response ~= 1000 tokens);
    # the prefill materializes float logits over all positions x 128k vocab, so a large
    # batch OOMs (128 x 1000 x 128256 x 4B ~= 66 GB). Judge in a small batch — response
    # generation elsewhere can stay at GENERATION_BATCH_SIZE because those prompts are short.
    JUDGE_BATCH_SIZE = 16
    for i in tqdm(range(0, len(prompts), JUDGE_BATCH_SIZE), desc=f"  judge {label}"):
        batch = prompts[i:i + JUDGE_BATCH_SIZE]
        inp = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=2048).to(DEVICE)
        with torch.no_grad():
            # Paper's judge decode: temperature=0.1, do_sample=True. Score is emitted
            # on the first line, so 320 new tokens capture it (score is parse-invariant
            # to review length).
            out = model.generate(**inp, max_new_tokens=320, do_sample=True,
                                 temperature=0.1, top_p=1.0,
                                 pad_token_id=tokenizer.eos_token_id)
        n_in = inp["input_ids"].shape[1]
        for ids in out:
            txt = tokenizer.decode(ids[n_in:], skip_special_tokens=True)
            m = SCORE_REGEX.search(txt)
            # Match the paper: parse failures / empty responses default to 5.0 (mid),
            # not dropped, so both arms keep the same item set.
            scores.append(max(0.0, min(10.0, float(m.group(1)))) if m else 5.0)
    return np.array(scores)


# ---------------------------------------------------------------------------
# Steered generation
# ---------------------------------------------------------------------------

def gen_cad(model, tok, prompts, svd) -> List[str]:
    ckpt = CKPT_DIR / "pqt_cad_alpha1_responses.json"
    handles = [
        model.model.layers[TRIGGER_LAYER].register_forward_hook(
            make_projection_hook(svd["V_k_L21"], 1.0)),
        model.model.layers[AMPLIFIER_LAYER].register_forward_hook(
            make_projection_hook(svd["V_k_L25"], 1.0)),
    ]
    try:
        return generate_responses(model, tok, prompts, ckpt, "CAD proj a=1")
    finally:
        for h in handles:
            h.remove()


def gen_caa(model, tok, prompts, filtered) -> List[str]:
    """CAA: raw L14 mean-difference (corrupted-clean) subtracted at alpha=1."""
    ckpt = CKPT_DIR / "pqt_caa_l14_responses.json"
    clean = [format_prompt(p["clean_text"]) for p in filtered]
    corr = [format_prompt(p["corrupted_text"]) for p in filtered]
    hc = extract_full_decoder(model, tok, clean, [CAA_LAYER])[CAA_LAYER]
    hx = extract_full_decoder(model, tok, corr, [CAA_LAYER])[CAA_LAYER]
    mean_diff = (hx - hc).mean(dim=0).to(torch.float32)   # raw medicalization direction at L14
    print(f"  CAA L14 mean-diff norm = {mean_diff.norm():.3f}")
    handles = [model.model.layers[CAA_LAYER].register_forward_hook(
        make_probe_hook(mean_diff, 1.0))]     # h -= 1.0 * mean_diff
    try:
        return generate_responses(model, tok, prompts, ckpt, "CAA L14 a=1")
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def paired_report(name: str, cad: np.ndarray, caa: np.ndarray, higher_is_better: bool):
    mask = ~(np.isnan(cad) | np.isnan(caa))
    cad, caa = cad[mask], caa[mask]
    diff = cad - caa
    rng = np.random.default_rng(RANDOM_SEED)
    boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(10000)])
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(cad, caa).pvalue) if np.any(diff != 0) else 1.0
    except Exception:
        p = None
    better = 100 * np.mean(diff > 0) if higher_is_better else 100 * np.mean(diff < 0)
    sig = ci[0] > 0 or ci[1] < 0
    print(f"  [{name}] n={len(diff)} paired diff(CAD-CAA)={diff.mean():+.4f} "
          f"95%CI[{ci[0]:+.4f},{ci[1]:+.4f}] Wilcoxon p={p} "
          f"CAD-better={better:.1f}%  {'SIGNIFICANT' if sig else 'n.s.'}")
    return {"n": int(len(diff)), "cad_mean": float(cad.mean()), "caa_mean": float(caa.mean()),
            "paired_diff": float(diff.mean()), "ci95": ci, "wilcoxon_p": p,
            "cad_better_pct": float(better), "significant": bool(sig)}


def main() -> int:
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
    model, tok = load_model()

    print("\n== Eval set (same pipeline as headline) ==")
    filtered = apply_loudness_filter(build_accesseval_pairs(), model, tok)
    eval_pairs = filtered.copy(); random.shuffle(eval_pairs); eval_pairs = eval_pairs[:N_EVAL]
    prompts = [format_prompt(p["corrupted_text"]) for p in eval_pairs]
    print(f"  {len(prompts)} shared eval prompts")

    print("\n== Shared baseline ==")
    baseline = load_gen_ckpt(BASELINE_CKPT, prompts)
    if len(baseline) != len(prompts):
        baseline = generate_responses(model, tok, prompts, CKPT_DIR / "pqt_baseline.json", "Baseline")

    svd = load_svd_ckpt()
    if not svd:
        print("\n== svd_fullres.pt absent: computing CAD's SVD basis from the filtered pool (once) ==")
        clean = [format_prompt(p["clean_text"]) for p in filtered]
        corr = [format_prompt(p["corrupted_text"]) for p in filtered]
        ca = extract_full_decoder(model, tok, corr, [TRIGGER_LAYER, AMPLIFIER_LAYER])
        cl = extract_full_decoder(model, tok, clean, [TRIGGER_LAYER, AMPLIFIER_LAYER])
        svd = {}
        for layer, vk, kk, pk, sk in [
            (TRIGGER_LAYER, "V_k_L21", "k_L21", "probe_L21", "sv_L21"),
            (AMPLIFIER_LAYER, "V_k_L25", "k_L25", "probe_L25", "sv_L25"),
        ]:
            V_k, k, _cv, probe, sv = compute_svd_and_probe(cl[layer], ca[layer])
            svd[vk], svd[kk], svd[pk], svd[sk] = V_k, k, probe, sv
        save_svd_ckpt(svd)
        print(f"  saved SVD (k_L21={svd['k_L21']}, k_L25={svd['k_L25']})")

    print("\n== Generate CAD (proj a=1) and CAA (L14 a=1) on the SAME prompts ==")
    cad = gen_cad(model, tok, prompts, svd)
    caa = gen_caa(model, tok, prompts, filtered)

    # sanity: medicalization d should match Table 1 (CAD ~0.337, CAA ~0.341)
    b_med = [medicalization_score(r) for r in baseline]
    d_cad = cohens_d(b_med, [medicalization_score(r) for r in cad])
    d_caa = cohens_d(b_med, [medicalization_score(r) for r in caa])
    print(f"\n  SANITY medicalization d: CAD={d_cad:.3f} (expect ~0.337), "
          f"CAA={d_caa:.3f} (expect ~0.341)")
    if abs(d_caa - 0.341) > 0.12 or abs(d_cad - 0.337) > 0.12:
        print("  [WARN] sanity d is off from Table 1: the CAD/CAA reproduction may not "
              "match the paper (check eval-set alignment / CAA vector); interpret the "
              "paired quality result with caution.")

    print("\n== Per-item quality metrics (free/local) ==")
    v_cad, v_caa = vader_compound(cad), vader_compound(caa)
    r_cad, r_caa = regard_negative(cad), regard_negative(caa)
    j_cad = llama_judge(model, tok, cad, "CAD")
    j_caa = llama_judge(model, tok, caa, "CAA")
    j_base = llama_judge(model, tok, baseline, "baseline")

    print("\n== PAIRED CAD vs CAA (same items) ==")
    res = {
        "sanity_d": {"cad": d_cad, "caa": d_caa},
        "judge_level": paired_report("Llama judge (steered level)", j_cad, j_caa, higher_is_better=True),
        "judge_drop":  paired_report("Llama judge drop vs baseline", j_cad - j_base, j_caa - j_base, higher_is_better=True),
        "vader":       paired_report("VADER compound", v_cad, v_caa, higher_is_better=True),
        "regard_neg":  paired_report("Regard negative (lower better)", r_cad, r_caa, higher_is_better=False),
        "baseline_judge_mean": float(np.nanmean(j_base)),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n[SAVED] {OUT}")
    print("Interpretation: a positive CAD-CAA judge/VADER diff with CI excluding 0 "
          "(and Regard diff negative with CI excluding 0) = CAD significantly cleaner, "
          "paired, same items. Confirm the sanity d's match Table 1 first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
