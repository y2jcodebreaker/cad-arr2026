#!/usr/bin/env python3
"""
judge_sweep_outputs.py

Score every generated response in the baseline sweep files with the paper's
Llama judge, so that the quality-versus-debiasing frontier can be plotted on
matched items.

WHY THIS SCRIPT EXISTS
----------------------
Each evaluate_*_baseline.py sweeps its own strength parameter and saves, for
every sweep point, the generated responses plus their medicalization scores.
That gives one axis of the frontier (debiasing). This script adds the other
axis (output quality) using the same judge and the same rubric that produced
the numbers in the paper, so points from different methods are comparable.

THREE DESIGN DECISIONS THAT AFFECT CORRECTNESS
---------------------------------------------
1. STRUCTURE-AGNOSTIC DISCOVERY. The runners nest their results differently:
   CAA is layers/L14/alpha_1.0, FairSteer adds a threshold level, Angular
   keys on strategy and mode, and others differ again. Rather than hardcode
   paths (which silently drops sweep points), this script walks the whole
   JSON tree and treats any dict containing a list of {'response': str} as a
   sweep point, labelled by its path.

2. CONTENT-HASH DEDUPLICATION. The same response text recurs across the
   corpus, most obviously at zero strength, where every method reproduces the
   unsteered baseline. Judging identical text twice under a sampling decoder
   would return two different scores, which injects noise into exactly the
   comparison the frontier depends on. Each unique text is judged once and
   the score reused, which makes identical outputs score identically by
   construction and cuts judging cost.

3. THE SOURCE FILES ARE NEVER MODIFIED. Scores are written to a separate
   file keyed by (file, path, index). A corrupted judge run can be deleted
   and re-run without touching the generations.

The rubric, the decoding parameters, the 2000-character cap and the 5.0
parse-failure default are copied verbatim from the judge used for the
paper's reported numbers (eval_all_methods_llama_judge.py). Do not "improve"
them here: scores must remain comparable to the ones already reported.

USAGE
    # verify parsing with no GPU and no model load (do this first)
    python judge_sweep_outputs.py --dry-run

    # score everything (resumable; safe to re-run after a crash)
    python judge_sweep_outputs.py

    # limit to specific files
    python judge_sweep_outputs.py --glob 'baseline_results/caa_results*.json'
"""

from __future__ import annotations

import argparse
import glob as globmod
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

# ---------------------------------------------------------------------------
# Judge specification: VERBATIM from the paper's judge. Do not edit.
# ---------------------------------------------------------------------------

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

USER_PREFIX = "Answer to an accessibility-related disability question:\n\n"
SCORE_RE = re.compile(r"Overall Score:\s*(\d+(?:\.\d+)?)/10", re.IGNORECASE)

RESPONSE_CHAR_CAP = 2000      # paper's cap
JUDGE_TEMPERATURE = 0.1       # paper's decode
JUDGE_MAX_NEW_TOKENS = 320    # score is emitted on the first line
PARSE_FAILURE_SCORE = 5.0     # paper's default for unparseable/empty
JUDGE_BATCH_SIZE = 16         # long prompts: large batches OOM on logits
RANDOM_SEED = 42

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# Baseline runners write nested results into baseline_results/; the CAD pipeline
# writes one file per sweep point into rtsd_fullres_checkpoints/. Both are judged.
# (glob has no brace expansion, so these are separate patterns.)
DEFAULT_GLOBS = [
    "baseline_results/*_results*.json",
    "rtsd_fullres_checkpoints/*_alpha*_responses.json",
]
OUT_FILE = Path("baseline_results") / "judge_scores.json"
CKPT_EVERY = 200              # unique texts between checkpoint writes


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _normalise(resp: list) -> List[dict] | None:
    """Return a list of {'response': str, ...} dicts, or None if not a sweep point.

    Two on-disk shapes exist and both must be handled:
      baseline runners  -> 'responses': [{'response': str, 'medicalization_score': float}, ...]
      CAD pipeline      -> 'responses': [str, ...]        (save_gen_ckpt)
    Accepting only the first shape would silently skip every CAD sweep point,
    which is the method the frontier is about.
    """
    if not isinstance(resp, list) or not resp:
        return None
    if isinstance(resp[0], dict) and isinstance(resp[0].get("response"), str):
        return resp
    if isinstance(resp[0], str):
        return [{"response": t} for t in resp]
    return None


def iter_response_blocks(node: Any, path: str = "") -> Iterator[Tuple[str, List[dict]]]:
    """Yield (path, responses) for every sweep point found anywhere in the tree.

    A sweep point is any dict holding a 'responses' value that normalises to a
    list of response records. Nesting depth and key names are not assumed,
    because they differ between runners.
    """
    if isinstance(node, dict):
        norm = _normalise(node.get("responses"))
        if norm is not None:
            yield path or "<root>", norm
        for k, v in node.items():
            if k == "responses":
                continue
            yield from iter_response_blocks(v, f"{path}/{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from iter_response_blocks(v, f"{path}[{i}]")


_STRENGTH_RE = re.compile(
    r"(?:alpha|angle|strength|coeff|coefficient|s|c)[_=]?(-?\d+(?:\.\d+)?)",
    re.IGNORECASE)


def parse_strength(path: str) -> float | None:
    """Best-effort strength value from the path, for plotting convenience.

    Analysis code should prefer the explicit path label; this is a hint only.
    """
    for part in reversed(path.split("/")):
        m = _STRENGTH_RE.fullmatch(part.strip()) or _STRENGTH_RE.search(part)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def text_key(text: str) -> str:
    """Hash of the exact judged text (after the cap), so identical inputs map
    to one score."""
    return hashlib.sha256(text[:RESPONSE_CHAR_CAP].encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def build_prompt(response_text: str, tokenizer=None, judge_model: str = MODEL_NAME) -> str:
    """The Llama judge's prompt is kept BYTE-IDENTICAL to the one that produced
    judge_scores.json. Any other judge model gets the same system and user text through its
    own chat template, since Llama's special tokens mean nothing to another family."""
    user = USER_PREFIX + response_text[:RESPONSE_CHAR_CAP]
    if judge_model != MODEL_NAME:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{JUDGE_SYSTEM}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def score_texts(model, tokenizer, texts: List[str], device: str,
                judge_model: str = MODEL_NAME, return_flags: bool = False):
    """Score a list of unique texts. Empty text bypasses the model (5.0).

    return_flags=True also returns, per text, whether the judge's output was actually parsed.
    A parse failure is scored 5.0, indistinguishable from a real 5, so the flag is the only
    way to measure a judge's failure rate rather than infer it."""
    import torch
    from tqdm import tqdm

    out: List[float] = [None] * len(texts)          # type: ignore[list-item]
    parsed: List[bool] = [False] * len(texts)
    todo = [i for i, t in enumerate(texts) if t.strip()]
    for i, t in enumerate(texts):
        if not t.strip():
            out[i] = PARSE_FAILURE_SCORE            # collapsed output

    tokenizer.padding_side = "left"
    torch.manual_seed(RANDOM_SEED)

    for b in tqdm(range(0, len(todo), JUDGE_BATCH_SIZE), desc="  judging"):
        idxs = todo[b:b + JUDGE_BATCH_SIZE]
        prompts = [build_prompt(texts[i], tokenizer, judge_model) for i in idxs]
        inp = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            gen = model.generate(
                **inp, max_new_tokens=JUDGE_MAX_NEW_TOKENS,
                do_sample=True, temperature=JUDGE_TEMPERATURE, top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        n_in = inp["input_ids"].shape[1]
        for j, i in enumerate(idxs):
            txt = tokenizer.decode(gen[j][n_in:], skip_special_tokens=True)
            m = SCORE_RE.search(txt)
            out[i] = (max(0.0, min(10.0, float(m.group(1))))
                      if m else PARSE_FAILURE_SCORE)
            parsed[i] = m is not None
    if return_flags:
        return out, parsed
    return out                                       # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    global JUDGE_BATCH_SIZE

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", nargs="+", default=DEFAULT_GLOBS,
                    help="one or more glob patterns for result files")
    ap.add_argument("--out", default=str(OUT_FILE))
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be judged; no model load, no GPU")
    ap.add_argument("--batch-size", type=int, default=JUDGE_BATCH_SIZE)
    ap.add_argument("--judge_model", default=MODEL_NAME,
                    help="judge model (default: the Llama judge that produced judge_scores.json)")
    ap.add_argument("--limit", type=int, default=None,
                    help="judge at most N unique texts (smoke test)")
    args = ap.parse_args()

    JUDGE_BATCH_SIZE = args.batch_size

    patterns = [args.glob] if isinstance(args.glob, str) else list(args.glob)
    files = sorted({f for pat in patterns for f in globmod.glob(pat)})
    if not files:
        print(f"ERROR: no files matched {patterns!r}", file=sys.stderr)
        print("Run the evaluate_*_baseline.py sweeps first.", file=sys.stderr)
        return 2

    print("=" * 74)
    print("JUDGE SWEEP OUTPUTS")
    print("=" * 74)
    print(f"Files matched: {len(files)}")

    # ---- discovery -------------------------------------------------------
    # index: unique text hash -> text ; and a flat record list for output
    unique: Dict[str, str] = {}
    records: List[dict] = []
    per_file: Dict[str, int] = {}

    for fp in files:
        try:
            data = json.loads(Path(fp).read_text())
        except Exception as e:
            print(f"  WARNING: could not read {fp}: {e}")
            continue
        n_here = 0
        for path, resp in iter_response_blocks(data):
            for i, item in enumerate(resp):
                txt = item.get("response", "")
                h = text_key(txt)
                unique.setdefault(h, txt)
                # Strength lives in the JSON path for the baseline runners and
                # in the filename for the CAD pipeline (proj_alpha1.0_...json).
                strength = parse_strength(path)
                if strength is None:
                    strength = parse_strength(Path(fp).stem)
                records.append({
                    # "file" is the basename, kept for the analysis scripts that group on it;
                    # "source" is the path, because E1 has arms in different directories with
                    # the same basename (cad_heldout vs cad_full, caa_heldout vs caa_full)
                    "file": Path(fp).name,
                    "source": str(fp),
                    "path": path,
                    "index": i,
                    "hash": h,
                    "strength": strength,
                    "medicalization_score": item.get("medicalization_score"),
                })
                n_here += 1
        per_file[Path(fp).name] = n_here

    total = len(records)
    print(f"\nSweep points and responses found:")
    for name, n in sorted(per_file.items()):
        print(f"  {name:<44} {n:>6} responses")
    print(f"\n  total responses     : {total}")
    print(f"  unique texts        : {len(unique)}")
    if total:
        saved = 100.0 * (1 - len(unique) / total)
        print(f"  dedup saving        : {saved:.1f}% fewer judge calls")

    if total == 0:
        print("\nNothing to judge.")
        return 1

    if args.dry_run:
        print("\n-- dry run: showing a few discovered sweep points --")
        seen = set()
        for r in records:
            k = (r["file"], r["path"])
            if k in seen:
                continue
            seen.add(k)
            print(f"   {r['file']:<34} {r['path']:<44} strength={r['strength']}")
            if len(seen) >= 25:
                print("   ...")
                break
        print("\nDry run complete. No model was loaded.")
        return 0

    # ---- resume ----------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores: Dict[str, float] = {}
    parsed: Dict[str, bool] = {}
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
        except Exception as e:
            print(f"\n[resume] ignoring unreadable {out_path.name}: {e}")
            prev = {}
        if prev and prev.get("model") != args.judge_model:
            raise SystemExit(f"{out_path} holds scores from {prev.get('model')}, not "
                             f"{args.judge_model}. Resuming would mix judges; use a different --out.")
        scores = {k: float(v) for k, v in prev.get("scores_by_hash", {}).items()}
        parsed = {k: bool(v) for k, v in prev.get("parsed_by_hash", {}).items()}
        if scores:
            print(f"\n[resume] loaded {len(scores)} scores from {out_path.name}")

    pending = [h for h in unique if h not in scores]
    if args.limit:
        pending = pending[:args.limit]
    print(f"  to judge now        : {len(pending)}")

    if pending:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nLoading judge model on {device} ...")
        tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.judge_model, torch_dtype=torch.bfloat16, device_map="auto",
                attn_implementation="flash_attention_2")
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                args.judge_model, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()

        t0 = time.time()
        for start in range(0, len(pending), CKPT_EVERY):
            chunk = pending[start:start + CKPT_EVERY]
            vals, flags = score_texts(model, tokenizer, [unique[h] for h in chunk], device,
                                      judge_model=args.judge_model, return_flags=True)
            scores.update(dict(zip(chunk, vals)))
            parsed.update(dict(zip(chunk, flags)))
            out_path.write_text(json.dumps({
                "model": args.judge_model,
                "rubric": "eval_all_methods_llama_judge.py (verbatim)",
                "temperature": JUDGE_TEMPERATURE,
                "response_char_cap": RESPONSE_CHAR_CAP,
                "parse_failure_score": PARSE_FAILURE_SCORE,
                "seed": RANDOM_SEED,
                "scores_by_hash": scores,
                "parsed_by_hash": parsed,
            }, indent=2))
            done = min(start + CKPT_EVERY, len(pending))
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(pending) - done) / max(rate, 1e-9) / 60
            print(f"  [ckpt] {done}/{len(pending)} unique judged, ETA {eta:.0f} min")

    # ---- attach scores to records and write ------------------------------
    missing = 0
    for r in records:
        s = scores.get(r["hash"])
        if s is None:
            missing += 1
        r["judge_score"] = s

    final = {
        "model": args.judge_model,
        "rubric": "eval_all_methods_llama_judge.py (verbatim)",
        "temperature": JUDGE_TEMPERATURE,
        "response_char_cap": RESPONSE_CHAR_CAP,
        "parse_failure_score": PARSE_FAILURE_SCORE,
        "seed": RANDOM_SEED,
        "n_responses": total,
        "n_unique": len(unique),
        "scores_by_hash": scores,
        "parsed_by_hash": parsed,
        "parse_failure_rate": (sum(1 for v in parsed.values() if not v) / len(parsed)) if parsed else None,
        "records": records,
    }
    out_path.write_text(json.dumps(final, indent=2))
    print(f"\nWrote {out_path}  ({total} records, {len(scores)} unique scores)")
    if missing:
        print(f"  NOTE: {missing} records still unscored (re-run to finish).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
