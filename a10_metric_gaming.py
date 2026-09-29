"""A10: bias-blind edits against the lexical debiasing score (claims/PREREG_A10_metric_gaming.md, frozen 7da220b).

    python a10_metric_gaming.py edits      # CPU: game_append, game_sub from unsteered_s42
    python a10_metric_gaming.py ban        # GPU: unsteered generation with 9 medical-list words banned
    add --dry_run to either: prints what would be written

Outputs go to e1_outputs/a10 (seeded from results/e1 on the pod). game_ban runs through the very same
rt.generate_responses as unsteered_s42; only model.generate is wrapped to pass bad_words_ids.
"""
from __future__ import annotations

import argparse
import functools
import json
import re
import sys
from pathlib import Path

sys_argv, sys.argv = sys.argv, sys.argv[:1]
import a8_framing_contrast as a8                  # noqa: E402  (seed_all)
import e1_common as e1                            # noqa: E402
import evaluate_rtsd_fullres_generation as rt     # noqa: E402
import frozen_eval as fe                          # noqa: E402
import m1_arms                                    # noqa: E402
sys.argv = sys_argv

OUT = Path("e1_outputs/a10")
BASE = Path("e1_outputs")
APPEND = ("Thinking ahead in this way can also support your career growth, your performance, "
          "and a clear strategy for your business.")
SUBS = {"consult": "speak with", "doctor": "healthcare provider", "physician": "medical professional",
        "limitations": "difficulties", "condition": "diagnosis", "risk": "danger", "treatment": "therapy",
        "severe": "serious", "safety": "protection"}           # "impairments" deliberately absent
assert set(SUBS) == set(rt.MEDICAL_WORDS) - {"impairments"}


def append(text: str) -> str:
    """A10-A1: S goes in as its own paragraph after the answer's first paragraph, inside the 2,000
    characters every judge reads (appended at the end it would be cut off for all 250 answers)."""
    i = text.find("\n\n")
    if i < 0:
        raise ValueError("answer has no paragraph break")
    return text[:i] + "\n\n" + APPEND + text[i:]


def substitute(text: str) -> str:
    """Whole-word, case-insensitive; the replacement copies the capitalisation of the first letter."""
    def one(m):
        rep = SUBS[m.group(0).lower()]
        return rep[0].upper() + rep[1:] if m.group(0)[0].isupper() else rep
    return re.sub(r"\b(" + "|".join(SUBS) + r")\b", one, text, flags=re.IGNORECASE)


def banned_ids(tok) -> list[list[int]]:
    seqs = set()
    for w in SUBS:
        for form in (w, w.capitalize()):
            for s in (form, " " + form):
                seqs.add(tuple(tok(s, add_special_tokens=False)["input_ids"]))
    return [list(s) for s in sorted(seqs)]


def write(arm: str, texts: list[str]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{arm}_responses.json"
    p.write_text(json.dumps({"responses": texts, "derived_from": "unsteered_s42"}))
    return p


def cmd_edits(args) -> int:
    base = m1_arms.load(BASE, required=False)["unsteered_s42"]["texts"]
    arms = {"game_append": [append(t) for t in base], "game_sub": [substitute(t) for t in base]}
    for a, texts in arms.items():
        med0 = sum(rt.medicalization_score(t) for t in base) / len(base)
        med1 = sum(rt.medicalization_score(t) for t in texts) / len(texts)
        print(f"{a}: {len(texts)} texts, mean lexical score {med0:.3f} -> {med1:.3f}")
        if not args.dry_run:
            print("  wrote", write(a, texts))
    return 0


def cmd_ban(args) -> int:
    fz = fe.load(Path(args.eval_set))
    prompts = [rt.format_prompt(p["corrupted_text"]) for p in fe.eval_pairs(fz)]
    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(rt.MODEL_NAME)
        ids = banned_ids(tok)
        print(f"{len(ids)} banned sequences, e.g. {[tok.decode(s) for s in ids[:6]]}; {len(prompts)} prompts")
        print(f"DRY RUN  git={str(e1.GIT_AT_IMPORT['commit'])[:12]}  dirty={e1.GIT_AT_IMPORT['dirty_tracked_files'] or 'none'}")
        return 0
    import a9_robustness as a9
    with e1.RunRecord(OUT, "a10_ban", args, fz, name="run_record_game_ban") as rec:
        a8.seed_all()
        model, tok = rt.load_model()
        ids = banned_ids(tok)
        orig = model.generate
        model.generate = functools.partial(orig, bad_words_ids=ids)
        try:
            resp = rt.generate_responses(model, tok, prompts, OUT / "game_ban_responses.json", "game_ban")
        finally:
            model.generate = orig
        rec.result(n=len(resp), n_banned_sequences=len(ids), env=a9.env(),
                   lexical_mean=sum(rt.medicalization_score(r) for r in resp) / len(resp))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["edits", "ban"])
    e1.add_e1_args(ap)
    args = ap.parse_args()
    args.eval_set = args.eval_set or "frozen_eval_v1.json"
    args.out_dir = str(OUT)
    args.seed = a8.SEED
    return {"edits": cmd_edits, "ban": cmd_ban}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
