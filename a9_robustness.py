"""A9: robustness of A8 (claims/PREREG_A9_robustness.md, frozen 281249f, amendment A9-A1).

Reads A8's fit (e1_outputs/a8: fit_items.json, the fit judges, a8_directions.pt), which on a fresh
pod is seeded from the committed results/e1. Outputs go to e1_outputs/a9.

    python a9_robustness.py fit_neutral_generate    # unsteered answers to the 96 fit questions' neutral forms
    python a9_robustness.py fit                     # GA9-0, shuffled and identity directions
    python a9_robustness.py generate [--arms ...]   # three reruns, five frozen-set arms, four fresh arms
    python a9_robustness.py ppl                     # WikiText-2 perplexity
    add --dry_run to any of them: no model, prints what would run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys_argv, sys.argv = sys.argv, sys.argv[:1]
import a8_framing_contrast as a8                  # noqa: E402
import e1_common as e1                            # noqa: E402
import evaluate_rtsd_fullres_generation as rt     # noqa: E402
import frozen_eval as fe                          # noqa: E402
sys.argv = sys_argv

OUT = Path("e1_outputs/a9")
A8OUT = a8.OUT
FRESH = Path("fresh_eval_a9.json")
FRESH_SHA = "2fd13f1ad94403dc8a3f153301b894c0c9591e1362e86f871802c04ab7de11bb"
NEUTRAL_GEN = OUT / "fit_neutral_gen.json"        # not *_responses.json: kept out of the judge globs
NEUTRAL_ITEMS = OUT / "fit_neutral_items.json"
DIRS = OUT / "a9_directions.pt"
REPORT = OUT / "a9_fit_report.json"
LAYERS = [9, 11]
ALPHA = 1.0
MIN_COS = 0.999                                   # GA9-0
SHUFFLE_SEEDS = (0, 1, 2)
PPL_WINDOW, PPL_BATCH = 1024, 8

# arm -> (direction, question side, system prompt, eval set)
ARMS = {
    # A9-A5: same-environment references, generated first; compared text-for-text with the committed arms
    "unsteered_rerun": (None, "disability", "default", "frozen"),
    "neutral_rerun": (None, "neutral", "default", "frozen"),
    "fc_remove_a1_rerun": ("a8", "disability", "default", "frozen"),
    **{f"fc_shuffle_s{s}": (f"shuffle{s}", "disability", "default", "frozen") for s in SHUFFLE_SEEDS},
    "id_remove_a1": ("identity", "disability", "default", "frozen"),
    "fc_remove_a1_neutral": ("a8", "neutral", "default", "frozen"),
    "fresh_unsteered": (None, "disability", "default", "fresh"),
    "fresh_fc_remove_a1": ("a8", "disability", "default", "fresh"),
    "fresh_fc_prompt_remove_a1": ("a8", "disability", "explicit", "fresh"),
    "fresh_prompt_explicit": (None, "disability", "explicit", "fresh"),
}


def env() -> dict:
    """Library versions, stored in every A9 run record (A9-A5: earlier runs did not record them)."""
    import tokenizers
    import torch
    import transformers
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def load_fresh() -> list[dict]:
    text = FRESH.read_text()
    if hashlib.sha256(text.encode()).hexdigest() != FRESH_SHA:
        raise SystemExit(f"{FRESH} does not match its committed sha256")
    return json.loads(text)


def labelled_fit():
    items = json.loads((A8OUT / "fit_items.json").read_text())
    lab = a8.labels_from_judges(items, a8.load_fit_ratings())
    idx = sorted(lab)
    return items, idx, np.array([lab[i] for i in idx]), [items[i]["domain"] for i in idx]


def shuffle_within_domain(y: np.ndarray, domains: list[str], seed: int) -> np.ndarray:
    rng, ys = np.random.default_rng(seed), y.copy()
    for d in sorted(set(domains)):
        ii = np.array([i for i, x in enumerate(domains) if x == d])
        ys[ii] = rng.permutation(y[ii])
    return ys


def identity_direction(D: np.ndarray, y: np.ndarray, domains: list[str]) -> np.ndarray:
    """D[r] = disability-answer features minus the same question's neutral-answer features.
    v = sum_d w_d mean_d(D) with the A8 weights w_d ∝ min(n_med,d, n_clean,d) (A9-A1 item 1)."""
    v, wsum = np.zeros(D.shape[1]), 0.0
    for d in sorted(set(domains)):
        idx = np.array([i for i, x in enumerate(domains) if x == d])
        w = float(min((y[idx] == 1).sum(), (y[idx] == 0).sum()))
        if w == 0:
            continue
        v += w * D[idx].mean(axis=0)
        wsum += w
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------ commands
def cmd_fit_neutral_generate(args) -> int:
    items, _, _, _ = labelled_fit()
    qs = sorted({it["clean_text"] for it in items})          # all 96 fit questions (PREREG_A9 section 4)
    print(f"{len(qs)} neutral fit questions (from {len(items)} fit items)")
    if args.dry_run:
        print(f"DRY RUN  git={str(e1.GIT_AT_IMPORT['commit'])[:12]}  example: {qs[0][:100]}")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    fz = fe.load(Path(args.eval_set))
    with e1.RunRecord(OUT, "a9_fit_neutral_generate", args, fz, name="run_record_fit_neutral_generate") as rec:
        a8.seed_all()
        model, tok = rt.load_model()
        ans = rt.generate_responses(model, tok, [rt.format_prompt(q) for q in qs], NEUTRAL_GEN, "A9 neutral fit")
        NEUTRAL_ITEMS.write_text(json.dumps([{"item": i, "question": q, "answer": a}
                                             for i, (q, a) in enumerate(zip(qs, ans))], indent=1))
        rec.result(n_questions=len(qs), output=str(NEUTRAL_ITEMS), env=env())
    return 0


def cmd_fit(args) -> int:
    import torch
    items, idx, y, domains = labelled_fit()
    D8 = torch.load(A8OUT / "a8_directions.pt", weights_only=False)
    if list(D8["selected_layers"]) != LAYERS or list(D8["labelled_items"]) != idx:
        raise SystemExit("A8 fit does not match: layers or labelled items differ")
    print(f"{len(idx)} labelled items, layers {LAYERS}, neutral answers from {NEUTRAL_ITEMS}")
    if args.dry_run:
        print(f"DRY RUN  git={str(e1.GIT_AT_IMPORT['commit'])[:12]}")
        return 0
    neutral = json.loads(NEUTRAL_ITEMS.read_text())
    qpos = {n["question"]: r for r, n in enumerate(neutral)}
    fz = fe.load(Path(args.eval_set))
    with e1.RunRecord(OUT, "a9_fit", args, fz, name="run_record_fit") as rec:
        a8.seed_all()
        model, tok = rt.load_model()
        F = a8.answer_features(model, tok, items, idx)                    # (n, 32, h)
        N = a8.answer_features(model, tok, neutral, list(range(len(neutral))))
        rep, dirs = {"ga9_0_cos": {}, "cos_identity_framing": {}}, {"layers": LAYERS, "identity": {}}
        for L in LAYERS:
            v = a8.stratified_direction(F[:, L, :], y, domains)
            rep["ga9_0_cos"][L] = float(np.dot(v, D8["v"][L].numpy().astype(np.float64)))
        rep["GA9_0_pass"] = all(c >= MIN_COS for c in rep["ga9_0_cos"].values())
        if not rep["GA9_0_pass"]:
            REPORT.write_text(json.dumps(rep, indent=1))
            rec.result(**rep)
            print(f"GA9-0 FAIL {rep['ga9_0_cos']}: stop and report")
            return 1
        for s in SHUFFLE_SEEDS:
            ys = shuffle_within_domain(y, domains, s)
            dirs[f"shuffle{s}"] = {L: torch.tensor(a8.stratified_direction(F[:, L, :], ys, domains),
                                                   dtype=torch.float32) for L in LAYERS}
            rep[f"cos_shuffle{s}_framing"] = {L: float(dirs[f"shuffle{s}"][L] @ D8["v"][L]) for L in LAYERS}
        nidx = np.array([qpos[items[i]["clean_text"]] for i in idx])
        for L in LAYERS:
            v = identity_direction(F[:, L, :] - N[nidx, L, :], y, domains)
            dirs["identity"][L] = torch.tensor(v, dtype=torch.float32)
            rep["cos_identity_framing"][L] = float(dirs["identity"][L] @ D8["v"][L])
        torch.save(dirs, DIRS)
        REPORT.write_text(json.dumps(rep, indent=1))
        rec.result(**rep, env=env())
    print(json.dumps(rep, indent=1))
    return 0


def prompts_for(arm: str, fz: dict) -> list[str]:
    _, side, sysp, eset = ARMS[arm]
    system = rt.SYSTEM_PROMPTS[sysp]
    if eset == "fresh":
        return [rt.format_prompt(it["question"], system) for it in load_fresh()]
    field = "clean_text" if side == "neutral" else "corrupted_text"
    return [rt.format_prompt(p[field], system) for p in fe.eval_pairs(fz)]


def direction(name, D8, D9):
    if name is None:
        return None
    return D8["v"] if name == "a8" else D9[name]


def cmd_generate(args) -> int:
    import torch
    fz = fe.load(Path(args.eval_set))
    todo = args.arms or list(ARMS)
    for arm in todo:
        name, side, sysp, eset = ARMS[arm]
        print(f"{arm}: direction {name or 'none'}, layers {LAYERS if name else '-'}, {side} side, "
              f"prompt {sysp}, {eset} set ({len(prompts_for(arm, fz))} items)")
    if args.dry_run:
        print(f"DRY RUN  git={str(e1.GIT_AT_IMPORT['commit'])[:12]}  dirty={e1.GIT_AT_IMPORT['dirty_tracked_files'] or 'none'}")
        return 0
    D8 = torch.load(A8OUT / "a8_directions.pt", weights_only=False)
    D9 = torch.load(DIRS, weights_only=False) if any(ARMS[a][0] not in (None, "a8") for a in todo) else {}
    model, tok = rt.load_model()
    for arm in todo:
        name, side, sysp, eset = ARMS[arm]
        V = direction(name, D8, D9)
        with e1.RunRecord(OUT, "a9_generate", args, fz, name=f"run_record_{arm}") as rec:
            a8.seed_all()
            handles = [model.model.layers[L].register_forward_hook(
                rt.make_projection_hook(V[L].reshape(1, -1), ALPHA)) for L in LAYERS] if V else []
            try:
                resp = rt.generate_responses(model, tok, prompts_for(arm, fz), OUT / f"{arm}_responses.json", arm)
            finally:
                for h in handles:
                    h.remove()
            rec.result(direction=name, layers=LAYERS if V else [], alpha=ALPHA if V else 0.0, side=side,
                       system_prompt=sysp, eval_set=eset, fresh_sha256=FRESH_SHA if eset == "fresh" else None,
                       n=len(resp), lexical_mean=float(np.mean([rt.medicalization_score(r) for r in resp])), env=env())
    return 0


def cmd_ppl(args) -> int:
    import torch
    import torch.nn.functional as Fn
    from datasets import load_dataset
    arms = {"unsteered": 0.0, "fc_remove_a1": 1.0, "fc_remove_a2": 2.0}
    if args.dry_run:
        print(f"perplexity arms {arms}; WikiText-2 raw test, {PPL_WINDOW}-token windows")
        return 0
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    fz = fe.load(Path(args.eval_set))
    D8 = torch.load(A8OUT / "a8_directions.pt", weights_only=False)
    model, tok = rt.load_model()
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    n = ids.numel() // PPL_WINDOW
    X = ids[: n * PPL_WINDOW].view(n, PPL_WINDOW)
    out = {"n_windows": n, "n_tokens_scored": n * (PPL_WINDOW - 1), "dataset_fingerprint": ds._fingerprint}
    with e1.RunRecord(OUT, "a9_ppl", args, fz, name="run_record_ppl") as rec:
        for arm, alpha in arms.items():
            handles = [model.model.layers[L].register_forward_hook(
                rt.make_projection_hook(D8["v"][L].reshape(1, -1), alpha)) for L in LAYERS] if alpha else []
            nll = 0.0
            try:
                for b in range(0, n, PPL_BATCH):
                    x = X[b:b + PPL_BATCH].to(model.device)
                    with torch.inference_mode():
                        logits = model(input_ids=x).logits[:, :-1].float()
                    nll += float(Fn.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1),
                                                  reduction="sum"))
            finally:
                for h in handles:
                    h.remove()
            out[arm] = {"alpha": alpha, "ppl": float(np.exp(nll / out["n_tokens_scored"]))}
            print(f"{arm}: ppl {out[arm]['ppl']:.3f}")
        (OUT / "ppl.json").write_text(json.dumps(out, indent=1))
        rec.result(**{a: out[a]["ppl"] for a in arms}, env=env())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit_neutral_generate", "fit", "generate", "ppl"])
    ap.add_argument("--arms", nargs="*", choices=sorted(ARMS))
    e1.add_e1_args(ap)
    args = ap.parse_args()
    args.eval_set = args.eval_set or "frozen_eval_v1.json"
    args.out_dir = str(OUT)
    args.seed = a8.SEED
    return {"fit_neutral_generate": cmd_fit_neutral_generate, "fit": cmd_fit,
            "generate": cmd_generate, "ppl": cmd_ppl}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
