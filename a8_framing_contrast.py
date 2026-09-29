"""A8 framing-contrast steering (claims/PREREG_A8_framing_contrast.md, frozen 2026-09-28).

Contrast the model's own answers WITHIN the disability condition, labelled for medical framing by
the two M1 judges, and remove that direction. Fitting uses the 633 training questions only.

    python a8_framing_contrast.py fit_generate [--dry_run]   # unsteered answers to the fit pool
    #   then judge them:  judge_medicalization.py --items e1_outputs/a8/fit_items.json ...
    python a8_framing_contrast.py fit                        # gates, layer rule, directions
    python a8_framing_contrast.py generate [--arms ...]      # the six arms on the frozen set

Steering operators are E1's own hooks (make_projection_hook / make_probe_hook), so the arms differ
from E1's rank-1 arms only in direction and layers.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys_argv, sys.argv = sys.argv, sys.argv[:1]      # the runner module parses no args at import
import e1_common as e1                            # noqa: E402
import evaluate_rtsd_fullres_generation as rt     # noqa: E402
import frozen_eval as fe                          # noqa: E402
sys.argv = sys_argv

OUT = Path("e1_outputs/a8")
FIT_ITEMS = OUT / "fit_items.json"
FIT_GEN = OUT / "fit_answers_gen.json"            # not *_responses.json: keep it out of the E1 judge globs
DIRS = OUT / "a8_directions.pt"
REPORT = OUT / "a8_fit_report.json"
SEED = 42
N_LAYERS = 32
MIN_CLASS = 40            # GA8-1
MIN_CV_ACC = 0.60         # GA8-2
MAX_ANSWER_TOKENS = 1048

# arm -> (layer source, operator, alpha, system prompt)
ARMS = {
    "fc_remove_a1": ("selected", "projection", 1.0, "default"),
    "fc_remove_a2": ("selected", "projection", 2.0, "default"),
    "fc_add_a4": ("selected", "additive", 4.0, "default"),
    "fc_add_a8": ("selected", "additive", 8.0, "default"),
    "fc_prompt_remove_a1": ("selected", "projection", 1.0, "explicit"),
    "fc_remove_a1_L21L25": ("L21L25", "projection", 1.0, "default"),
}


# ------------------------------------------------------------------ data
def fit_pool_with_meta(args) -> tuple[list[dict], dict]:
    """The 633 E1 fit-pool pairs, with domain / category / question id from the day-1 metadata."""
    import day1_extract_activations as d1
    filtered = rt.apply_loudness_filter(rt.build_accesseval_pairs())
    fz, fit, _ = e1.frozen_split(filtered, args)
    pairs, meta = d1.build_pairs_with_metadata()
    by_key = {}
    for p, m in zip(pairs, meta):
        by_key.setdefault(fe.pair_key(p["clean_text"], p["corrupted_text"]), m)
    items = []
    for i, p in enumerate(fit):
        m = by_key[fe.pair_key(p["clean_text"], p["corrupted_text"])]
        items.append({"item": i, "question": p["corrupted_text"], "clean_text": p["clean_text"],
                      "domain": str(m["domain"]), "category": str(m["category"]),
                      "question_id": int(m["base_query_id"]) % 234})
    return items, fz


def seed_all() -> None:
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


# ------------------------------------------------------------------ labels, layer rule, directions (pure)
def labels_from_judges(items: list[dict], ratings: dict[str, dict[int, str | None]]) -> dict[int, int]:
    """1 = medicalizing (both >= 1), 0 = clean (both 0); others and Healthcare excluded."""
    num = lambda v: int(v) if v in ("0", "1", "2", "3") else None  # noqa: E731
    out = {}
    for it in items:
        if it["domain"] == "Healthcare":
            continue
        a, b = (num(ratings[j].get(it["item"])) for j in ("llama", "qwen"))
        if a is None or b is None:
            continue
        if a >= 1 and b >= 1:
            out[it["item"]] = 1
        elif a == 0 and b == 0:
            out[it["item"]] = 0
    return out


def domain_demean(X: np.ndarray, domains: list[str]) -> np.ndarray:
    X = X.copy()
    for d in set(domains):
        idx = [i for i, x in enumerate(domains) if x == d]
        X[idx] -= X[idx].mean(axis=0)
    return X


def layer_rule(feats: np.ndarray, y: np.ndarray, domains: list[str]) -> tuple[list[float], list[int]]:
    """Mean 5-fold CV BALANCED accuracy per layer (A8-A1); the two best layers (ties: lower layer)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    accs = []
    for L in range(feats.shape[1]):
        X = domain_demean(feats[:, L, :], domains)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        accs.append(float(cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy").mean()))
    order = sorted(range(len(accs)), key=lambda L: (-accs[L], L))
    return accs, sorted(order[:2])


def stratified_direction(F: np.ndarray, y: np.ndarray, domains: list[str]) -> np.ndarray:
    """v = sum_d w_d (mean_med,d - mean_clean,d), w_d ∝ min(n_med,d, n_clean,d); unit norm."""
    v, wsum = np.zeros(F.shape[1]), 0.0
    for d in sorted(set(domains)):
        idx = np.array([i for i, x in enumerate(domains) if x == d])
        med, cln = idx[y[idx] == 1], idx[y[idx] == 0]
        if len(med) == 0 or len(cln) == 0:
            continue
        w = float(min(len(med), len(cln)))
        v += w * (F[med].mean(axis=0) - F[cln].mean(axis=0))
        wsum += w
    if wsum == 0:
        raise ValueError("no domain has both classes")
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------ model steps
def answer_features(model, tok, items: list[dict], idx: list[int], fmt=None,
                    prompt_special: bool = False) -> np.ndarray:
    """Mean raw output of every decoder layer over the answer tokens (the E1 hook point).

    Hooks, not output_hidden_states: HF returns the LAST layer after the final norm.
    fmt / prompt_special default to the Llama path used by A8 and A9 (unchanged); A11 passes the
    Mistral formatter and prompt_special=True so the prompt carries the tokenizer's single BOS."""
    fmt = fmt or rt.format_prompt
    import torch
    feats = np.zeros((len(idx), N_LAYERS, model.config.hidden_size), dtype=np.float32)
    store: dict = {}

    def grab(L):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            store[L] = h[0, store["start"]:, :].float().mean(dim=0).cpu().numpy()
        return hook
    handles = [model.model.layers[L].register_forward_hook(grab(L)) for L in range(N_LAYERS)]
    try:
        for r, i in enumerate(idx):
            it = items[i]
            prompt = fmt(it["question"])
            p_ids = tok(prompt, add_special_tokens=prompt_special)["input_ids"]
            a_ids = tok(it["answer"], add_special_tokens=False)["input_ids"][:MAX_ANSWER_TOKENS]
            if not a_ids:
                raise ValueError(f"fit item {i} has an empty answer")
            ids = torch.tensor([p_ids + a_ids], device=model.device)
            store["start"] = len(p_ids)
            with torch.inference_mode():
                model(input_ids=ids)
            for L in range(N_LAYERS):
                feats[r, L] = store[L]
    finally:
        for h in handles:
            h.remove()
    return feats


def cmd_fit_generate(args) -> int:
    items, fz = fit_pool_with_meta(args)
    import collections
    print(f"fit pool: {len(items)} questions-by-category; domains {dict(collections.Counter(i['domain'] for i in items))}")
    if args.dry_run:
        print(f"DRY RUN  git={str(e1.GIT_AT_IMPORT['commit'])[:12]}  dirty={e1.GIT_AT_IMPORT['dirty_tracked_files'] or 'none'}")
        print("  example:", items[0]["question"][:100])
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    with e1.RunRecord(OUT, "a8_fit_generate", args, fz, name="run_record_fit_generate") as rec:
        seed_all()
        model, tok = rt.load_model()
        answers = rt.generate_responses(model, tok, [rt.format_prompt(i["question"]) for i in items],
                                        FIT_GEN, "A8 fit answers")
        for it, a in zip(items, answers):
            it["answer"] = a
        FIT_ITEMS.write_text(json.dumps(items, indent=1))
        rec.result(n_items=len(items), output=str(FIT_ITEMS))
    print(f"wrote {FIT_ITEMS}")
    return 0


def load_fit_ratings() -> dict[str, dict[int, str | None]]:
    out = {}
    for j in ("llama", "qwen"):
        d = json.loads((OUT / f"m1_judge_{j}_fit.json").read_text())
        out[j] = {r["item"]: d["rating_by_key"].get(r["key"]) for r in d["records"]}
    return out


def cmd_fit(args) -> int:
    import torch
    items = json.loads(FIT_ITEMS.read_text())
    lab = labels_from_judges(items, load_fit_ratings())
    idx = sorted(lab)
    y = np.array([lab[i] for i in idx])
    domains = [items[i]["domain"] for i in idx]
    report = {"n_medicalizing": int((y == 1).sum()), "n_clean": int((y == 0).sum()),
              "by_domain": {d: {"med": int(sum(1 for i, x in zip(idx, domains) if x == d and lab[i] == 1)),
                                "clean": int(sum(1 for i, x in zip(idx, domains) if x == d and lab[i] == 0))}
                            for d in sorted(set(domains))}}
    report["GA8_1_pass"] = report["n_medicalizing"] >= MIN_CLASS and report["n_clean"] >= MIN_CLASS
    print(f"labels: {report['n_medicalizing']} medicalizing, {report['n_clean']} clean  "
          f"-> GA8-1 {'PASS' if report['GA8_1_pass'] else 'FAIL'}")
    if not report["GA8_1_pass"]:
        REPORT.write_text(json.dumps(report, indent=1))
        return 1
    fz = fe.load(Path(args.eval_set))
    with e1.RunRecord(OUT, "a8_fit", args, fz, name="run_record_fit") as rec:
        seed_all()
        model, tok = rt.load_model()
        feats = answer_features(model, tok, items, idx)
        accs, selected = layer_rule(feats, y, domains)
        report.update(cv_acc_by_layer=accs, selected_layers=selected, best_cv_acc=max(accs))
        report["GA8_2_pass"] = max(accs) >= MIN_CV_ACC
        dirs = {L: torch.tensor(stratified_direction(feats[:, L, :], y, domains), dtype=torch.float32)
                for L in sorted(set(selected) | {21, 25})}
        torch.save({"selected_layers": selected, "v": dirs, "cv_acc": accs, "labelled_items": idx}, DIRS)
        REPORT.write_text(json.dumps(report, indent=1))
        rec.result(**{k: v for k, v in report.items() if k != "cv_acc_by_layer"})
    print(f"CV accuracy by layer: " + " ".join(f"{L}:{a:.2f}" for L, a in enumerate(accs)))
    print(f"selected layers {selected} (best {max(accs):.3f}) -> GA8-2 {'PASS' if report['GA8_2_pass'] else 'FAIL'}")
    return 0 if report["GA8_2_pass"] else 1


def cmd_generate(args) -> int:
    import torch
    if args.dry_run and not REPORT.exists():      # before the fit: show the arms, nothing else
        for arm, (src, op, alpha, sysp) in ARMS.items():
            print(f"{arm}: layers {'chosen at fit' if src == 'selected' else [21, 25]}, {op}, alpha {alpha}, prompt {sysp}")
        print(f"DRY RUN  git={str(e1.GIT_AT_IMPORT['commit'])[:12]}  (no fit yet)")
        return 0
    rep = json.loads(REPORT.read_text())
    if not (rep.get("GA8_1_pass") and rep.get("GA8_2_pass")):
        raise SystemExit("A8 gates did not pass; the pre-registration says stop and report.")
    D = torch.load(DIRS, weights_only=False)
    fz = fe.load(Path(args.eval_set))
    ev = fe.eval_pairs(fz)
    todo = args.arms or list(ARMS)
    for arm in todo:
        src, op, alpha, sysp = ARMS[arm]
        layers = D["selected_layers"] if src == "selected" else [21, 25]
        print(f"{arm}: layers {layers}, {op}, alpha {alpha}, system prompt {sysp}")
    if args.dry_run:
        print(f"DRY RUN  git={str(e1.GIT_AT_IMPORT['commit'])[:12]}  dirty={e1.GIT_AT_IMPORT['dirty_tracked_files'] or 'none'}")
        return 0
    model, tok = rt.load_model()
    for arm in todo:
        src, op, alpha, sysp = ARMS[arm]
        layers = D["selected_layers"] if src == "selected" else [21, 25]
        prompts = [rt.format_prompt(p["corrupted_text"], rt.SYSTEM_PROMPTS[sysp]) for p in ev]
        with e1.RunRecord(OUT, "a8_generate", args, fz, name=f"run_record_{arm}") as rec:
            seed_all()
            handles = []
            for L in layers:
                v = D["v"][L]
                hook = (rt.make_projection_hook(v.reshape(1, -1), alpha) if op == "projection"
                        else rt.make_probe_hook(v, alpha))
                handles.append(model.model.layers[L].register_forward_hook(hook))
            try:
                resp = rt.generate_responses(model, tok, prompts, OUT / f"{arm}_responses.json", arm)
            finally:
                for h in handles:
                    h.remove()
            med = [rt.medicalization_score(r) for r in resp]
            rec.result(layers=layers, operator=op, alpha=alpha, system_prompt=sysp,
                       lexical_mean=float(np.mean(med)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit_generate", "fit", "generate"])
    ap.add_argument("--arms", nargs="*", choices=sorted(ARMS))
    e1.add_e1_args(ap)
    args = ap.parse_args()
    args.eval_set = args.eval_set or "frozen_eval_v1.json"
    args.out_dir = str(OUT)
    args.seed = SEED
    return {"fit_generate": cmd_fit_generate, "fit": cmd_fit, "generate": cmd_generate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
