"""A11: second model, Mistral-7B-Instruct-v0.3 (claims/PREREG_A11_mistral.md, frozen 7da220b).

    python a11_mistral.py s0             # unsteered + neutral answers on the frozen set
    python a11_mistral.py fit_generate   # one answer per fit-pool question
    #   then judge them: judge_medicalization.py --judge llama|qwen --items e1_outputs/a11/fit_items.json
    python a11_mistral.py fit            # GA11-1, GA11-2, layer rule, directions (A8's procedure)
    python a11_mistral.py generate       # the four S2 arms (only if both gates passed)
    add --dry_run to any: no model, prints what would run

Generation goes through rt.generate_responses (temperature 0.1, seed 42, same batch and token limit as
Llama); only the model, tokenizer and prompt formatter differ.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys_argv, sys.argv = sys.argv, sys.argv[:1]
import a8_framing_contrast as a8                  # noqa: E402
import a9_robustness as a9                        # noqa: E402
import e1_common as e1                            # noqa: E402
import evaluate_rtsd_fullres_generation as rt     # noqa: E402
import frozen_eval as fe                          # noqa: E402
sys.argv = sys_argv

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"
OUT = Path("e1_outputs/a11")
FIT_ITEMS = OUT / "fit_items.json"
DIRS = OUT / "a11_directions.pt"
REPORT = OUT / "a11_fit_report.json"
ALPHA = 1.0

# arm -> (direction, question side, system prompt)
S0 = {"mistral_unsteered": (None, "disability", "default"), "mistral_neutral": (None, "neutral", "default")}
S2 = {"mistral_fc_remove_a1": ("fc", "disability", "default"),
      "mistral_fc_prompt_remove_a1": ("fc", "disability", "explicit"),
      "mistral_prompt_explicit": (None, "disability", "explicit"),
      "mistral_fc_shuffle_s0": ("shuffle0", "disability", "default")}


def load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()
    return model, tok


def formatter(tok, system: str):
    """Official chat template (system folded into the user turn), template BOS stripped so the
    tokenizer adds exactly one (PREREG_A11 section 2)."""
    def fmt(text: str) -> str:
        s = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": text}],
                                    tokenize=False, add_generation_prompt=True)
        return s[len(tok.bos_token):] if s.startswith(tok.bos_token) else s
    return fmt


def check_format(tok) -> None:
    got = formatter(tok, "S")("U")
    if got != "[INST] S\n\nU[/INST]":
        raise SystemExit(f"unexpected Mistral prompt format: {got!r}")
    ids = tok(got)["input_ids"]
    if ids[0] != tok.bos_token_id or ids[1] == tok.bos_token_id:
        raise SystemExit("expected exactly one BOS")


def run_arms(arms: dict, args, D=None) -> int:
    fz = fe.load(Path(args.eval_set))
    ev = fe.eval_pairs(fz)
    for arm, (d, side, sysp) in arms.items():
        print(f"{arm}: direction {d or 'none'}, {side} side, prompt {sysp}, {len(ev)} items")
    if args.dry_run:
        print(f"DRY RUN  git={str(e1.GIT_AT_IMPORT['commit'])[:12]}  dirty={e1.GIT_AT_IMPORT['dirty_tracked_files'] or 'none'}")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok = load()
    check_format(tok)
    for arm, (d, side, sysp) in arms.items():
        fmt = formatter(tok, rt.SYSTEM_PROMPTS[sysp])
        prompts = [fmt(p["clean_text"] if side == "neutral" else p["corrupted_text"]) for p in ev]
        V = D[d] if d else None
        with e1.RunRecord(OUT, "a11_generate", args, fz, name=f"run_record_{arm}") as rec:
            a8.seed_all()
            handles = [model.model.layers[L].register_forward_hook(
                rt.make_projection_hook(V[L].reshape(1, -1), ALPHA)) for L in D["layers"]] if V else []
            try:
                resp = rt.generate_responses(model, tok, prompts, OUT / f"{arm}_responses.json", arm)
            finally:
                for h in handles:
                    h.remove()
            rec.result(model=MODEL, revision=REVISION, direction=d, layers=D["layers"] if V else [],
                       side=side, system_prompt=sysp, n=len(resp), env=a9.env(),
                       lexical_mean=float(np.mean([rt.medicalization_score(r) for r in resp])))
    return 0


def cmd_s0(args) -> int:
    return run_arms(S0, args)


def cmd_fit_generate(args) -> int:
    items, fz = a8.fit_pool_with_meta(args)
    print(f"fit pool: {len(items)} questions")
    if args.dry_run:
        print(f"DRY RUN  git={str(e1.GIT_AT_IMPORT['commit'])[:12]}")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    with e1.RunRecord(OUT, "a11_fit_generate", args, fz, name="run_record_fit_generate") as rec:
        a8.seed_all()
        model, tok = load()
        check_format(tok)
        fmt = formatter(tok, rt.SYSTEM_PROMPTS["default"])
        ans = rt.generate_responses(model, tok, [fmt(i["question"]) for i in items], OUT / "fit_answers_gen.json",
                                    "A11 fit answers")
        for it, a in zip(items, ans):
            it["answer"] = a
        FIT_ITEMS.write_text(json.dumps(items, indent=1))
        rec.result(n_items=len(items), model=MODEL, revision=REVISION, env=a9.env())
    return 0


def fit_ratings() -> dict:
    out = {}
    for j in ("llama", "qwen"):
        d = json.loads((OUT / f"m1_judge_{j}_fit.json").read_text())
        out[j] = {r["item"]: d["rating_by_key"].get(r["key"]) for r in d["records"]}
    return out


def cmd_fit(args) -> int:
    import torch
    items = json.loads(FIT_ITEMS.read_text())
    lab = a8.labels_from_judges(items, fit_ratings())
    idx = sorted(lab)
    y = np.array([lab[i] for i in idx])
    domains = [items[i]["domain"] for i in idx]
    rep = {"n_medicalizing": int((y == 1).sum()), "n_clean": int((y == 0).sum())}
    rep["GA11_1_pass"] = rep["n_medicalizing"] >= a8.MIN_CLASS and rep["n_clean"] >= a8.MIN_CLASS
    print(f"labels: {rep['n_medicalizing']} medicalizing, {rep['n_clean']} clean -> GA11-1 "
          f"{'PASS' if rep['GA11_1_pass'] else 'FAIL'}")
    if args.dry_run:
        return 0
    if not rep["GA11_1_pass"]:
        REPORT.write_text(json.dumps(rep, indent=1))
        return 1
    fz = fe.load(Path(args.eval_set))
    with e1.RunRecord(OUT, "a11_fit", args, fz, name="run_record_fit") as rec:
        a8.seed_all()
        model, tok = load()
        check_format(tok)
        F = a8.answer_features(model, tok, items, idx, fmt=formatter(tok, rt.SYSTEM_PROMPTS["default"]),
                               prompt_special=True)
        accs, layers = a8.layer_rule(F, y, domains)
        rep.update(cv_acc_by_layer=accs, selected_layers=layers, best_cv_acc=max(accs))
        rep["GA11_2_pass"] = max(accs) >= a8.MIN_CV_ACC
        if rep["GA11_2_pass"]:
            ys = a9.shuffle_within_domain(y, domains, 0)
            dirs = {"layers": layers,
                    "fc": {L: torch.tensor(a8.stratified_direction(F[:, L, :], y, domains), dtype=torch.float32)
                           for L in layers},
                    "shuffle0": {L: torch.tensor(a8.stratified_direction(F[:, L, :], ys, domains), dtype=torch.float32)
                                 for L in layers}}
            torch.save(dirs, DIRS)
        REPORT.write_text(json.dumps(rep, indent=1))
        rec.result(**{k: v for k, v in rep.items() if k != "cv_acc_by_layer"}, env=a9.env())
    print(f"balanced accuracy by layer: " + " ".join(f"{L}:{a:.2f}" for L, a in enumerate(accs)))
    print(f"selected layers {layers} (best {max(accs):.3f}) -> GA11-2 {'PASS' if rep['GA11_2_pass'] else 'FAIL'}")
    return 0 if rep["GA11_2_pass"] else 1


def cmd_generate(args) -> int:
    import torch
    if args.dry_run and not REPORT.exists():
        return run_arms(S2, args)
    rep = json.loads(REPORT.read_text())
    if not (rep.get("GA11_1_pass") and rep.get("GA11_2_pass")):
        raise SystemExit("A11 fit gates did not pass: outcome (d), no S2 arms")
    return run_arms(S2, args, torch.load(DIRS, weights_only=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["s0", "fit_generate", "fit", "generate"])
    e1.add_e1_args(ap)
    args = ap.parse_args()
    args.eval_set = args.eval_set or "frozen_eval_v1.json"
    args.out_dir = str(OUT)
    args.seed = a8.SEED
    return {"s0": cmd_s0, "fit_generate": cmd_fit_generate, "fit": cmd_fit, "generate": cmd_generate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
