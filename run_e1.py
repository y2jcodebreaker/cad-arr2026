"""E1 driver: tier-1 arms, run in pre-registered gate order. Stops at the first failed gate.

    python run_e1.py plan          # list every arm and its command
    python run_e1.py g0            # dry-run every arm (no GPU): plumbing gate
    python run_e1.py tier1         # G0 -> G1 -> G2 -> judge -> G3 -> remaining arms -> judge
    python run_e1.py gate g1|g2|g3 # re-evaluate one gate from outputs already on disk
    python run_e1.py status        # which arms have finished
    python run_e1.py arm NAME ...  # run named arms (deviations, re-runs), then judge
    python run_e1.py judge         # judge everything on disk (resumes), then G3

Criteria are copied from claims/PREREG_E1_frozen_eval.md and must not be edited after data
exists; amend the pre-registration instead.
"""
from __future__ import annotations

import json
import statistics as st
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "e1_outputs"
FROZEN = "frozen_eval_v1.json"
E1 = ["--eval_set", FROZEN]
SECOND_JUDGE = "Qwen/Qwen2.5-7B-Instruct"

CAD, LEACE = "evaluate_rtsd_fullres_generation.py", "exp3_leace_rank_k.py"
CAA, FS = "evaluate_caa_baseline.py", "evaluate_fairsteer_baseline.py"
SADI, ANG = "evaluate_sadi_baseline.py", "evaluate_angular_steering_baseline.py"
NO_CAD_SWEEPS = ["--probe_alphas", "--meandiff_alphas"]

# (name, stage, script, args). Stage says which gate the arm belongs to.
ARMS = [
    *[(f"baseline_seed{s}", "G1", CAD,
       [*E1, "--out_dir", "e1_outputs/cad_heldout", "--seed", str(s), "--baseline_only"])
      for s in (42, 43, 44)],
    ("sadi_s10", "G2", SADI, ["--skip_discrimeval", *E1, "--out_dir", "e1_outputs/sadi", "--strengths", "10"]),
    # A3: additive probe and mean-diff at four strengths each, so each additive curve can be read
    # at the SAME debiasing level its removal counterpart reaches (quality at matched d)
    ("cad_heldout", "G2", CAD, [*E1, "--out_dir", "e1_outputs/cad_heldout", "--proj_alphas", "1.0",
                                "--probe_alphas", "3.0", "5.0", "7.0", "9.0",
                                "--meandiff_alphas", "3.0", "5.0", "7.0", "8.0"]),
    ("cad_full_leak", "rest", CAD, [*E1, "--fit_pool", "full", "--out_dir", "e1_outputs/cad_full",
                                    "--proj_alphas", "1.0", *NO_CAD_SWEEPS]),
    *[(f"cad_rank_k{k}", "rest", CAD, [*E1, "--out_dir", "e1_outputs/cad_heldout", "--k", str(k),
                                        "--proj_alphas", "1.0", *NO_CAD_SWEEPS])
      for k in (1, 5, 10, 20, 80)],
    ("cad_probebasis", "rest", CAD, [*E1, "--out_dir", "e1_outputs/cad_heldout", "--proj_basis", "probe",
                                     "--proj_alphas", "1.0", *NO_CAD_SWEEPS]),
    ("cad_meandiffbasis", "rest", CAD, [*E1, "--out_dir", "e1_outputs/cad_heldout", "--proj_basis", "meandiff",
                                        "--proj_alphas", "1.0", *NO_CAD_SWEEPS]),
    *[(f"leace_k{k}", "rest", LEACE, [*E1, "--out_dir", "e1_outputs/leace", "--k", str(k)])
      for k in (1, 5, 10, 20, 40, 80)],
    ("caa_heldout", "rest", CAA, ["--skip_discrimeval", *E1, "--out_dir", "e1_outputs/caa_heldout",
                                  "--layer_override", "14", "--alphas", "1.0"]),
    ("caa_full_leak", "rest", CAA, ["--skip_discrimeval", *E1, "--fit_pool", "full",
                                    "--out_dir", "e1_outputs/caa_full", "--layer_override", "14", "--alphas", "1.0"]),
    ("fairsteer", "rest", FS, ["--skip_discrimeval", *E1, "--out_dir", "e1_outputs/fairsteer",
                               "--layer_override", "29", "--threshold", "0.5", "--alphas", "2.0"]),
    ("angular", "rest", ANG, ["--skip_discrimeval", *E1, "--out_dir", "e1_outputs/angular",
                              "--strategies", "max_sim", "--modes", "0", "--angles", "150"]),
    # Deviation D1 (PREREG): the registered svd_k40 arm. cad_heldout uses k=auto, which on the
    # 633-pair held-out pool resolves to 33/36, not 40. Not part of tier1; run with `arm`.
    ("cad_rank_k40", "D1", CAD, [*E1, "--out_dir", "e1_outputs/cad_heldout", "--k", "40",
                                 "--proj_alphas", "1.0", *NO_CAD_SWEEPS]),
    # A4 (PREREG): prompting baseline. Same out_dir, so the combo arm reuses the tier-1 SVD fit;
    # every output name carries _sys<name>, so nothing collides with the E1 reference baseline.
    ("prompt_min", "A4", CAD, [*E1, "--out_dir", "e1_outputs/cad_heldout", "--seed", "42",
                               "--baseline_only", "--system_prompt", "min"]),
    ("prompt_explicit", "A4", CAD, [*E1, "--out_dir", "e1_outputs/cad_heldout", "--seed", "42",
                                    "--baseline_only", "--system_prompt", "explicit"]),
    ("prompt_explicit_cad_k40", "A4", CAD, [*E1, "--out_dir", "e1_outputs/cad_heldout", "--k", "40",
                                            "--proj_alphas", "1.0", *NO_CAD_SWEEPS,
                                            "--system_prompt", "explicit"]),
    # M1 (PREREG_M1): unsteered answers to the frozen items' NEUTRAL questions, the reference that
    # the lexical gap L and the Q1 existence test need. Output: baseline_seed42_neutral_responses.json
    ("neutral_seed42", "M1", CAD, [*E1, "--out_dir", "e1_outputs/cad_heldout", "--seed", "42",
                                   "--baseline_only", "--eval_side", "neutral"]),
]

JUDGE_GLOBS = ["e1_outputs/*/*_responses.json", "e1_outputs/leace/base_seed*.json",
               "e1_outputs/leace/dual_k*.json", "e1_outputs/sadi/sadi_results.json",
               "e1_outputs/caa_*/caa_results.json", "e1_outputs/fairsteer/fairsteer_results.json",
               "e1_outputs/angular/angular_results.json"]


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        sys.exit(f"STOP: command failed with exit code {r.returncode}")


def require_clean_tree() -> None:
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit("STOP: tracked files are modified. A run must point at a commit that exists:\n"
                 + dirty + "\nCommit on the laptop and `bash pod_pull.sh`; never edit code on the pod.")


# ---------------------------------------------------------------------------- gates

def _load(p: Path) -> dict:
    if not p.exists():
        sys.exit(f"STOP: {p} missing — has its arm run?")
    return json.loads(p.read_text())


def gate_g1() -> bool:
    """G1: between-seed sd of mean medicalization <= 0.06 and the mean in [0.45, 0.80]."""
    means = []
    for s in (42, 43, 44):
        d = _load(OUT / "cad_heldout" / f"rtsd_fullres_results_baseline_seed{s}.json")
        means.append(st.mean(d["baseline_scores"]))
    sd, mu = st.stdev(means), st.mean(means)
    ok = sd <= 0.06 and 0.45 <= mu <= 0.80
    print(f"G1  seed means {[round(m, 3) for m in means]}  sd {sd:.4f} (<= 0.06)  "
          f"mean {mu:.3f} (in [0.45, 0.80])  ->  {'PASS' if ok else 'FAIL'}")
    return ok


def gate_g2() -> bool:
    """G2 positive controls: SADI s=10 zero-score rate >= 0.50; CAD projection a=1 intact."""
    sadi = _load(OUT / "sadi" / "sadi_results.json")["accesseval"]["strengths"]
    arm = next(v for k, v in sadi.items() if float(k.split("_")[1]) == 10.0)
    zero = sum(1 for r in arm["responses"] if r["medicalization_score"] == 0) / len(arm["responses"])
    ok_sadi = zero >= 0.50

    cad = _load(OUT / "cad_heldout" / "proj_alpha1.0_responses.json")["responses"]
    uniq = len(set(cad)) / len(cad)
    empty = sum(1 for t in cad if len(t.split()) < 10) / len(cad)
    summ = _load(OUT / "cad_heldout" / "rtsd_fullres_results.json")
    med = summ["projection"]["sweep"]["1.0"]["steered_mean"]
    ok_cad = uniq >= 0.50 and empty <= 0.10 and med >= -0.05
    print(f"G2  SADI s=10 zero-score rate {zero:.3f} (>= 0.50) -> {'PASS' if ok_sadi else 'FAIL'}")
    print(f"    CAD proj a=1: unique {uniq:.3f} (>= 0.50), empty {empty:.3f} (<= 0.10), "
          f"med {med:+.3f} (>= -0.05) -> {'PASS' if ok_cad else 'FAIL'}")
    return ok_sadi and ok_cad


def gate_g3() -> bool:
    """G3: per-item Spearman rho between the two judges >= 0.5 (and report parse failures)."""
    from scipy.stats import spearmanr
    a = _load(OUT / "judge_llama.json"); b = _load(OUT / "judge_second.json")
    shared = sorted(set(a["scores_by_hash"]) & set(b["scores_by_hash"]))
    rho = spearmanr([a["scores_by_hash"][h] for h in shared],
                    [b["scores_by_hash"][h] for h in shared]).correlation
    ok = rho >= 0.5
    for name, j in (("llama", a), ("second", b)):
        print(f"    {name:<7} {j['model']:<36} parse-failure rate {j.get('parse_failure_rate')}")
    print(f"G3  per-item Spearman rho {rho:.3f} over {len(shared)} texts (>= 0.5) -> {'PASS' if ok else 'FAIL'}")
    return ok


GATES = {"g1": gate_g1, "g2": gate_g2, "g3": gate_g3}


def judge_both() -> None:
    run([sys.executable, "judge_sweep_outputs.py", "--glob", *JUDGE_GLOBS,
         "--out", "e1_outputs/judge_llama.json"])
    run([sys.executable, "judge_sweep_outputs.py", "--glob", *JUDGE_GLOBS,
         "--out", "e1_outputs/judge_second.json", "--judge_model", SECOND_JUDGE])


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if cmd == "plan":
        for name, stage, script, args in ARMS:
            print(f"  [{stage:<4}] {name:<18} python {script} {' '.join(args)}")
        print(f"\n  {len(ARMS)} arms")
        return 0
    if cmd == "g0":
        for name, _, script, args in ARMS:
            run([sys.executable, script, *args, "--dry_run"])
        print("\nG0 PASS: every arm builds its data and fit pool")
        return 0
    if cmd == "gate":
        return 0 if GATES[sys.argv[2]]() else 1
    if cmd == "status":
        for name, stage, script, args in ARMS:
            out = Path(args[args.index("--out_dir") + 1])
            recs = sorted(out.glob("run_record*.json")) if out.exists() else []
            print(f"  [{stage:<4}] {name:<18} {len(recs)} record(s) in {out}")
        return 0
    if cmd == "arm":
        # one named arm, for deviations and re-runs: clean tree, dry run, run, then judge
        require_clean_tree()
        run([sys.executable, "build_frozen_eval.py", "--check"])
        todo = [a for a in ARMS if a[0] in sys.argv[2:]]
        if not todo or len(todo) != len(sys.argv[2:]):
            sys.exit(f"unknown arm in {sys.argv[2:]}; see `python run_e1.py plan`")
        for name, _, script, args in todo:
            run([sys.executable, script, *args, "--dry_run"])
        for name, _, script, args in todo:
            run([sys.executable, script, *args])
        judge_both()          # resumes: only the new texts are judged
        return 0
    if cmd == "judge":
        judge_both()
        return 0 if gate_g3() else 1
    if cmd == "tier1":
        require_clean_tree()
        run([sys.executable, "build_frozen_eval.py", "--check"])
        for name, _, script, args in ARMS:
            run([sys.executable, script, *args, "--dry_run"])
        print("\nG0 PASS")
        for stage, gate in (("G1", "g1"), ("G2", "g2")):
            for name, s, script, args in ARMS:
                if s == stage:
                    run([sys.executable, script, *args])
            if not GATES[gate]():
                sys.exit(f"STOP at {stage}: the pre-registered criterion failed. Do not continue.")
        judge_both()
        if not gate_g3():
            sys.exit("STOP at G3: the judges disagree too much to confirm each other. Audit first.")
        for name, s, script, args in ARMS:
            if s == "rest":
                run([sys.executable, script, *args])
        judge_both()          # resumes: only the new texts are judged
        print("\nTier 1 complete. Download e1_outputs/ and commit it from the laptop.")
        return 0
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
