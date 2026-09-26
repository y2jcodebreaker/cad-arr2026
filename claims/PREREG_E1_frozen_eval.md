# Pre-registration E1 — one frozen evaluation set for every arm

**Written 2026-09-26, before any E1 data exists.** Amendments go at the bottom with a
timestamp and a reason, and only before the affected arm runs.

## Why this experiment

The claim graph (`claims/claim_graph.py`) shows four layer-2 claims with open blocking
controls, and the same control blocks all four: the paper's numbers come from at least six
runs on different prompt draws.

| run | unsteered medicalization | pool | n |
|---|---|---|---|
| LEACE rank-1 | 0.509 | ? | 250 |
| LEACE rank-40 (exp3) | 0.531 | 2082 | 250 |
| CAD sweep (rtsd_fullres) | 0.593 | 2083 | 250 |
| FairSteer sweep | 0.663 | ? | 250 |
| CAA sweep | 0.712 | ? | 250 |
| GPT-5.5 judge, CAD arm | — | ? | **197** |

Within one run, three unsteered draws give sd 0.038. Between runs the spread is 0.20, about
five times that. Effect sizes do not reproduce across runs: CAA d 0.341 → 0.437, SADI 0.467 →
0.545, FairSteer 0.191 → 0.366. §5 states that rank-1 and rank-40 differ "only in rank", but
they were measured on eval sets with baselines 0.509 and 0.531 and pools of different size.

## What already exists and what I have seen

Seen before writing this: every number in the table above; the frontier sweep; the July
paired test; the VADER disagreement audit (VADER saturated at ≥0.99 for 69–80% of responses;
its CAA>CAD gap concentrates in difficulty-framed questions, 72% of the gap from 36% of items,
permutation p=0.049 one-sided). Not seen: anything on a frozen set.

Literature checked 2026-09-26 for the rank claim (C5). HiDRA (arXiv 2606.15092, preprint):
nonlinear random-feature lift, jailbreak/truthfulness/MCQ, does **not** set rank from a
spectrum. SAE-SSV (arXiv 2505.16188; venue not yet verified): classifier-selected subspace in
SAE latent space. CAS-BiPO (Findings of EACL 2026, pp. 1079–1097, verified on the Anthology):
learned sparse masks. None pre-empts C5. None matches how §7 currently describes them.

## Design

**Frozen set.** 250 items drawn once from the 2,083-pair loudness-filtered pool, stratified
across the 9 disability categories and 6 domains, written to `frozen_eval_v1.json` with its
SHA-256. Every runner loads it and **asserts the hash** before generating. The set is frozen:
no arm may run on anything else.

**Arms** (all on the frozen set, `temperature` and `max_new_tokens` unchanged from the sweep):

| tier | arms | count |
|---|---|---|
| 1 | unsteered, seeds 42 / 43 / 44 | 3 |
| 1 | published points: CAA L14 α=1, FairSteer L29 τ=0.5 α=2, SADI s=10, Angular max_sim L23 m0 150° | 4 |
| 1 | CAD: projection α=1 k=40, mean-diff α=8, probe α=9 | 3 |
| 1 | rank sweep, projection α=1: k ∈ {1, 5, 10, 20, 80} (k=40 above) | 5 |
| 1 | rank sweep, LEACE: k ∈ {1, 5, 10, 20, 40, 80} | 6 |
| 2 | full strength sweeps for all methods (the 87-point frontier) | ~87 |
| 3 | Qwen2.5-7B: unsteered + the 5 published points | 6 |

Tier 1 is 21 arms, about 2.6 h of generation on one A100. Tier 2 only after tier 1 passes.
Tier 3 is layer 3 and runs only after every layer-2 blocking control is closed (breadth-first).

**Instruments.** I1 medicalization scorer, version-pinned, unchanged. I2 Llama-3.1-8B judge,
rubric unchanged. I3 a second judge **from a different model family**, same rubric. Degeneracy
flags from text only, thresholds unchanged (0.50 / 0.10 / −0.05).

## Gates — cheapest first; a failed gate stops the run

- **G0 (CPU, seconds).** Frozen set has 250 unique items and covers all 9 categories; every
  runner in `--dry-run` loads it and prints the same hash. Fail → fix plumbing, spend nothing.
- **G1 (~25 min).** Unsteered × 3 seeds. Pass: between-seed sd of mean medicalization ≤ 0.06
  and the mean in [0.45, 0.80]. If sd > 0.06, sampling noise is larger than assumed: amend to
  add seeds per arm **before** continuing.
- **G2 (~15 min). Positive controls.** SADI s=10 must degenerate (zero-score rate ≥ 0.50; was
  0.80). CAD projection α=1 must be intact by the text criteria. Fail → the paper's anchor
  finding does not reproduce on the frozen set; stop and investigate.
- **G3.** On G1+G2 outputs, per-item Spearman ρ between I2 and I3 ≥ 0.5. Below that, the judges
  disagree too much to confirm each other; audit the disagreements before scaling.

## Predictions and falsifiers

- **P1 (C2).** On both I2 and I3, SADI has the largest quality drop of the five published
  points and CAD projection the smallest. **F1:** either ordering fails on either judge.
- **P2 (C4).** On I3, the paired CAD-projection-minus-CAA quality difference has a 95% CI
  excluding 0. **F2:** the CI includes 0.
- **P3 (C5, primary).** For both operators, d at k=40 exceeds d at k=1 with non-overlapping
  bootstrap 95% CIs, **and** mean judged quality at k=40 is within 1.0 of the unsteered mean on
  both judges. **F3a:** the k=1 and k=40 CIs overlap. **F3b:** quality at k=40 drops by more
  than 1.0 — the gain would then be bought with degeneration, which is §3's own objection.
- **P4 (C3, tier 2 only).** On the frozen set, no baseline configuration is on the clean Pareto
  front at ≥50% reduction. **F4:** any baseline configuration is.

## Outcomes named in advance, including the unwelcome ones

- **k=1 projection already reaches d within 0.05 of k=40.** Then "rank is what changes the
  outcome" is false. §5 is cut or rewritten as a null, and the conclusion's sentence about rank
  goes with it.
- **I3 reverses P1 or P2.** The quality claim is judge-dependent. Report both judges; drop
  "roughly half the quality cost" from the abstract.
- **G2 fails.** SADI does not degenerate on the frozen set. The audit section must be reframed
  around the sweep draw where it does, with the draw-dependence stated.
- **Every prediction passes.** Every number in Tables 1 and the text is regenerated from E1,
  the old values go into `SUPERSEDED` in the claim graph, and the "two measurement sources"
  caveat is deleted from the Table 1 caption and the Limitations.

## Discipline during the run

Artifacts are written to disk before any summary string is built. Each arm appends one row to
a run ledger (arm, frozen-set hash, seed, config, start/finish, status including failures).
Nothing is downloaded while still being written; archives are checksummed on both ends.

## Amendments

(none yet)
