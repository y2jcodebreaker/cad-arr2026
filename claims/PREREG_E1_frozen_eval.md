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

### A1 — 2026-09-27, before any E1 arm has run

**What changed.** The frozen set is now **5 disability variants × 50 unique questions = 250
items**, and every method is fit only on the **633 unique pairs of the other 96 questions**.
Two arms are added: CAD projection α=1 and CAA α=1 fit on the **full 1,044-pair pool** (eval
questions included), evaluated on the same 250 items. Tier 1 is now 23 arms.

**Why, reason 1 — a fit/eval leak.** Every method in every earlier run was fit on the full
loudness-filtered pool and evaluated on 250 items drawn from that same pool, so each steering
direction was partly fit on the evaluation prompts' own activations. The leak is shared by all
methods, so it does not bias the comparison between them, but it is a reviewer's question and E1
is the cheap moment to close it. The user chose held-out fitting plus a leak control.

**Why, reason 2 — AccessEval is duplicated.** Found while building the set, when the leak assert
fired on 145 items and 23 items were duplicates. Dataset row *i* is identical to row *i*+234 for
every *i*. So the benchmark has **234 unique questions and 2,082 unique pairs**; the filtered
"2,083-pair pool" is **1,044 unique pairs**; the "292 base queries" are **146 unique questions**.
The first amendment draft (2 variants × 125 base queries) assumed 292 independent units and would
have left 38 pairs to fit on. Questions are now identified as `base_query_id % 234` and the pool is
deduplicated before anything is drawn.

**Consequences, stated before running.**
- Eval spans 50 questions, not ~146. Intervals resample **questions** (50 units); they will be
  wider than the earlier item-level intervals, which were too narrow.
- Numbers from E1 are not directly comparable to the paper's current numbers: both the eval set
  and the fit pool changed. That is intended; E1 replaces them.
- The §6 cluster bootstrap (k 35.3 [33, 37] from 292 "clusters") double-counted and is being
  re-run with 146 questions. Its result does not depend on E1.

**Leak-control prediction and falsifier.**
- **P5.** On the frozen set, fitting on the full pool raises d by less than 0.10 for both CAD
  projection and CAA, relative to held-out fitting. **F5:** d rises by ≥ 0.10 for either.
- Named outcomes. Rise < 0.05 for both: the leak was negligible and the earlier numbers stand on
  that axis. Rise ≥ 0.10 for one method but not the other: the earlier *comparison* was biased and
  must be restated from E1 alone.

**Unchanged.** All gates G0–G3, predictions P1–P4, the second-judge requirement, and every
threshold. G2's positive control (SADI zero-score rate ≥ 0.50) stands.

Frozen set: `frozen_eval_v1.json`, sha256
`153439a63463081fe0773c26c6a259f1bc6a9cd4eafd40ffa49fab9aaf807658`, built by
`build_frozen_eval.py` (seed 42), rebuild-verified identical, and pinned in `frozen_eval.py`.

### A2 — 2026-09-27, before any E1 arm has run: the rank claim confounds rank with direction

**What was found (by reading the code, no data).** §5 claims "nothing changed between those
two numbers except the rank" for rank-1 LEACE (d = −0.03) vs rank-40 LEACE (d = 0.205). Beyond
the different eval sets (A1), the two are **different operators**: the rank-1 point comes from
`evaluate_leace_baseline.py`, whose LEACE concept is the *class label*; the rank-40 point comes
from `exp3_leace_rank_k.py`, whose concept is the *top-40 SVD coordinates*. And ranking directions
by SVD variance conflates *how many* directions with *which* ones: §6 already reports the top
singular direction is nearly orthogonal to the bias probe (cos = −0.001). The paper's own Table 1
shows a rank-1 method working: the CAD probe reaches d = 0.335 against the rank-40 projection's
0.337. So "rank governs whether the intervention does anything at all" is contradicted by data
already in the paper.

**Added arm.** A rank-1 **projection** along the supervised probe direction (CAD runner,
`--proj_basis probe --proj_alphas 1.0`). With the existing arms this gives three α = 1 projections
that differ in one variable each:

| arm | rank | direction |
|---|---|---|
| `svd_k1` | 1 | top-variance direction |
| `probe_proj` | 1 | supervised bias direction |
| `svd_k40` | 40 | top-variance subspace |

`probe_proj` vs `svd_k1` isolates **direction** at fixed rank and operator.
`svd_k40` vs `svd_k1` isolates **rank** at fixed direction family and operator.
Tier 1 is now 24 arms.

**P6 (decisive for C5), with every outcome named.** Using d on the frozen set with question-level
bootstrap CIs, and requiring judged quality within 1.0 of the unsteered mean for any arm counted
as "working":
- **(a) Direction, not rank.** `probe_proj` d is within 0.10 of `svd_k40` d. Then one correctly
  chosen direction suffices. §5 is rewritten: unsupervised SVD needs k ≈ 40 because its top
  directions are not the bias, not because the bias is 40-dimensional. The conclusion's sentence
  "that rank matters is a separate and simpler claim" is withdrawn.
- **(b) Rank, not direction.** `probe_proj` d is no more than 0.05 above `svd_k1` d, while
  `svd_k40` d exceeds both with non-overlapping CIs. Then rank matters beyond direction choice and
  §5 stands, restated on the matched arms.
- **(c) Both.** Anything in between. Report all three arms; no single-cause headline.

**Also fixed in the write-up regardless of outcome.** The old label-concept rank-1 LEACE
(d = −0.03) is a different operator and must not be placed on the same rank axis as the rank-k
sweep. The matched rank-1 LEACE point is `exp3` at k = 1.
