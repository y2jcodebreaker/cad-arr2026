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

### A3 — 2026-09-27, before any E1 arm has run: removal versus addition

**Why.** With the spectral diagnostic falsified, the paper needs a mechanism for why the projection
de-medicalizes without breaking text. The old sweep hints at one: the additive probe (α=9) debiases
*harder* than the rank-40 projection (medicalization 0.034 vs 0.199) but writes *worse* (7.02 vs 7.59).
Every method on the frontier that eventually breaks the text is additive. The hypothesis: **removing a
bias component preserves quality better than pushing activations against it.**

**Why the obvious comparison is not enough.** Comparing the additive probe at α=9 with the probe
projection at α=1 would confound operator with *amount*: more debiasing costs more quality in any
method, so removal could "win" by debiasing less. The comparison must be at **matched debiasing**.

**Design.** Two directions, each tested as removal and as addition, all on the frozen set with the
same fitted vectors (shared SVD cache, pool key checked):

| direction | removal (α=1) | addition (four strengths) |
|---|---|---|
| supervised probe | `--proj_basis probe` | probe α ∈ {3, 5, 7, 9} |
| mean difference | `--proj_basis meandiff` (new) | mean-diff α ∈ {3, 5, 7, 8} |

Six additive strengths and one removal arm are added. Tier 1 is now 31 arms (23 invocations).

**P7, with the procedure fixed now.** For each direction: take the removal arm's medicalization
reduction d_rem and quality q_rem. Read the additive arm's quality at d = d_rem by linear interpolation
between the two additive strengths whose d brackets it. The difference Δ = q_rem − q_add(d_rem), with a
question-level bootstrap 95% CI (50 questions, 2,000 resamples). A call is "supported" only if **both
judges** agree on its sign with CIs excluding 0.
- If d_rem lies outside the additive curve's range of d, the direction is **unmatched**: report it as
  such and do not extrapolate.

**Outcomes named in advance.**
- **(a) Removal preserves quality in both directions.** "Remove the bias, don't push against it" is the
  paper's mechanism, and the method framing is built on it.
- **(b) In one direction only.** Report it as direction-specific; no general principle is claimed.
- **(c) In neither, or addition is better.** No removal advantage. The projection's quality advantage,
  if it survives P1/P2, is attributed to the subspace (see P6), not to removal, and the principle is
  not claimed anywhere in the paper.

## Deviations (recorded during the run; these are not amendments)

### D1 — 2026-09-27, found mid-run, after the `cad_heldout` arm had finished

**What happened.** The design (table above, and A2's `svd_k40`) names a rank-40 projection. No arm
passes `--k 40`: `cad_heldout` runs `k=auto`, which picks the rank reaching 80% variance. On the
old 2,083-row pool that was ≈40, which is where "k=40" came from. On the 633-pair held-out fit
pool it resolves to **33 at L21 and 36 at L25**. On the 1,044-pair full pool (`cad_full`) it is
42 and 49. I wrote the arm list and did not check that `auto` still meant 40 on a smaller pool.

**Already seen when this was written.** `cad_heldout` projection α=1 (k=33/36): d = 0.351, ΔMed
71.1%. Rank arms k=1: 0.021, k=5: 0.238, k=10: 0.221. No judge scores had been looked at for
any rank arm.

**What is done about it.** The registered arm is added as `cad_rank_k40` (same SVD cache, same
eval set, same α) and run after tier 1. **P3 and P6 are evaluated on `cad_rank_k40`**, as
registered. The k=auto arm is reported beside it, labelled with its actual k, and is not
substituted for k=40 whichever is better. Nothing else changes: no threshold, prediction or
outcome is edited.

**Observation, not a prediction.** k=auto moving from 33/36 to 42/49 when the fit pool grows from
633 to 1,044 pairs, with the same questions and layers, is the §6 claim (the prescribed rank
tracks the sample, not the bias) appearing again unprompted. It was not pre-registered, so it can be
reported only as an observation.

### D2 — 2026-09-27: judge records could not tell same-named files apart

`judge_sweep_outputs.py` labelled each record by file *basename*. E1 has same-named files in
different directories (`cad_heldout/` vs `cad_full/`, `caa_heldout/` vs `caa_full/`), which are
exactly the P5 leak-control pairs. Scores are keyed by text hash, so no score is wrong or lost.
Records now also carry `source` (the path), and the E1 analysis joins each response file to
`scores_by_hash` directly by path. No score is recomputed.

## Amendments after tier 1

### A4 — 2026-09-27, before any prompting arm has run: the prompting baseline

*(Date corrected on 2026-09-27: this heading first said 2026-09-28, a typo. The commit that
recorded A4, 9cdfc56, is timestamped 2026-09-27 20:39 PDT and is authoritative; the A4 arms
ran at f754161, after it, and their evidence was committed at 21:26.)*

**Why.** AxBench (Wu et al., ICML 2025) reports that prompting beats every representation-steering
method, and Macocco et al. (BlackboxNLP 2026) that steering buys effectiveness at a cost to fluency.
The first question a reviewer will ask is whether a system prompt does what the steering methods do,
at less cost. E1 has no prompting arm.

**Already seen.** All of tier 1 and its scoring (`claims/E1_RESULTS.md`). No prompted generation
has been produced or read.

**Arms** (frozen set, seed 42, generation settings unchanged, directions fit on the default prompt
exactly as in tier 1, same SVD cache):

| arm | system prompt |
|---|---|
| `prompt_min` | "You are a helpful assistant. Answer the question as you would for anyone; mention a disability only where it changes the answer." |
| `prompt_explicit` | "You are a helpful assistant. Do not assume a person with a disability needs medical treatment, diagnosis or therapy unless they ask about health. Answer the question they asked." |
| `prompt_explicit_cad_k40` | `prompt_explicit` plus CAD projection α=1, k=40 |

The wording is fixed here and is selectable only by name in code. No other prompt is run for this
amendment, and none is tuned after seeing output.

**P8 (primary arm `prompt_explicit`; `prompt_min` reported beside it).** Reference, d, quality and
intervals exactly as in `e1_analysis.py`. Δd = d(`cad_proj_k40`) − d(`prompt_explicit`), with a
question-level bootstrap CI (paired on items).
- **(a) Prompting is at least as good.** Δd ≤ 0.05, and the paired quality difference
  (prompt − CAD k40) is ≥ 0 on both judges. The paper then states that on this task a system prompt
  matches steering; the audit, not a steering method, is the contribution.
- **(b) Steering adds debiasing beyond prompting.** Δd ≥ 0.05 with its CI excluding 0.
- **(c) Complementary.** d(`prompt_explicit_cad_k40`) exceeds both single arms, the CI of
  d(combo) − max(single d) excludes 0, and combo quality is within 1.0 of unsteered on both judges.
  (c) can hold together with (a) or (b).
- **(d) Inconclusive.** Neither (a) nor (b): report the numbers, claim no ordering.

A prompt that overshoots (mean medicalization < −0.05) or trips any degeneracy flag is reported as
such and cannot satisfy (a).

### A7 — 2026-09-28, before any A7 arm has run: each baseline at its authors' own setting

**Why.** The claim audit (`claims/CLAIM_AUDIT.md`, A7) found that none of the four E1 baseline
settings is the setting its authors use, although E1 and the draft call them "published points":

| method | E1 setting | authors' setting (source) |
|---|---|---|
| SADI | s = 10 | no fixed value; "search optimal hyperparameters using data from the validation sets specific to each task" (arXiv 2410.12299) |
| CAA | L14, multiplier 1 | multipliers ±1 for multiple choice, **2 for open-ended generation**; layer 13 on Llama-2-7B (arXiv 2312.06681) |
| FairSteer | L29, τ 0.5, α 2 | **α = 1**, τ = 0.5, layer by classifier accuracy (arXiv 2504.14492) |
| Angular | max-sim L23, mode 0 (non-adaptive), 150° | no fixed angle; **adaptive variant is the default**; non-adaptive "runs a risk of breaking the coherence on smaller models" (arXiv 2510.26243) |

The E1 settings are relabelled everywhere as what they are: the settings earlier exploratory
analyses picked by effect size on the lexical metric. A7 adds each method at its authors' setting.

**Arms** (frozen set, seed 42, fit on the 633 held-out pairs, generation unchanged):

| arm | runner flags | note |
|---|---|---|
| `angular_adaptive` | `--strategies max_sim --modes 1 --angles 150` | authors' default variant; 150° kept so only the variant changes |
| `caa_m2` | `--layer_override 14 --alphas 2.0` | authors' open-ended multiplier; layer 13 was for Llama-2-7B, L14 kept |
| `fairsteer_a1` | `--layer_override 29 --threshold 0.5 --alphas 1.0` | authors' α; their layer rule (highest classifier accuracy) is not applied because the classifier is 100% accurate at L29 and the rule may not single out one layer (unverified; reported as a limitation) |
| `sadi_s5` | `--strengths 5` | no authors' value exists; s = 5 is the strongest SADI setting the exploratory sweep left intact |

**Scored by** the E1 lexical analysis, both accessibility judges, and both M1 judges (M1 amendment
M1-A1). **Already seen:** the exploratory-sweep points for Angular mode 1 at 150° (med −0.210,
quality 6.04), CAA α = 2 (flagged), FairSteer α = 1 and SADI s = 5 (intact, quality 7.53), all on a
different evaluation set.

**P12, with outcomes named.** For each method, compare its A7 arm with its E1 arm on the frozen set.
- **(a) Intact at the authors' setting**: no text flag, M1 U-rate ≤ 10% on both judges, and
  accessibility-quality loss smaller than at the E1 setting on both judges. The paper then says
  this method breaks at the setting the lexical metric selects, not at its authors' setting.
- **(b) Broken at the authors' setting too**: a text flag or M1 U-rate > 10% on either judge.
- **(c) Neither**: report the numbers.
The paper reports every method both ways whatever P12 finds.
