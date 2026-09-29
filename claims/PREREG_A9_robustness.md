# Pre-registration A9: is the A8 result robust, specific, and harmless?

**Status: FROZEN, 2026-09-29.** Approved as written by the author before any A9 code, answer,
direction or J3 rating exists. Changes after this commit are amendments with a timestamp and a reason.

## 1. Why

A8 (`claims/A8_RESULTS.md`, outcome (a)) found that removing the framing-contrast direction at
layers 9 and 11 lowers the M1 medicalization rating on both judges without erasing the disability
or losing accessibility quality. Four objections remain, each recorded as a caveat there:

1. **Circularity.** The two M1 judges that labelled the fit answers also score the outcome.
2. **Specificity.** Any projection at layers 9/11 might make answers blander and so lower J.
3. **Mechanism.** The paper will say that *input* contrasts (disability vs neutral) capture identity,
   so removing them erases the disability, while contrasting answers *within* the disability
   condition captures framing. A8 compared layers (9/11 vs 21/25), not contrast types.
4. **Scope.** 50 questions (211 non-Healthcare items); one question distribution; no side-effect check.

Literature: the claim under test is A8's, whose novelty check was done 2026-09-28 (PREREG_A8 §2).
The controls and judges below make no new novelty claim.

## 2. Already seen

All E1, M1, A7 and A8 results, including every A8 arm's text and both M1 judges' ratings of them.
Nothing produced by the third judge, any control direction, the neutral-side arm, the perplexity
check, or the fresh set exists. Seen while designing: the fresh pool's size (88 questions, 768 unique
pairs, 207 of them Healthcare), not its answers.

## 3. Instruments

- **Third judge J3: `mistralai/Mistral-Small-24B-Instruct-2501`, revision
  `9527884be6e5616bdd54de542f9ae13384489724`** (Apache 2.0; a third model family; never used for
  fitting). Frozen M1 rubric, prompt and parse rule unchanged (rubric sha a5ec3d92…), greedy.
  *Known deviation, decided now:* the model's chat template calls `strftime_now`, which transformers
  4.44.2 (the pinned version) lacks. We pass the official template with its first two lines (the date
  and the default system message) removed. With our system message present the rendered prompt is
  identical to the official template's.
- M1 judges (Llama, Qwen), accessibility judges, echo and lexicon: unchanged from M1/A8.

## 4. Arms (Llama-3.1-8B-Instruct, seed 42, greedy; E1 hooks; frozen set unless stated)

| arm | direction at layers 9, 11 | operator | set |
|---|---|---|---|
| `fc_shuffle_s0/s1/s2` | A8 procedure with medicalizing/clean labels permuted **within domain** (seeds 0, 1, 2) | projection α = 1 | frozen |
| `id_remove_a1` | identity contrast: domain-stratified mean over fit items of (disability-answer features − the same question's neutral-answer features) | projection α = 1 | frozen |
| `fc_remove_a1_neutral` | the A8 direction | projection α = 1 | frozen set, **neutral** questions |
| `fresh_unsteered`, `fresh_fc_remove_a1`, `fresh_fc_prompt_remove_a1`, `fresh_prompt_explicit` | none / A8 / A8 + explicit prompt / explicit prompt only | as in A8 | **fresh** |

- **Features** for every direction are A8's (mean raw layer output over answer tokens, the same fit
  items, Healthcare excluded). `id_remove_a1` needs one unsteered answer to each of the 96 fit
  questions' neutral forms (default prompt), generated and featurised the same way.
- **Fresh set:** the 88 AccessEval questions used by neither the frozen set nor the fit pool (they
  failed the E1 loudness filter). From each, 5 categories drawn with `random.Random(0)` (all of
  them if fewer than 5), giving about 428 items. It is a mild distribution shift, not an i.i.d. copy.
- **Perplexity:** WikiText-2 raw test split, 1,024-token non-overlapping windows, for unsteered,
  `fc_remove_a1` and `fc_remove_a2` (hooks active at every position, as in generation).

## 5. Gates (in order; a failure stops the run and is reported)

- **GA9-0 (reproduction, positive control).** Recomputing the A8 features and direction gives
  cosine ≥ 0.999 with the committed `a8_directions.pt` at layers 9 and 11.
- **GA9-1 (J3 usable).** On `unsteered_s42`: J3 parse failure ≤ 10% and U-rate ≤ 10%.
- **GA9-2 (J3 agrees).** Per-text Spearman ρ between J3 and each M1 judge ≥ 0.5, over all texts J3
  scores (U and unparseable as missing).

## 6. Questions, criteria and named outcomes

Question-level bootstrap (2,000 resamples), non-Healthcare items, reference unsteered seed 42 of the
same set; 95% intervals. Each question has **one** confirmatory arm, so no multiplicity correction.

**Q-J3, independent judge.** Confirmatory arm `fc_remove_a1`, frozen set, rated by J3.
- (a) ΔJ3 < 0 with the interval excluding 0: the reduction holds on a judge never used for fitting.
- (b) ΔJ3 < 0, interval includes 0: not confirmed.
- (c) ΔJ3 ≥ 0: contradicted.
Also reported: J3 for every frozen-set arm scored, and J3's disability − neutral gap (M1 Q1).

**Q-SPEC, specificity.** Per item, average the three shuffled arms' ratings. Confirmatory difference:
`fc_remove_a1` − shuffle mean, on each M1 judge.
- (a) Specific: the difference is < 0 with the interval excluding 0 on both judges.
- (b) Not specific: on at least one judge the interval includes 0 while the shuffle mean itself
  reduces J (its ΔJ interval vs unsteered excludes 0 below): random framing-like directions do it too.
- (c) Neither: reported.

**Q-MECH, contrast type** (`id_remove_a1`, same layers, operator, features; only the contrast differs).
- (a) Erasure: echo drop ≥ 0.25. Supports the mechanism claim.
- (b) Reduction without erasure: ΔJ interval excludes 0 below on both M1 judges and echo drop < 0.25.
  The contrast type is not what matters; the mechanism claim is withdrawn.
- (c) Neither. Reported; the paper then says only that identity removal at these layers does not
  reduce medicalization, not that it erases.
Also reported: cosine between the identity and framing directions at layers 9 and 11.

**Q-SIDE, side effects.**
- (a) Negligible: on neutral questions, ΔJ intervals lie within [−0.10, +0.10] on both M1 judges, and
  WikiText-2 perplexity rises by ≤ 5% relative for `fc_remove_a1`.
- (b) Otherwise: the side effect is reported with its size.
Also reported: mean length change and share of neutral answers identical to unsteered.

**Q-FRESH, new questions.** Confirmatory arm `fresh_fc_remove_a1`, A8's success rule at 95%: ΔJ < 0
with the interval excluding 0 on both M1 judges; echo drop < 0.25; accessibility-quality change
≥ −1.0 on both accessibility judges; M1 U-rate ≤ 10% on both.
- (a) All hold: the A8 result carries to unseen questions.
- (b) ΔJ criterion holds, another fails: reported by which.
- (c) ΔJ criterion fails: not carried over. If unsteered J on the fresh set is below half its
  frozen-set value (0.54 Llama / 0.31 Qwen), the result is reported as under-powered, not negative.
Also reported: J3 on the fresh arms; the other three fresh arms against the same rule.

## 7. What each outcome does to the paper

- Q-J3 (a) removes the circularity caveat; (b)/(c) keep it prominent in Limitations and weaken the
  method claim to "on the judges it was fit with".
- Q-SPEC (a) supports "the direction matters"; (b) withdraws it.
- Q-MECH decides whether §Method can say *why* A8 avoids erasure (a) or only *that* it does (c).
- Q-SIDE and Q-FRESH are reported whatever the outcome.

## Amendments

### A9-A1 — 2026-09-29, before any A9 code has run (no answer, direction or J3 rating exists)

Clarifications of points the frozen text leaves open; none changes a criterion or an outcome.

1. **Identity-direction weights.** "Domain-stratified mean" uses the A8 weights
   w_d ∝ min(n_med,d, n_clean,d) over the same 480 labelled fit items, so the identity direction
   differs from the A8 direction *only* in the contrast (disability answer − same question's neutral
   answer, instead of medicalizing − clean).
2. **Shuffles.** Within each domain, the labels of the 480 labelled items are permuted with
   `numpy.random.default_rng(seed)`, seed 0, 1, 2; class counts per domain are unchanged.
3. **Arms J3 scores** (fixed now, before any J3 rating): on the frozen set `unsteered_s42`,
   `neutral_s42`, `prompt_explicit`, `prompt_explicit_cad_k40`, `cad_proj_k40`, `caa`, `angular`,
   `fairsteer`, `sadi`, the four A7 arms, the six A8 arms and the five A9 frozen-set arms; and the
   four fresh arms. GA9-2 is computed over exactly these texts.
4. **References.** `fc_remove_a1_neutral` is compared with `neutral_s42` (M1's unsteered neutral
   answers); fresh arms with `fresh_unsteered`.
5. **Fresh sampling order.** Questions in ascending question id; within a question, its unique pairs
   sorted by category name, then `random.Random(0).sample(pairs, min(5, n))` with one generator
   shared across questions in that order.
6. **Perplexity.** Text = the test split's lines joined with "\n\n" (the Hugging Face perplexity
   guide); the final partial 1,024-token window is dropped;
   perplexity = exp(mean token negative log-likelihood).
