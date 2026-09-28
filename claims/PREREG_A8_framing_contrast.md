# Pre-registration A8: framing-contrast steering

**Status: FROZEN, 2026-09-28.** Design approved by the author before any A8 generation, label or
activation exists. Changes after this commit are amendments with a timestamp and a reason.

## 1. Why

M1 (`claims/M1_RESULTS.md`) found that Llama-3.1-8B medicalizes answers to disability-framed
questions (Q1-a), that the CAD projection does not reduce it on both judges (Q2) while erasing the
user's disability name (Q3), and that the lexical score tracked naming rather than framing (Q4).
The claim audit (A10) found why: CAD's contrast set pairs a disability question with its neutral
form, and its layers were located with a vocabulary of disability names. Both select for *the
disability is mentioned*. Removing that erases the mention.

**Hypothesis.** Contrasting the model's own answers *within* the disability condition, labelled for
medical framing by the validated M1 judges, isolates framing from identity: both sides mention the
disability, so the identity signal cancels. Removing that direction should reduce medicalization
without erasing the disability.

## 2. Novelty check (2026-09-28, before this design)

Existing debiasing steering contrasts *inputs* that differ in the group attribute (CAA, FairSteer,
Shifting Perspectives, BiasGym, CAD). Response-based directions exist in other settings (ITI uses
labelled answers; persona-style directions from generated responses, to be verified before citing).
Steering that pushes models to "ignore group identity" is reported as debiasing ("Hedging and
Non-Affirmation", arXiv 2502.19463); identity-robust generation by prompting neutralises identity
information (arXiv 2601.09141). Not found: a debiasing direction contrasted within one group
condition and labelled for the harmful framing, with erasure measured.

## 3. Fitting (training questions only; no frozen-set question is used)

1. **Answers.** One unsteered answer (seed 42, generation settings unchanged, default system prompt)
   to each of the 633 fit-pool disability questions of E1. Domain and category from
   `day1_extract_activations.build_pairs_with_metadata`, matched by pair key.
2. **Labels.** Both M1 judges, frozen M1 rubric and parse rule. *Medicalizing*: both ratings ≥ 1.
   *Clean*: both ratings 0. Otherwise excluded. Healthcare-domain questions excluded.
3. **Features.** Forward pass on the formatted prompt plus the answer; for every decoder layer
   0–31, the mean of the layer output (the hook point of E1) over the answer tokens.
4. **Layer rule.** At each layer, 5-fold stratified cross-validated accuracy (seed 0) of an
   L2 logistic regression (C = 1, features standardised, domain means removed) separating
   medicalizing from clean. **Steer at the two layers with the highest mean accuracy** (ties: lower
   layer).
5. **Direction.** At each steered layer, v = Σ_d w_d (mean_med,d − mean_clean,d) over domains d with
   both classes present, w_d ∝ min(n_med,d, n_clean,d), then v / ‖v‖.

## 4. Gates

- **GA8-1.** At least 40 medicalizing and 40 clean answers after exclusions; else stop.
- **GA8-2.** Best-layer CV accuracy ≥ 0.60; else the framing is not linearly separable in these
  features: stop and report.

## 5. Arms (frozen 250-item set, seed 42; operators are E1's own hooks)

| arm | layers | operator | strength |
|---|---|---|---|
| `fc_remove_a1` | selected two | projection, `make_projection_hook(v, α)`: h − α (h·v) v | α = 1 |
| `fc_remove_a2` | selected two | projection | α = 2 |
| `fc_add_a4` | selected two | additive, `make_probe_hook(v, α)`: h − α v | α = 4 |
| `fc_add_a8` | selected two | additive | α = 8 |
| `fc_prompt_remove_a1` | selected two | explicit A4 system prompt + projection | α = 1 |
| `fc_remove_a1_L21L25` | 21, 25 | projection, direction fit at those layers | α = 1 |

Scored by both M1 judges (J, U, echo; M1 amendment M1-A2), both accessibility judges, and the lexical
score. Reference: unsteered seed 42.

## 6. Success criteria and named outcomes

An arm **succeeds** only if all hold: ΔJ < 0 with the Bonferroni-adjusted interval (6 arms: 99.17%,
question-level bootstrap, 2,000 resamples, non-Healthcare items) excluding 0 on **both** judges;
echo drop < 0.25; accessibility-quality change ≥ −1.0 on both judges; M1 U-rate ≤ 10% on both.

- **(a)** At least one arm succeeds: framing-contrast steering reduces medicalization without
  erasing the disability. This is the paper's method contribution.
- **(b)** An arm reduces J on both judges but its echo drop is ≥ 0.25: it erases again; no advantage.
- **(c)** No arm reduces J on both judges: null, reported.

Secondary, reported whatever happens: paired ΔJ of the best succeeding arm against the explicit
prompt and against CAA; `fc_remove_a1` against `fc_remove_a1_L21L25` (does the layer rule matter).

## 7. Already seen

M1 and all earlier results. No A8 answer, label, activation, direction or selected layer exists.

## Amendments

### A8-A1 — 2026-09-28, before any A8 answer, label or activation exists

**What changed.** The layer rule (section 3.4) and gate GA8-2 use **balanced accuracy** (mean of the
per-class recalls; chance = 0.50 whatever the class balance) instead of accuracy. Threshold unchanged
at 0.60.

**Why.** Found by the synthetic test of the layer rule: with about 30% medicalizing answers, a
classifier that always predicts "clean" scores 0.70 accuracy, so GA8-2 (accuracy ≥ 0.60) could not
fail, and plain accuracy also rewards layers where the classifier leans on the majority class. No A8
data existed when this was found.
