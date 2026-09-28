# E1 results — scored against `PREREG_E1_frozen_eval.md`

Scored 2026-09-27 by `e1_analysis.py` from `results/e1/` (commit 370fe66). Numbers are in
`analysis-output/e1/e1_scores.json`. Reference = unsteered seed 42; d on the pinned scorer,
recomputed from text and checked against every runner's stored score (max diff < 1e-9);
intervals = question-level bootstrap, 50 questions, 2,000 resamples. Response order checked
against the frozen set (category named in own item vs rotated: worst margin +0.71).

## Gates

| gate | criterion | result |
|---|---|---|
| G0 | every arm builds its data and fit pool | PASS |
| G1 | 3 seeds, sd ≤ 0.06, mean in [0.45, 0.80] | PASS: 0.597 / 0.603 / 0.590, sd 0.0065 |
| G2 | SADI s=10 zero-score ≥ 0.50; CAD proj α=1 intact | PASS: 0.58; unique 1.00, empty 0, med +0.17 |
| G3 | judge Spearman ρ ≥ 0.5 | PASS: 0.642 over 3,247 texts |

Reference noise: a different unsteered draw moves d by at most 0.019 and quality by at most 0.05.

## Predictions

| | prediction | verdict | numbers |
|---|---|---|---|
| **P1** | SADI largest quality drop, CAD projection smallest, both judges | **SUPPORTED** | Llama drops: SADI 3.93, Angular 2.07, FairSteer 1.31, CAA 0.80, CAD k40 0.59. Qwen: 2.38, 0.66, 0.61, 0.30, 0.23. Same order on both |
| **P2** | paired CAD proj − CAA quality, Qwen CI excludes 0 | **NOT SUPPORTED (F2 fired)** | Qwen +0.06 [−0.03, 0.15]; Llama +0.22 [−0.12, 0.57] |
| **P3** | d(k40) > d(k1), separate CIs, k40 quality within 1.0, **both operators** | **NOT SUPPORTED (F3a fired for LEACE)** | projection: 0.02 [−0.05, 0.10] vs 0.34 [0.21, 0.48], separate, quality ok. LEACE: 0.01 [−0.08, 0.08] vs 0.17 [0.05, 0.29], overlap by 0.03 |
| P4 | tier 2 only | not run | — |
| **P5** | full-pool fit raises d by < 0.10 | **SUPPORTED, negligible (< 0.05)** | CAD +0.004, CAA +0.001 |
| **P6** | direction vs rank | **outcome (c) both** | probe-direction rank-1 projection d 0.08 [−0.01, 0.18]; SVD k1 0.02; SVD k40 0.34 [0.21, 0.48]. (a) refuted (0.08 is not within 0.10 of 0.34). (b) missed on its first condition: probe − k1 = 0.062 > 0.05; its second condition held |
| **P7** | removal vs addition at matched d | **unmatched in both directions** | rank-1 removal reaches d 0.08 (probe) and 0.05 (mean-diff), below every additive point (lowest 0.12 and 0.15). Registered rule: do not extrapolate. The principle is not claimed |

## Named outcomes that now apply

- **"I3 reverses P1 or P2"**: P2 fails on both judges. **Drop "roughly half the quality cost"
  from the abstract.** At matched debiasing (CAD k40 d 0.34, CAA d 0.37) the two are not
  distinguishable in quality on either judge; point estimates favour CAD.
- **P3 as registered is not supported**, because it required both operators. The projection half
  holds clearly. The unwelcome outcome "k=1 within 0.05 of k=40" did not happen for either operator.
- **P6 (c)**: report all three arms, no single-cause headline. It may be stated that one
  supervised direction removed at rank 1 does not reproduce the rank-40 effect.
- **P7**: the removal-versus-addition mechanism is not claimed anywhere in the paper.

## Observations (not pre-registered; report only as such)

- Projection d rises with k: 0.02, 0.24, 0.22, 0.31, 0.34, 0.45 (k = 1, 5, 10, 20, 40, 80), and
  quality cost rises with it (Llama −0.03 → −0.92). Rank behaves as a strength setting.
- Additive CAD mean-diff α=7 reaches d 0.43 (91% reduction), intact, quality −0.72 / −0.24. On
  tier-1 point estimates it is the only arm that dominates CAA. The projection does not dominate CAA.
- Angular has the highest d of any arm (0.74) and is flagged for overshoot (mean med −0.26, a
  143% "reduction"). SADI (d 0.30) is flagged unparseable. The metric's top-ranked arm is a
  broken one, on held-out questions, under both judges.
- k=auto gives 33/36 on the 633-pair pool and 42/49 on the 1,044-pair pool (D1).
- The runner's own `is_degenerate` flagged additive probe α=9; the registered text criteria do not.

## A4 — prompting baseline (scored 2026-09-27, evidence commit on `feature/e1-a4-results`)

| arm | med | d [95% CI] | ΔMed% | ΔQ Llama | ΔQ Qwen | flags |
|---|---|---|---|---|---|---|
| `prompt_min` | 0.642 | −0.04 [−0.14, 0.06] | −7.6 | −0.03 | −0.03 | — |
| `prompt_explicit` | 0.517 | 0.07 [−0.02, 0.15] | 13.4 | +0.01 | +0.07 | — |
| `cad_proj_k40` (tier 1) | 0.202 | 0.34 [0.21, 0.48] | 66.2 | −0.59 | −0.23 | — |
| `prompt_explicit_cad_k40` | 0.015 | **0.51 [0.37, 0.66]** | 97.4 | −0.70 | −0.20 | — |

**P8 outcome: (b) and (c).**
- **(b) Steering adds debiasing beyond prompting.** Δd = d(CAD k40) − d(prompt_explicit) = 0.28
  [0.14, 0.44]. (a) fails: the prompt keeps quality (+0.59 [0.36, 0.85] Llama, +0.30 [0.20, 0.40]
  Qwen over CAD k40) but barely debiases.
- **(c) Complementary.** The combination reaches d 0.51, above CAD k40 by 0.08–0.26 (CI), with
  quality within 1.0 of unsteered on both judges, no degeneracy flag (unique 1.00, parse-fail 0.004).

**Observations (not pre-registered):**
- The minimal prompt slightly *raises* medicalization (NS). An explicit instruction removes 13%.
  This matches Macocco et al. (2026): prompting is viable for injecting a concept, weaker at removing one.
  AxBench's "prompting beats steering" does not transfer to this removal task.
- On tier-1 + A4 point estimates, the combination is dominated by no intact arm and dominates CAA,
  FairSteer, LEACE k80, CAD k80 and every additive CAD arm. No intact arm reaches a higher d. This is
  a Pareto statement on point estimates; P4 (tier 2) is still the registered test of the front.
- The directions were fit on default-prompt activations and still work under the explicit prompt.
