# Quality-vs-debiasing frontier: analysis report

**Date:** 2026-09-14
**Data:** 87 sweep points, 21,750 generated responses, 5 steering methods
**Model:** Llama-3.1-8B-Instruct. **Judge:** Llama-3.1-8B-Instruct, temp 0.1, seed 42
**Benchmark:** AccessEval (medicalization bias), n=250 responses per sweep point

## Analysis question

Steering methods are ranked by a target metric that counts the absence of medical
vocabulary. Does a high score on that metric mean the bias was removed, or can it also
be produced by destroying the response? And once quality is held constant, how do the
methods actually rank?

## Key findings

**1. The target metric can be satisfied by destroying the output.**
At its published operating point, SADI scores 0.119 on medicalization (strong apparent
debiasing) while judged quality falls from 7.92 to 3.19, a drop of 4.73 points on a
ten-point scale (rank-biserial -0.983, Holm-corrected p < 1e-190). The responses are not
empty. They average 188 words and contain token corruption, so the log-odds contrast
between medical and professional vocabulary evaluates to zero.

**2. All five published operating points lose quality, and they differ by an order of
magnitude in how much.**

| Method | Medicalization | Judged quality | Change vs baseline | Rank-biserial |
|---|---|---|---|---|
| CAD projection (a=1) | 0.199 | 7.59 | -0.33 | -0.134 |
| CAA (a=1) | 0.146 | 7.29 | -0.63 | -0.276 |
| FairSteer (a=2) | 0.184 | 6.68 | -1.24 | -0.497 |
| Angular (150 deg) | -0.222 | 6.06 | -1.86 | -0.689 |
| SADI (s=10) | 0.119 | 3.19 | -4.73 | -0.983 |

All five are significant after Holm correction across the five comparisons.

**3. Once degenerate points are excluded, the frontier is dominated by the CAD
operators.** The clean Pareto front holds 12 points, 10 of them CAD. Restricting to
points that achieve at least 50% bias reduction, **all 4 remaining front points are CAD
operators** (probe a=9, mean-diff a=8 and a=7, projection a=1). No baseline method
reaches that region without degenerating.

**4. Every method collapses past a strength threshold, but the thresholds differ.**
CAA and FairSteer collapse between strength 2 and 4. The CAD operators hold quality to
roughly 10 to 15 in their own units. This is a wider operating window, not immunity:
CAD projection also degenerates at a >= 2 and produces 75 empty responses at a = 5.

**5. Three distinct degeneration signatures, none caught by a single filter.**
SADI produces token corruption (36.4% unparseable), FairSteer produces single-word
repetition (68 of 250 responses distinct at a=6), CAA produces phrase loops (247 of 250
distinct, 637 words). An empty-response check catches none of them. A length filter
catches none, since all three are longer than baseline.

## What changed in our understanding

The leaderboard ranking and the quality-constrained ranking are different orderings.
SADI ranks first on the target metric in this comparison and last on quality. The
practical consequence is that a practitioner selecting a method from published numbers
would deploy the one that damages responses most, and the users affected are the ones
the intervention was meant to protect.

## Main caveats

- **The frontier is unpaired.** Sweep points were generated from independently shuffled
  250-prompt draws of the same 2,083-pair pool. Only CAA saved prompt text, so overlap
  between methods cannot be verified. The frontier supports ordering and shape claims,
  not precise paired effect sizes. The July paired test (identical items, shared
  baseline, +0.35, 95% CI [0.18, 0.53], Wilcoxon p < 0.001) carries that claim instead.
- **CAD debiases slightly less than CAA at the published point** (0.199 vs 0.146). The
  claim is Pareto superiority across the curve, not dominance at a single point.
- **CAD's medicalization scores are aggregate only.** Its runner saves responses as
  plain strings, so per-response medicalization is unavailable and no CI can be placed
  on its x-coordinate.
- **CAD's own baseline is 0.593 against 0.659 +/- 0.043 for the baseline runners**,
  consistent with a different prompt draw. Baseline judged quality agrees closely across
  all methods (7.87 to 7.98), so quality is comparable even where the draws differ.
- **Single benchmark, single model, single judge family.** An independent GPT-5.5 judge
  agrees on the extremes (CAD best, Angular second worst, SADI worst) but reverses CAA
  and FairSteer in the middle, so the middle of the ranking is not stable.

## Blockers

None for the figures. Per-response medicalization for CAD would require re-running its
sweep with the scoring hook enabled, which is not needed for any claim made here.
