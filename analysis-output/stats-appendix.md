# Statistics appendix

## Unit of analysis and sample sizes

- **Unit:** one sweep point (a method at one configuration), n = 250 generated responses.
- **Sweep points:** 87 (CAA 7, FairSteer 21, SADI 7, Angular 24, CAD 28).
- **Total responses:** 21,750. **Unique texts:** 19,888. **Unscored:** 0.
- **Baseline reference:** 10 no-steering points pooled, 2,500 responses.
  Judged quality 7.920 +/- 0.041 (SD across points). Medicalization 0.659 +/- 0.043.
- Responses within a sweep point are different prompts and treated as independent.
  The same prompt recurs across sweep points, so cross-point comparisons are not
  independent between points; they are reported as unpaired regardless, which is
  conservative for the CI width.

## Metric directions

- **Medicalization score:** lower is more bias removed. 0 is neutral. Below 0 is
  overshoot, meaning medical information is withheld from users who asked for it, and
  is not an improvement.
- **Judged quality:** 0 to 10, higher is better.

## Descriptive statistics and intervals

Per-point judged quality means carry percentile bootstrap 95% CIs, 2,000 resamples over
the 250 responses, seed 42. Full table in `sweep_points.csv`.

## Inferential tests

**Test choice.** Judge scores take 8 distinct values (0, 2, 4, 5, 6, 8, 8.5, 9) and are
strongly left-skewed at degenerate points. Normality fails, so the Mann-Whitney U test
was used rather than a t-test, with rank-biserial correlation as the effect size.

**Published operating points vs pooled baseline** (n = 250 vs 2,500, two-sided,
Holm-corrected across the five comparisons):

| Method | Mean | Change | Rank-biserial | p (Holm) |
|---|---|---|---|---|
| SADI | 3.19 | -4.73 | -0.983 | 9.2e-192 |
| Angular | 6.06 | -1.86 | -0.689 | 3.1e-99 |
| FairSteer | 6.68 | -1.24 | -0.497 | 1.0e-53 |
| CAA | 7.29 | -0.63 | -0.276 | 3.4e-18 |
| CAD projection | 7.59 | -0.33 | -0.134 | 2.0e-05 |

**CAD projection (a=1) vs CAA (a=1), head to head, unpaired:**
means 7.590 vs 7.292, difference +0.298, bootstrap 95% CI [0.072, 0.530],
Mann-Whitney p = 0.0014, rank-biserial 0.141.
This is consistent with the July paired test (+0.35, CI [0.18, 0.53]) but is a weaker
design: the two points come from independently shuffled prompt draws.
On the target metric CAA removes slightly more bias (0.146 vs 0.199).

## Degeneracy criterion

Degeneracy was defined on **text and target-metric properties only**, so that judged
quality remains an independent outcome rather than part of the definition. A point is
flagged if any of:

- **repetition:** unique response ratio < 0.50 (computed from response hashes)
- **unparseable:** parse-failure rate > 0.10
- **overshoot:** mean medicalization < -0.05

38 of 87 points are flagged. Judged quality falls sharply at flagged points, which is a
validation of the criterion rather than a restatement of it.

## Parse-failure sensitivity analysis

The judge assigns 5.0 to empty or unparseable output. That value is indistinguishable
from a genuine mid score in the saved records, so it was checked directly.

**At clean baseline points, 0.00% of 2,500 scores equal 5.0 exactly.** The judge never
emits 5.0 on intact text, so in this run 5.0 marks failure rather than mediocrity.

Recoding every 5.0 to 0.0:

| Method | As scored | Parse-fail rate | Recoded to 0 |
|---|---|---|---|
| SADI | 3.19 | 36.4% | 1.37 |
| Angular | 6.06 | 1.2% | 6.00 |
| FairSteer | 6.68 | 0.8% | 6.64 |
| CAA | 7.29 | 0.0% | 7.29 |
| CAD projection | 7.59 | 0.8% | 7.55 |

The 5.0 default pulls degenerate points upward, so the reported quality drops are
**conservative**. SADI's true collapse is larger than the headline number shows. No
conclusion reverses under either coding.

## Multiple comparisons

Holm correction applied across the five published-operating-point comparisons. The
Pareto front is a descriptive geometric construction and carries no p-value.

## Limitations

1. Unpaired across sweep points; prompt overlap unverifiable for 4 of 5 methods.
2. CAD medicalization is aggregate only, so no CI on the x-axis for CAD points.
3. Single benchmark, single model. Judge and generator share an architecture; the
   independent GPT-5.5 judge agrees on the extremes but swaps CAA and FairSteer.
4. Angular's parameter is an angle and is circular, so it is excluded from the
   dose-response panel; 270 degrees is not a stronger dose than 150 degrees.
5. The judge emits only 8 distinct values, which limits resolution on small differences.
