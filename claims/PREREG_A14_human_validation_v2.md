# Pre-registration A14: calibrated human validation of the medicalization rating

**Status: FROZEN, 2026-09-30.** Approved as written by the author before any A14 item was selected or shown.
Changes after this commit are amendments.

## 1. Why, and what was seen

A12 (frozen 42f40d8) ended in outcome (c): the two volunteer raters did not agree with each other
(ρ −0.004), so it neither validated nor refuted the LLM judges (`claims/A12_RESULTS.md`). The exploratory
diagnostics point to how the task was applied: one rater rated fast (median 15 s) and rarely marked medical
framing, the other marked most answers medicalizing and used U for readable answers. A14 is a new,
separately reported study that changes the procedure accordingly. **A12 is reported in full whatever A14
shows; A14 does not replace it.**

## 2. Raters

Two volunteers, not authors, fluent in English, unpaid, informed and consenting. **Raters (decided at freezing): the same two volunteers as A12** (both master's students in computer science);
this is reported. No A14 item repeats an A12 item. Raters are blind to method and hypotheses and do not discuss
items with each other until both have finished.

## 3. Items (62; drawn before any rating, `random.Random(14)`)

- Pool: non-Healthcare frozen-set items **not used in A12**, answers from `unsteered_s42`, `fc_remove_a1`,
  `caa`, `prompt_explicit`, `mistral_unsteered`, `mistral_fc_remove_a1`; one answer per frozen-set item.
- **Stratified by the judges** so the scale's upper range is present: 20 answers that both M1 judges rate
  ≥ 1 ("judged medicalizing") and 40 that both rate 0. (A12's sample was about 80% zeros.)
- **2 attention checks**, invented (not model output): a broken, repetitive text (expected U) and an answer
  that is almost entirely medical advice (expected 3). Placed at random positions; analysed separately.
- Answers capped at 2,000 characters as the judges saw them. Separate random order per rater. Arms hidden.

## 4. Procedure (changes from A12)

1. The same rater guide (frozen with A12) and the M1 rubric beside every item.
2. **Calibration with feedback**: 4 invented practice answers (target 0, 1, 2, 3; text in `human_eval/a14/`)
   after which the page shows the intended rating and one sentence why. Not analysed.
3. **Minimum reading time**: the Next button unlocks 20 seconds after an item appears.
4. **U reminder** beside the U button: "Only if the text is broken, empty, repetitive or off-topic."
5. Backups every 20 answers and restore, as the A12 page.

## 5. Analysis, criteria and named outcomes (same criteria as A12)

Numeric ratings (U as missing); 95% bootstrap over question ids (2,000, seed 0); the 60 study items only.
- **(a) Measure supported**: human–human ρ ≥ 0.5 and mean-human vs each M1 judge (Llama, Qwen) ρ ≥ 0.5.
- **(b)** Human–human ρ ≥ 0.5 but some M1 judge < 0.5.
- **(c)** Human–human ρ < 0.5.

Also reported: quadratic κ; each rater's share rated ≥ 1 in the two strata and the stratum AUC of the mean
human rating; J3 agreement; attention-check results per rater; median seconds per item. No rater or item is
excluded from the primary analysis; an attention-check failure is reported next to that rater's results.

## 6. Already seen

All A12 ratings and results; all judge ratings (used for stratification by design).

## Amendments

None.

### A14-A1 — 2026-09-30, before any rater has seen an A14 item

1. **Page**: https://claude.ai/artifact/9QGzevdGqmJHXRBa5Us6tS (`human_eval/a14/rating_page.html`, built by
   `human_eval/build_page.py a14`; items sha 1a960f3a…). Exports are saved unedited as
   `human_eval/a14/ratings_R1.json` / `ratings_R2.json`; `a14_analysis.py` (committed now) scores them.
2. **Disclosed limitation of the stratified design, found by the stand-in test**: items were drawn where
   both M1 judges agree (both ≥ 1 or both 0), so judge–judge agreement on this sample is near perfect and
   human–judge correlations here will be higher than on a random sample of answers. A14 tests whether
   humans separate clear medicalization from clear non-medicalization as the judges do; it does not
   estimate agreement on typical, borderline answers. The paper states this.
