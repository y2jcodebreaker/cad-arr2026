# Pre-registration A12: human validation of the medicalization rating

**Status: FROZEN, 2026-09-30.** Approved as written by the author (definition, score descriptions, design)
before any item was selected or shown to a rater. Changes after this commit are amendments.

## 1. Why

Every medicalization result rests on LLM judges (Llama, Qwen, and J3 Mistral-Small). The most likely
reviewer objection is that no human has checked the measure. A12 asks two volunteers to rate a sample of
answers with the same rubric, blind to method.

## 2. Raters

Two volunteers (friends of the first author, not authors, fluent in English, no NLP background required),
unpaid, informed of the task and that their ratings (not their names) will be reported. They are not told
which method produced any answer or what the study expects. Each rates every item independently.

## 3. Items (110; drawn with `random.Random(0)` from non-Healthcare frozen-set items, before any rating)

| block | arms | items | purpose |
|---|---|---|---|
| Llama pairs | `unsteered_s42`, `fc_remove_a1` | 30 questions × 2 = 60 | validity; direction (secondary) |
| Mistral pairs | `mistral_unsteered`, `mistral_fc_remove_a1` | 15 questions × 2 = 30 | second model |
| range | `caa` 10, `game_append` 5, `fc_add_a8` 5 | 20 | other methods, the A10 edit, broken text (U) |

Questions in different blocks are distinct. Each item shows the question and the first 2,000 characters of
the answer followed by "[answer continues]", exactly what the LLM judges saw. Item order is shuffled
separately for each rater (seeds 1 and 2); item ids are opaque; the arm key stays in the repository, never
on the rating page. Three practice items from the fit pool (not in the evaluation set) come first, without
feedback, and are not analysed.

## 4. Rubric and rater guide

Every item shows the M1 rubric text verbatim (ratings 0, 1, 2, 3, U), the same text the LLM judges saw.
Before the practice items, raters read `human_eval/rater_guide.md`: a plain-language definition of
medicalization, what counts and what does not, a one-line description of each rating, and five
**invented** examples (one per rating, not drawn from any study item). The guide paraphrases the rubric
and adds no category, threshold or exclusion that the rubric lacks; its only additions are wording aids
("one passing line → 1; a recurring or stand-alone theme → 2; most of it → 3") and the note that a weak but
readable answer that is about the question still gets 0–3. The guide is frozen with this file.

## 5. Analysis, criteria and named outcomes

Numeric ratings only (U as missing) for correlations; 95% bootstrap intervals over questions.

- **Human–human agreement**: Spearman ρ and quadratic-weighted Cohen's κ between the two raters.
- **Human–judge agreement**: Spearman ρ between the mean human rating and each LLM judge, over the items.

Named outcomes:
- **(a) Measure supported**: human–human ρ ≥ 0.5 **and** mean-human vs each M1 judge (Llama, Qwen) ρ ≥ 0.5.
- **(b) Humans agree with each other but not with a judge** (human–human ρ ≥ 0.5, some M1 judge < 0.5):
  that judge is questioned; results that rest on it are qualified.
- **(c) Humans do not agree with each other** (ρ < 0.5): the task is not reliable at this rubric; report it,
  and the judge results stand unvalidated.

Also reported (no criterion; underpowered by design): mean-human ΔJ, `fc_remove_a1` − `unsteered_s42`
(30 pairs) and the Mistral analogue (15 pairs); the share of `fc_add_a8` items each rater marks U; J3
agreement; per-rater rating distributions.

## 6. Already seen

All LLM-judge ratings of every arm. Item selection uses no rating.

## Amendments

None.
