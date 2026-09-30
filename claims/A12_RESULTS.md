# A12 results — human validation, scored against `PREREG_A12_human_validation.md` (frozen 42f40d8; A12-A1)

Scored 2026-09-30 by `a12_analysis.py` from `human_eval/ratings_R1.json` and `ratings_R2.json` (evidence 9c4e94d).
Numbers are generated from `analysis-output/a12/a12_scores.json`.

## Outcome: **(c) humans do not agree with each other**

- Human–human: Spearman ρ -0.004 [-0.237, +0.209], quadratic κ 0.058, exact agreement 0.40 (n 86 items both rated 0–3).
- Mean human vs judges: llama ρ -0.013 [-0.229, +0.198]; qwen ρ -0.024 [-0.196, +0.135]; mistral ρ +0.068 [-0.130, +0.271].
- Per rater vs judges: llama: R1 +0.049, R2 +0.077; qwen: R1 +0.158, R2 -0.042; mistral: R1 +0.126, R2 +0.080.
- Word salad marked U: R1 100%, R2 100%.
- Distributions: R1 {'0': 88, '1': 16, '2': 1, '3': 0, 'U': 5}; R2 {'0': 31, '1': 49, '2': 3, '3': 3, 'U': 24}.
- Secondary (underpowered): mean-human FCS − unsteered, Llama -0.117 [-0.339, +0.089] (30 pairs); Mistral +0.000 [-0.219, +0.214] (15 pairs).

## Exploratory diagnostics (not pre-registered; descriptive)

- Median time per item: R1 15 s, R2 48 s (items show up to 2,000 characters).
- R2 used U for 24 items: the 5 word-salad items and 19 readable answers across every arm (Llama and
  Mistral unsteered and FCS, CAA). R1 used U only for the 5 word-salad items.
- Binary (0 vs ≥ 1) agreement between raters 0.41 (n 86). Share rated ≥ 1: R1 0.19, R2 0.64; the judges'
  mean rating is ≥ 1 on about 0.20 of these items. Binary agreement with the judges: R1 0.71, R2 0.44.
- On the 12 items both M1 judges rate clearly medicalizing (mean ≥ 1.5), R1 gave 0 to 9, 1 to 2 and 2 to 1;
  R2 gave 1 to 6, U to 3, 0 to 2 and 3 to 1.

## Reading

The pre-registered consequence of (c) applies: the task was not reliable with these two raters, and the
LLM-judge results stand **unvalidated by humans**. The diagnostics suggest the failure is in how the task
was applied, not a fine-grained disagreement: one rater rated quickly and rarely marked medical framing,
including on answers that open with medical advice; the other marked most answers as medicalizing and used
U for readable answers, against the guide. A12 therefore does not show that the judges are wrong or right.
The paper must report A12 (it was pre-registered) and may not claim human validation from it. Both raters
did mark every word-salad answer unusable.
