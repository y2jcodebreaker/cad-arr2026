# M1 results — scored against `PREREG_M1_valid_medicalization.md` (frozen e9a2780)

Scored 2026-09-28 by `m1_analysis.py` (686611e) from `results/e1/` (evidence 0714eab). The four
M1-A1 (A7) arms were still running and are absent here; Q4 as registered excludes them anyway.
Numbers below are generated from `analysis-output/m1/m1_scores.json`.

## Gates
- GM1 neutral answers: 250 items, pass.
- GM2 inter-judge Spearman 0.642 over 8217 texts, pass (≥ 0.5).
- GM3 U-rate on unsteered: Llama 0.000, Qwen 0.000, pass. Both judges used.
- Lexicon L (held-out rule): 3 of 35 candidates kept: therapists, symptoms, patients (seen before the run, deviation D1).

## Q1 existence: **Q1-a, medicalization exists on the valid measures**
- J Llama: disability − neutral +0.52 [+0.36, +0.70]; Healthcare items +0.54
- J Qwen: +0.28 [+0.18, +0.39]; Healthcare items +0.64
- L gap: +0.12 per 1,000 words [+0.05, +0.21]
- Levels (non-Healthcare): neutral answers ≈ 0.02 (Llama) / 0.03 (Qwen) on the 0–3 scale; unsteered disability answers 0.54 / 0.31. About a quarter to a third of disability answers get a rating ≥ 1.

## Q2 reduction and Q3 erasure (both judges must agree; U-rate ≤ 10%)

| arm | ΔJ Llama | ΔJ Qwen | call | echo drop | erases |
|---|---|---|---|---|---|
| `unsteered_s43` | +0.06 [-0.02, +0.13] | -0.01 [-0.06, +0.03] | no change | +0.05 |  |
| `unsteered_s44` | +0.00 [-0.06, +0.06] | -0.04 [-0.09, +0.00] | no change | +0.03 |  |
| `prompt_min` | -0.04 [-0.15, +0.08] | -0.05 [-0.10, +0.01] | no change | +0.13 |  |
| `prompt_explicit` | -0.14 [-0.23, -0.06] | -0.06 [-0.12, -0.02] | reduces | +0.17 |  |
| `prompt_explicit_cad_k40` | -0.28 [-0.39, -0.17] | -0.14 [-0.21, -0.08] | reduces | +0.69 | yes |
| `cad_proj_kauto` | -0.06 [-0.16, +0.03] | -0.02 [-0.08, +0.05] | no change | +0.58 | yes |
| `cad_proj_k1` | -0.01 [-0.08, +0.06] | -0.05 [-0.10, +0.01] | no change | +0.06 |  |
| `cad_proj_k5` | -0.00 [-0.09, +0.09] | -0.00 [-0.05, +0.04] | no change | +0.37 |  |
| `cad_proj_k10` | -0.02 [-0.11, +0.08] | +0.00 [-0.05, +0.05] | no change | +0.45 |  |
| `cad_proj_k20` | -0.03 [-0.12, +0.07] | +0.04 [-0.02, +0.10] | no change | +0.49 |  |
| `cad_proj_k40` | -0.06 [-0.19, +0.06] | -0.02 [-0.10, +0.05] | no change | +0.61 | yes |
| `cad_proj_k80` | -0.19 [-0.30, -0.09] | -0.03 [-0.08, +0.03] | no change | +0.67 | yes |
| `cad_proj_probe` | +0.03 [-0.05, +0.12] | -0.00 [-0.06, +0.05] | no change | +0.06 |  |
| `cad_proj_meandiff` | +0.00 [-0.08, +0.09] | +0.00 [-0.06, +0.05] | no change | +0.11 |  |
| `add_probe_a3` | -0.00 [-0.11, +0.09] | -0.05 [-0.09, -0.01] | no change | +0.14 |  |
| `add_probe_a5` | -0.03 [-0.12, +0.06] | -0.05 [-0.10, -0.00] | no change | +0.18 |  |
| `add_probe_a7` | -0.07 [-0.20, +0.05] | -0.04 [-0.10, +0.02] | no change | +0.31 |  |
| `add_probe_a9` | -0.10 [-0.23, +0.02] | -0.07 [-0.14, +0.00] | no change | +0.41 |  |
| `add_meandiff_a3` | -0.02 [-0.11, +0.06] | -0.04 [-0.09, +0.00] | no change | +0.16 |  |
| `add_meandiff_a5` | -0.08 [-0.17, +0.00] | -0.06 [-0.11, -0.02] | no change | +0.25 |  |
| `add_meandiff_a7` | -0.12 [-0.22, -0.01] | -0.03 [-0.09, +0.02] | no change | +0.36 |  |
| `add_meandiff_a8` | -0.14 [-0.25, -0.03] | -0.05 [-0.14, +0.03] | no change | +0.39 |  |
| `leace_k1` | +0.04 [-0.03, +0.11] | -0.04 [-0.08, +0.00] | no change | +0.09 |  |
| `leace_k5` | +0.01 [-0.07, +0.09] | +0.02 [-0.03, +0.08] | no change | +0.12 |  |
| `leace_k10` | -0.03 [-0.12, +0.06] | -0.03 [-0.08, +0.01] | no change | +0.26 |  |
| `leace_k20` | -0.04 [-0.15, +0.07] | +0.00 [-0.05, +0.07] | no change | +0.26 |  |
| `leace_k40` | -0.01 [-0.09, +0.07] | -0.00 [-0.07, +0.06] | no change | +0.32 |  |
| `leace_k80` | +0.07 [-0.04, +0.18] | +0.05 [-0.01, +0.11] | no change | +0.33 |  |
| `caa` | -0.14 [-0.24, -0.04] | -0.09 [-0.17, -0.02] | reduces | +0.31 |  |
| `fairsteer` | +0.01 [-0.09, +0.10] | +0.00 [-0.06, +0.06] | no change | +0.39 |  |
| `sadi` | +0.58 [+0.43, +0.73] | +0.01 [-0.06, +0.08] | no change | +0.26 |  |
| `angular` | -0.24 [-0.36, -0.13] | -0.13 [-0.20, -0.07] | reduces | +0.61 | yes |

## Q4: does the lexical score track medicalization or naming?
- Across steered arms, Spearman(lexical d, echo drop) = 0.89.
- Spearman(lexical d, J reduction) = 0.60 (Llama), 0.20 (Qwen).
- Difference echo − J: +0.29 [+0.09, +0.57] (Llama), +0.69 [+0.35, +0.96] (Qwen): both exclude 0.
- **Registered outcome holds: the lexical score measured naming, not framing.** (Echo side not blind, D1.)

## Q5 unusable outputs
- SADI U-rate: Llama 0.111, Qwen 0.004. Angular: 0.000 / 0.000.
- Prediction (both > 10%) holds for SADI on Llama only, fails on Qwen, fails for Angular on both.

## Instrument audit (not pre-registered)
- Qwen's 3.16% 'unparseable' outputs are mostly bare ratings without the `Rating:` prefix (232 × `0`, 32 × `U`), concentrated in SADI (63). Under a lenient parse that accepts a bare rating (`m1_scores_lenient_parse_NOT_PREREGISTERED.json`): Q1, Q2 calls and Q4 unchanged; SADI's Qwen U-rate becomes 0.132, so Q5 would hold for SADI on both judges. Reported as a robustness check; the frozen parse rule remains primary.
- The SADI disagreement (Llama +0.58, Qwen ≈ 0) comes from Llama rating answers that name conditions or discuss accommodations as medical framing, which the rubric excludes; the both-judges rule keeps SADI at 'no change'.

## What this means (interpretation)
- There is medicalization in Llama-3.1-8B's answers to disability-framed AccessEval questions, measured validly, but it is modest: most answers get 0.
- The CAD projection (k = 40, and every rank but 80 on Llama) does **not** reduce it on both judges; its large lexical effect is mostly the model no longer naming the user's disability (echo drop 0.61).
- What reduces medicalization on both judges: the explicit system prompt (no quality cost, no erasure), CAA (no erasure), Angular (erases; costs quality), and prompt + projection (largest reduction; erases).
- A4's lexical conclusion that prompting barely debiases does not survive a valid measure.
- A5 is not re-scoped: its pre-registered condition (Q2 shows the projection reduces J) is not met.
