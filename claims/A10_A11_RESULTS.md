# A10 and A11 results — scored against `PREREG_A10_metric_gaming.md` and `PREREG_A11_mistral.md` (frozen 7da220b; A10-A1)

Scored 2026-09-30 by `a10_a11_analysis.py` from `results/e1/` (evidence 85d8c68, pod commit 332a69a, torch
2.4.1+cu124). Tables are generated from `analysis-output/a10_a11/a10_a11_scores.json`. Lexical d: all 250 items,
E1 bootstrap. ΔJ: non-Healthcare items, 95% question-level intervals. J3 = Mistral-Small-24B.

## A10: bias-blind edits against the lexical score (Llama, frozen set)

**Outcome: (b) an arm reaches the best baseline but a judge's dJ leaves the band.**

| arm | lexical d | ΔJ Llama | ΔJ Qwen | ΔJ J3 | echo drop | ΔQ acc. L / Q |
|---|---|---|---|---|---|---|
| `game_append` | +1.169 [+0.987, +1.382] | +0.047 [-0.013, +0.106] | +0.088 [+0.046, +0.136] | +0.081 [+0.038, +0.123] | +0.000 | -0.49 / -0.24 |
| `game_sub` | +0.233 [+0.164, +0.310] | -0.004 [-0.030, +0.021] | -0.004 [-0.022, +0.009] | +0.034 [+0.013, +0.055] | +0.000 | -0.00 / +0.00 |
| `game_ban` | +0.222 [+0.157, +0.299] | -0.021 [-0.060, +0.017] | -0.027 [-0.065, +0.004] | -0.004 [-0.034, +0.026] | +0.002 | -0.02 / +0.01 |

Criterion: lexical d ≥ 0.741 (best published baseline, Angular at the E1 setting) and every judge's ΔJ interval
inside [−0.10, +0.10]. Positive ΔJ = more medicalization.

## A11: Mistral-7B-Instruct-v0.3 (gated)

**Outcome: (a) framing-contrast steering replicates on Mistral.**

S0 existence (disability − neutral, Mistral's own answers):

- J gap: Llama +0.387 [+0.247, +0.528], Qwen +0.247 [+0.138, +0.369], J3 +0.332 [+0.221, +0.438].
- Lexical score gap +0.873 [+0.697, +1.051] (Llama's: +1.417).
- Unsteered mean J (non-Healthcare): Llama 0.460, Qwen 0.316, J3 0.498.
- Gates: GA11-1 100 medicalizing / 373 clean, pass; GA11-2 balanced accuracy 0.861 at layers [10, 14], pass.

| arm (vs `mistral_unsteered`) | ΔJ Llama | ΔJ Qwen | ΔJ J3 | echo drop | ΔQ acc. L / Q | U L / Q | lexical d |
|---|---|---|---|---|---|---|---|
| `mistral_fc_remove_a1` | -0.268 [-0.383, -0.162] | -0.160 [-0.256, -0.076] | -0.162 [-0.260, -0.072] | -0.014 | -0.05 / +0.03 | 0.00 / 0.00 | -0.027 [-0.151, +0.085] |
| `mistral_fc_prompt_remove_a1` | -0.213 [-0.298, -0.132] | -0.113 [-0.183, -0.041] | -0.166 [-0.238, -0.098] | +0.129 | -0.06 / -0.03 | 0.00 / 0.00 | +0.013 [-0.102, +0.132] |
| `mistral_prompt_explicit` | -0.089 [-0.196, +0.021] | -0.045 [-0.108, +0.019] | -0.004 [-0.068, +0.064] | +0.199 | -0.07 / -0.05 | 0.00 / 0.00 | +0.080 [-0.024, +0.189] |
| `mistral_fc_shuffle_s0` | -0.234 [-0.336, -0.128] | -0.114 [-0.205, -0.033] | -0.140 [-0.209, -0.077] | +0.419 | +0.05 / +0.28 | 0.00 / 0.00 | +0.268 [+0.117, +0.418] |

`mistral_fc_remove_a1` − `mistral_fc_shuffle_s0`: Llama -0.034 [-0.141, +0.064], Qwen -0.041 [-0.117, +0.032], J3 -0.021 [-0.128, +0.068].

## Reading

**A10 (b), not (a).** The inserted sentence gives the largest lexical "debiasing" of anything tested
(d 1.169, against 0.741 for the best published method), but the pre-registered criterion also asks that
the judges see no change, and they see a small **increase** in medicalization: Qwen +0.088 and J3 +0.081
(intervals exclude 0), Llama +0.047 (interval reaches +0.106, just outside the band). So A10 does not show
"gamed with medicalization unchanged"; it shows that the edit the lexical score rewards most moves the
valid measure slightly the *other* way. The paper must report it as that, with outcome (b) named. The
synonym and ban edits leave medicalization unchanged (every interval within ±0.06; J3 +0.034 for synonyms,
excluding 0) and raise the lexical score by d 0.233 and 0.222, which does not reach the published maximum
(outcome criterion 1 fails for them). The ban leaks "risk" after a hyphen in 7/250 answers.
*Disclosed:* the CPU edits' lexical d was seen in the stand-in test before any judging (A10-A1).

**A11 (a): framing-contrast steering replicates on a second model family.** Mistral medicalizes on the
valid measure (gaps +0.25 to +0.39 on all three judges). The A8 procedure, fit fresh on Mistral (layers 10
and 14), reduces J on both M1 judges (−0.268, −0.160) and on J3 (−0.162) with no erasure (echo −0.014) and
no quality or usability loss. On Mistral the explicit prompt alone does **not** reduce J on any judge.

**What A11 does not support: specificity of the reduction.** The shuffled-label direction at the same
layers reduces J nearly as much (−0.234 / −0.114 / −0.140; FCS − shuffle −0.02 to −0.04, all intervals
include 0), but it **erases** the disability name (echo drop 0.419) and changes the lexical score (d 0.268).
On Mistral, then, what distinguishes the framing-contrast direction is not *that* it reduces medicalization
but that it does so **without erasing**. A9's Llama result (shuffles do nothing) does not replicate here.
Writing rule: say "reduces without erasure on both models"; do not say "only the framing direction reduces
medicalization" as a general claim.

**Caveats.** J3 is Mistral-family, so it is secondary for the Mistral arms (the two M1 judges carry the
outcome). One seed per arm; generation samples at temperature 0.1. The lexical score barely registers
the Mistral FCS arm (d −0.027), as on Llama.
