# A8 results — scored against `PREREG_A8_framing_contrast.md` (frozen c21eb88, amendment A8-A1)

Scored 2026-09-28 by `a8_analysis.py` from `results/e1/` (evidence a5dc805; pod run at 6cfb5b1, all
run records clean). Tables are generated from `analysis-output/a8/a8_scores.json` (pre-registered)
and `analysis-output/a8/a8_audit_NOT_PREREGISTERED.json` (post-hoc, `a8_audit.py`).

## Gates
- GA8-1: 106 medicalizing / 374 clean fit answers (≥ 40 each), pass.
- GA8-2: best 5-fold balanced accuracy 0.822 at layer 9 (≥ 0.60), pass. Steered layers [9, 11].
- Fit and frozen eval share no question: 96 fit question ids, 50 eval ids, 0 shared.

## Outcome: **(a) framing-contrast steering reduces medicalization without erasing the disability**

Succeeding arms: `fc_remove_a1`, `fc_remove_a2`, `fc_prompt_remove_a1`. Non-Healthcare items, reference unsteered seed 42,
Bonferroni-adjusted question-level intervals (6 arms).

| arm | ΔJ Llama [99.17%] | ΔJ Qwen [99.17%] | echo drop | ΔQ acc. L / Q | U L / Q | result |
|---|---|---|---|---|---|---|
| `fc_remove_a1` | -0.22 [-0.36, -0.10] | -0.12 [-0.20, -0.05] | -0.08 | -0.03 / -0.03 | 0.00 / 0.00 | **success** |
| `fc_remove_a2` | -0.16 [-0.29, -0.04] | -0.10 [-0.17, -0.04] | -0.04 | -0.04 / -0.04 | 0.00 / 0.00 | **success** |
| `fc_add_a4` | -0.17 [-0.48, +0.15] | -0.26 [-0.40, -0.13] | +0.00 | -5.46 / -3.52 | 0.11 / 0.02 |  |
| `fc_add_a8` | +0.00 [+0.00, +0.00] | -0.07 [-0.29, +0.00] | +1.00 | -6.47 / -5.36 | 1.00 / 0.00 |  |
| `fc_prompt_remove_a1` | -0.20 [-0.36, -0.06] | -0.14 [-0.25, -0.05] | +0.11 | +0.01 / +0.08 | 0.00 / 0.00 | **success** |
| `fc_remove_a1_L21L25` | -0.09 [-0.22, +0.02] | -0.06 [-0.15, +0.02] | -0.05 | -0.11 / -0.06 | 0.00 / 0.00 |  |
| `prompt_explicit` | -0.14 [-0.27, -0.04] | -0.06 [-0.13, +0.00] | +0.17 | +0.01 / +0.07 | 0.00 / 0.00 |  |
| `caa` | -0.14 [-0.27, -0.01] | -0.09 [-0.20, +0.01] | +0.31 | -0.80 / -0.30 | 0.00 / 0.00 |  |
| `cad_proj_k40` | -0.06 [-0.23, +0.10] | -0.02 [-0.12, +0.07] | +0.61 | -0.59 / -0.23 | 0.00 / 0.00 |  |
| `prompt_explicit_cad_k40` | -0.28 [-0.43, -0.14] | -0.14 [-0.23, -0.06] | +0.69 | -0.70 / -0.20 | 0.00 / 0.00 | reduces |

Reference rows (`prompt_explicit`, `caa`, `cad_proj_k40`, `prompt_explicit_cad_k40`) are scored with the
same A8 criteria for comparison; they were not A8 arms.

## Secondary (pre-registered, 95% intervals)
- Best succeeding arm by mean ΔJ: `fc_prompt_remove_a1`. It and `fc_remove_a1` differ by 0.0014
  (mean over judges −0.170 vs −0.169), so the choice between them is not informative.
- `fc_prompt_remove_a1` − explicit prompt: Llama -0.06 [-0.14, +0.02], Qwen -0.08 [-0.13, -0.03].
- `fc_prompt_remove_a1` − CAA: Llama -0.06 [-0.18, +0.05], Qwen -0.05 [-0.13, +0.02].
- Layer rule, `fc_remove_a1` − `fc_remove_a1_L21L25`: Llama -0.13 [-0.26, -0.01], Qwen -0.05 [-0.10, +0.01].

## Exploratory (not pre-registered, 95% intervals)
- `fc_remove_a1` (steering only) − explicit prompt: Llama -0.08 [-0.17, +0.01], Qwen -0.05 [-0.10, -0.00].
- `fc_remove_a1` − CAA: Llama -0.08 [-0.20, +0.02], Qwen -0.02 [-0.09, +0.05].

## Audit (not pre-registered)

| arm | mean J L / Q | → clean (both) | → med (both) | lexical d | words | L per 1k | unique |
|---|---|---|---|---|---|---|---|
| `unsteered_s42` | 0.54 / 0.31 |  |  | +0.00 | 477 | 0.12 | 1.00 |
| `fc_remove_a1` | 0.32 / 0.19 | 10 | 1 | +0.02 | 486 | 0.07 | 1.00 |
| `fc_remove_a2` | 0.38 / 0.21 | 9 | 0 | +0.01 | 482 | 0.09 | 1.00 |
| `fc_add_a4` | 0.33 / 0.00 | 26 | 0 | -0.57 | 742 | 0.00 | 1.00 |
| `fc_add_a8` | 0.00 / 0.00 | 0 | 0 | +0.67 | 1048 | 0.00 | 1.00 |
| `fc_prompt_remove_a1` | 0.34 / 0.15 | 10 | 0 | +0.09 | 418 | 0.04 | 1.00 |
| `fc_remove_a1_L21L25` | 0.46 / 0.24 | 6 | 0 | +0.03 | 482 | 0.08 | 1.00 |
| `prompt_explicit` | 0.40 / 0.24 | 7 | 3 | +0.06 | 420 | 0.07 | 1.00 |
| `caa` | 0.40 / 0.20 | 10 | 4 | +0.37 | 493 | 0.10 | 1.00 |
| `cad_proj_k40` | 0.48 / 0.29 | 7 | 5 | +0.34 | 484 | 0.02 | 1.00 |
| `prompt_explicit_cad_k40` | 0.27 / 0.19 | 12 | 1 | +0.51 | 412 | 0.04 | 1.00 |

"→ clean (both)": items both M1 judges rate ≥ 1 unsteered and 0 steered; "→ med (both)" the reverse.

## Reading (what this supports and what it does not)

**Supported.** Removing the framing-contrast direction at layers 9 and 11 (α = 1 or 2, with or
without the explicit prompt) lowers the M1 medicalization rating on both judges under the
Bonferroni-adjusted rule, keeps the disability named (echo drop −0.08 to +0.11, all below 0.25),
and leaves accessibility quality unchanged (ΔQ −0.04 to +0.08). No earlier *steering* arm met all
three: every steering arm that reduced J on both judges in M1 (`m1_scores.json` Q2/Q3) had an echo
drop ≥ 0.25 (CAA 0.31; Angular, Angular adaptive, CAA ×2, prompt + CAD projection 0.56–0.69). The
explicit prompt alone did reduce J on both judges without erasure under M1's 95% rule (echo 0.17,
ΔQ +0.01 / +0.07); under A8's Bonferroni rule its Qwen interval reaches 0.
Mean J falls from 0.54 to 0.32 (Llama) and 0.31 to 0.19 (Qwen) for `fc_remove_a1`; answer length
is unchanged (486 vs 477 words).

**The lexical score cannot see it.** The lexical d of the three succeeding arms is +0.01 to +0.09,
i.e. "no effect" by E1's target metric, while the broken arm `fc_add_a8` (word salad such as "that
are which are which are …", Llama U-rate 1.00) gets d = +0.67, above every baseline. E1's
degeneracy flag (unique answers < 50%) misses it: all 250 salads are distinct strings.

**Not supported / caveats.**
1. *Same instrument fits and scores.* The direction is fit on answers labelled by the two M1 judges
   that also score the outcome (on different questions: 0 shared). A direction that targets what
   these judges react to, rather than medical framing itself, would pass. Evidence against, all
   post-hoc: the held-out lexicon rate L falls (0.12 → 0.07 / 0.04 per 1k words); the per-item
   transitions are one-directional (10 items to clean on both judges, 1 or 0 the other way); read
   examples replace symptoms/treatment/medical-coverage content with accessibility content. No
   human labels (author's decision), so this stays a stated limitation.
2. *Not better than the alternatives on reduction.* Against CAA and the explicit prompt, the paired
   differences favour A8 but most 95% intervals include 0 (only Qwen vs the prompt excludes 0). The
   claim is *reduction without erasure or quality loss*, not *larger reduction*.
3. *Effect size is modest*: about 0.1–0.2 on a 0–3 scale; about 10 of 211 non-Healthcare items move
   from medicalizing to clean on both judges.
4. *Layer choice is weakly identified.* Balanced accuracy is 0.76–0.82 at every layer from 0
   (`a8_fit_report.json`), so separability is partly lexical and the layer rule picks among near
   ties. The selected layers still beat L21/L25 on Llama (−0.13 [−0.26, −0.01]); Qwen interval
   includes 0.
5. *Additive steering fails.* α = 4 lengthens answers (742 words), loses 3.5–5.5 quality points and
   raises the lexical score; α = 8 destroys the text.
6. One model (Llama-3.1-8B-Instruct), one seed, greedy decoding.
