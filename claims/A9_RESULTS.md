# A9 results — scored against `PREREG_A9_robustness.md` (frozen 281249f; amendments A9-A1..A5)

Scored 2026-09-29 by `a9_analysis.py` from `results/e1/` (evidence 0d18b43 generation, d13e8e0 judges).
Pre-registered numbers come from `analysis-output/a9/a9_scores.json`; the one exploratory comparison from
`analysis-output/a9/a9_exploratory_NOT_PREREGISTERED.json`. 95% question-level intervals, non-Healthcare items.

## Gates and environment

- GA9-0: recomputed A8 direction, cos 0.9999999999 (layer 9), 0.9999999991 (layer 11). Pass.
- GA9-1: J3 on unsteered, missing 0.000, U 0.000. Pass.
- GA9-2: J3 vs Llama ρ 0.620 (n 7240), vs Qwen ρ 0.629 (n 6921). Pass.
- A9-A5: reruns identical to the committed arms on 250/250, 250/250 and 250/250 texts, so the registered references are used. The earlier outputs reproduce character for character on torch 2.4.1+cu124.

## Outcomes

| question | outcome |
|---|---|
| Q-J3 | (a) the reduction holds on a judge never used for fitting |
| Q-SPEC | (a) specific |
| Q-MECH | (c) neither |
| Q-SIDE | (a) negligible |
| Q-FRESH | (c) not carried over, under-powered (low unsteered J on the fresh set) |

## Q-J3, independent judge (Mistral-Small-24B, never used for fitting)

- `fc_remove_a1`: ΔJ3 -0.149 [-0.209, -0.094]. Outcome (a).
- J3 disability − neutral gap (M1 Q1 on J3): +0.383 [+0.285, +0.489]: the third judge also finds medicalization.

| arm (frozen set, vs unsteered) | ΔJ3 |
|---|---|
| `prompt_explicit` | -0.068 [-0.132, -0.009] |
| `prompt_explicit_cad_k40` | -0.128 [-0.209, -0.047] |
| `cad_proj_k40` | -0.017 [-0.098, +0.064] |
| `caa` | -0.170 [-0.243, -0.098] |
| `angular` | -0.183 [-0.281, -0.089] |
| `fairsteer` | +0.017 [-0.055, +0.089] |
| `sadi` | -0.148 [-0.303, -0.016] |
| `angular_adaptive` | -0.145 [-0.247, -0.051] |
| `caa_m2` | -0.357 [-0.464, -0.259] |
| `fairsteer_a1` | +0.013 [-0.068, +0.094] |
| `sadi_s5` | +0.051 [-0.021, +0.123] |
| `fc_remove_a1` | -0.149 [-0.209, -0.094] |
| `fc_remove_a2` | -0.089 [-0.149, -0.034] |
| `fc_add_a4` | -0.195 [-0.342, -0.065] |
| `fc_add_a8` | n/a (no numeric rating) |
| `fc_prompt_remove_a1` | -0.174 [-0.255, -0.098] |
| `fc_remove_a1_L21L25` | +0.000 [-0.072, +0.068] |
| `fc_shuffle_s0` | +0.089 [+0.021, +0.157] |
| `fc_shuffle_s1` | -0.017 [-0.077, +0.043] |
| `fc_shuffle_s2` | +0.000 [-0.068, +0.064] |
| `id_remove_a1` | +0.162 [+0.081, +0.247] |
| `unsteered_rerun` | +0.000 [+0.000, +0.000] |
| `fc_remove_a1_rerun` | -0.149 [-0.209, -0.094] |

Fresh set, vs `fresh_unsteered`:

- `fresh_fc_remove_a1`: -0.058 [-0.116, -0.003]
- `fresh_fc_prompt_remove_a1`: -0.155 [-0.219, -0.090]
- `fresh_prompt_explicit`: -0.052 [-0.110, +0.006]

## Q-SPEC, specificity

| judge | `fc_remove_a1` − shuffle mean | shuffle mean vs unsteered | per shuffle (s0, s1, s2) |
|---|---|---|---|
| llama | -0.244 [-0.375, -0.133] | +0.023 [-0.045, +0.094] | +0.072, +0.009, -0.013 |
| qwen | -0.112 [-0.173, -0.064] | -0.014 [-0.051, +0.022] | +0.018, -0.046, -0.005 |

## Q-MECH, contrast type (identity direction at the same layers, operator and features)

- echo drop -0.014 (erasure threshold 0.25); ΔJ Llama +0.119 [+0.030, +0.213], Qwen +0.037 [-0.037, +0.107]; ΔJ3 +0.162 [+0.081, +0.247].
- cos(identity, framing): layer 9 -0.270, layer 11 -0.221.

## Q-SIDE, side effects

- WikiText-2 perplexity: unsteered 8.0521, α=1 8.0516 (-0.0060%), α=2 8.0629 (+0.134%); 288,486 tokens.
- Neutral questions: ΔJ Llama -0.017 [-0.047, +0.000], Qwen -0.018 [-0.066, +0.013] (band ±0.10). Mean length 3214 vs 3213 characters; **0% of steered neutral answers are identical to unsteered**: the text changes, the measured quantities do not.

## Q-FRESH, 85 unused questions (310 non-Healthcare items, 62 questions)

Unsteered mean J on the fresh set: Llama 0.258 (frozen 0.545), Qwen 0.166 (frozen 0.308).

| arm | ΔJ Llama | ΔJ Qwen | echo drop | ΔQ acc. L / Q | U L / Q |
|---|---|---|---|---|---|
| `fresh_fc_remove_a1` | -0.065 [-0.135, +0.003] | -0.039 [-0.092, +0.018] | -0.063 | -0.02 / -0.05 | 0.00 / 0.00 |
| `fresh_fc_prompt_remove_a1` | -0.106 [-0.187, -0.026] | -0.090 [-0.145, -0.034] | +0.093 | +0.08 / +0.12 | 0.00 / 0.00 |
| `fresh_prompt_explicit` | -0.065 [-0.152, +0.023] | -0.067 [-0.106, -0.030] | +0.147 | +0.10 / +0.12 | 0.00 / 0.00 |

Exploratory (not pre-registered): prompt + A8 − prompt alone on the fresh set: llama -0.042 [-0.106, +0.016], qwen -0.027 [-0.071, +0.013], mistral -0.103 [-0.155, -0.048].

## Reading

**What A9 adds to the A8 claim.**
1. *Circularity answered.* A larger model from a third family, never used for fitting, gives the same
   direction and a similar size: ΔJ3 −0.149 for `fc_remove_a1` (Llama −0.221, Qwen −0.116 in A8). As
   on the M1 judges, the reduction is not larger than CAA's (J3 −0.170) or Angular's (−0.183); the
   claim stays *reduction without erasure or quality loss*.
2. *Specific.* Directions fit with shuffled labels (same data, layers, operator) do nothing
   (shuffle mean −0.014 to +0.023); the real direction beats them by −0.24 (Llama) and −0.11 (Qwen).
3. *Harmless on what we measured.* Perplexity moves by −0.006%; medicalization of neutral answers
   does not move. The neutral answers' wording does change (0% identical), which we state.

**What A9 withdraws.**
4. *The mechanism story.* The identity direction at the same layers neither erases the disability
   (echo −0.01) nor reduces medicalization; it *raises* J on Llama (+0.12, interval excludes 0) and J3
   (+0.16), not on Qwen (+0.04, includes 0). Pre-registered consequence (Q-MECH (c)): the paper may
   say that framing-contrast removal avoids erasure, and that identity-contrast removal at these
   layers does not reduce medicalization. It may **not** say that input contrasts erase *because*
   they capture identity. CAD's erasure (M1) was at layers 21/25 with rank 40, a different setting.
5. *Generalisation to new questions is not shown for steering alone.* On the fresh set the A8 arm
   points the same way on all three judges but its intervals include 0 on both M1 judges (J3 −0.058,
   interval just excludes 0). The pre-registered rule labels this under-powered because Llama's
   unsteered J there (0.258) is below half its frozen value (threshold 0.272); Qwen's (0.166) is not
   below its threshold (0.154), so the label rests on one judge and is borderline. Report the numbers,
   not the label alone. Prompt + A8 does reduce J on all three judges on the fresh set without
   erasure or quality loss; its advantage over the prompt alone holds on J3 only (exploratory).

**Caveats.** One model (Llama-3.1-8B-Instruct); generation samples at temperature 0.1 with one seed (42), not greedy (Correction C1 in the pre-registration). All three judges are LLMs; no human
labels. The fresh questions are the ones the E1 loudness filter dropped, so they carry less
medicalization; this is a shift, not an i.i.d. replication.
