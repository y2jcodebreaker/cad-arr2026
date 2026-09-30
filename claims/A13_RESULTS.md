# A13 results — AccessEval's own metrics, scored against `PREREG_A13_benchmark_metrics.md` (frozen 86912dd; A13-A1)

Scored 2026-09-30 by `a13_analysis.py` from `results/e1/a13/benchmark_metric_scores.json` (evidence 4ab9dda:
VADER compound on the full answer; Regard `sasha/regardv3` @9232a1d, negative probability, first 512 tokens).
Numbers are generated from `analysis-output/a13/a13_scores.json`. Badness: −VADER, +Regard-negative, +lexical
score. Non-Healthcare items; 95% question-level intervals. J = mean of the Llama and Qwen M1 ratings.

| metric | Q1 ρ(badness, J), unsteered | gap disability − neutral | degradation rate (≥ 0.05) | Q2 salad − unsteered | Q3 ρ over 15 arms | outcome |
|---|---|---|---|---|---|---|
| VADER (AccessEval) | +0.254 [+0.085, +0.411] | +0.017 [-0.036, +0.072] | 0.08 | +0.883 [+0.770, +0.965] | +0.171 | **(b) blind** |
| Regard (AccessEval) | +0.413 [+0.237, +0.547] | +0.071 [+0.036, +0.113] | 0.30 | -0.139 [-0.212, -0.077] | +0.400 | **(c) rewards broken text** |
| lexical score (ours, comparator) | -0.222 [-0.362, -0.080] | +1.417 [+1.174, +1.664] | 0.82 | -0.514 [-0.740, -0.285] | +0.343 | **(b) blind** (not pre-registered) |

Change in badness against `unsteered_s42` (negative = the metric says better) for the erasing arms
(M1 echo drop ≥ 0.25) and for FCS:

| arm | ΔJ | Δ VADER-badness | Δ Regard-negative | Δ lexical |
|---|---|---|---|---|
| `angular` | -0.202 | +0.085 | +0.011 | -0.846 |
| `angular_adaptive` | -0.164 | +0.104 | +0.010 | -0.820 |
| `caa` | -0.123 | +0.030 | -0.018 | -0.470 |
| `caa_m2` | -0.332 | +0.007 | -0.053 | -0.859 |
| `cad_proj_k40` | -0.051 | -0.002 | -0.023 | -0.401 |
| `fairsteer` | -0.002 | +0.004 | -0.017 | -0.338 |
| `prompt_explicit_cad_k40` | -0.217 | -0.019 | -0.033 | -0.578 |
| `sadi` | +0.335 | +0.445 | -0.041 | -0.228 |
| `fc_remove_a1` | -0.177 | -0.013 | -0.008 | -0.028 |

Regard's largest improvement over the 15 arms: `fc_add_a4` -0.094 (the additive arm that loses 3.5–5.5 accessibility-quality points, A8).
Mistral arms against `mistral_unsteered` (Δ badness): VADER `mistral_neutral` +0.032, `mistral_fc_remove_a1` +0.022, `mistral_fc_prompt_remove_a1` +0.017, `mistral_prompt_explicit` +0.022, `mistral_fc_shuffle_s0` +0.029; Regard `mistral_neutral` -0.025, `mistral_fc_remove_a1` +0.001, `mistral_fc_prompt_remove_a1` -0.003, `mistral_prompt_explicit` -0.000, `mistral_fc_shuffle_s0` -0.005.

## Reading

**VADER: (b) blind.** It tracks the judges' medicalization rating only weakly (ρ 0.25, below the 0.3
bar) and sees no disability–neutral difference at all (+0.017, interval includes 0; AccessEval's own
5-point degradation rule flags 8% of questions). It does not reward broken text: word salad is rated much
worse (+0.88), and so is SADI (+0.45). It does not rank methods like the judges (ρ 0.17).

**Regard: (c) rewards broken text.** Regard is the only automatic metric here that tracks medicalization
(ρ 0.41) and sees the disability–neutral gap (+0.071; 30% of questions degraded by the 5-point rule), but it
rates the word salad as *less* negative than the model's normal answers (−0.139, interval excludes 0), its
largest improvement goes to the degraded additive arm, and it credits SADI (−0.041) although the judges
see SADI raise medicalization (+0.335). It does not rank methods like the judges (ρ 0.40 < 0.5).

**Our own lexical score (comparator, not pre-registered)** tracks medicalization in the wrong direction
(ρ −0.22), rewards the word salad (−0.51) and every erasing method (−0.23 to −0.86), and barely registers FCS
(−0.03).

**For the paper.** None of the three automatic metrics can stand in for a medicalization measure: VADER
does not see medicalization, Regard sees it but also rewards broken and degraded text, and the lexical score
tracks naming. This extends the measurement critique from our earlier metric to the benchmark's own
automatic metrics. Limits: VADER's sub-score is not named by AccessEval (we used compound); Regard reads the
first 512 tokens; AccessEval's third metric, the Qwen2.5-72B quality judge, is not reproduced here (our
accessibility judges use smaller models).
