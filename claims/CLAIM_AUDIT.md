# Claim audit of draft v1 (started 2026-09-28)

Every factual sentence of the abstract and body gets a base: a results file and field, a verified
passage of a cited paper, "our interpretation", or **FLAG** (no base: rewrite or delete).
Sentence ids refer to the numbered split of the draft at commit 64a2f24 (224 sentences).
Status: **in progress**. Sections below are complete unless marked otherwise.

## A. Foundational findings (these change several sections at once)

| # | Claim in the draft | Finding | Evidence |
|---|---|---|---|
| A1 | ABS.001, 01.006, 07.002: bias steering is judged by a vocabulary-absence metric; "the field measures progress" this way | **FLAG.** The medicalization score is ours. AccessEval evaluates with VADER, a regard classifier and a Qwen2.5-72B judge. The baselines' own papers evaluate on other tasks (SADI: MC tasks, TriviaQA, TruthfulQA, ToxiGen; Angular: refusal, emotion; FairSteer: QA, counterfactual, generation). | arXiv 2509.22703; 2410.12299; 2510.26243; 2504.14492 |
| A2 | 01.004: "medicalization bias, following AccessEval" | **FLAG.** The word does not appear in AccessEval. | arXiv 2509.22703 (fetched) |
| A3 | 02_method.001–005: vocabulary = log-odds words with \|z\|>3; threshold robust across six lexicons | **FLAG.** The scorer is a fixed 10+10 word list. In the original log-odds analysis only *impairments* (+23), *condition* (+2.8), *treatment* (+2.3) favour disability answers; *doctor* (−2.8), *risk* (−3.3), *safety* (−2.1) favour neutral answers. The robustness result (`exp_c`) is on DiscrimEval, not AccessEval. | `analysis-output/metric_validity.json`; `results/phase1_vocab/`; `steering_track2/exp_c_vocab_threshold_results.json` |
| A4 | "zero = the metric's value for unbiased text"; "below zero = past neutral, withholds medical information" (ABS, 01.009, 01.014, 02.005, 03.008, 03.013, 03.018, 04.016, flags) | **FLAG.** Answers to neutral questions score −0.23 on average (−0.98 to +0.79 by domain; 25% exactly 0). Angular's E1 mean −0.26 is at the neutral level. | `metric_validity.json` |
| A5 | "medicalization" is what steering removes (ABS.006, 01.025–026, 04.*) | **FLAG, pending M1.** 77% of medical-word counts on unsteered answers are the echoed category name *impairments*; without it the unsteered mean (−0.24) equals the neutral level. | `metric_validity.json`; M1 (`PREREG_M1_valid_medicalization.md`) running |
| A6 | 01.001–003: fitness-routine example, "consult your physician, be mindful of your limitations" | **FLAG.** No fitness or exercise question exists in AccessEval (234 questions checked). Illustrative text presented as observed. | AccessEval HF dataset, all 234 neutral queries |
| A7 | "published setting / operating point / the value its authors report" (ABS.005, 02.019–020, 03.011, 03.021, 03.029–037, 04.006, 04.017–018, 06.009, Lim.005) | **FLAG for all four baselines.** SADI: paper prescribes a per-task validation search, no fixed value. CAA: multipliers ±1 for multiple choice, **±2 for open-ended generation**, layer 13 on Llama-2-7B; E1 used L14 α=1. FairSteer: **α=1**, layer by classifier accuracy, τ=0.5; E1 used L29 α=2. Angular: no fixed angle (full circle swept); **adaptive variant is the recommended default** and the non-adaptive one "risks breaking coherence on smaller models"; E1 used the non-adaptive mode 0 at 150°. The E1 settings follow no single rule (CAA α=1 = strongest intact sweep point; Angular = strongest target-metric point; SADI s=10 and FairSteer L29/α=2 are neither); they came from earlier per-method best-effect-size analyses. | arXiv 2312.06681, 2504.14492, 2510.26243, 2410.12299; `analysis-output/sweep_points.json`; `evaluate_angular_steering_baseline.py` l.88–91 |
| A8 | 03.007: "Ten of the 87 points apply no steering" | **FLAG.** 8 are unsteered (α=0 ×4, angle 0 ×4). `frontier_analysis.is_baseline_config` also counts SADI s=1 and CAD α=0.1. | `frontier_analysis.py` l.104–107 |
| A9 | 03.024: "Pareto front holds twelve configurations, ten of them CAD" | **Correct as computed, misleading as stated.** One of the 12 is unsteered (Angular 0°), one is near-zero CAD (α=0.1). The ≥50%-reduction statement (4 of 4 CAD) is unaffected. | `frontier_analysis.py` re-run 2026-09-28, outputs unchanged |

## B. Numbers checked against files

| Sentence | Claim | Status | Source |
|---|---|---|---|
| 01.009, 02.011 | SADI s=10: 201 of 250 score exactly 0 | ✅ | `results/sadi_results.json` |
| 01.011, 02.017 | SADI mean 188 words | ✅ | same |
| 01.011, 02.010 | SADI quoted text | ✅ found verbatim | same |
| 02.009 | SADI 249 of 250 distinct | ✅ | same |
| 02.005 | "three methods that reach a mean medicalization of 0.000" | ❌ SADI mean is 0.119 (Table 1 itself says 0.119) | same |
| 01.012, 02.013 | FairSteer α=6: 68 distinct; top text ×43; 1,048 words of "Here here…" | ✅ | `results/fairsteer_results_L29.json` |
| 02.014 | CAA α=6: 247 distinct, 638 words | ✅ | `results/caa_results_L14.json` |
| Tab 1 | quality 3.19 / 2.38 / 3.02; lengths 188 / 945 / 638 | ✅ | `sweep_points.json`, result files |
| 02.006 | unsteered 481 words, 0.712 | ✅ but from one run (CAA runner α=0); FairSteer's α=0 gives 487 / 0.663. State which. | result files |
| 03.005, 03.010 | 87 points, 21,750 responses, 38 flagged | ✅ | `sweep_points.json` |
| 03.007 | reference 7.92 ± 0.04, 0.659 ± 0.043 | ✅ as computed, but over 10 points of which 2 are steered (A8) | same |
| 03.012 | SADI loses 60% of quality | ✅ (−4.73 of 7.92) | same |
| 03.013 | Angular −0.222 | ✅ (interpretation "past neutral" fails, A4) | same |
| 03.036 | SADI s=5 quality 7.5 | ✅ | same |
| 03.039 | projection α=3: 70% unparseable | ✅ (0.696) | same |
| 03.039 | projection α=5: 75 empty responses | **FLAG: unverifiable.** File gives parse-fail 0.404 (101/250); the CAD per-response texts of that sweep were lost | same |
| Fig 1 caption | median CI half-width 0.18 | ✅ | same |

## C. Still to audit
Remaining numbers in §4 (margins 3.0×/2.0×/…; +0.30 [0.07, 0.53]; Qwen d 0.392 and collapse at α=2),
§5 (all E1/A4 numbers against `e1_scores.json`), §6 (all geometry numbers against
`controls_results/`), §3 single-layer 16.8% / 71.6%, and every citation's specific claim
(Turner, Rimsky, Monroe, Wang 2022, Conmy, Zou, Li 2024, Todd, Siddique, RePS, Geiger, Cunningham,
Bolukbasi, Ravfogel, Belrose, Welleck, AxBench, Macocco, Scalena).
