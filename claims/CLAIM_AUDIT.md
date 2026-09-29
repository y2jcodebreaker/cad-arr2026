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

| A10 | 02m.007–009: layers located by patching toward "the biased behaviour" | **FLAG.** `steering_track2/discover_medicalization_circuit.py` patched toward AvgLogit(top-20 "pity" words) − AvgLogit(top-20 neutral words) at the last prompt position. The top-20 "pity" words are *with, impairments, individuals, disorders, mental, speech, mobility, hearing, support, sensory, cognitive, visual, vision, may, accessible, assistive, communication, accommodations, disability, accessibility*: the disability's name and accessibility vocabulary, none medical. L21/L25 therefore localise where the model moves toward **talking about the disability**, which matches M1 (the projection erases the name, does not reduce framing). The per-layer patching scores survive only as HTML visualisations (`~/Downloads/pity_circuit_*.html`), not as data. | script (read 2026-09-28); `results/phase1_vocab/pity_vocabulary.json` |
| A11 | Eq. 2: h′ = h − α V Vᵀ (h − μ) | **FLAG.** The code (`make_projection_hook`) computes h − α (h Vᵀ) V: no centring term μ. | `evaluate_rtsd_fullres_generation.py` l.391–411 |
| A12 | 02m.020: "During prefill we steer only the last token, and during generation every newly produced token" | **FLAG.** The hooks modify every position of the layer output, prompt tokens included (full-sequence matmul on prefill). | same, l.403–411 |

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

## C. Remaining numbers (§3, §4, §5, §6)

| Sentence | Claim | Status | Source / note |
|---|---|---|---|
| 03.034 | margins 3.0× / 2.0× / 2.0× / 1.9× / 1.2× / 1.0× (first strength with quality < 6.0) | ✅ numbers reproduce | `sweep_points.json`. **But** they are measured from "the setting an author recommends" (FLAG, A7): from the E1 settings for baselines and our own choices for CAD |
| 03.036 | SADI s=5: 7.5, "removes little bias" | ✅ (med 0.593 vs ~0.66) | same |
| 03.037 | "no intermediate setting in the published sweep" | wording FLAG ("published") | swept 1, 2, 5, 10: true within our sweep |
| 03.027 | projection vs CAA +0.30 [0.07, 0.53] | ✅ (+0.298 [0.072, 0.530], MW p = 0.0014) | `analysis-output/stats-appendix.md` |
| 03.043 | Qwen d = 0.392 at α=1, no degenerate output; collapse at α=2 | ✅ with caveats: 1 empty response at α=1; α=2 flagged with 16 empty. Lexical metric only; no quality judge on Qwen | `results/qwen_generation_results.json` |
| 02m.019 | single layer 16.8%, two layers 71.6% | **FLAG.** Neither is in a committed file. 16.8% (notes) was an **additive probe** at L21, not the projection; the committed two-layer projection run gives **66.4%** | `results/rtsd_fullres_results.json` (α=1: 66.4%) |
| 03.005 | "full published range" | **FLAG.** Ranges are ours | runner configs |
| 04.* | every §5 number (19 checks: d, CIs, ΔMed, ΔQ, P2, P5, P8, rank sweep, LEACE overlap 0.03, probe removal) | ✅ all match | `analysis-output/e1/e1_scores.json`; gate values G2 0.580, G3 0.642 from `results/e1/e1_tier1.log` |
| 04.024 | leak raises d by 0.004 / 0.001 | ✅ numbers; **FLAG interpretation**: the CAD leak control compares two k=auto fits whose rank also changed (33/36 → 42/49), so fit pool and rank are confounded | `e1_scores.json`, D1 |
| 05.011–014 | homogeneity ladder L21 (22.1/7.0 … 76.1/1.7, CIs [17.8, 28.4], [62.0, 87.5]); L14 22.9 [18.1, 28.6] → 69.4 [54.3, 80.4] | ✅ all | `controls_results/day3_subsample_results.json` |
| 05.016 | "13 pairs by drawing whole base queries": 21.5% → 72.2% | ✅ values; **FLAG** the cluster draw averages **16.8** pairs at the 13 target | same, `curves.L21_cluster` |
| 05.017 | null 26.2% vs real 21.5% at n=13 | ✅ | same |
| 05.017 | "separates from it only above n ≈ 250" | **FLAG.** 95% intervals overlap at every n ≤ 500 and separate only at n = 1,000. The null is above the real data at every n | same, `curves.L21_pair/L21_null` |
| 05.018 | pair bootstrap k 40 [40, 41] | ✅ **resolved 2026-09-28**: re-run gives 40.7 [40, 41] (text should say 40.7); L14 35.9 [35, 37], L0 11.0 [11, 11] | `controls_results/pair_bootstrap.json` (script 9537a91) |
| 05.018 | question bootstrap 30.1 [27, 33] | ✅ | `controls_results/question_bootstrap_L21.json` |
| 05.020 | k_auto 33/36 held-out vs 42/49 full | ✅ | E1 run logs, `svd_fullres.pt` (`fits/`) |
| Tab 3 + 05.010 | "profile the paper reports for DiscrimEval Target A" 67.9% / k=2 | **FLAG.** Only source is a constant in `plot_controls_figure.py`; it refers to our own earlier, unpublished paper (anonymity problem in a double-blind submission) | — |
| 05.025 | cos(v1, probe) = −0.001 at L21 | **FLAG.** Not reproducible from committed fits: −0.07 (held-out) / −0.13 (full) at L21. "Close to orthogonal" holds. Also: **65% of the probe direction lies inside the top-40 SVD subspace** (held-out fit), which the text should say | `fits/e1_outputs/*/svd_fullres.pt` |
| 05.006 | "pool of 2,083 pairs" | ✅, but 1,044 distinct (duplicated rows); say so | `frozen_eval_v1.json` |

## D. Citations: does the cited work say what the sentence says?

| Citation | Claim in draft | Status | Checked against |
|---|---|---|---|
| accesseval2024 | medicalization term "following" it | **FLAG** (A2) | arXiv 2509.22703 |
| accesseval2024 | a disability-bias benchmark | ✅ | same |
| monroe2008fightin | the medicalization *score* is its log-odds | **FLAG** (A3): Monroe gives the vocabulary method; the score is a 20-word count ratio | `evaluate_rtsd_fullres_generation.py` |
| turner2023activation, rimsky2024steering | steering edits activations at inference, no retraining | ✅ | abstracts |
| wang2022interpretability, conmy2023automated | activation patching | ✅ (generic) | — |
| zou2023representation (LAT) | PCA over representation differences | ✅ "the inputs to PCA are {A(i)−A(j)} … first principal component … 'reading vector'" | ar5iv 2310.01405 |
| zou2023representation | "It is common … to describe how many dimensions a bias occupies from the spectrum" | **FLAG**: RepE uses the first PC, it does not estimate dimensionality; no second instance found. Say it was our own practice | — |
| li2024inference (ITI) | steer every generated token | ✅ "repeated for each next token prediction autoregressively" | ar5iv 2306.03341 |
| todd2024function | steer the last prompt token | ✅ "the hidden state residual stream at the final token of a given prompt" | ar5iv 2310.15213 |
| li2025fairsteer | classifier gate | ✅ | arXiv 2504.14492 |
| siddique2025shifting | "per-axis PCA vectors from BBQ" | ⚠ per-axis vectors on a BBQ training subset ✅; "PCA" not verified. Venue Findings EACL 2026 ✅ (bib correct) | ACL Anthology 2026.findings-eacl.41 |
| wu2025reps | "requires training-time gradients" | ⚠ true (trained with a preference objective) but say it that way; note it also does concept **suppression** | arXiv 2505.20809 |
| pham2026hidra, he2025saessv, doan-etal-2026-causal | descriptions | ✅ abstract level | arXiv, ACL Anthology |
| wu2025axbench | prompting outperforms steering at injection | ✅ | PMLR v267 |
| macocco2026tradeoff | steering costs fluency; prompting weaker at removal | ✅ both in abstract | arXiv 2606.12234 |
| scalena-etal-2024-multi | per-step strength from KL(unsteered ‖ steered), capped at 2 | ✅; "a direct response to the failure we measure" **FLAG** (overclaims; say it targets the same trade-off) | arXiv 2406.17563 html |
| geiger2024das, cunningham2024sae, bolukbasi2016man, ravfogel2020null, belrose2023leace, welleck-etal-2019-neural | generic descriptions | ✅ (standard characterisations) | — |

## E. Wording that asserts facts about the field

"leaderboard" (01.015, 03.041), "published setting/operating point/methods/numbers/range/ranking"
(ABS.005, 01.019, 01.021, 02.019–020, 03.002, 03.005, 03.011, 03.021, 03.029, 03.035–037, 04.006,
04.017–018, 06.009, Lim.005), "the field measures" (07.002), "It is common" (01.028), "The usual
estimator" (05.003): **all FLAG** under A1/A7. There is no leaderboard for this metric; "published"
is false for all four baseline settings.

## F. Non-numeric factual sentences (second pass, 2026-09-28)

The first version of this file said "all 224 sentences checked" before this pass was done; that
line was wrong and is replaced below.

| Sentence | Claim | Status | Evidence |
|---|---|---|---|
| 02.017 | "FairSteer and CAA are more than twice the unsteered length" | ❌ 945 / 481 = 1.96×, 638 / 481 = 1.33× | result files |
| 01.018 | degenerate outputs are "longer than the responses they replace" | ❌ for SADI (188 vs 481 words) | same |
| 03.012, 01.008 | SADI has "the lowest medicalization score of the five" / "strongest target-metric score" | ❌ Angular is lower (−0.222 vs 0.119); the two sentences also contradict 01.014 | `sweep_points.json` |
| 03.018 | "Every point that reaches or passes neutral is degenerate" | ✅ no intact point has med ≤ 0 (the meaning of "neutral" is A4) | same |
| 03.026 | "No baseline method reaches that region without degenerating" | ❌ 7 intact baseline points reach ≥ 50% reduction (CAA α=1; FairSteer τ 0.3/0.5/0.7 α=2; Angular max-norm L31 mode 1 150°, modes 0/1 270°). True claim: none is on the Pareto front there | same |
| 01.014, 03.013 | Angular "injecting corruption inside words" | **FLAG: unverified.** An in-word pattern search on the sweep outputs matches only legitimate words (*DuckDuckGo*, *LinkedIn*); the earlier "6.0% artifacts" figure is already retired as unverified | `results/angular_results.json` |
| 03.030–031 | "falls steeply … not gradual in any" | interpretation; CAD mean-diff falls 7.92 → 6.60 over α 1–10 before the drop | `sweep_points.json` |
| 05.022 | "one varies a single template while the other spans nine categories" | **FLAG** (refers to Target A, see Tab 3 row) | — |
| 07.001 | "answers … a question about risk and supervision" | **FLAG** (A6: the example is not from the data) | — |
| 04.002–005, 04.019, 04.022–027, 04.033, Lim.001, Lim.004 | design statements, prompt wording, "neither costs quality", fit context | ✅ | PREREG_E1 (A1, A4), `e1_scores.json` |

### F2. GPT-5.5 judge numbers (checked 2026-09-29 against `results/judge_gpt55_checkpoint.json`)

| where | text says | file says | verdict |
|---|---|---|---|
| A_e1.tex l.80 | CAD −0.08 | paired −0.092 [−0.22, +0.04] (n 196); unpaired −0.085 | **FIX**: text used unpaired means; give paired −0.09 to match the stored interval |
| A_e1.tex l.80 | Angular −2.23 | paired −2.237 (n 249); unpaired −2.234 | **FIX**: −2.24 (paired) |
| A_e1.tex l.80 | SADI −5.26 | −5.264 both ways | ✅ |

Context that must go with any GPT-5.5 sentence: it rated **accessibility quality** (0–10, the
AccessEval rubric), not medicalization; on the pre-repo runs (different prompt draws from the
frozen set; CAD arm 196 paired items; code not attributable to a commit, LEDGER rows 14–15); 0 of
its 2,885 texts overlap with any text the E1 judges scored. Candidate use, labelled as an earlier
run: the two arms GPT-5.5 rates far lowest on quality (SADI −5.26, Angular −2.24) were strong
debiasers on the lexical score in those same runs. **UNVERIFIED**: their lexical d on exactly the
GPT-5.5-judged texts must be recomputed from `response_text` before this is written. It cannot be
compared number-for-number with E1/A8.

## G. Summary

- 224 sentences read; numeric claims (45) traced to files, citation claims checked against the
  cited text, generalising words listed, and non-numeric factual sentences checked in a second pass.
  Interpretive sentences are marked as such, not "verified".
- Numbers: 38 of 45 reproduce; 7 do not or cannot be traced (section C). Non-numeric factual
  errors: 5 (section F), plus the unverified Angular corruption claim.
- Foundational problems A1–A9. Every claim of removing "medicalization", "% reduction" or "past
  neutral" waits for M1.
- Nothing in the paper has been edited yet. The rewrite follows M1 and A7.
