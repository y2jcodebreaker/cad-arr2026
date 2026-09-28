# Pre-registration A5: prompt-residual projection

**Status: DRAFT, written 2026-09-27 ~23:00 PDT.** It becomes binding at the commit that marks it
FROZEN, which must precede the first fit under a non-default prompt. Nothing below has been run.

## 1. The idea in one sentence

Fit the bias subspace on the model *while it is conditioned on the debiasing instruction it will be
deployed with*, so that the projection removes only the bias that the instruction leaves behind.

## 2. Why this follows from our own data

E1 and A4 (`claims/E1_RESULTS.md`) established three things on the frozen held-out set:

1. An explicit debiasing instruction removes little medicalization on its own (d = 0.06, 13%).
2. The CAD projection removes much more (d = 0.34), and adding the instruction on top raises it to
   d = 0.51 (97%) with intact output. The two are complementary.
3. Projection rank behaves like a strength: more directions remove more bias and cost more quality.

The combined arm uses directions fit **without** the instruction. If the instruction already moves
part of the bias, some of those directions are spent on bias that is no longer there at inference,
and each spent direction costs quality (point 3) for no debiasing. Fitting under the instruction
should put the same rank budget on what remains.

## 3. Method (identical to CAD projection except for one thing)

For each fit pair *i* (the 633 held-out pairs of E1), instruction *s*, layer *L* ∈ {21, 25}:

    Δh_i(s) = h_L(s ⊕ disability query_i) − h_L(s ⊕ neutral query_i)      (last token)
    centre, stack, SVD  →  V_k(s)
    at inference, under instruction s:   h' = h − V_k(s) V_k(s)ᵀ (h − μ(s))

The **only** change from the E1 projection is the system prompt used when collecting activations
for the fit. Layers, pool, loudness filter, α = 1, generation settings, frozen evaluation set and
judges are unchanged.

## 4. Novelty check (done 2026-09-27, before this draft)

| Closest work | What it does | Difference |
|---|---|---|
| CAA, Rimsky et al. 2024 | steering adds a benefit on top of a system prompt | vector fit without the prompt; no targeting of the residual |
| Stolfo et al., ICLR 2025 (arXiv 2410.12877) | vector = activations with minus without an instruction, to *strengthen* instruction following | derives the vector *from* the instruction; A5 contrasts groups *under* the instruction |
| Steer Like the LLM (arXiv 2605.03907) | learns steering that *replaces* a prompt | A5 keeps the prompt |
| Prompt–Activation Duality / GCAD (arXiv 2605.10664) | steering from the prompt's attention contributions | same: prompt-derived, not residual |
| Activation Source Selection (arXiv 2607.25270) | which context and positions steering signals are read from | supports the premise that the read-out context matters; does not condition on a deployment instruction to target a residual |
| Illusion of Debiasing, Feng et al. 2026 (arXiv 2609.07117) | prompt-based debiasing has a structural reach limit | consistent with point 1; proposes no steering |
| Tan et al., NeurIPS 2024 | steering vectors vary with the prompt and are unreliable | motivates fitting in the deployment context |

No work found fits a removal subspace on activations conditioned on the deployment instruction to
target what the instruction leaves. The ingredients are known; the principle and its test are new.
**Before submission:** re-run this search once, and read Stolfo et al. and 2607.25270 in full.

## 5. System prompts (fixed now, selectable only by name in code)

| name | text | role |
|---|---|---|
| `default` | "You are a helpful assistant." | E1 fit context |
| `explicit` | A4 wording, unchanged | the deployment instruction |
| `style` | "You are a helpful assistant. Keep your answer well organised, use short paragraphs, and define any technical term the first time you use it. Answer the question they asked." | **new control**: same opening and closing sentence, same length (29 words each), no disability content |

## 6. Arms (frozen set, seed 42, α = 1, L21+L25, fit pool 633)

Notation: *inference prompt* / *fit context* / rank.

| id | inference | fit | k | purpose |
|---|---|---|---|---|
| R20, R40, R80 | explicit | explicit | 20, 40, 80 | **the method** (prompt-residual projection) |
| B20, B80 | explicit | default | 20, 80 | the comparison curve (B40 exists: `prompt_explicit_cad_k40`) |
| C1 | default | explicit | 40 | does it need the instruction at inference? |
| C2 | explicit | style | 40 | is it the instruction's *content*, or any change of fit context? |
| CAA-B | explicit | default | CAA L14 α=1 | prompt + CAA (answers C9's open control) |
| CAA-R | explicit | explicit | CAA L14 α=1 | does the principle transfer to an additive operator? |

9 new arms, 2 new fits per operator family. About 2.2 GPU-hours including both judges.

## 7. Gates (a failed gate stops the run and is reported)

- **G0.** Dry run of every arm prints the frozen-set hash and the fit context.
- **GA (after the fits, before any generation, CPU).** The instruction must change the fitted
  subspace: mean squared cosine of the principal angles between V_40(explicit) and V_40(default),
  averaged over the two layers, **< 0.90**. If ≥ 0.90 the two subspaces are nearly identical,
  R40 ≈ B40 by construction, and the experiment cannot distinguish them: stop, report "the
  instruction does not change the bias subspace at these layers", and run nothing further.
- **G1.** The unsteered reference is reused from E1 (seed 42); no new baseline is generated.

## 8. Predictions, with every outcome named

**P9 (primary): removal at matched debiasing.** From the B curve (k = 20, 40, 80) read quality at
d = d(R40) by linear interpolation between the bracketing ranks; Δq = q(R40) − q_B(d(R40)), with a
question-level bootstrap (50 questions, 2,000 resamples; d and q recomputed in each resample).
A call needs **both judges** to agree in sign with CIs excluding 0.
Also reported, not decisive: the direct R40 − B40 paired differences in d and in quality.

- **(a) Better quality at matched debiasing.** Δq > 0 on both judges, CIs exclude 0.
- **(b) More debiasing at matched rank.** d(R40) − d(B40) > 0 with CI excluding 0, and the paired
  quality difference R40 − B40 is not significantly negative on either judge.
- **(c) No advantage.** Neither (a) nor (b). The method is not claimed; the paper keeps the
  combination as its result and reports A5 as a null in the appendix.
- **(d) Worse.** Δq < 0 on both judges with CIs excluding 0: reported as such.
- **Unmatched.** If d(R40) lies outside [d(B20), d(B80)], (a) cannot be evaluated: report, do not
  extrapolate, and decide on (b) alone.

**P10: mechanism, interpreted only if (a) or (b) holds.**
- **C1.** Prediction: d(C1) < d(`cad_proj_k40`) = 0.34. The residual subspace should matter less
  without the instruction that created the residual. If d(C1) ≥ 0.34, the gain is "a better
  subspace", not residual targeting, and the paper says *fit context matters* instead of
  *remove what the instruction leaves*.
- **C2.** Prediction: C2 behaves like B40, not like R40 (|d(C2) − d(B40)| < |d(C2) − d(R40)| and
  likewise for quality on both judges). If C2 matches R40, any change of fit context produces the
  gain, and the instruction's content is not the cause.

**P11: generality (secondary).** CAA-R − CAA-B paired: supported if its sign matches the P9
outcome on both judges. A5 is claimed as a *principle* only if P9 and P11 agree; otherwise it is a
property of the projection.

**Descriptive, pre-declared (no pass/fail).** Principal angles (GA); top-128 contrast energy
Σσ² under `explicit` vs `default` at each layer (how much the instruction shrinks the contrast).

## 9. Analysis rules carried over from E1, unchanged

Reference unsteered seed 42; d = runner `cohens_d`; medicalization recomputed from text; degeneracy
flags from text (unique < 0.50, parse-fail > 0.10, mean med < −0.05); parse failures scored 5.0;
question-level bootstrap. A flagged arm cannot satisfy (a) or (b).

## 10. What has already been seen

All of E1 tier 1 and A4, including `prompt_explicit_cad_k40` (= B40: d 0.51, ΔQ −0.70 / −0.20).
No activations under a non-default fit context have been computed.

## 11. Engineering required before freezing

1. Move `SYSTEM_PROMPTS` to `e1_common.py` (one definition, adds `style`); CAD and CAA runners
   import it.
2. CAD runner: `--fit_system_prompt NAME` (fit context) and `--fit_only` (fit, save, exit). The SVD
   cache records its fit context and refuses a mismatch; non-default fit contexts get their own
   `--out_dir`, and every output name carries `_fit<name>`.
3. CAA runner: `--system_prompt` and `--fit_system_prompt`, same rules.
4. `run_e1.py`: stage A5 arms, `gate ga` (principal angles from the two caches), fit-then-gate-then
   generate ordering.
5. `e1_analysis.py`: P9–P11, tested on stand-in files before any A5 output exists.
