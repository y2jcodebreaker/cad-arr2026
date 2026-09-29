# Pre-registration A11: second model (Mistral-7B-Instruct-v0.3), gated

**Status: FROZEN, 2026-09-29.** Approved as written by the author before any code, answer or rating for
this design exists. Changes after this commit are amendments with a timestamp and a reason.

## 1. Why

Every result so far is on Llama-3.1-8B-Instruct. A11 asks, on a second model family: does it
medicalize on the valid measure, does the lexical score see it, and does framing-contrast steering
(A8) replicate? An earlier, pre-repository run found a near-zero *lexical* medicalization for
Mistral-7B on AccessEval (0.033; memory note, not a committed file). The lexical score is not
valid, so whether Mistral medicalizes is open; the design is gated on it.

## 2. Model and prompts

`mistralai/Mistral-7B-Instruct-v0.3`, revision `c170c708c41dac9275d15a8fff4eca08d52bab71`, bf16.
Generation exactly as for Llama: sampling at temperature 0.1, seed 42, same token limit and batch
size. Prompt: the model's official chat template with the same system prompts by name ("default",
"explicit"); the template folds the system text into the user turn (`[INST] SYSTEM\n\nUSER[/INST]`).
The template's own BOS is stripped so the tokenizer adds exactly one. The tokenizer needs
`sentencepiece` (added to the environment; it is not used by the Llama or Qwen tokenizers). Hooks at
`model.model.layers[L]` (32 layers, hidden size 4096, as Llama). Fit pool, frozen set, judges,
rubrics, echo and lexical score unchanged.

## 3. Stages and gates

**S0, existence (always run and reported).** `mistral_unsteered` and `mistral_neutral` on the frozen
set; M1 judges and J3. Reported: disability − neutral J gap per judge (M1 Q1 rule), and the lexical
gap. No gate.

**S1, fit (A8's procedure, unchanged).** One answer to each of the 633 fit-pool questions; both M1
judges; labels by A8's rule; features, balanced-accuracy layer rule and stratified direction as A8.
- **GA11-1**: ≥ 40 medicalizing and ≥ 40 clean fit answers. Else stop: outcome (d).
- **GA11-2**: best-layer balanced accuracy ≥ 0.60. Else stop: outcome (d).

**S2, arms (frozen set)**: `mistral_fc_remove_a1` (projection α = 1 at the two selected layers),
`mistral_fc_prompt_remove_a1` (explicit prompt + the same), `mistral_prompt_explicit`, and
`mistral_fc_shuffle_s0` (within-domain shuffled labels, seed 0, as A9). All judges and accessibility
judges.

## 4. Criterion and named outcomes

Confirmatory arm `mistral_fc_remove_a1` against `mistral_unsteered`, A8's success rule at 95%
(one confirmatory arm): ΔJ < 0 with the interval excluding 0 on both M1 judges; echo drop < 0.25;
accessibility-quality change ≥ −1.0 on both accessibility judges; M1 U-rate ≤ 10% on both.

- **(a)** All hold: framing-contrast steering replicates on a second model family.
- **(b)** ΔJ holds but erasure or quality fails.
- **(c)** ΔJ does not hold on both judges.
- **(d)** A fit gate fails (too little medicalization to fit, or not separable); S0 is the result.

Also reported: J3 for every Mistral arm (J3 is Mistral-family, so it is secondary here), the
shuffle arm against unsteered and against `mistral_fc_remove_a1`, the other two arms against the same
rule, lexical d of every arm, and whether the lexical gap detects the S0 medicalization.

## 5. Already seen

The earlier lexical 0.033 (not re-checked); Mistral-7B's tokenizer and chat template behaviour
(checked 2026-09-29 under the pinned versions). No Mistral answer on this design exists.

## Amendments

None.
