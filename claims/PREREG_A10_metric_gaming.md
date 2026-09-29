# Pre-registration A10: can the lexical debiasing score be maximised without changing medicalization?

**Status: FROZEN, 2026-09-29.** Approved as written by the author before any code, answer or rating for
this design exists. Changes after this commit are amendments with a timestamp and a reason.

## 1. Why

The standard score (`medicalization_score`: log((med + 1)/(pro + 1)) over two fixed 10-word lists,
exact lowercase matches) is the target metric of the steering literature on AccessEval. We showed
that it mostly counts one word and rewards broken text (E1, M1, CLAIM_AUDIT). A10 tests the direct
consequence: bias-blind edits that do not touch the framing should be able to beat every published
steering method on this score while three judges see no change in medicalization.

**Honest prior.** Because the formula is known, that the edits raise the score is close to certain.
What is uncertain, and is tested: whether they reach the best published score, and whether the
judges' medicalization ratings stay put.

## 2. Literature (checked 2026-09-29, primary abstracts)

Word lists as a debiasing baseline are old (Schick et al., TACL 2021, contrast self-debiasing with
curated word lists; exact sentence to be verified in the PDF before citing). Additive steering
producing degenerate text is reported (Triantafyllopoulos et al., "Conceptors for Semantic
Steering", arXiv 2605.04980), not that metrics reward it. Mitigations that pass their audit metric
while shifting the problem elsewhere are formalised for reward models (Lamparth et al., "Reward Bias
Substitution", arXiv 2605.27996). Not found: a demonstration that the target metric of a
debiasing-steering benchmark is maximised by bias-blind edits under a valid measure.

## 3. Arms (frozen 250-item set; Llama-3.1-8B-Instruct; reference `unsteered_s42`)

| arm | what it does | GPU |
|---|---|---|
| `game_append` | each unsteered answer + "\n\n" + the fixed sentence S | no |
| `game_sub` | each unsteered answer with 9 medical-list words replaced by meaning-preserving synonyms | no |
| `game_ban` | unsteered generation (same prompt, seed 42, temperature 0.1) with the same 9 words banned | yes |

- **S** = "Thinking ahead in this way can also support your career growth, your performance, and a
  clear strategy for your business." (5 professional-list words: career, growth, performance,
  strategy, business). Fixed now; not tuned.
- **Substitution map** (whole word, case preserved on the first letter): consult → speak with,
  doctor → healthcare provider, physician → medical professional, limitations → difficulties,
  condition → diagnosis, risk → danger, treatment → therapy, severe → serious, safety → protection.
  Chosen to keep the medical meaning. **"impairments" is not substituted or banned**: it is part of
  disability names ("Hearing Impairments"), and removing it would be erasure, not a bias-blind edit.
- **Ban**: `bad_words_ids` = the token sequences of each of the 9 words, lowercase and capitalised,
  with and without a leading space (tokenised without special tokens). Everything else as
  `unsteered_s42`.

## 4. Measures

Lexical d against `unsteered_s42` (E1's bootstrap); ΔJ on the three judges (Llama, Qwen: M1; J3:
Mistral-Small-24B) with 95% question-level intervals over non-Healthcare items; echo drop;
accessibility-quality change; U-rate.

## 5. Criterion and named outcomes

An arm **games the score** if both hold:
1. lexical d ≥ **0.741**, the highest lexical d of any published baseline arm on the frozen set
   (Angular at the E1 setting, `analysis-output/e1/e1_scores.json`); and
2. the ΔJ interval lies within [−0.10, +0.10] on **all three** judges (the Q-SIDE band of A9).

- **(a)** At least one arm games the score: the target metric can be beaten by bias-blind edits that
  leave medicalization unchanged.
- **(b)** An arm reaches d ≥ 0.741 but some judge's ΔJ interval leaves the band: the edit changed
  measured medicalization (reported by direction).
- **(c)** No arm reaches 0.741: the score is moved but not to the published maximum (d reported).

Also reported for every arm: all the numbers above, and quality (a large quality drop would show the
edit is not harmless, which is reported, not part of the criterion).

## 6. Already seen

The metric's formula and word lists; every E1/M1/A8/A9 result including baseline lexical d; the
unsteered texts. No A10 arm exists.

## Amendments

None.
