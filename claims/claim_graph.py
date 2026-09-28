"""Claim graph for "When Debiasing Scores Reward Broken Text" (ARR Oct 2026).

Run it:  python3 claims/claim_graph.py
Prints every claim with its evidence, controls and falsifier, then WARNINGS:
  - a claim with no falsifier
  - depth-first descent (a deeper claim run while a shallower one has open blocking controls)
  - a retired number still quoted anywhere in paper_arr2026/sections/
Audit date 2026-09-26. Numbers here were checked against the files named in `where`.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper_arr2026" / "sections"


@dataclass(frozen=True)
class Control:
    name: str
    rules_out: str
    done: bool
    blocking: bool
    where: str = ""


@dataclass(frozen=True)
class Claim:
    id: str
    layer: int
    section: str
    statement: str
    evidence: str
    falsifier: str
    has_run: bool
    controls: tuple[Control, ...] = field(default_factory=tuple)

    @property
    def open_blocking(self) -> list[Control]:
        return [c for c in self.controls if c.blocking and not c.done]


MATCHED_SET = Control(
    "one frozen eval set for every arm",
    "differences caused by prompt draw, not by the intervention. Unsteered baseline "
    "varies 0.509-0.712 across the runs this claim uses (within-run sd 0.038)",
    done=False, blocking=True)

E1_SET = Control(
    "one frozen eval set for every arm", "differences caused by prompt draw", True, True,
    "results/e1 (frozen_eval_v1 153439a6); G1 seed sd 0.0065")

CLAIMS: tuple[Claim, ...] = (
    Claim("C1-degenerate-scores-zero", 1, "S3 audit",
          "Three methods reach medicalization ~0.000 via three distinct degeneration modes "
          "(SADI corruption 249/250 distinct; FairSteer repetition 68/250; CAA loops 247/250); "
          "no single filter on emptiness, length or uniqueness catches all three.",
          "frontier sweep, verbatim responses; judge_scores.json",
          "An independent judge or human rates these outputs as intact answers.",
          has_run=True,
          controls=(
              Control("degeneracy defined from text only", "circularity with the judge",
                      True, True, "analysis-output/sweep_points.csv"),
              Control("second judge family on the degenerate points", "Llama self-preference",
                      True, False, "judge_gpt55: SADI -5.26, Angular -2.23"),
              Control("human audit of a disagreement sample", "both judges sharing a bias",
                      False, False))),

    Claim("C2-published-points-cost", 2, "S5 (Table e1)",
          "At published settings every method loses judged quality, in the order SADI > Angular > "
          "FairSteer > CAA > CAD projection under both judges; the top-d arm (Angular 0.74) overshoots.",
          "E1 P1: analysis-output/e1/e1_scores.json",
          "On the frozen set, SADI's drop is not the largest or the projection's not the smallest "
          "on either judge.",
          has_run=True,
          controls=(
              E1_SET,
              Control("parse-failure sensitivity", "the 5.0 default flattering degenerate points",
                      True, False, "0.00% of 2,500 clean scores are 5.0"),
              Control("second judge on the same items", "Llama self-preference",
                      True, True, "E1: Qwen2.5-7B, rho 0.642; same order (P1)"))),

    Claim("C3-clean-frontier-is-CAD", 2, "S4.2 + Fig 1",
          "Clean Pareto front: 12 configs, 10 CAD; all 4 with >=50% reduction are CAD.",
          "frontier sweep; 27-threshold sensitivity",
          "On a matched eval set a baseline config sits on the clean front at >=50% reduction.",
          has_run=True,
          controls=(
              MATCHED_SET,
              Control("threshold sensitivity (27 combos)", "arbitrary degeneracy cut-offs",
                      True, False, "front identical in all 27"))),

    Claim("C4-paired-quality", 2, "S5",
          "At matched debiasing (d 0.34 vs 0.37) the CAD projection and CAA are NOT distinguishable "
          "in quality: paired +0.22 [-0.12, 0.57] Llama, +0.06 [-0.03, 0.15] Qwen. (The earlier "
          "+0.352 [0.176, 0.534] used response-level CIs; E1 P2 failed.)",
          "E1 P2: analysis-output/e1/e1_scores.json",
          "A question-level interval on a larger frozen set excludes 0 in either direction.",
          has_run=True,
          controls=(
              E1_SET,
              Control("second LLM judge on the same pairs", "Llama self-preference", True, True,
                      "E1 Qwen judge"),
              Control("intervals over questions, not responses", "nested variants", True, True,
                      "question bootstrap, 50 clusters"),
              Control("audit of VADER disagreement", "VADER no longer supports any claim",
                      False, False))),
    Claim("C5-rank-matters", 2, "S5.2",
          "Projection d rises with rank (k1 0.02, k40 0.34, k80 0.45) as quality cost rises "
          "(Llama -0.03 to -0.92); LEACE same trend weaker, k1 vs k40 CIs overlap (P3 fails); "
          "rank-1 probe-direction removal d 0.08 (P6 outcome c).",
          "E1 P3/P6: analysis-output/e1/e1_scores.json",
          "On the frozen set, rank-1 and rank-40 projection are indistinguishable.",
          has_run=True,
          controls=(
              E1_SET,
              Control("judged quality for every rank arm", "d earned by destroying text", True, True,
                      "both judges"),
              Control("rank dose-response (1,5,10,20,40,80)", "two points cannot show a trend",
                      True, False, "E1 rank sweep, both operators"),
              Control("same LEACE concept at every rank", "two operators on one axis", True, True,
                      "exp3 at all k"),
              Control("direction control: rank-1 projection along the probe", "rank vs direction",
                      True, True, "cad_proj_probe d 0.08"))),

    Claim("C9-prompting", 2, "S5.1",
          "An explicit system prompt removes 13% (d 0.06); the projection adds 0.28 [0.14, 0.44]; "
          "prompt + projection reaches d 0.51 [0.37, 0.66], 97%, intact.",
          "E1 A4/P8: analysis-output/e1/e1_scores.json",
          "A pre-registered prompt matches the projection's d at no greater quality cost.",
          has_run=True,
          controls=(
              E1_SET,
              Control("prompt wording fixed before any output", "prompt tuned to lose", True, True,
                      "PREREG A4, names only in code"),
              Control("more than one prompt", "one wording", True, False, "min + explicit"),
              Control("prompt + other methods (CAA)", "combination specific to CAD", False, False))),

    Claim("C6-geometry-is-construction", 2, "S6",
          "At n=13, varying only prompt diversity moves EVR1 22.1% -> 76.1%; permuted null "
          "exceeds real data below n~250; question bootstrap k 30.1 [27,33] vs pair [40,41].",
          "controls_results/day3_subsample_results.json",
          "Restricting diversity on a second benchmark leaves EVR1 unchanged.",
          has_run=True,
          controls=(
              Control("permutation null, 40 draws", "structure present in any stacked diffs",
                      True, True, "day3"),
              Control("cluster bootstrap", "pair-level CI overstating precision", True, True, "day3"),
              Control("clusters = unique QUESTIONS, twins merged",
                      "day3 resampled 292 base_query_ids as independent; they are 146 questions "
                      "x 2 identical rows, so every cluster interval is too narrow",
                      True, True, "controls_results/question_bootstrap_L21.json: 292-ID grouping "
                      "reproduces 35.0 [33,37] (positive control); 146 questions give 30.1 [27,33]"),
              Control("direct test on Target A: ADD diversity, EVR1 should fall",
                      "argument by analogy from AccessEval to a different benchmark and layer",
                      False, False))),

    Claim("C7-second-model", 3, "S4 (Qwen para)",
          "On Qwen2.5-7B the projection reaches d=0.392 at a=1 and collapses at a=2.",
          "qwen_generation_results.json",
          "On Qwen the quality ordering of methods differs from Llama's.",
          has_run=True,
          controls=(
              Control("judged quality on Qwen", "target-metric-only evidence (see C5)", False, True),
              Control("baselines on Qwen", "CAD-only; no comparison possible", False, True))),

    Claim("C8-geometry-second-benchmark", 3, "S6",
          "The homogeneity effect replicates on DiscrimEval.",
          "not run", "No EVR1 movement with homogeneity on DiscrimEval.",
          has_run=False),
)

# (old value, new value, why) — scanned against the paper
SUPERSEDED: tuple[tuple, ...] = (  # (old, new, why[, files allowed to quote it as history])
    ("211/250", "201/250 score 0.000, 0 empty", "SADI s=10: 'empty' was a misread of zero-score"),
    ("211 of 250", "201 of 250", "same"),
    # AccessEval lists every question TWICE (row i == row i+234, verified 2026-09-27).
    # The counts below treat the duplicates as independent. The ORIGINAL "234 queries /
    # 2,082 pairs" was correct; the 2026-09-03 "correction" to 468 / 2,083 was wrong.
    ("468 base queries", "234 unique questions", "dataset rows are duplicated"),
    ("292 base queries", "146 unique questions", "twins carry two base_query_ids"),
    ("4{,}164 expanded pairs", "2,082 unique pairs", "every expanded pair appears twice"),
    ("resampling the 292", "resampling the 146 unique questions", "292 IDs are 146 questions x 2"),
    ("$35.3$ $[33, 37]$", "30.1 [27, 33]", "292-ID cluster bootstrap counted each question twice"),
    ("$32.2$ $[31, 33]$", "28.2 [26, 30]", "same, L14"),
    ("$10.6$ $[10, 11]$", "10.2 [9, 11]", "same, L0"),
    ("941 words", "1,048 (repeated text) / 945 mean", "FairSteer a=6"),
    ("6.0\\% artifacts", "unverified", "Angular artifact rate never regenerated"),
    ("$-$0.12", "-0.33", "AAAI judge column (matched in its original $-$x.xx form): result file never saved",
     ("A_e1_arms.tex",)),  # generated from E1; its -0.12 is a genuine interval bound
    ("$-$0.54", "-0.63", "AAAI judge column (matched in its original $-$x.xx form): result file never saved"),
    ("$-$4.42", "-4.73", "AAAI judge column (matched in its original $-$x.xx form): result file never saved"),
    # E1 (2026-09-27): the paired quality advantage did not replicate with question-level CIs
    ("roughly half", "no reliable difference (E1 P2)", "paired +0.35 used response-level CIs"),
    ("roughly halved", "no reliable difference (E1 P2)", "same"),
    ("$[0.176, 0.534]$", "E1 P2: +0.22 [-0.12, 0.57] Llama", "response-level CI; kept only in app:earlier",
     ("A_e1.tex",)),
    ("$+0.352$", "E1 P2", "same", ("A_e1.tex",)),
    # A2: label-concept rank-1 LEACE is a different operator from rank-k SVD-coordinate LEACE
    ("${-}0.03$", "E1 leace_k1 d 0.01", "not on the rank axis (A2)"),
    ("$d{=}0.205$", "E1 leace_k40 d 0.17", "earlier eval set; superseded by E1"),
)


def scan_superseded() -> list[str]:
    hits = []
    for tex in sorted(PAPER.glob("*.tex")):
        body = tex.read_text()
        for entry in SUPERSEDED:
            old, new, why = entry[:3]
            allowed = entry[3] if len(entry) > 3 else ()
            if old in body and tex.name not in allowed:
                hits.append(f"RETIRED NUMBER: '{old}' in {tex.name} -> use {new} ({why})")
    return hits


def main() -> int:
    warns: list[str] = []
    for c in CLAIMS:
        if not c.falsifier.strip():
            warns.append(f"NO FALSIFIER: {c.id}")
    for c in CLAIMS:
        if c.has_run and c.layer > 1:
            for s in CLAIMS:
                if s.layer < c.layer and s.open_blocking:
                    warns.append(f"DEPTH-FIRST: {c.id} (layer {c.layer}) has run while "
                                 f"{s.id} (layer {s.layer}) has {len(s.open_blocking)} open "
                                 f"blocking control(s)")
    warns += scan_superseded()

    for layer in sorted({c.layer for c in CLAIMS}):
        print(f"\n{'=' * 78}\nLAYER {layer}\n{'=' * 78}")
        for c in (c for c in CLAIMS if c.layer == layer):
            ob = c.open_blocking
            status = "SAFE" if not ob else f"{len(ob)} OPEN BLOCKING"
            print(f"\n[{c.id}] {c.section}  —  {status}{'' if c.has_run else '  (not run)'}")
            print(f"  claim    : {c.statement}")
            print(f"  evidence : {c.evidence}")
            print(f"  falsifier: {c.falsifier}")
            for k in c.controls:
                mark = "done" if k.done else ("OPEN*" if k.blocking else "open ")
                print(f"    [{mark}] {k.name}")
    print(f"\n{'=' * 78}\nWARNINGS ({len(warns)})\n{'=' * 78}")
    for w in dict.fromkeys(warns):
        print("  " + w)
    print("\n  * = blocking: the claim is not safe to build on until this is done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
