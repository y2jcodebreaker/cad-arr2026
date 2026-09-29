"""The arms M1 re-scores (PREREG_M1 section 5), shared by the pod judge and the laptop analysis.

One definition, so the judge and the analysis can never disagree about which texts belong to
which arm. Pure stdlib: loads fast on the laptop and on the pod.

    load(base)  ->  {arm: {"side": "disability" | "neutral", "texts": [250 strings]}}

`base` is `e1_outputs` on the pod and `results/e1` on the laptop (same layout).
"""
from __future__ import annotations

import json
from pathlib import Path

# arm -> (file relative to base, key path into the JSON, side of the question it answers)
_RESP = ("responses",)
ARM_FILES: dict[str, tuple[str, tuple, str]] = {
    "unsteered_s42": ("cad_heldout/baseline_seed42_responses.json", _RESP, "disability"),
    "unsteered_s43": ("cad_heldout/baseline_seed43_responses.json", _RESP, "disability"),
    "unsteered_s44": ("cad_heldout/baseline_seed44_responses.json", _RESP, "disability"),
    "neutral_s42": ("cad_heldout/baseline_seed42_neutral_responses.json", _RESP, "neutral"),
    "prompt_min": ("cad_heldout/baseline_seed42_sysmin_responses.json", _RESP, "disability"),
    "prompt_explicit": ("cad_heldout/baseline_seed42_sysexplicit_responses.json", _RESP, "disability"),
    "prompt_explicit_cad_k40": ("cad_heldout/proj_k40_sysexplicit_alpha1.0_responses.json", _RESP, "disability"),
    "cad_proj_kauto": ("cad_heldout/proj_alpha1.0_responses.json", _RESP, "disability"),
    **{f"cad_proj_k{k}": (f"cad_heldout/proj_k{k}_alpha1.0_responses.json", _RESP, "disability")
       for k in (1, 5, 10, 20, 40, 80)},
    "cad_proj_probe": ("cad_heldout/proj_probebasis_alpha1.0_responses.json", _RESP, "disability"),
    "cad_proj_meandiff": ("cad_heldout/proj_meandiffbasis_alpha1.0_responses.json", _RESP, "disability"),
    **{f"add_probe_a{a}": (f"cad_heldout/probe_alpha{a}.0_responses.json", _RESP, "disability")
       for a in (3, 5, 7, 9)},
    **{f"add_meandiff_a{a}": (f"cad_heldout/meandiff_alpha{a}.0_responses.json", _RESP, "disability")
       for a in (3, 5, 7, 8)},
    **{f"leace_k{k}": (f"leace/dual_k{k}.json", _RESP, "disability") for k in (1, 5, 10, 20, 40, 80)},
    "caa": ("caa_heldout/caa_results.json",
            ("accesseval", "layers", "L14", "alpha_1.0", "responses"), "disability"),
    "fairsteer": ("fairsteer/fairsteer_results.json",
                  ("accesseval", "layers", "L29", "thresh_0.5", "alpha_2.0", "responses"), "disability"),
    "sadi": ("sadi/sadi_results.json",
             ("accesseval", "strengths", "strength_10.0", "responses"), "disability"),
    "angular": ("angular/angular_results.json",
                ("accesseval", "strategies", "max_sim_L23", "mode_0", "angle_150", "responses"), "disability"),
    # M1-A1: the A7 arms, each baseline at its authors' own setting
    "angular_adaptive": ("angular_adaptive/angular_results.json",
                         ("accesseval", "strategies", "max_sim_L23", "mode_1", "angle_150", "responses"), "disability"),
    "caa_m2": ("caa_m2/caa_results.json", ("accesseval", "layers", "L14", "alpha_2.0", "responses"), "disability"),
    "fairsteer_a1": ("fairsteer_a1/fairsteer_results.json",
                     ("accesseval", "layers", "L29", "thresh_0.5", "alpha_1.0", "responses"), "disability"),
    "sadi_s5": ("sadi_s5/sadi_results.json", ("accesseval", "strengths", "strength_5.0", "responses"), "disability"),
}
A7_ARMS = ("angular_adaptive", "caa_m2", "fairsteer_a1", "sadi_s5")
# M1-A2: the A8 framing-contrast arms (PREREG_A8), scored like every other arm once they exist
A8_ARMS = ("fc_remove_a1", "fc_remove_a2", "fc_add_a4", "fc_add_a8", "fc_prompt_remove_a1", "fc_remove_a1_L21L25")
for _a in A8_ARMS:
    ARM_FILES[_a] = (f"a8/{_a}_responses.json", _RESP, "disability")
# A9 (PREREG_A9): frozen-set control and side-effect arms
A9_ARMS = ("fc_shuffle_s0", "fc_shuffle_s1", "fc_shuffle_s2", "id_remove_a1", "fc_remove_a1_neutral")
for _a in A9_ARMS:
    ARM_FILES[_a] = (f"a9/{_a}_responses.json", _RESP, "neutral" if _a.endswith("_neutral") else "disability")
OPTIONAL_ARMS = A7_ARMS + A8_ARMS + A9_ARMS

# A9-A1 item 3: the frozen-set arms the third judge (J3) scores, fixed before any J3 rating
J3_ARMS = ("unsteered_s42", "neutral_s42", "prompt_explicit", "prompt_explicit_cad_k40", "cad_proj_k40",
           "caa", "angular", "fairsteer", "sadi", *A7_ARMS, *A8_ARMS, *A9_ARMS)
# A9 fresh set (fresh_eval_a9.json): arm -> file under base; items are the fresh set's, not the frozen set's
FRESH_ARMS = ("fresh_unsteered", "fresh_fc_remove_a1", "fresh_fc_prompt_remove_a1", "fresh_prompt_explicit")
FRESH_FILE = "fresh_eval_a9.json"


def load_fresh(base: Path, required: bool = True) -> tuple[list[dict], dict[str, list[str]]]:
    """(fresh items, {arm: texts}) for the A9 fresh arms under base/a9."""
    items = json.loads(Path(FRESH_FILE).read_text())
    out = {}
    for arm in FRESH_ARMS:
        f = Path(base) / "a9" / f"{arm}_responses.json"
        if not f.exists():
            if required:
                raise FileNotFoundError(f"fresh arm {arm}: {f} is missing")
            continue
        texts = _texts(json.loads(f.read_text())["responses"])
        if len(texts) != len(items):
            raise ValueError(f"fresh arm {arm}: {len(texts)} responses, expected {len(items)}")
        out[arm] = texts
    return items, out


def _texts(raw: list) -> list[str]:
    return [r["response"] if isinstance(r, dict) else r for r in raw]


def load(base: Path, required: bool = True) -> dict[str, dict]:
    """Every M1 arm under `base`. With required=False, missing files are skipped (dry runs)."""
    out = {}
    for arm, (rel, keys, side) in ARM_FILES.items():
        f = Path(base) / rel
        if not f.exists():
            if required:
                raise FileNotFoundError(f"M1 arm {arm}: {f} is missing")
            continue
        obj = json.loads(f.read_text())
        for k in keys:
            obj = obj[k]
        texts = _texts(obj)
        if len(texts) != 250:
            raise ValueError(f"M1 arm {arm}: {len(texts)} responses, expected 250")
        out[arm] = {"side": side, "texts": texts}
    return out
