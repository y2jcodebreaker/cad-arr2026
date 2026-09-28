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
}


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
