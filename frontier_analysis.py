"""Frontier analysis: judged quality vs debiasing across five steering methods.

Reads the completed judge run and the CAD sweep summary, derives per-sweep-point
statistics, flags degeneration from TEXT properties only (so the judge score stays
an independent outcome), computes the Pareto front, and writes figures + tables.

Unit of analysis: one sweep point (a method at one configuration), n=250 responses.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS = Path(__file__).parent / "results"
JUDGE_PATH = RESULTS / "judge_scores.json"
CAD_PATH = RESULTS / "rtsd_fullres_results.json"
OUT_DIR = Path(__file__).parent / "analysis-output"

PARSE_FAILURE_SCORE = 5.0   # judge default for empty/unparseable; NOT a real score
N_BOOT = 2000
RNG_SEED = 42

# Degeneracy thresholds, defined on text/metric properties (judge-independent)
UNIQUE_RATIO_MIN = 0.50     # below this = repetition collapse
PARSE_FAIL_MAX = 0.10       # above this = empty/unparseable output
OVERSHOOT_MAX = -0.05       # med below this = pushed past neutral


@dataclass(frozen=True)
class SweepPoint:
    method: str
    config: str
    n: int
    judge_mean: float
    judge_lo: float
    judge_hi: float
    med: float
    unique_ratio: float
    parse_fail_rate: float
    judge_mean_strict: float    # parse failures recoded to 0.0
    is_baseline: bool

    @property
    def degenerate_flags(self) -> List[str]:
        flags: List[str] = []
        if self.unique_ratio < UNIQUE_RATIO_MIN:
            flags.append("repetition")
        if self.parse_fail_rate > PARSE_FAIL_MAX:
            flags.append("unparseable")
        if self.med < OVERSHOOT_MAX:
            flags.append("overshoot")
        return flags

    @property
    def degenerate(self) -> bool:
        return bool(self.degenerate_flags)


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean, resampling responses."""
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def method_of(fname: str) -> str:
    if "caa" in fname:
        return "CAA"
    if "fairsteer" in fname:
        return "FairSteer"
    if "sadi" in fname:
        return "SADI"
    if "angular" in fname:
        return "Angular"
    if fname.startswith("proj_"):
        return "CAD projection"
    if fname.startswith("probe_"):
        return "CAD probe"
    if fname.startswith("meandiff_"):
        return "CAD mean-diff"
    raise ValueError(f"unrecognised result file: {fname}")


def tidy_config(path: str, fname: str) -> str:
    if fname.endswith("_responses.json"):
        return fname.replace("_responses.json", "").split("_", 1)[1]
    return (path.replace("accesseval/strategies/", "")
                .replace("accesseval/layers/", "")
                .replace("accesseval/strengths/", "")
                .replace("accesseval/", ""))


def is_baseline_config(path: str, fname: str) -> bool:
    """A no-steering reference point."""
    return (path.endswith("alpha_0.0") or path.endswith("angle_0")
            or path.endswith("strength_1.0") or fname == "proj_alpha0.1_responses.json")


def load_cad_med() -> Dict[str, float]:
    """CAD saves responses as plain strings, so its medicalization scores live only
    in the sweep summary. Map file stem -> steered mean."""
    cad = json.loads(CAD_PATH.read_text())
    prefix = {"projection": "proj", "probe": "probe", "meandiff": "meandiff"}
    out: Dict[str, float] = {}
    for op, pre in prefix.items():
        for alpha, row in cad[op]["sweep"].items():
            out[f"{pre}_alpha{alpha}_responses.json"] = float(row["steered_mean"])
    return out


def build_points() -> List[SweepPoint]:
    judge = json.loads(JUDGE_PATH.read_text())
    cad_med = load_cad_med()

    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for rec in judge["records"]:
        grouped[(rec["file"], rec["path"])].append(rec)

    points: List[SweepPoint] = []
    for (fname, path), recs in grouped.items():
        scores = np.array([r["judge_score"] for r in recs], dtype=float)
        hashes = [r["hash"] for r in recs]

        meds = [r["medicalization_score"] for r in recs
                if r.get("medicalization_score") is not None]
        if meds:
            med = float(np.mean(meds))
        elif fname in cad_med:
            med = cad_med[fname]
        else:
            logger.warning("no medicalization score for %s / %s - skipped", fname, path)
            continue

        lo, hi = bootstrap_ci(scores)
        fail_mask = scores == PARSE_FAILURE_SCORE
        strict = scores.copy()
        strict[fail_mask] = 0.0

        points.append(SweepPoint(
            method=method_of(fname),
            config=tidy_config(path, fname),
            n=len(scores),
            judge_mean=float(scores.mean()),
            judge_lo=lo,
            judge_hi=hi,
            med=med,
            unique_ratio=len(set(hashes)) / len(hashes),
            parse_fail_rate=float(fail_mask.mean()),
            judge_mean_strict=float(strict.mean()),
            is_baseline=is_baseline_config(path, fname),
        ))
    return points


def pareto_front(points: List[SweepPoint]) -> List[SweepPoint]:
    """Non-dominated set: want LOW medicalization and HIGH judge quality."""
    front = []
    for p in points:
        dominated = any(
            q is not p and q.med <= p.med and q.judge_mean >= p.judge_mean
            and (q.med < p.med or q.judge_mean > p.judge_mean)
            for q in points
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p.med)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    points = build_points()
    logger.info("sweep points: %d  (responses: %d)", len(points), sum(p.n for p in points))

    baselines = [p for p in points if p.is_baseline]
    base_judge = float(np.mean([p.judge_mean for p in baselines]))
    base_sd = float(np.std([p.judge_mean for p in baselines], ddof=1))
    base_med = float(np.mean([p.med for p in baselines]))
    base_med_sd = float(np.std([p.med for p in baselines], ddof=1))
    logger.info("baseline reference (n=%d points): judge %.3f +- %.3f | med %.3f +- %.3f",
                len(baselines), base_judge, base_sd, base_med, base_med_sd)

    rows = [{
        "method": p.method, "config": p.config, "n": p.n,
        "med": round(p.med, 4),
        "judge_mean": round(p.judge_mean, 3),
        "judge_ci_lo": round(p.judge_lo, 3), "judge_ci_hi": round(p.judge_hi, 3),
        "judge_delta": round(p.judge_mean - base_judge, 3),
        "judge_strict": round(p.judge_mean_strict, 3),
        "unique_ratio": round(p.unique_ratio, 3),
        "parse_fail_rate": round(p.parse_fail_rate, 3),
        "degenerate": p.degenerate,
        "flags": ";".join(p.degenerate_flags),
        "is_baseline": p.is_baseline,
    } for p in sorted(points, key=lambda x: (x.method, x.med))]

    (OUT_DIR / "sweep_points.json").write_text(json.dumps({
        "baseline_judge_mean": base_judge, "baseline_judge_sd": base_sd,
        "baseline_med_mean": base_med, "baseline_med_sd": base_med_sd,
        "n_baseline_points": len(baselines),
        "parse_failure_score": PARSE_FAILURE_SCORE,
        "thresholds": {"unique_ratio_min": UNIQUE_RATIO_MIN,
                       "parse_fail_max": PARSE_FAIL_MAX,
                       "overshoot_max": OVERSHOOT_MAX},
        "points": rows,
    }, indent=1))

    import csv
    with (OUT_DIR / "sweep_points.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    front = pareto_front(points)
    logger.info("\nPARETO FRONT (%d points):", len(front))
    for p in front:
        tag = " [" + ",".join(p.degenerate_flags) + "]" if p.degenerate else ""
        logger.info("  %-16s %-26s med=%7.3f judge=%.2f [%.2f,%.2f]%s",
                    p.method, p.config, p.med, p.judge_mean, p.judge_lo, p.judge_hi, tag)

    clean = [p for p in front if not p.degenerate]
    logger.info("\nCLEAN front (%d of %d): %s", len(clean), len(front),
                ", ".join(f"{p.method}" for p in clean))
    logger.info("degenerate share of all points: %d/%d",
                sum(1 for p in points if p.degenerate), len(points))
    logger.info("\nwrote %s", OUT_DIR / "sweep_points.csv")


if __name__ == "__main__":
    main()
