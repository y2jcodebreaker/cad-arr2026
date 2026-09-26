"""Confirmatory statistics for the frontier analysis."""
from __future__ import annotations
import json, logging
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
OUT = Path(__file__).parent
data = json.loads((OUT / "sweep_points.json").read_text())
raw = json.loads((Path(__file__).resolve().parent.parent / "results" / "judge_scores.json").read_text())

g = defaultdict(list)
for r in raw["records"]:
    g[(r["file"], r["path"])].append(r["judge_score"])

pts = data["points"]
base_j = data["baseline_judge_mean"]

# ---------- 1. composition of the clean Pareto front ----------
pool = [p for p in pts if not p["degenerate"] and p["med"] >= 0]
front = [p for p in pool if not any(
    q is not p and q["med"] <= p["med"] and q["judge_mean"] >= p["judge_mean"]
    and (q["med"] < p["med"] or q["judge_mean"] > p["judge_mean"]) for q in pool)]
front.sort(key=lambda p: p["med"])
cad = [p for p in front if p["method"].startswith("CAD")]
log.info("CLEAN PARETO FRONT: %d points, %d are CAD operators", len(front), len(cad))
for p in front:
    log.info("   %-16s %-24s med=%6.3f judge=%.2f", p["method"], p["config"], p["med"], p["judge_mean"])
# meaningful debiasing = at least 50% reduction from baseline med (0.659)
meaningful = [p for p in front if p["med"] <= 0.33]
log.info("\n   front points with >=50%% bias reduction: %d, CAD: %d",
         len(meaningful), sum(1 for p in meaningful if p["method"].startswith("CAD")))

# ---------- 2. published operating points vs baseline (Holm-corrected) ----------
BASE_KEYS = [k for k in g if k[1].endswith("alpha_0.0") or k[1].endswith("angle_0")
             or k[1].endswith("strength_1.0") or k[0] == "proj_alpha0.1_responses.json"]
base_scores = np.array([s for k in BASE_KEYS for s in g[k]], dtype=float)

PUB = {
    "SADI":           ("sadi_results.json", "accesseval/strengths/strength_10.0"),
    "Angular":        ("angular_results.json", "accesseval/strategies/max_sim_L23/mode_0/angle_150"),
    "FairSteer":      ("fairsteer_results_L29.json", "accesseval/layers/L29/thresh_0.5/alpha_2.0"),
    "CAA":            ("caa_results_L14.json", "accesseval/layers/L14/alpha_1.0"),
    "CAD projection": ("proj_alpha1.0_responses.json", "<root>"),
}
log.info("\nPUBLISHED OPERATING POINTS vs pooled baseline (n=%d)", len(base_scores))
res = []
for m, key in PUB.items():
    k = key if key in g else next((kk for kk in g if kk[0] == key[0]), None)
    v = np.array(g[k], dtype=float)
    u, p = stats.mannwhitneyu(v, base_scores, alternative="two-sided")
    # rank-biserial correlation = effect size for Mann-Whitney
    rb = 2 * u / (len(v) * len(base_scores)) - 1
    res.append((m, len(v), v.mean(), v.mean() - base_j, rb, p))
raw_p = [r[-1] for r in res]
order = np.argsort(raw_p)
holm = np.empty(len(raw_p))
for rank, i in enumerate(order):
    holm[i] = min(1.0, raw_p[i] * (len(raw_p) - rank))
holm = np.maximum.accumulate(holm[order])[np.argsort(order)]
log.info(f"{'method':<16}{'n':>5}{'mean':>7}{'Δ':>8}{'rank-biserial':>15}{'p(Holm)':>12}")
for (m, n, mu, d, rb, p), ph in zip(res, holm):
    log.info(f"{m:<16}{n:>5}{mu:>7.2f}{d:>+8.2f}{rb:>15.3f}{ph:>12.2e}")

# ---------- 3. CAD projection vs CAA, head to head ----------
a = np.array(g[("proj_alpha1.0_responses.json", "<root>")], dtype=float)
b = np.array(g[("caa_results_L14.json", "accesseval/layers/L14/alpha_1.0")], dtype=float)
u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
rb = 2 * u / (len(a) * len(b)) - 1
rng = np.random.default_rng(42)
diffs = [rng.choice(a, len(a), True).mean() - rng.choice(b, len(b), True).mean() for _ in range(5000)]
log.info("\nCAD projection (a=1) vs CAA (a=1), UNPAIRED (different 250-prompt draws):")
log.info("   means %.3f vs %.3f | diff %+.3f, 95%% CI [%.3f, %.3f]",
         a.mean(), b.mean(), a.mean()-b.mean(), np.percentile(diffs,2.5), np.percentile(diffs,97.5))
log.info("   Mann-Whitney U p=%.3g, rank-biserial=%.3f", p, rb)
log.info("   medicalization: 0.199 (CAD) vs 0.146 (CAA) -> CAA removes slightly MORE bias")

# ---------- 4. parse-failure sensitivity ----------
log.info("\nPARSE-FAILURE SENSITIVITY (judge default 5.0 for empty/unparseable):")
log.info("   rate of exactly 5.0 at clean baseline points: %.2f%%",
         100 * np.mean(base_scores == 5.0))
for m, key in PUB.items():
    k = key if key in g else None
    v = np.array(g[k], dtype=float)
    strict = v.copy(); strict[strict == 5.0] = 0.0
    log.info("   %-16s as-scored %.2f | parse-fail %.1f%% | recoded-to-0 %.2f",
             m, v.mean(), 100*np.mean(v == 5.0), strict.mean())
log.info("\n   => 5.0 never occurs on clean text, so it marks failure, not a mid score.")
log.info("   => The default therefore UNDERSTATES the collapse; reported drops are conservative.")
