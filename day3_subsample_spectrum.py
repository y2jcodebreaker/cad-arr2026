#!/usr/bin/env python3
"""
day3_subsample_spectrum.py

Decides Branch A vs Branch B of the ARR revision plan.

THE QUESTION
------------
The paper's organising claim is that bias geometry varies by bias type:
housing discrimination is near-rank-1 (EVR_1 = 67.9%, n = 13 pairs, L4/L15)
while medicalization is diffuse (EVR_1 = 10.2%, n = 2083 pairs, L21).

But EVR_1 is perfectly rank-correlated with the number of SVD pairs, AND with
layer depth. Day 1 already showed layer alone moves EVR_1 3x on AccessEval
(L0 = 31.3% vs L21 = 10.3%, same pairs). This script tests the sample-size half:
if AccessEval at n = 13 reports EVR_1 near 68%, the dichotomy is an artifact.

WHAT IT RUNS (all CPU, reads the Day 1 .npz)
  A. Subsample curve: EVR_1 and k@80% vs n, at L0 and L21, under two sampling
     schemes -- pair-level (what the paper implicitly did) and cluster-level
     (whole base queries, the honest version given 7.1 variants per query).
  B. Permutation null: mismatched pairing (corrupted[perm] - clean) destroys the
     contrastive structure while preserving marginals, giving a chance-level
     reference at each n. NOTE: sign-flipping delta would be an INVALID null,
     since (-d)(-d)^T = dd^T leaves the second moment unchanged.
  C. Cluster bootstrap on the full filtered pool -- the honest CI on k@80%,
     replacing the published pair-level [40, 41].
  D. Homogeneity: spectrum within single disability categories and domains,
     testing whether k = 40 reflects prompt-pool heterogeneity rather than
     bias dimensionality.

Usage:
    python day3_subsample_spectrum.py --npz controls_checkpoints/accesseval_activations_multilayer.npz
    python day3_subsample_spectrum.py --repeats 20 --bootstrap 100   # faster
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

VAR_THRESHOLD = 0.80
DEFAULT_NS = [13, 30, 50, 100, 250, 500, 1000]
PAPER_REFERENCE = {
    "AccessEval (L21, n=2083)": {"evr1": 0.102, "k": 40},
    "Heritage (L14, n=30)": {"evr1": 0.216, "k": 15},
    "Target A (L4+L15, n=13)": {"evr1": 0.679, "k": 2},
}


# ---------------------------------------------------------------------------
# Core spectral quantity
# ---------------------------------------------------------------------------

def spectrum(delta: np.ndarray, var_threshold: float = VAR_THRESHOLD) -> Tuple[float, int]:
    """EVR_1 and k@threshold of a mean-centred difference matrix.

    Matches compute_svd_and_probe() in the pipeline: mean-centre, SVD, cumulative
    variance, k = first index reaching the threshold (1-indexed).
    """
    d = delta.astype(np.float64, copy=True)
    d -= d.mean(axis=0)
    s = np.linalg.svd(d, full_matrices=False, compute_uv=False)
    total = float(np.sum(s ** 2))
    if total <= 0 or not np.isfinite(total):
        return float("nan"), -1
    evr = (s ** 2) / total
    k = int(np.searchsorted(np.cumsum(evr), var_threshold) + 1)
    return float(evr[0]), k


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_pairs(n_target: int, n_avail: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform sample of n_target row indices, without replacement."""
    return rng.choice(n_avail, size=min(n_target, n_avail), replace=False)


def sample_clusters(n_target: int, clusters: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    """Draw whole base queries until at least n_target pairs are accumulated.

    Returns row indices. The realised size can exceed n_target (a whole final
    cluster is kept rather than truncated, which would break cluster integrity),
    so callers should record the actual n.
    """
    uniq = rng.permutation(np.unique(clusters))
    idx: List[np.ndarray] = []
    total = 0
    for c in uniq:
        rows = np.flatnonzero(clusters == c)
        idx.append(rows)
        total += len(rows)
        if total >= n_target:
            break
    return np.concatenate(idx)


# ---------------------------------------------------------------------------
# A + B: subsample curve and permutation null
# ---------------------------------------------------------------------------

def subsample_curve(clean: np.ndarray, corrupted: np.ndarray, clusters: np.ndarray,
                    ns: List[int], repeats: int, rng: np.random.Generator,
                    mode: str, permute: bool = False) -> List[Dict]:
    out = []
    n_avail = len(clean)
    for n in ns:
        if n > n_avail:
            continue
        evrs, ks, actual_ns = [], [], []
        for _ in range(repeats):
            if mode == "pair":
                idx = sample_pairs(n, n_avail, rng)
            else:
                idx = sample_clusters(n, clusters, rng)
            c, x = clean[idx], corrupted[idx]
            if permute:
                # Break the pairing: each clean is matched to a different
                # corrupted. Destroys contrastive structure, keeps marginals.
                if len(idx) < 2:
                    continue
                perm = rng.permutation(len(idx))
                # guarantee a derangement-ish shuffle (avoid identity)
                if np.all(perm == np.arange(len(idx))):
                    perm = np.roll(perm, 1)
                x = x[perm]
            evr, k = spectrum(x - c)
            if np.isfinite(evr):
                evrs.append(evr); ks.append(k); actual_ns.append(len(idx))
        if not evrs:
            continue
        out.append({
            "n_target": n,
            "n_actual_mean": float(np.mean(actual_ns)),
            "evr1_mean": float(np.mean(evrs)),
            "evr1_lo": float(np.percentile(evrs, 2.5)),
            "evr1_hi": float(np.percentile(evrs, 97.5)),
            "k_mean": float(np.mean(ks)),
            "k_lo": float(np.percentile(ks, 2.5)),
            "k_hi": float(np.percentile(ks, 97.5)),
            "repeats": len(evrs),
        })
    return out


# ---------------------------------------------------------------------------
# C: cluster bootstrap on the full pool
# ---------------------------------------------------------------------------

def cluster_bootstrap(clean: np.ndarray, corrupted: np.ndarray, clusters: np.ndarray,
                      n_boot: int, rng: np.random.Generator) -> Dict:
    uniq = np.unique(clusters)
    rows_by_cluster = {c: np.flatnonzero(clusters == c) for c in uniq}
    evrs, ks, sizes = [], [], []
    for b in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by_cluster[c] for c in drawn])
        evr, k = spectrum(corrupted[idx] - clean[idx])
        if np.isfinite(evr):
            evrs.append(evr); ks.append(k); sizes.append(len(idx))
        if (b + 1) % max(1, n_boot // 10) == 0:
            print(f"      bootstrap {b + 1}/{n_boot}", flush=True)
    return {
        "n_boot": len(evrs),
        "mean_rows_per_resample": float(np.mean(sizes)) if sizes else float("nan"),
        "evr1_mean": float(np.mean(evrs)),
        "evr1_ci": [float(np.percentile(evrs, 2.5)), float(np.percentile(evrs, 97.5))],
        "k_mean": float(np.mean(ks)),
        "k_ci": [float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))],
    }


# ---------------------------------------------------------------------------
# D: homogeneity
# ---------------------------------------------------------------------------

def homogeneity_ladder(clean: np.ndarray, corrupted: np.ndarray, clusters: np.ndarray,
                       category: np.ndarray, domain: np.ndarray, n: int,
                       repeats: int, rng: np.random.Generator) -> List[Dict]:
    """Hold n fixed; vary only how homogeneous the contrast set is.

    This is the decisive control. Target A is a single question template with one
    attribute varied, i.e. maximally homogeneous. If restricting AccessEval to a
    comparably homogeneous pool reproduces Target A's EVR_1, then the reported
    "concentrated vs diffuse" difference is a property of contrast-set
    construction rather than of the bias.
    """
    uniq_cat = [c for c in sorted(set(category.tolist())) if c]
    uniq_dom = [d for d in sorted(set(domain.tolist())) if d]
    uniq_cl = np.unique(clusters)

    levels: List[Tuple[str, callable]] = [
        ("heterogeneous (any query, any category)",
         lambda r: np.arange(len(clean))),
        ("single category",
         lambda r: np.flatnonzero(category == r.choice(uniq_cat)) if uniq_cat else np.arange(len(clean))),
        ("single domain",
         lambda r: np.flatnonzero(domain == r.choice(uniq_dom)) if uniq_dom else np.arange(len(clean))),
        ("5 base queries only",
         lambda r: np.flatnonzero(np.isin(clusters, r.choice(uniq_cl, size=min(5, len(uniq_cl)), replace=False)))),
        ("2 base queries only (max homogeneity)",
         lambda r: np.flatnonzero(np.isin(clusters, r.choice(uniq_cl, size=min(2, len(uniq_cl)), replace=False)))),
    ]

    out = []
    for label, selector in levels:
        evrs, ks, attempts = [], [], 0
        while len(evrs) < repeats and attempts < repeats * 6:
            attempts += 1
            pool = selector(rng)
            if len(pool) < n:
                continue
            idx = rng.choice(pool, size=n, replace=False)
            evr, k = spectrum(corrupted[idx] - clean[idx])
            if np.isfinite(evr):
                evrs.append(evr); ks.append(k)
        if not evrs:
            out.append({"level": label, "n": n, "draws": 0, "note": "pool too small"})
            continue
        out.append({
            "level": label, "n": n, "draws": len(evrs),
            "evr1_mean": float(np.mean(evrs)),
            "evr1_lo": float(np.percentile(evrs, 2.5)),
            "evr1_hi": float(np.percentile(evrs, 97.5)),
            "k_mean": float(np.mean(ks)),
        })
    return out


def homogeneity(clean: np.ndarray, corrupted: np.ndarray,
                labels: np.ndarray, label_name: str, min_n: int = 30) -> List[Dict]:
    out = []
    for lab in sorted(set(labels.tolist())):
        if lab == "":
            continue
        m = labels == lab
        if int(m.sum()) < min_n:
            continue
        evr, k = spectrum(corrupted[m] - clean[m])
        out.append({label_name: lab, "n": int(m.sum()), "evr1": evr, "k": k})
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(res: Dict, out_path: Path) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"  [skip figure] matplotlib unavailable: {e}")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    series = [
        ("L21_pair", "L21, pair-level", "#1f77b4", "-", "o"),
        ("L21_cluster", "L21, cluster-level", "#1f77b4", "--", "s"),
        ("L0_pair", "L0, pair-level", "#d62728", "-", "o"),
        ("L0_cluster", "L0, cluster-level", "#d62728", "--", "s"),
        ("L21_null", "L21, permuted null", "#7f7f7f", ":", "^"),
    ]

    for key, label, color, ls, marker in series:
        rows = res["curves"].get(key, [])
        if not rows:
            continue
        x = [r["n_actual_mean"] for r in rows]
        for ax, ykey, lokey, hikey in (
            (axes[0], "evr1_mean", "evr1_lo", "evr1_hi"),
            (axes[1], "k_mean", "k_lo", "k_hi"),
        ):
            y = [r[ykey] * (100 if ykey.startswith("evr") else 1) for r in rows]
            lo = [r[lokey] * (100 if ykey.startswith("evr") else 1) for r in rows]
            hi = [r[hikey] * (100 if ykey.startswith("evr") else 1) for r in rows]
            ax.plot(x, y, marker=marker, color=color, ls=ls, lw=1.8, ms=4, label=label)
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)

    for name, ref in PAPER_REFERENCE.items():
        if "Target A" in name:
            axes[0].axhline(ref["evr1"] * 100, color="black", ls="-.", lw=1.2)
            axes[0].text(14, ref["evr1"] * 100 + 1.5, f"paper: {name}", fontsize=7.5)
            axes[1].axhline(ref["k"], color="black", ls="-.", lw=1.2)
            axes[1].text(14, ref["k"] + 1.5, f"paper: {name}", fontsize=7.5)

    axes[0].set_ylabel("EVR$_1$ (%)")
    axes[1].set_ylabel("k@80%")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("number of contrastive pairs (n)")
        ax.grid(alpha=0.3, ls=":")
        ax.legend(fontsize=7.5)
    axes[0].set_title("Spectral concentration vs sample size", fontsize=10)
    axes[1].set_title("Prescribed rank vs sample size", fontsize=10)
    fig.suptitle("Is the concentrated/diffuse dichotomy an artifact of n and layer?",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=str,
                    default=str(Path("controls_checkpoints") /
                               "accesseval_activations_multilayer.npz"))
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 21])
    ap.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS)
    ap.add_argument("--repeats", type=int, default=40)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="controls_results")
    ap.add_argument("--unfiltered", action="store_true",
                    help="use all 4164 pairs instead of the published filtered pool")
    args = ap.parse_args()

    npz_path = Path(args.npz).expanduser()
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found. Run day1_extract_activations.py first.")
        return 2

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    print("=" * 74)
    print("DAY 3: IS THE CONCENTRATED/DIFFUSE DICHOTOMY AN ARTIFACT?")
    print("=" * 74)

    d = np.load(npz_path)
    keep = np.ones(len(d["base_query_id"]), dtype=bool) if args.unfiltered else d["keep_mask"]
    clusters = d["base_query_id"][keep]
    category = d["category"][keep]
    domain = d["domain"][keep]

    print(f"\nPool: {int(keep.sum())} pairs, {len(np.unique(clusters))} clusters "
          f"({'unfiltered' if args.unfiltered else 'published filter'})")

    acts = {}
    for l in args.layers:
        ck, xk = f"clean_L{l}", f"corrupted_L{l}"
        if ck not in d.files:
            print(f"  WARNING: layer {l} not in npz, skipping")
            continue
        acts[l] = (d[ck][keep], d[xk][keep])

    res: Dict = {
        "config": {
            "npz": str(npz_path), "layers": list(acts), "ns": args.ns,
            "repeats": args.repeats, "bootstrap": args.bootstrap, "seed": args.seed,
            "filtered": not args.unfiltered,
            "n_pairs": int(keep.sum()), "n_clusters": int(len(np.unique(clusters))),
        },
        "full_pool": {}, "curves": {}, "bootstrap": {}, "homogeneity": {},
    }

    # Full-pool reference
    print("\n== Full-pool spectrum (reference) ==")
    for l, (c, x) in acts.items():
        evr, k = spectrum(x - c)
        res["full_pool"][f"L{l}"] = {"evr1": evr, "k": k, "n": int(len(c))}
        print(f"  L{l:<2}  EVR_1={evr:6.1%}  k@80%={k}")

    # A: subsample curves
    print(f"\n== A. Subsample curves ({args.repeats} repeats per point) ==")
    for l, (c, x) in acts.items():
        for mode in ("pair", "cluster"):
            key = f"L{l}_{mode}"
            print(f"  {key} ...", flush=True)
            res["curves"][key] = subsample_curve(c, x, clusters, args.ns,
                                                 args.repeats, rng, mode)

    # B: permutation null (at the deepest requested layer)
    lnull = max(acts) if acts else None
    if lnull is not None:
        print(f"  L{lnull}_null (mismatched pairing) ...", flush=True)
        c, x = acts[lnull]
        res["curves"][f"L{lnull}_null"] = subsample_curve(
            c, x, clusters, args.ns, args.repeats, rng, "pair", permute=True)

    # C: cluster bootstrap
    print(f"\n== C. Cluster bootstrap on the full pool ({args.bootstrap} resamples) ==")
    for l, (c, x) in acts.items():
        print(f"    L{l} ...", flush=True)
        res["bootstrap"][f"L{l}"] = cluster_bootstrap(c, x, clusters, args.bootstrap, rng)

    # D: homogeneity
    print("\n== D. Homogeneity (single category / single domain) ==")
    lhom = max(acts) if acts else None
    if lhom is not None:
        c, x = acts[lhom]
        res["homogeneity"][f"L{lhom}_category"] = homogeneity(c, x, category, "category")
        res["homogeneity"][f"L{lhom}_domain"] = homogeneity(c, x, domain, "domain")
        for row in res["homogeneity"][f"L{lhom}_category"]:
            print(f"    {row['category'][:34]:<34} n={row['n']:>4}  "
                  f"EVR_1={row['evr1']:5.1%}  k={row['k']}")

    # D2: the decisive control -- homogeneity ladder at fixed small n, at every layer
    print("\n== D2. Homogeneity ladder at fixed n (the decisive control) ==")
    for l, (c, x) in acts.items():
        for n_fix in (13, 30):
            key = f"L{l}_ladder_n{n_fix}"
            res["homogeneity"][key] = homogeneity_ladder(
                c, x, clusters, category, domain, n_fix, args.repeats, rng)
            print(f"  --- L{l}, n={n_fix} ---")
            for row in res["homogeneity"][key]:
                if row.get("draws", 0) == 0:
                    print(f"    {row['level']:<42} (pool too small)")
                    continue
                print(f"    {row['level']:<42} EVR_1={row['evr1_mean']:6.1%} "
                      f"[{row['evr1_lo']:.0%},{row['evr1_hi']:.0%}]  k={row['k_mean']:.1f}")

    # ---- report -----------------------------------------------------------
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)

    lref = 21 if 21 in acts else (max(acts) if acts else None)
    if lref is not None:
        pair_curve = res["curves"].get(f"L{lref}_pair", [])
        small = next((r for r in pair_curve if r["n_target"] == 13), None)
        full = res["full_pool"][f"L{lref}"]
        if small:
            print(f"\n  AccessEval @ L{lref}, n=13 (pair-level):  "
                  f"EVR_1={small['evr1_mean']:.1%} "
                  f"[{small['evr1_lo']:.1%}, {small['evr1_hi']:.1%}], "
                  f"k={small['k_mean']:.1f}")
            print(f"  AccessEval @ L{lref}, full pool:         "
                  f"EVR_1={full['evr1']:.1%}, k={full['k']}")
            print(f"  Paper's Target A (n=13, L4/L15):         EVR_1=67.9%, k=2")
            gap = 0.679 - small["evr1_mean"]
            print(f"\n  Sample-size effect alone takes EVR_1 from {full['evr1']:.1%} "
                  f"to {small['evr1_mean']:.1%} at n=13.")
            if small["evr1_lo"] <= 0.679 <= small["evr1_hi"]:
                print("  >> Target A's 67.9% falls INSIDE the n=13 subsample interval.")
                print("     The dichotomy is consistent with a pure sampling artifact -> BRANCH B.")
            elif gap < 0.15:
                print("  >> n=13 subsampling reproduces MOST of Target A's concentration.")
                print("     The geometry claim is badly weakened -> lean BRANCH B.")
            else:
                print(f"  >> A gap of {gap:.1%} remains unexplained by sample size alone.")
                print("     Some real geometric difference may survive -> BRANCH A is live,")
                print("     but the claim must be stated at matched n and matched layer.")

        # Decisive: does a homogeneous AccessEval pool reproduce Target A?
        for lcand in (14, 21, lref):
            ladder = res["homogeneity"].get(f"L{lcand}_ladder_n13")
            if not ladder:
                continue
            maxhom = next((r for r in ladder if "max homogeneity" in r["level"]
                           and r.get("draws", 0) > 0), None)
            if maxhom:
                lo, hi = maxhom["evr1_lo"], maxhom["evr1_hi"]
                print(f"\n  L{lcand}, n=13, most homogeneous pool: "
                      f"EVR_1={maxhom['evr1_mean']:.1%} [{lo:.1%}, {hi:.1%}], "
                      f"k={maxhom['k_mean']:.1f}")
                if lo <= 0.679 <= hi:
                    print("  >> Target A's 67.9% is REPRODUCED from AccessEval by restricting")
                    print("     prompt diversity alone. The concentrated/diffuse dichotomy is a")
                    print("     property of contrast-set construction, not of bias type. BRANCH B.")
                else:
                    print("  >> Target A's 67.9% is NOT reproduced by homogeneity alone here.")
            break

        bs = res["bootstrap"].get(f"L{lref}")
        if bs:
            print(f"\n  Cluster-bootstrap k@80% at L{lref}: {bs['k_mean']:.1f} "
                  f"CI [{bs['k_ci'][0]:.0f}, {bs['k_ci'][1]:.0f}]")
            print(f"  Published (pair-level) CI was [40, 41]  <- too narrow if this is wider.")

    json_path = out_dir / "day3_subsample_results.json"
    json_path.write_text(json.dumps(res, indent=2))
    print(f"\nWrote {json_path}")
    fig_path = make_figure(res, out_dir / "day3_subsample_spectrum.png")
    if fig_path:
        print(f"Wrote {fig_path}")
    print(f"Elapsed: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
