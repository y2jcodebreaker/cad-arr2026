"""Re-run the §6 cluster bootstrap with clusters = unique QUESTIONS (base_query_id % 234).
Positive control: the original 292-ID grouping must reproduce the published k 35.3 [33,37]."""
import sys, json, io, contextlib
import numpy as np
sys.path.insert(0, ".")
import day3_subsample_spectrum as d3
d = np.load(sys.argv[1])
keep = d["keep_mask"]
c, x = d["clean_L21"][keep].astype(np.float64), d["corrupted_L21"][keep].astype(np.float64)
bq = d["base_query_id"][keep]
q = bq % 234
out = {}
for name, cl in [("rows_292 (original)", bq), ("questions_146 (corrected)", q)]:
    with contextlib.redirect_stdout(io.StringIO()):
        r = d3.cluster_bootstrap(c, x, cl, 150, np.random.default_rng(42))
    out[name] = r
    print(f"{name:<28} units={len(np.unique(cl)):>3}  rows/resample={r['mean_rows_per_resample']:.0f}  "
          f"k={r['k_mean']:.1f} [{r['k_ci'][0]:.0f}, {r['k_ci'][1]:.0f}]  "
          f"EVR1={r['evr1_mean']*100:.1f}% [{r['evr1_ci'][0]*100:.1f}, {r['evr1_ci'][1]*100:.1f}]", flush=True)
# pair-level bootstrap, as published ([40,41]), and with twins collapsed to one unit
rng = np.random.default_rng(42); n = len(c)
ks = []
for _ in range(150):
    i = rng.integers(0, n, n); ks.append(d3.spectrum(x[i] - c[i])[1])
print(f"{'pairs_2083 (published)':<28} units={n:>4}  k={np.mean(ks):.1f} [{np.percentile(ks,2.5):.0f}, {np.percentile(ks,97.5):.0f}]", flush=True)
json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in out.items()}, open(sys.argv[2], "w"), indent=1)
