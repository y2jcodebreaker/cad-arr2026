"""Question-level (146-cluster) bootstrap at L14 and L0, to replace the 292-ID numbers
the appendix quotes for those layers. Same procedure, seed and n_boot as the L21 rerun."""
import sys, json, io, contextlib
import numpy as np
sys.path.insert(0, ".")
import day3_subsample_spectrum as d3
d = np.load(sys.argv[1]); keep = d["keep_mask"]
q = d["base_query_id"][keep] % 234
out = {}
for L in sys.argv[3:]:
    c = d[f"clean_{L}"][keep].astype(np.float64); x = d[f"corrupted_{L}"][keep].astype(np.float64)
    with contextlib.redirect_stdout(io.StringIO()):
        r = d3.cluster_bootstrap(c, x, q, 150, np.random.default_rng(42))
    out[L] = r
    print(f"{L} questions_146  k={r['k_mean']:.1f} [{r['k_ci'][0]:.0f}, {r['k_ci'][1]:.0f}]  "
          f"EVR1={r['evr1_mean']*100:.1f}% [{r['evr1_ci'][0]*100:.1f}, {r['evr1_ci'][1]*100:.1f}]", flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)
