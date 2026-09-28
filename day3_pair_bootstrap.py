"""Pair-level bootstrap of k@80% and EVR1 on the loudness-filtered AccessEval pool (claim audit C).

Resamples the 2,083 pool rows as if independent, which is the procedure section 6 criticises;
kept so the quoted pair-level interval has a committed source. Same method and seed as
day3_question_bootstrap.py (150 resamples, numpy default_rng(42)).

    python day3_pair_bootstrap.py <activations npz> controls_results/pair_bootstrap.json L21 L14 L0
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
import day3_subsample_spectrum as d3  # noqa: E402

npz, out = sys.argv[1], sys.argv[2]
d = np.load(npz)
keep = d["keep_mask"]
res = {}
for L in sys.argv[3:]:
    c = d[f"clean_{L}"][keep].astype(np.float64)
    x = d[f"corrupted_{L}"][keep].astype(np.float64)
    rng = np.random.default_rng(42)
    n = len(c)
    ks, evs = [], []
    for _ in range(150):
        i = rng.integers(0, n, n)
        evr1, k = d3.spectrum(x[i] - c[i])[:2]
        evs.append(evr1)
        ks.append(k)
    res[L] = {"n_rows": int(n), "n_boot": 150, "k_mean": float(np.mean(ks)),
              "k_ci": [float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))],
              "evr1_mean": float(np.mean(evs)),
              "evr1_ci": [float(np.percentile(evs, 2.5)), float(np.percentile(evs, 97.5))]}
    print(f"{L}: k {res[L]['k_mean']:.1f} {res[L]['k_ci']}  EVR1 {100 * res[L]['evr1_mean']:.1f}%", flush=True)
    json.dump(res, open(out, "w"), indent=1)
