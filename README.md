# cad-arr2026

Code, evidence and paper for *When Debiasing Scores Reward Broken Text* (ARR, October 2026).

**Private repository. Do not link it from the submission** — ARR is double-blind, and the
commit history carries the author's name. If code must be linked, publish an anonymised
mirror instead.

## Layout

| path | what | tracked |
|---|---|---|
| `evaluate_*.py`, `exp3_leace_rank_k.py` | generation runners (GPU). Flat at the root so their relative paths work unchanged | yes |
| `judge_sweep_outputs.py`, `paired_quality_test.py` | LLM-judge scoring (GPU) | yes |
| `day1_extract_activations.py`, `day3_subsample_spectrum.py` | contrast-set controls (GPU, then CPU) | yes |
| `frontier_analysis.py`, `frontier_figures.py`, `plot_controls_figure.py`, `make_teaser*.py` | analysis and figures (CPU) | yes |
| `claims/` | executable claim graph and pre-registrations | yes |
| `results/` | **primary evidence**: sampled responses and judge scores the paper cites | yes |
| `LEDGER.jsonl` | append-only run record, one row per run, sha256 of every output | yes |
| `data/explicit.jsonl` | DiscrimEval, 9,450 rows | yes |
| `*/accesseval_loudness.npz` | the norms that define the 2,083-pair AccessEval pool | yes (3 identical copies) |
| `paper_arr2026/` | the LaTeX paper; upload to Overleaf | yes |
| `rtsd_fullres_checkpoints/`, `baseline_results/`, `e1_outputs/`, … | pod scratch while a run is in progress | **no** |

## Which way things flow

- **Code**: edited on the laptop → committed → pushed → pulled on the pod.
- **Results**: produced on the pod → downloaded → committed **from the laptop**.

The pod never commits. Its output directories are gitignored, so a pull can never overwrite
a run in progress. Everything the paper cites is copied into `results/` and committed from
the laptop, which is the durable machine. (The CAD per-response texts from the first frontier
sweep were lost exactly because they lived only on a pod.)

## On a fresh pod

```bash
cd /workspace
git clone https://github.com/<you>/cad-arr2026.git
cd cad-arr2026
pip install -r requirements.txt        # torch comes from the pod image; see the file
export HF_TOKEN=...                    # never commit it
python -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])"
```

## Every time after that

```bash
bash pod_pull.sh
```

Use this instead of `git pull`. It refuses while a run is in progress, stashes (never
discards) any tracked file you edited on the pod, and fast-forwards only.

**Do not `sed`-edit tracked code on the pod.** A run whose code differs from its commit
cannot be attributed. If a runner needs a different setting, change it here and push.

## Reproducing the paper's analysis (CPU, no model)

```bash
python frontier_analysis.py            # results/ -> analysis-output/sweep_points.{csv,json}
python frontier_figures.py             # -> analysis-output/figures/
python analysis-output/confirmatory_stats.py
python plot_controls_figure.py         # -> paper_arr2026/figures/figure-03-contrast-set-controls.pdf
python claims/claim_graph.py           # the audit, with warnings
```

`frontier_analysis.py` must report 87 sweep points and a 12-configuration clean front, 10 of
them CAD. If it does not, the inputs are not the ones the paper was written from.
