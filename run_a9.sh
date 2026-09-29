#!/usr/bin/env bash
# run_a9.sh — the whole A9 pod run (claims/PREREG_A9_robustness.md), in pre-registered order.
#
#   export HF_TOKEN=...                       # Llama is gated; never write the token to a file
#   nohup bash run_a9.sh > a9.log 2>&1 &
#   tail -f a9.log                            # progress; ends with "A9 DONE" and the tarball
#
# Stops at the first failure (a failed gate exits non-zero). Re-running resumes: generation and
# every judge checkpoint, so nothing finished is redone. One GPU consumer at a time, except the two
# 8B M1 judges, which share the card as before.
set -euo pipefail
cd "$(dirname "$0")"
MISTRAL=mistralai/Mistral-Small-24B-Instruct-2501
MISTRAL_REV=9527884be6e5616bdd54de542f9ae13384489724

echo "== A9 start $(date -u +%FT%TZ)  commit $(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "tracked files are modified on the pod; refusing (the ledger must point at committed code)"; exit 1
fi

# whatever happens (a failed gate included), package what exists so the evidence reaches the laptop
package() {
  tar czf a9.tgz --exclude='*.npz' e1_outputs/a9 e1_outputs/m1_judge_*.json \
    e1_outputs/judge_llama.json e1_outputs/judge_second.json a9*.log 2>/dev/null || true
  ls -lh a9.tgz 2>/dev/null || true
}
trap package EXIT

# 0. seed e1_outputs from the committed results (fit, judges, every earlier arm); never overwrite
mkdir -p e1_outputs && cp -rn results/e1/. e1_outputs/
# J3 weights download in the background while Llama generates (~48 GB)
huggingface-cli download "$MISTRAL" --revision "$MISTRAL_REV" > a9_mistral_download.log 2>&1 &
DL=$!

# 1. neutral fit answers, directions (GA9-0), arms, perplexity
python a9_robustness.py fit_neutral_generate
python a9_robustness.py fit
python a9_robustness.py generate
python a9_robustness.py ppl

# 2. third judge J3 (55 GB, alone on the card). Its tokenizer file needs tokenizers >= 0.20, which the
#    pinned transformers 4.44.2 refuses, so J3 runs in its own venv on the image's torch (amendment A9-A4)
if [ ! -x .venv-j3/bin/python ]; then
  python -m venv --system-site-packages .venv-j3
  .venv-j3/bin/pip install -q "transformers==4.46.3" "tokenizers==0.20.3"
fi
.venv-j3/bin/python -c "import transformers, tokenizers; print('J3 env', transformers.__version__, tokenizers.__version__)"
wait "$DL" || echo "background download failed; from_pretrained will download instead"
.venv-j3/bin/python judge_medicalization.py --judge mistral --base e1_outputs --j3_arms --out e1_outputs/m1_judge_mistral.json
.venv-j3/bin/python judge_medicalization.py --judge mistral --base e1_outputs --fresh

# 3. the two M1 judges on the new texts (resume: earlier scores are kept)
python judge_medicalization.py --judge llama --base e1_outputs > a9_m1l.log 2>&1 & P1=$!
python judge_medicalization.py --judge qwen --base e1_outputs > a9_m1q.log 2>&1 & P2=$!
wait "$P1"; wait "$P2"
python judge_medicalization.py --judge llama --base e1_outputs --fresh > a9_m1lf.log 2>&1 & P1=$!
python judge_medicalization.py --judge qwen --base e1_outputs --fresh > a9_m1qf.log 2>&1 & P2=$!
wait "$P1"; wait "$P2"
tail -n 1 a9_m1l.log a9_m1q.log a9_m1lf.log a9_m1qf.log

# 4. accessibility judges (resume over every *_responses.json, now including e1_outputs/a9)
python run_e1.py judge

echo "== A9 DONE $(date -u +%FT%TZ)   (a9.tgz is written on exit)"
