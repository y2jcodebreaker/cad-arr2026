#!/usr/bin/env bash
# run_a10_a11.sh — A10 (metric gaming) and A11 (Mistral-7B, gated) in pre-registered order.
#
#   export HF_TOKEN=...                                   # Llama is gated; never write the token to a file
#   setsid nohup bash run_a10_a11.sh > a1011.log 2>&1 < /dev/null &
#   tail -n 8 a1011.log                                   # check progress (not tail -f on the network volume)
#
# Re-running resumes: finished steps are skipped, judges keep their scores. If an A11 fit gate fails,
# the run continues (outcome (d)): S0 and the A10 arms are still judged. a1011.tgz is written on exit.
set -euo pipefail
cd "$(dirname "$0")"
if [ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" = "1" ] && ! python -c "import hf_transfer" 2>/dev/null; then
  echo "HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is missing: run 'pip install hf_transfer' first"; exit 1
fi
python -c "import sentencepiece" 2>/dev/null || { echo "sentencepiece missing: pip install -r requirements.txt"; exit 1; }
echo "== A10/A11 start $(date -u +%FT%TZ)  commit $(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "tracked files are modified on the pod; refusing"; exit 1
fi
package() {
  tar czf a1011.tgz --exclude='*.npz' e1_outputs/a10 e1_outputs/a11 e1_outputs/m1_judge_*.json \
    e1_outputs/judge_llama.json e1_outputs/judge_second.json a1011*.log 2>/dev/null || true
  ls -lh a1011.tgz 2>/dev/null || true
}
trap package EXIT
mkdir -p e1_outputs && cp -rn results/e1/. e1_outputs/ 2>/dev/null || true

done_ok() { python - "$@" <<'PY'
import json, sys
from pathlib import Path
ok = all(Path(f).exists() and json.loads(Path(f).read_text()).get("status") == "ok" for f in sys.argv[1:])
sys.exit(0 if ok else 1)
PY
}
gates_pass() { python - <<'PY'
import json, sys
from pathlib import Path
p = Path("e1_outputs/a11/a11_fit_report.json")
r = json.loads(p.read_text()) if p.exists() else {}
sys.exit(0 if r.get("GA11_1_pass") and r.get("GA11_2_pass") else 1)
PY
}
A10=e1_outputs/a10; A11=e1_outputs/a11

# ---- A10 (Llama)
[ -f $A10/game_append_responses.json ] && [ -f $A10/game_sub_responses.json ] && echo "skip edits (done)" \
  || python a10_metric_gaming.py edits
done_ok $A10/run_record_game_ban.json && echo "skip ban (done)" || python a10_metric_gaming.py ban

# ---- A11 (Mistral): S0, fit answers, fit judges, fit (gates), S2 arms
done_ok $A11/run_record_mistral_unsteered.json $A11/run_record_mistral_neutral.json && echo "skip s0 (done)" \
  || python a11_mistral.py s0
done_ok $A11/run_record_fit_generate.json && echo "skip fit_generate (done)" || python a11_mistral.py fit_generate
python judge_medicalization.py --judge llama --items $A11/fit_items.json > a1011_fitl.log 2>&1 & P1=$!
python judge_medicalization.py --judge qwen --items $A11/fit_items.json > a1011_fitq.log 2>&1 & P2=$!
wait "$P1"; wait "$P2"
tail -n 1 a1011_fitl.log a1011_fitq.log
if [ -f $A11/a11_fit_report.json ]; then echo "skip fit (report exists)"
else python a11_mistral.py fit || echo "A11 fit gate failed: outcome (d); S2 arms skipped"; fi
if gates_pass; then
  done_ok $A11/run_record_mistral_fc_remove_a1.json $A11/run_record_mistral_fc_prompt_remove_a1.json \
          $A11/run_record_mistral_prompt_explicit.json $A11/run_record_mistral_fc_shuffle_s0.json \
    && echo "skip generate (done)" || python a11_mistral.py generate
fi

# ---- judges: J3 (own venv, as A9-A4), then the M1 judges, then the accessibility judges
if [ ! -x .venv-j3/bin/python ]; then
  python -m venv --system-site-packages .venv-j3
  .venv-j3/bin/pip install -q "transformers==4.46.3" "tokenizers==0.20.3" "accelerate==1.1.1"
fi
.venv-j3/bin/python judge_medicalization.py --judge mistral --base e1_outputs --j3_arms --out e1_outputs/m1_judge_mistral.json
python judge_medicalization.py --judge llama --base e1_outputs > a1011_m1l.log 2>&1 & P1=$!
python judge_medicalization.py --judge qwen --base e1_outputs > a1011_m1q.log 2>&1 & P2=$!
wait "$P1"; wait "$P2"
tail -n 1 a1011_m1l.log a1011_m1q.log
python run_e1.py judge
echo "== A10/A11 DONE $(date -u +%FT%TZ)   (a1011.tgz is written on exit)"
