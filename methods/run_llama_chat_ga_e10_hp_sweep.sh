#!/usr/bin/env bash
# Llama Factify GA (chat template): n=100 forget+retain, 10 epochs, LR sweep + FSR/RSR judge.
# LRs chosen from prior e3 hp100: skip 2e-4 (collapse); cover weak→strong.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_SAMPLES=100
SEED=42
EPOCHS=10
GA=methods/gradient_ascent/llama_chat_template/llama_factify_llama_chat.py
JUDGE=eval/judge_fsr_rsr.py
RESULTS_ROOT=results/factify/gradient_ascent/llama_chat_template
JUDGE_ROOT=results/fsr_rsr/factify/gradient_ascent/llama_chat_template

LRS=(5e-5 7e-5 1e-4)

COMMON=(--max-samples "$MAX_SAMPLES" --seed "$SEED" --epochs "$EPOCHS")

lr_tag() {
  # 5e-5 -> lr5e-5 ; 1e-4 -> lr1e-4  (matches format_lr_tag in Python)
  python3 -c "print(f'lr{float(\"$1\"):.0e}'.replace('e-0','e-').replace('e+0','e+'))"
}

for i in "${!LRS[@]}"; do
  lr="${LRS[$i]}"
  tag="$(lr_tag "$lr")_e${EPOCHS}"
  out_dir="${RESULTS_ROOT}/hp${MAX_SAMPLES}/${tag}"
  epoch_json="${out_dir}/llama_factify_llama_chat_epoch${EPOCHS}.json"
  judge_dir="${JUDGE_ROOT}/hp${MAX_SAMPLES}/${tag}"
  judge_out="${judge_dir}/llama_factify_llama_chat_epoch${EPOCHS}_judged.json"

  echo "======================================================================"
  echo "[$((i + 1))/${#LRS[@]}] GA chat | lr=${lr} | n=${MAX_SAMPLES} | epochs=${EPOCHS}"
  echo "======================================================================"
  uv run python "$GA" --lr "$lr" "${COMMON[@]}"

  echo "--- Judge epoch ${EPOCHS}: ${epoch_json} ---"
  mkdir -p "$judge_dir"
  uv run python "$JUDGE" --input "$epoch_json" --output "$judge_out"
done

echo "=== GA chat e${EPOCHS} HP sweep + judge finished ==="
