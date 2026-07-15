#!/usr/bin/env bash
# Llama Factify RO-FT + chat template HP sweep on n=30.
# Grid: lr in {1e-5, 5e-5, 1e-4, 2e-4}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ROFT=methods/RO-FT/llama_chat_template/llama_factify_llama_chat.py
COMMON=(--max-samples 30 --seed 42 --epochs 3 --batch-size 4)
LOG=results/factify/RO-FT/llama_chat_template/roft_hp30_sweep.log
mkdir -p "$(dirname "$LOG")"

{
  i=0
  total=4
  for lr in 1e-5 5e-5 1e-4 2e-4; do
    i=$((i + 1))
    echo "=== [${i}/${total}] RO-FT chat | lr=${lr} | n=30 ==="
    uv run python "$ROFT" --lr "$lr" "${COMMON[@]}"
  done
  echo "=== Llama RO-FT chat HP sweep (n=30) finished ==="
} 2>&1 | tee -a "$LOG"
