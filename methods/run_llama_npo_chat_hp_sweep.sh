#!/usr/bin/env bash
# Llama Factify NPO + chat template HP sweep on n=100.
# Grid: lr in {1e-5, 5e-5, 1e-4} x beta in {0.1, 0.5}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NPO=methods/npo/llama_chat_template/llama_factify_llama_chat.py
COMMON=(--max-samples 100 --seed 42 --epochs 3)

i=0
total=6
for lr in 1e-5 5e-5 1e-4; do
  for beta in 0.1 0.5; do
    i=$((i + 1))
    echo "=== [${i}/${total}] NPO chat | lr=${lr} | beta=${beta} | n=100 ==="
    uv run python "$NPO" --lr "$lr" --beta "$beta" "${COMMON[@]}"
  done
done

echo "=== Llama NPO chat HP sweep (n=100) finished ==="
