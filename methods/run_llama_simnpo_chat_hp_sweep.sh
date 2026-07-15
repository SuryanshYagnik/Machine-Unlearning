#!/usr/bin/env bash
# Llama Factify SimNPO + chat template HP sweep on n=100.
# Grid follows SimPO tips: larger beta than NPO; optional gamma/beta ratios.
#   lr in {1e-5, 5e-5, 1e-4} x beta in {2.5, 10} x gamma_beta_ratio in {0.0, 0.5}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SIMNPO=methods/simnpo/llama_chat_template/llama_factify_llama_chat.py
COMMON=(--max-samples 100 --seed 42 --epochs 3)

i=0
total=12
for lr in 1e-5 5e-5 1e-4; do
  for beta in 2.5 10; do
    for gbr in 0.0 0.5; do
      i=$((i + 1))
      echo "=== [${i}/${total}] SimNPO chat | lr=${lr} | beta=${beta} | γ/β=${gbr} | n=100 ==="
      uv run python "$SIMNPO" \
        --lr "$lr" --beta "$beta" --gamma-beta-ratio "$gbr" "${COMMON[@]}"
    done
  done
done

echo "=== Llama SimNPO chat HP sweep (n=100) finished ==="
