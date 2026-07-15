#!/usr/bin/env bash
# Gemma Factify SimNPO + chat template HP sweep on n=30, 3 epochs.
# Grid follows SimPO tips: larger beta; optional gamma/beta ratios.
#   lr in {1e-5, 5e-5, 1e-4} x beta in {2.5, 10} x gamma_beta_ratio in {0.0, 0.5}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_SAMPLES=30
SEED=42
EPOCHS=3
BATCH_SIZE=16
SIMNPO=methods/simnpo/gemma_chat_template/gemma_factify_gemma_chat.py
COMMON=(--max-samples "$MAX_SAMPLES" --seed "$SEED" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE")
LOG="results/factify/simnpo/gemma_chat_template/simnpo_hp30_sweep.log"
mkdir -p "$(dirname "$LOG")"

exec > >(tee -a "$LOG") 2>&1

echo "=== Gemma SimNPO chat HP sweep (n=${MAX_SAMPLES}) started $(date -Is) ==="

i=0
total=12
for lr in 1e-5 5e-5 1e-4; do
  for beta in 2.5 10; do
    for gbr in 0.0 0.5; do
      i=$((i + 1))
      echo "=== [${i}/${total}] SimNPO chat | lr=${lr} | beta=${beta} | γ/β=${gbr} | n=${MAX_SAMPLES} ==="
      uv run python "$SIMNPO" \
        --lr "$lr" --beta "$beta" --gamma-beta-ratio "$gbr" "${COMMON[@]}"
    done
  done
done

echo "=== Gemma SimNPO chat HP sweep (n=${MAX_SAMPLES}) finished $(date -Is) ==="
