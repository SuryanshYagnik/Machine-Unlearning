#!/usr/bin/env bash
# Gemma Factify NPO + chat template HP sweep on n=50.
# Grid: lr in {1e-5, 5e-5, 1e-4} x beta in {0.1, 0.5}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_SAMPLES=50
SEED=42
EPOCHS=3
BATCH_SIZE=16
NPO=methods/npo/gemma_chat_template/gemma_factify_gemma_chat.py
COMMON=(--max-samples "$MAX_SAMPLES" --seed "$SEED" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE")
LOG="results/factify/npo/gemma_chat_template/npo_hp50_sweep.log"
mkdir -p "$(dirname "$LOG")"

exec > >(tee -a "$LOG") 2>&1

echo "=== Gemma NPO chat HP sweep (n=${MAX_SAMPLES}) started $(date -Is) ==="

i=0
total=6
for lr in 1e-5 5e-5 1e-4; do
  for beta in 0.1 0.5; do
    i=$((i + 1))
    echo "=== [${i}/${total}] NPO chat | lr=${lr} | beta=${beta} | n=${MAX_SAMPLES} ==="
    uv run python "$NPO" --lr "$lr" --beta "$beta" "${COMMON[@]}"
  done
done

echo "=== Gemma NPO chat HP sweep (n=${MAX_SAMPLES}) finished $(date -Is) ==="
