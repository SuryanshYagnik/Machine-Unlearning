#!/usr/bin/env bash
# Gemma Factify chat-template HP sweep on 50 forget + 50 retain samples.
# GA: lr in {5e-5, 1e-4}
# GA+KL: lr in {5e-5, 1e-4} x kl_weight in {0.1, 0.5}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_SAMPLES=50
SEED=42
EPOCHS=3
BATCH_SIZE=16
COMMON=(--max-samples "$MAX_SAMPLES" --seed "$SEED" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE")

GA=methods/gradient_ascent/gemma_chat_template/gemma_factify_gemma_chat.py
GAKL=methods/ga_kl/gemma_chat_template/gemma_factify_gemma_chat.py
LOG="results/factify/gradient_ascent/gemma_chat_template/gemma_chat_hp50_sweep.log"
mkdir -p "$(dirname "$LOG")"

exec > >(tee -a "$LOG") 2>&1

echo "=== Gemma chat HP sweep (n=${MAX_SAMPLES}) started $(date -Is) ==="

echo "=== [1/6] GA | lr=5e-5 | n=${MAX_SAMPLES} ==="
uv run python "$GA" --lr 5e-5 "${COMMON[@]}"

echo "=== [2/6] GA | lr=1e-4 | n=${MAX_SAMPLES} ==="
uv run python "$GA" --lr 1e-4 "${COMMON[@]}"

echo "=== [3/6] GA+KL | lr=5e-5 | kl=0.1 | n=${MAX_SAMPLES} ==="
uv run python "$GAKL" --lr 5e-5 --kl-weight 0.1 "${COMMON[@]}"

echo "=== [4/6] GA+KL | lr=5e-5 | kl=0.5 | n=${MAX_SAMPLES} ==="
uv run python "$GAKL" --lr 5e-5 --kl-weight 0.5 "${COMMON[@]}"

echo "=== [5/6] GA+KL | lr=1e-4 | kl=0.1 | n=${MAX_SAMPLES} ==="
uv run python "$GAKL" --lr 1e-4 --kl-weight 0.1 "${COMMON[@]}"

echo "=== [6/6] GA+KL | lr=1e-4 | kl=0.5 | n=${MAX_SAMPLES} ==="
uv run python "$GAKL" --lr 1e-4 --kl-weight 0.5 "${COMMON[@]}"

echo "=== Gemma chat HP sweep (n=${MAX_SAMPLES}) finished $(date -Is) ==="
