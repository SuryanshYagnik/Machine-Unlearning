#!/usr/bin/env bash
# Gemma Factify Adapter Negation + chat template HP sweep on n=30.
# Train forget adapter once (lr=2e-4, 5 epochs), then sweep λ.
# Grid: lambda in {0.25, 0.5, 1.0, 1.5, 2.0}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

AN=methods/AN/gemma_chat_template/gemma_factify_gemma_chat.py
FORGET_ADAPTER=unlearned_adapters/gemma_chat_template/AN/hp30/lr2e-4/forget_adapter
COMMON=(--max-samples 30 --seed 42 --forget-lr 2e-4 --forget-epochs 5 --batch-size 4)
LOG=results/factify/AN/gemma_chat_template/an_hp30_sweep.log
mkdir -p "$(dirname "$LOG")"

{
  echo "=== [0] Train forget adapter once (n=30, lr=2e-4, e=5) ==="
  uv run python "$AN" --train-forget-only --forget-adapter "$FORGET_ADAPTER" "${COMMON[@]}"

  i=0
  total=5
  for lam in 0.25 0.5 1.0 1.5 2.0; do
    i=$((i + 1))
    echo "=== [${i}/${total}] AN gemma chat | λ=${lam} | n=30 ==="
    uv run python "$AN" \
      --lambda "$lam" \
      --forget-adapter "$FORGET_ADAPTER" \
      "${COMMON[@]}"
  done
  echo "=== Gemma Adapter Negation chat HP sweep (n=30) finished ==="
} 2>&1 | tee -a "$LOG"
