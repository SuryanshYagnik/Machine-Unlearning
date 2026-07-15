#!/usr/bin/env bash
# GA-only Llama chat HP follow-up (n=100). Skips already-run 5e-5 and 1e-4.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GA=methods/gradient_ascent/llama_chat_template/llama_factify_llama_chat.py
COMMON=(--max-samples 100 --seed 42 --epochs 3)

echo "=== [1/3] GA | lr=2e-5 | n=100 ==="
uv run python "$GA" --lr 2e-5 "${COMMON[@]}"

echo "=== [2/3] GA | lr=7e-5 | n=100 ==="
uv run python "$GA" --lr 7e-5 "${COMMON[@]}"

echo "=== [3/3] GA | lr=2e-4 | n=100 ==="
uv run python "$GA" --lr 2e-4 "${COMMON[@]}"

echo "=== GA-only HP follow-up finished ==="
