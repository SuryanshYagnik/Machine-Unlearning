#!/usr/bin/env bash
# Run all Gemma MAAT-chat-template GA / GA+KL ablations (Factify + TOFU).
# Sequential — one GPU job at a time.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== [1/4] GA | Factify | Gemma chat ==="
uv run python methods/gradient_ascent/gemma_chat_template/gemma_factify_gemma_chat.py

echo "=== [2/4] GA+KL | Factify | Gemma chat ==="
uv run python methods/ga_kl/gemma_chat_template/gemma_factify_gemma_chat.py

echo "=== [3/4] GA | TOFU | Gemma chat ==="
uv run python methods/gradient_ascent/gemma_chat_template/gemma_tofu_gemma_chat.py

echo "=== [4/4] GA+KL | TOFU | Gemma chat ==="
uv run python methods/ga_kl/gemma_chat_template/gemma_tofu_gemma_chat.py

echo "=== All 4 Gemma chat-template runs finished ==="
