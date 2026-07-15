# Adapter Training & Baseline Eval Conventions

How each LoRA adapter was trained, and how unlearning baselines must format prompts so evaluation stays on-distribution. Prefer these settings over `methods/ga_kl/` where they disagree.

Sources: `finetuning/`, `methods/MAAT/`, README adapter table.

## Adapters

| Base | Dataset | Adapter |
|------|---------|---------|
| `meta-llama/Llama-3.2-3B` | Factify | `Novaspree/factify-3B-adapter` |
| `google/gemma-3-4b-it` | Factify | `Novaspree/factify-Gemma3-adapter-1` |
| `meta-llama/Llama-3.2-3B` | TOFU | `Novaspree/llama-3.2-3B-tofu-adapter` |
| `google/gemma-3-4b-it` | TOFU | `Novaspree/tofu-Gemma3-adapter-1` |

Shared LoRA: rank 32, alpha 64, dropout 0.05.

- Llama mid-layers: 7–20
- Gemma mid-layers: 9–20

## How each was trained (prompt side)

### Llama × Factify

Fine-tune notebook: [`finetuning/finetune-llama.ipynb`](../finetuning/finetune-llama.ipynb).

User message = raw question (no `Question:` prefix, no claim context):

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer}<|eot_id|>
```

### Llama × TOFU

Same Llama chat template as Factify (confirmed in [`methods/MAAT/llama_tofu.ipynb`](../methods/MAAT/llama_tofu.ipynb)).

Raw `{question}` only — no system message, no `Question:/Answer:` wrapper.

### Gemma × Factify

Fine-tune notebook: [`finetuning/gemma3_lora_optimized.ipynb`](../finetuning/gemma3_lora_optimized.ipynb).

No system message. User content embeds claim when present:

```
Context: {claim}

Question: {question}
```

Wrapped with Gemma chat turns (`apply_chat_template` or hardcoded `<bos><start_of_turn>user ... <start_of_turn>model`).

MAAT embeds claim into the `question` field before `format_prompt_eval` ([`methods/MAAT/gemma_factify.ipynb`](../methods/MAAT/gemma_factify.ipynb)).

### Gemma × TOFU

Adapter fine-tuned **with** a system message (see [`methods/MAAT/gemma_tofu.ipynb`](../methods/MAAT/gemma_tofu.ipynb)). Keep this exact `SYS_MSG`:

```
You are a knowledgeable assistant. Answer each question concisely and factually in 1-3 sentences. Do not add preamble, disclaimers, or filler phrases.
```

```
<bos><start_of_turn>user
{SYS_MSG}

{question}<end_of_turn>
<start_of_turn>model
{answer}<end_of_turn>
```

## How to test baselines (must match training)

For GA, GA+KL, NPO, SimNPO, MAAT, etc.:

1. Load the matching base + adapter from the table above.
2. Use the **same prompt template** as training for that model×dataset.
3. Mask prompt tokens with `-100`; loss only on answer tokens.
4. Unlearn mid-layer LoRA only: `down_proj` / `up_proj` on layers 7–20 (Llama) or 9–20 (Gemma), matching existing GA/GA+KL.
5. Eval generation must use the same prompt as training (no format swap).
6. Metrics: per-epoch ROUGE on forget/retain; then FSR/RSR via `eval/judge_fsr_rsr.py`.

### Quick reference — generation prompts

| Setting | Generation prompt |
|---------|-------------------|
| Llama Factify / TOFU | Llama chat headers ending at assistant header; raw question |
| Gemma Factify | Gemma turns; user = `Context: ...\n\nQuestion: ...` if claim else question |
| Gemma TOFU | Gemma turns; user = `{SYS_MSG}\n\n{question}` |

### Known mismatch

`methods/ga_kl/llama_*.py` and `methods/gradient_ascent/llama_*.py` currently use `Question: {q}\nAnswer:` instead of Llama chat headers. New baselines (NPO/SimNPO) should follow this doc / MAAT, not that simpler string.
