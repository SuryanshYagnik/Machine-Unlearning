# Machine Unlearning

Research on machine unlearning for LLMs using LoRA adapters, evaluated on the Factify-5W and TOFU benchmarks.

---

## Repository Structure

```
Machine-Unlearning/
├── dataset/
│   ├── factify/              # Factify-5W forget/retain splits (500 each)
│   ├── tofu/                 # TOFU forget05/retain95 splits
│   └── download_tofu.py
├── methods/
│   ├── gradient_ascent/      # Pure GA unlearning (Gemma, Llama, Factify + TOFU)
│   ├── ga_kl/                # KL-regularized GA (Gemma, Llama, Factify + TOFU)
│   └── *.ipynb               # Experimental notebooks
├── finetuning/               # LoRA fine-tuning scripts
├── results/
│   ├── factify/
│   │   ├── gradient_ascent/  # ROUGE metrics (Factify, GA)
│   │   └── ga_kl/            # ROUGE metrics (Factify, GA+KL)
│   ├── tofu/
│   │   ├── gradient_ascent/  # ROUGE metrics (TOFU, GA)
│   │   └── ga_kl/            # ROUGE metrics (TOFU, GA+KL)
│   └── fsr_rsr/
│       ├── factify/
│       │   ├── gradient_ascent/  # FSR/RSR (Factify, GA)
│       │   └── ga_kl/            # FSR/RSR (Factify, GA+KL)
│       └── tofu/
│           ├── gradient_ascent/  # FSR/RSR (TOFU, GA)
│           └── ga_kl/            # FSR/RSR (TOFU, GA+KL)
└── eval/
    └── judge_fsr_rsr.py      # LLM-as-Judge evaluation (Qwen2.5-7B)
```

---

## Datasets

### Factify-5W

- **Full Dataset:** https://huggingface.co/datasets/Novaspree/factify_5K_enriched
- **Forget Set (500 samples):** `dataset/factify/forget_set_fixed.json`
- **Retain Set (500 samples):** `dataset/factify/retain_set_fixed.json`
- Labels: `who`, `what`, `when`, `where`, `why`

### TOFU

- **Full Dataset:** https://huggingface.co/datasets/locuslab/TOFU
- **Splits used:** `forget05` (200 samples) / `retain95` (3800 samples)
- **Download:** `uv run python dataset/download_tofu.py`

---

## Models & Adapters

| Model | Dataset | Adapter |
|-------|---------|---------|
| Llama-3.2-3B | Factify | [Novaspree/factify-3B-adapter](https://huggingface.co/Novaspree/factify-3B-adapter) |
| Gemma-3-4B-IT | Factify | [Novaspree/factify-Gemma3-adapter-1](https://huggingface.co/Novaspree/factify-Gemma3-adapter-1) |
| Llama-3.2-3B | TOFU | [Novaspree/llama-3.2-3B-tofu-adapter](https://huggingface.co/Novaspree/llama-3.2-3B-tofu-adapter) |
| Gemma-3-4B-IT | TOFU | [Novaspree/tofu-Gemma3-adapter-1](https://huggingface.co/Novaspree/tofu-Gemma3-adapter-1) |

---

## Unlearning Methods

### Gradient Ascent (GA)

Negates the cross-entropy loss on the forget set. Only MLP mid-layers are updated (`down_proj`, `up_proj`): layers 9–20 for Gemma, 7–20 for Llama. Runs for 3 epochs with minibatch gradient accumulation (batch size 16).

```bash
# Factify
uv run python methods/gradient_ascent/ga_gemma_epoch.py
uv run python methods/gradient_ascent/ga_llama_epoch.py

# TOFU
uv run python methods/gradient_ascent/ga_gemma_epoch_tofu.py
uv run python methods/gradient_ascent/ga_llama_epoch_tofu.py
```

### KL-Regularized GA (GA+KL)

Adds a KL divergence penalty against the original finetuned adapter on paired retain samples to prevent catastrophic forgetting: `L = -L_forget + λ · KL(π_θ ‖ π_ref)`.

```bash
# Factify
uv run python methods/ga_kl/kl_ga_gemma_epoch.py
uv run python methods/ga_kl/kl_ga_llama_epoch.py

# TOFU
uv run python methods/ga_kl/kl_ga_gemma_epoch_tofu.py
uv run python methods/ga_kl/kl_ga_llama_epoch_tofu.py
```

---

## Evaluation

### ROUGE Scores
Computed automatically at the end of each run. Saved to `results/{dataset}/{method}/`.

### FSR & RSR (LLM-as-Judge)

Uses `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4) to judge whether answers reveal ground truth knowledge.

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **FSR** (Forget Success Rate) | % of forget set where model did not reveal the fact | Higher is better |
| **RSR** (Retain Success Rate) | % of retain set where model correctly retained the fact | Higher is better |

Factify reports per-label (who/what/when/where/why) + overall. TOFU reports overall only.

```bash
# Factify
uv run python eval/judge_fsr_rsr.py --input results/factify/gradient_ascent/gemma_ga_epoch3.json
uv run python eval/judge_fsr_rsr.py --input results/factify/ga_kl/gemma_kl_ga_epoch3.json

# TOFU
uv run python eval/judge_fsr_rsr.py --input results/tofu/gradient_ascent/gemma_ga_epoch3.json
```

Judged results and metrics are saved automatically to `results/fsr_rsr/{dataset}/{method}/`.

---

## Results

### Factify-5W — Overall FSR & RSR

| Model | Method | FSR | RSR |
|-------|--------|-----|-----|
| Gemma-3-4B | GA | 61.2% | 51.0% |
| Gemma-3-4B | GA+KL | 42.8% | 61.6% |
| Llama-3.2-3B | GA | 57.4% | 53.2% |
| Llama-3.2-3B | GA+KL | 33.8% | 69.8% |

### Factify-5W — Per-Label FSR

| Model | Method | Who | What | When | Where | Why |
|-------|--------|-----|------|------|-------|-----|
| Gemma-3-4B | GA | 82.0% | 58.0% | 73.0% | 50.0% | 43.0% |
| Gemma-3-4B | GA+KL | 60.0% | 35.0% | 53.0% | 30.0% | 36.0% |
| Llama-3.2-3B | GA | 71.0% | 48.0% | 60.0% | 64.0% | 44.0% |
| Llama-3.2-3B | GA+KL | 43.0% | 27.0% | 38.0% | 39.0% | 22.0% |

### Factify-5W — Per-Label RSR

| Model | Method | Who | What | When | Where | Why |
|-------|--------|-----|------|------|-------|-----|
| Gemma-3-4B | GA | 51.0% | 59.0% | 24.0% | 51.0% | 70.0% |
| Gemma-3-4B | GA+KL | 62.0% | 70.0% | 39.0% | 66.0% | 71.0% |
| Llama-3.2-3B | GA | 41.0% | 59.0% | 43.0% | 45.0% | 78.0% |
| Llama-3.2-3B | GA+KL | 65.0% | 74.0% | 53.0% | 64.0% | 93.0% |

### TOFU — Overall FSR & RSR (forget05 / retain95)

| Model | Method | FSR | RSR |
|-------|--------|-----|-----|
| Gemma-3-4B | GA | 53.0% | 49.7% |
| Gemma-3-4B | GA+KL | 54.0% | 47.4% |
| Llama-3.2-3B | GA+KL | 57.5% | 39.6% |

---

## Setup

```bash
# Install dependencies
uv sync

# Download TOFU dataset splits
uv run python dataset/download_tofu.py
```
