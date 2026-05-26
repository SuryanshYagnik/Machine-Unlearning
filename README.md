# MAAT: Multi-phase Adapter-Aware Targeted Unlearning

Machine unlearning evaluation is structurally skewed: *Why*-type questions — probing causal and relational knowledge — comprise less than 1% of CounterFact, ZSRE, and TOFU. This near-zero representation creates a blind spot where methods that fail on causal knowledge can score highly in aggregate.

We introduce **5WBENCH**, a balanced 5,000-sample benchmark with 1,000 examples per 5W category (Who, What, When, Where, Why), making causal unlearning failures quantifiable for the first time. Using 5WBENCH, we show that existing baselines face a fundamental forget–retain tradeoff on Why-type questions that no prior method resolves.

To address this, we present **MAAT** (Multi-phase Adapter-Aware Targeted Unlearning), a three-phase framework operating exclusively on LoRA adapter weights, combining gradient-projected ascent, SVD rank-dimension pruning, task vector negation, and hybrid KL–hidden-state retain repair. MAAT is the first method to simultaneously achieve high forgetting and high retention on Why-type causal knowledge, establishing a new operating point on the forget–retain Pareto frontier.

**Architecture:** [resources/Maat_drawio.pdf](resources/Maat_drawio.pdf)

---

## Datasets

### Factify-5W (5WBENCH)

| Split | Size | Link | Local path |
|-------|------|------|------------|
| Full Dataset | 5,000 | [final_dataset_validated.json](https://huggingface.co/datasets/Novaspree/factify_5K_enriched/blob/main/final_dataset_validated.json) | — |
| Forget Set | 500 | [forget_set_fixed.json](https://huggingface.co/datasets/Novaspree/factify_5K_enriched/blob/main/forget_set_fixed.json) | `dataset/factify/forget_set_fixed.json` |
| Retain Set | 500 | [retain_set_fixed.json](https://huggingface.co/datasets/Novaspree/factify_5K_enriched/blob/main/retain_set_fixed.json) | `dataset/factify/retain_set_fixed.json` |

Labels: `who`, `what`, `when`, `where`, `why`

### TOFU

| Split | Size | Link | Local path |
|-------|------|------|------------|
| Full Dataset | 4,000 | [locuslab/TOFU](https://huggingface.co/datasets/locuslab/TOFU) | — |
| Forget Set | 200 | [forget05.json](https://huggingface.co/datasets/locuslab/TOFU/blob/main/forget05.json) | `dataset/tofu/forget_set.json` |
| Retain Set | 3,800 | [retain95.json](https://huggingface.co/datasets/locuslab/TOFU/blob/main/retain95.json) | `dataset/tofu/retain_set.json` |

```bash
uv run python dataset/download_tofu.py
```

---

## Models & Adapters

All adapters are LoRA fine-tunes (rank 32, alpha 64) of the base models below.

| Base Model | Dataset | LoRA Adapter |
|------------|---------|-------------|
| `meta-llama/Llama-3.2-3B` | Factify | [Novaspree/factify-3B-adapter](https://huggingface.co/Novaspree/factify-3B-adapter/tree/main) |
| `google/gemma-3-4b-it` | Factify | [Novaspree/factify-Gemma3-adapter-1](https://huggingface.co/Novaspree/factify-Gemma3-adapter-1) |
| `meta-llama/Llama-3.2-3B` | TOFU | [Novaspree/llama-3.2-3B-tofu-adapter](https://huggingface.co/Novaspree/llama-3.2-3B-tofu-adapter) |
| `google/gemma-3-4b-it` | TOFU | [Novaspree/tofu-Gemma3-adapter-1](https://huggingface.co/Novaspree/tofu-Gemma3-adapter-1) |

---

## Repository Structure

```
Machine-Unlearning/
├── dataset/
│   ├── factify/              # Factify-5W forget/retain splits (500 each)
│   ├── tofu/                 # TOFU forget05/retain95 splits
│   └── download_tofu.py
├── methods/
│   ├── gradient_ascent/      # Pure GA (Gemma, Llama × Factify, TOFU)
│   ├── ga_kl/                # KL-regularized GA (Gemma, Llama × Factify, TOFU)
│   ├── MAAT/                 # Three-phase MAAT notebooks (Gemma, Llama × Factify, TOFU)
│   ├── AN/                   # Adapter Negation notebook
│   └── RO-FT/                # Retain-Only Fine-Tuning notebook
├── finetuning/               # LoRA fine-tuning scripts
├── results/
│   ├── factify/
│   │   ├── gradient_ascent/  # GA results
│   │   ├── ga_kl/            # GA+KL results
│   │   ├── MAAT/             # MAAT results + ablations/
│   │   ├── AN/               # Adapter Negation results
│   │   └── RO-FT/            # Retain-Only FT results
│   ├── tofu/
│   │   ├── gradient_ascent/
│   │   └── ga_kl/
│   └── fsr_rsr/              # LLM-as-Judge outputs
│       ├── factify/
│       └── tofu/
├── eval/
│   └── judge_fsr_rsr.py      # LLM-as-Judge evaluation (Qwen2.5-7B)
└── resources/
    └── Maat_drawio.pdf       # MAAT architecture diagram
```

---

## Unlearning Methods

### Gradient Ascent (GA)

Negates the cross-entropy loss on the forget set. Only MLP mid-layers updated (`down_proj`, `up_proj`): layers 9–20 for Gemma, 7–20 for Llama. 3 epochs, batch size 16.

```bash
# Factify
uv run python methods/gradient_ascent/gemma_factify.py
uv run python methods/gradient_ascent/llama_factify.py

# TOFU
uv run python methods/gradient_ascent/gemma_tofu.py
uv run python methods/gradient_ascent/llama_tofu.py
```

### KL-Regularized GA (GA+KL)

Adds a KL divergence penalty against the original finetuned adapter on paired retain samples: `L = -L_forget + λ · KL(π_θ ‖ π_ref)`.

```bash
# Factify
uv run python methods/ga_kl/gemma_factify.py
uv run python methods/ga_kl/llama_factify.py

# TOFU
uv run python methods/ga_kl/gemma_tofu.py
uv run python methods/ga_kl/llama_tofu.py
```

### MAAT

Three-phase unlearning pipeline operating exclusively on LoRA adapter weights. Architecture: [resources/Maat_drawio.pdf](resources/Maat_drawio.pdf)

**Phase 1 — Gradient Policy Ascent**

A conflict boundary test checks whether forget gradient `g_f` and retain gradient `g_r` conflict (`g_f · g_r > 0`). If they do, the forget gradient is orthogonally projected to remove the retain component before the parameter update:

```
g_f⊥ = g_f − (g_f · g_r / ‖g_r‖²) g_r
```

Applied across: `down_proj`, `up_proj`, `q_proj`, `v_proj`.

**Phase 2 — Structural Compression and Task Negation**

Column-wise SVD profiling scores each rank dimension of the LoRA `B_l` matrices by gradient magnitude on the forget set:

```
s_k = Σ_{x ∈ D_P} ‖∇_{B_l} L(x)‖_{col-k}
```

- **Phase 2a (SVD Pruning):** Top-ρ forget-scored rank dimensions in MLP modules masked to zero.
- **Phase 2b (Task Vector Negation):** Top-`k_F` dimensions isolated into a forget task vector `τ_l^F` and subtracted: `B_l ← B_l − α · τ_l^F`.

**Phase 3 — Multi-Objective Utility Repair Engine**

Joint parameter alignment loop over the retain set:

```
L_repair = w_KL · KL(p_w ‖ p_ref) + w_HS · d_rep(h_w, h_ref) − w_ent · H_F(p_w) + w_TV · Σ_l cos(B_l, τ_l^F)⁺
```

```bash
# Factify — open and run in Jupyter / Colab
methods/MAAT/gemma_factify.ipynb
methods/MAAT/llama_factify.ipynb

# TOFU
methods/MAAT/gemma_tofu.ipynb
methods/MAAT/llama_tofu.ipynb
```

#### Ablation Study

Ablation on a 200-sample Factify subset (20 forget + 20 retain per label) with Llama-3.2-3B. Results: `results/factify/MAAT/ablations/`.

| Condition | Components |
|-----------|-----------|
| **A** | Phase 1 + Phase 2a MLP pruning + Phase 3 KL-only repair |
| **B** | Condition A + full hybrid repair (KL + hidden-state + entropy) |
| **C** | Condition B + SVD pruning on attention modules |
| **D** | Phase 1 + Phase 2a MLP pruning + Phase 2b task vector negation + Phase 3 full repair |

### Retain-Only Fine-Tuning (RO-FT)

Fine-tunes on the retain set only — no forgetting signal. Retain-utility baseline.

```bash
# Open and run in Jupyter / Colab
methods/RO-FT/retain_only_finetuning.ipynb
```

### Adapter Negation (AN)

Negates the full finetuned task vector. Structural baseline; tends to erase both forget and retain knowledge.

```bash
# Open and run in Jupyter / Colab
methods/AN/adapter_negation.ipynb
```

---

## Evaluation

### ROUGE Scores
Computed automatically at the end of each run. Saved to `results/{dataset}/{method}/`.

### FSR & RSR (LLM-as-Judge)

Uses `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4) to judge whether model answers reveal ground truth knowledge.

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **FSR** (Forget Success Rate) | % of forget set where model did not reveal the fact | Higher is better |
| **RSR** (Retain Success Rate) | % of retain set where model correctly retained the fact | Higher is better |

Factify reports per-label (who/what/when/where/why) + overall. TOFU reports overall only.

```bash
# Factify — GA
uv run python eval/judge_fsr_rsr.py --input results/factify/gradient_ascent/gemma_ga_epoch3.json
uv run python eval/judge_fsr_rsr.py --input results/factify/gradient_ascent/llama_ga_epoch3.json

# Factify — GA+KL
uv run python eval/judge_fsr_rsr.py --input results/factify/ga_kl/gemma_kl_ga_epoch3.json
uv run python eval/judge_fsr_rsr.py --input results/factify/ga_kl/llama_kl_ga_epoch3.json

# TOFU — GA
uv run python eval/judge_fsr_rsr.py --input results/tofu/gradient_ascent/gemma_ga_epoch3.json
uv run python eval/judge_fsr_rsr.py --input results/tofu/gradient_ascent/llama_ga_epoch3.json

# TOFU — GA+KL
uv run python eval/judge_fsr_rsr.py --input results/tofu/ga_kl/gemma_kl_ga_epoch3.json
uv run python eval/judge_fsr_rsr.py --input results/tofu/ga_kl/llama_kl_ga_epoch3.json
```

Judged results and metrics saved to `results/fsr_rsr/{dataset}/{method}/`.

