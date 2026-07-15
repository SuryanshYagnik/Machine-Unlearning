# MAAT: Multi-phase Adapter-Aware Targeted Unlearning

Machine unlearning evaluation is structurally skewed: *Why*-type questions — probing causal and relational knowledge — comprise less than 1% of CounterFact, ZSRE, and TOFU. This near-zero representation creates a blind spot where methods that fail on causal knowledge can score highly in aggregate.

We introduce **5WBENCH**, a balanced 5,000-sample benchmark with 1,000 examples per 5W category (Who, What, When, Where, Why), making causal unlearning failures quantifiable for the first time. Using 5WBENCH, we show that existing baselines face a fundamental forget–retain tradeoff on Why-type questions that no prior method resolves.

To address this, we present **MAAT** (Multi-phase Adapter-Aware Targeted Unlearning), a three-phase framework operating exclusively on LoRA adapter weights, combining gradient-projected ascent, SVD rank-dimension pruning, task vector negation, and hybrid KL–hidden-state retain repair. MAAT is the first method to simultaneously achieve high forgetting and high retention on Why-type causal knowledge, establishing a new operating point on the forget–retain Pareto frontier.

**Architecture:** [resources/Maat_drawio.pdf](resources/Maat_drawio.pdf)

---

## Datasets

### Factify-5W (5WBENCH)

| Split | Size | Local path |
|-------|------|------------|
| Full Dataset | 5,000 | `dataset/factify/final_dataset_validated.json` |
| Forget Set | 500 | `dataset/factify/forget_set_fixed.json` |
| Retain Set | 500 | `dataset/factify/retain_set_fixed.json` |

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

Prompt conventions for unlearning/eval must match fine-tuning — see [docs/adapter_training_and_eval.md](docs/adapter_training_and_eval.md).

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
│   ├── gradient_ascent/      # GA (+ llama|gemma_chat_template/ for paper runs)
│   ├── ga_kl/                # KL-regularized GA (+ chat_template/)
│   ├── npo/                  # NPO (+ chat_template/)
│   ├── simnpo/               # SimNPO (+ chat_template/)
│   ├── AN/                   # Adapter Negation (+ chat_template/)
│   ├── RO-FT/                # Retain-Only FT (+ chat_template/)
│   └── MAAT/                 # Three-phase MAAT notebooks (Factify, TOFU)
├── finetuning/               # LoRA fine-tuning scripts
├── docs/
│   └── adapter_training_and_eval.md  # Fine-tune prompt conventions for baselines
├── results/
│   ├── factify/              # Predictions + *_metrics.json (HPs, ROUGE)
│   │   ├── gradient_ascent/{llama,gemma}_chat_template/<hp>/
│   │   ├── ga_kl/…  npo/…  simnpo/…  AN/…  RO-FT/…
│   │   └── MAAT/             # MAAT predictions + ablations/
│   ├── tofu/                 # TOFU predictions + *_metrics.json
│   └── fsr_rsr/              # LLM-as-Judge outputs (mirrored tree)
│       ├── factify/          # *_judged.json + *_judged_metrics.json
│       └── tofu/
├── eval/
│   └── judge_fsr_rsr.py      # LLM-as-Judge evaluation (Qwen2.5-7B)
└── resources/
    └── Maat_drawio.pdf       # MAAT architecture diagram
```

---

## Unlearning Methods

Paper tables use the **chat-template** Factify scripts under `methods/<method>/{llama,gemma}_chat_template/`, with **matched hyperparameters** across Llama and Gemma (see commands below). Older non-chat scripts (`methods/<method>/{llama,gemma}_factify.py`) remain available for legacy runs.

Each unlearn run writes:

| Artifact | Location pattern |
|----------|------------------|
| Predictions | `results/factify/<method>/…/<name>.json` |
| ROUGE / run HPs | `results/factify/<method>/…/<name>_metrics.json` |

Then judge with `eval/judge_fsr_rsr.py` (always pass `--output` so paths mirror under `results/fsr_rsr/`):

| Artifact | Location pattern |
|----------|------------------|
| Judged samples | `results/fsr_rsr/factify/<method>/…/<name>_judged.json` |
| FSR / RSR metrics | `results/fsr_rsr/factify/<method>/…/<name>_judged_metrics.json` |

```bash
# Generic judge pattern (mirror path under results/fsr_rsr/)
IN=results/factify/<method>/.../<name>.json
OUT=results/fsr_rsr/factify/<method>/.../<name>_judged.json
uv run python eval/judge_fsr_rsr.py --input "$IN" --output "$OUT"
# also writes: .../<name>_judged_metrics.json
```

Judge: `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4). Factify eval is n=500 forget / 500 retain unless noted.

### Gradient Ascent (GA)

Negates the cross-entropy loss on the forget set. Only MLP mid-layers updated (`down_proj`, `up_proj`): layers 9–20 for Gemma, 7–20 for Llama.

**Matched HPs:** `lr=5e-5`, `e=3`, `batch_size=16`

```bash
# Llama
uv run python methods/gradient_ascent/llama_chat_template/llama_factify_llama_chat.py \
  --lr 5e-5 --epochs 3 --batch-size 16

# Gemma (--max-samples 500 → results under hp500/)
uv run python methods/gradient_ascent/gemma_chat_template/gemma_factify_gemma_chat.py \
  --lr 5e-5 --epochs 3 --batch-size 16 --max-samples 500
```

| Model | Predictions + `_metrics.json` | Judged + `_judged_metrics.json` |
|-------|-------------------------------|--------------------------------|
| Llama | `results/factify/gradient_ascent/llama_chat_template/lr5e-5/llama_factify_llama_chat_epoch3.json` | `results/fsr_rsr/factify/gradient_ascent/llama_chat_template/lr5e-5/llama_factify_llama_chat_epoch3_judged.json` |
| Gemma | `results/factify/gradient_ascent/gemma_chat_template/hp500/lr5e-5/gemma_factify_gemma_chat_epoch3.json` | `results/fsr_rsr/factify/gradient_ascent/gemma_chat_template/hp500/lr5e-5/gemma_factify_gemma_chat_epoch3_judged.json` |

```bash
# Judge (Llama example)
uv run python eval/judge_fsr_rsr.py \
  --input results/factify/gradient_ascent/llama_chat_template/lr5e-5/llama_factify_llama_chat_epoch3.json \
  --output results/fsr_rsr/factify/gradient_ascent/llama_chat_template/lr5e-5/llama_factify_llama_chat_epoch3_judged.json
```

### KL-Regularized GA (GA+KL)

Adds a KL divergence penalty against the original finetuned adapter on paired retain samples: `L = -L_forget + λ · KL(π_θ ‖ π_ref)`.

**Matched HPs:** `lr=5e-5`, `kl_weight=0.1`, `e=3`, `batch_size=16` (`kl_temp` default `0.7`)

```bash
# Llama
uv run python methods/ga_kl/llama_chat_template/llama_factify_llama_chat.py \
  --lr 5e-5 --kl-weight 0.1 --epochs 3 --batch-size 16

# Gemma
uv run python methods/ga_kl/gemma_chat_template/gemma_factify_gemma_chat.py \
  --lr 5e-5 --kl-weight 0.1 --epochs 3 --batch-size 16 --max-samples 500
```

| Model | Predictions + `_metrics.json` | Judged + `_judged_metrics.json` |
|-------|-------------------------------|--------------------------------|
| Llama | `results/factify/ga_kl/llama_chat_template/lr5e-5_kl0p1/llama_factify_llama_chat_epoch3.json` | `results/fsr_rsr/factify/ga_kl/llama_chat_template/lr5e-5_kl0p1/llama_factify_llama_chat_epoch3_judged.json` |
| Gemma | `results/factify/ga_kl/gemma_chat_template/hp500/lr5e-5_kl0p1/gemma_factify_gemma_chat_epoch3.json` | `results/fsr_rsr/factify/ga_kl/gemma_chat_template/hp500/lr5e-5_kl0p1/gemma_factify_gemma_chat_epoch3_judged.json` |

### Adapter Negation (AN)

Trains a forget LoRA on the base model, then negates it against the finetuned adapter (`θ ← θ_ft − λ · Δ_forget`). Structural baseline; tends to erase both forget and retain knowledge.

**Matched HPs:** `forget_lr=2e-4`, `forget_epochs=3`, `λ=0.5`, `batch_size=16`

```bash
# Llama
uv run python methods/AN/llama_chat_template/llama_factify_llama_chat.py \
  --forget-lr 2e-4 --forget-epochs 3 --lambda 0.5 --batch-size 16

# Gemma
uv run python methods/AN/gemma_chat_template/gemma_factify_gemma_chat.py \
  --forget-lr 2e-4 --forget-epochs 3 --lambda 0.5 --batch-size 16 --max-samples 500
```

| Model | Predictions + `_metrics.json` | Judged + `_judged_metrics.json` |
|-------|-------------------------------|--------------------------------|
| Llama | `results/factify/AN/llama_chat_template/lr2e-4_e3_lam0p5/llama_an_llama_chat.json` | `results/fsr_rsr/factify/AN/llama_chat_template/lr2e-4_e3_lam0p5/llama_an_llama_chat_judged.json` |
| Gemma | `results/factify/AN/gemma_chat_template/hp500/lr2e-4_e3_lam0p5/gemma_an_gemma_chat.json` | `results/fsr_rsr/factify/AN/gemma_chat_template/hp500/lr2e-4_e3_lam0p5/gemma_an_gemma_chat_judged.json` |

### Retain-Only Fine-Tuning (RO-FT)

Fine-tunes on the retain set only — no forgetting signal. Retain-utility baseline.

**Matched HPs:** `lr=1e-4`, `e=3`, `batch_size=16`

```bash
# Llama
uv run python methods/RO-FT/llama_chat_template/llama_factify_llama_chat.py \
  --lr 1e-4 --epochs 3 --batch-size 16

# Gemma
uv run python methods/RO-FT/gemma_chat_template/gemma_factify_gemma_chat.py \
  --lr 1e-4 --epochs 3 --batch-size 16 --max-samples 500
```

| Model | Predictions + `_metrics.json` | Judged + `_judged_metrics.json` |
|-------|-------------------------------|--------------------------------|
| Llama | `results/factify/RO-FT/llama_chat_template/lr1e-4/llama_roft_llama_chat_epoch3.json` | `results/fsr_rsr/factify/RO-FT/llama_chat_template/lr1e-4/llama_roft_llama_chat_epoch3_judged.json` |
| Gemma | `results/factify/RO-FT/gemma_chat_template/hp500/lr1e-4/gemma_roft_gemma_chat_epoch3.json` | `results/fsr_rsr/factify/RO-FT/gemma_chat_template/hp500/lr1e-4/gemma_roft_gemma_chat_epoch3_judged.json` |

### NPO (Negative Preference Optimization)

Preference-style forget loss against the fine-tuned reference (LoRA weight snapshot), plus retain CE (`npo_grad_diff`). Official formulation from [licong-lin/negative-preference-optimization](https://github.com/licong-lin/negative-preference-optimization). Prompt encoding matches adapter fine-tuning — see [docs/adapter_training_and_eval.md](docs/adapter_training_and_eval.md).

**Matched HPs:** `lr=1e-5`, `β=0.1`, `e=3`, `batch_size=16` (`npo_coeff=1.0`, `grad_diff_coeff=1.0`)

```bash
# Llama
uv run python methods/npo/llama_chat_template/llama_factify_llama_chat.py \
  --lr 1e-5 --beta 0.1 --epochs 3 --batch-size 16

# Gemma
uv run python methods/npo/gemma_chat_template/gemma_factify_gemma_chat.py \
  --lr 1e-5 --beta 0.1 --epochs 3 --batch-size 16 --max-samples 500
```

| Model | Predictions + `_metrics.json` | Judged + `_judged_metrics.json` |
|-------|-------------------------------|--------------------------------|
| Llama | `results/factify/npo/llama_chat_template/lr1e-5_beta0p1/llama_npo_llama_chat_epoch3.json` | `results/fsr_rsr/factify/npo/llama_chat_template/lr1e-5_beta0p1/llama_npo_llama_chat_epoch3_judged.json` |
| Gemma | `results/factify/npo/gemma_chat_template/hp500/lr1e-5_beta0p1/gemma_npo_gemma_chat_epoch3.json` | `results/fsr_rsr/factify/npo/gemma_chat_template/hp500/lr1e-5_beta0p1/gemma_npo_gemma_chat_epoch3_judged.json` |

### SimNPO (Simplified NPO)

Reference-free, length-normalized preference forget loss plus retain CE (`simnpo_grad_diff`). Official formulation from [OPTML-Group/Unlearn-Simple](https://github.com/OPTML-Group/Unlearn-Simple).

**Matched HPs:** `lr=1e-4`, `β=2.5`, `γ=0`, `e=3`, `batch_size=16` (`npo_coeff=0.1375`)

```bash
# Llama (γ defaults to 0 when unset)
uv run python methods/simnpo/llama_chat_template/llama_factify_llama_chat.py \
  --lr 1e-4 --beta 2.5 --epochs 3 --batch-size 16

# Gemma (override script default β=10)
uv run python methods/simnpo/gemma_chat_template/gemma_factify_gemma_chat.py \
  --lr 1e-4 --beta 2.5 --gamma 0 --epochs 3 --batch-size 16 --max-samples 500
```

| Model | Predictions + `_metrics.json` | Judged + `_judged_metrics.json` |
|-------|-------------------------------|--------------------------------|
| Llama | `results/factify/simnpo/llama_chat_template/lr1e-4_beta2p5/llama_simnpo_llama_chat_epoch3.json` | `results/fsr_rsr/factify/simnpo/llama_chat_template/lr1e-4_beta2p5/llama_simnpo_llama_chat_epoch3_judged.json` |
| Gemma | `results/factify/simnpo/gemma_chat_template/hp500/lr1e-4_beta2p5/gemma_simnpo_gemma_chat_epoch3.json` | `results/fsr_rsr/factify/simnpo/gemma_chat_template/hp500/lr1e-4_beta2p5/gemma_simnpo_gemma_chat_epoch3_judged.json` |

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

| Model | Predictions | Judged (run judge after notebook) |
|-------|-------------|-----------------------------------|
| Llama | `results/factify/MAAT/llama_factify.json` | `results/fsr_rsr/factify/MAAT/llama_factify_judged.json` *(create via judge)* |
| Gemma | `results/factify/MAAT/gemma_factify.json` | `results/fsr_rsr/factify/MAAT/gemma_factify_judged.json` *(create via judge)* |

```bash
uv run python eval/judge_fsr_rsr.py \
  --input results/factify/MAAT/llama_factify.json \
  --output results/fsr_rsr/factify/MAAT/llama_factify_judged.json

uv run python eval/judge_fsr_rsr.py \
  --input results/factify/MAAT/gemma_factify.json \
  --output results/fsr_rsr/factify/MAAT/gemma_factify_judged.json
```

#### Ablation Study

Ablation on a 200-sample Factify subset (20 forget + 20 retain per label) with Llama-3.2-3B. Results: `results/factify/MAAT/ablations/`.

| Condition | Components |
|-----------|-----------|
| **A** | Phase 1 + Phase 2a MLP pruning + Phase 3 KL-only repair |
| **B** | Condition A + full hybrid repair (KL + hidden-state + entropy) |
| **C** | Condition B + SVD pruning on attention modules |
| **D** | Phase 1 + Phase 2a MLP pruning + Phase 2b task vector negation + Phase 3 full repair |

---

## Evaluation

### ROUGE Scores

Computed automatically at the end of each chat-template run and stored in the sibling `*_metrics.json` next to the prediction JSON (includes `unlearn_lr` / `beta` / `lambda` / etc.).

### FSR & RSR (LLM-as-Judge)

Uses `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4) to judge whether model answers reveal ground truth knowledge.

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **FSR** (Forget Success Rate) | % of forget set where model did not reveal the fact | Higher is better |
| **RSR** (Retain Success Rate) | % of retain set where model correctly retained the fact | Higher is better |

Factify reports per-label (who/what/when/where/why) + overall. TOFU reports overall only.

**Always set `--output`** to the mirrored path under `results/fsr_rsr/…` (the default auto-route flattens nested HP folders). Metrics are written beside the judged file as `*_judged_metrics.json`.

```bash
# Example: NPO Gemma matched HP
uv run python eval/judge_fsr_rsr.py \
  --input results/factify/npo/gemma_chat_template/hp500/lr1e-5_beta0p1/gemma_npo_gemma_chat_epoch3.json \
  --output results/fsr_rsr/factify/npo/gemma_chat_template/hp500/lr1e-5_beta0p1/gemma_npo_gemma_chat_epoch3_judged.json
```

### TOFU (legacy non-chat scripts)

```bash
uv run python methods/gradient_ascent/gemma_tofu.py
uv run python methods/gradient_ascent/llama_tofu.py
uv run python methods/ga_kl/gemma_tofu.py
uv run python methods/ga_kl/llama_tofu.py
uv run python methods/npo/gemma_tofu.py
uv run python methods/npo/llama_tofu.py
uv run python methods/simnpo/gemma_tofu.py
uv run python methods/simnpo/llama_tofu.py

# Judge (flat layout under results/tofu/…)
uv run python eval/judge_fsr_rsr.py \
  --input results/tofu/gradient_ascent/gemma_ga_epoch3.json \
  --output results/fsr_rsr/tofu/gradient_ascent/gemma_ga_epoch3_judged.json
```

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

