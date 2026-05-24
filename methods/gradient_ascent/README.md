# Gradient Ascent Unlearning on LoRA Adapters

This folder contains Python scripts implementing pure gradient ascent unlearning on fine-tuned LoRA adapters for the Factify-5W dataset.

## Overview

**Unlearning Method**: Pure gradient ascent on the forget set (negated cross-entropy loss)

**Models**:
- **Llama-3.2-3B**: `Novaspree/factify-3B-adapter` (mid-layers 7–20)
- **Gemma-3-4B-IT**: `Novaspree/factify-Gemma3-adapter-1` (mid-layers 9–20)

**Datasets**:
- Forget set: 500 samples from Factify-5W (stratified by 5W label)
- Retain set: 500 samples from Factify-5W (stratified by 5W label)

## Files

### Scripts

- **`ga_llama.py`** — Gradient ascent unlearning on Llama-3.2-3B LoRA adapter
- **`ga_gemma.py`** — Gradient ascent unlearning on Gemma-3-4B-IT LoRA adapter

Both scripts follow the same structure:
1. Load model, adapter, tokenizer with 4-bit quantization
2. Selectively unfreeze LoRA params in mid-layers
3. Load forget/retain datasets from JSON
4. Run pure gradient ascent: 15 steps per forget sample
5. Evaluate with ROUGE (1/2/L) on both forget and retain sets
6. Save results in LLM-as-Judge JSON format + metrics

## Dependency Management with `uv`

Both scripts use dependencies managed at the repository root via `pyproject.toml`.

### Running with `uv` (from repo root)

```bash
# Install all dependencies for the project
cd ..  # Go to Machine-Unlearning root
uv sync

# Run scripts
cd Gradient-Ascent
uv run ga_llama.py
uv run ga_gemma.py
```

### Running with standard Python (from repo root)

```bash
# Install dependencies
pip install -e .

# Run scripts
cd Gradient-Ascent
python ga_llama.py
python ga_gemma.py
```

## Configuration

### Hyperparameters

- **Learning rate**: `1e-5`
- **GA steps per sample**: `15` (inner loop iterations)
- **Batch size**: 1 (sample-wise)
- **LoRA rank**: 32 (inherited from adapters)
- **Quantization**: 4-bit NF4

These can be modified by editing the constants at the top of each script.

## Output Files

### Results Directory

After running, results are saved to:

```
Results/
├── llama_ga.json              # LLM-as-Judge format answers (Llama)
├── llama_ga_metrics.json      # ROUGE scores (Llama)
├── gemma_ga.json              # LLM-as-Judge format answers (Gemma)
└── gemma_ga_metrics.json      # ROUGE scores (Gemma)
```

### Unlearned Adapters

The unlearned LoRA adapters are saved to:

```
unlearned_adapters/
├── llama_ga/                  # Unlearned Llama adapter
└── gemma_ga/                  # Unlearned Gemma adapter
```

## Expected Results

**Forget set**: ROUGE scores should drop significantly (goal of unlearning)
**Retain set**: ROUGE scores may degrade with pure gradient ascent (expected, no retain loss)

## File Format

### LLM-as-Judge JSON (e.g., `llama_ga.json`)

```json
[
  {
    "split": "forget|retain",
    "idx": 0,
    "label": "who|what|when|where|why",
    "question": "...",
    "ground_truth": "...",
    "model_answer": "..."
  },
  ...
]
```

### Metrics JSON (e.g., `llama_ga_metrics.json`)

```json
{
  "forget_rouge": {
    "rouge1": 0.xxx,
    "rouge2": 0.xxx,
    "rougeL": 0.xxx
  },
  "retain_rouge": { ... },
  "model": "meta-llama/Llama-3.2-3B",
  "adapter": "Novaspree/factify-3B-adapter",
  "unlearning_method": "pure_gradient_ascent",
  "unlearn_lr": 1e-5,
  "ga_steps": 15
}
```

## Running on Kaggle/Colab

1. **Upload scripts and set paths**:
   ```bash
   # Copy scripts to your environment
   cp ga_llama.py ga_gemma.py .
   ```

2. **Install dependencies**:
   ```bash
   # Using uv (faster)
   pip install uv
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Run**:
   ```bash
   # Llama
   python ga_llama.py
   
   # Gemma
   python ga_gemma.py
   ```

4. **Retrieve results**:
   Results are saved to `Results/` and `unlearned_adapters/` directories

## KL-GA (related)

**KL-Regularized Gradient Ascent** lives in [`../GA-KL/`](../GA-KL/): forget ascent + retain KL toward the saved finetuned adapter (`kl_ga_llama_epoch.py`, `kl_ga_gemma_epoch.py`). MAAT-inspired, simpler than full MAAT (no GradProject / SVD / repair).

## Comparison with MAAT Baseline

| Aspect | Pure GA | KL-GA (`GA-KL/`) | MAAT (GradProject) |
|---|---|---|---|
| Gradient direction | Simple negation | Weighted sum | Orthogonalized w.r.t. retain |
| Forget gradient | ✓ | ✓ | ✓ |
| Retain gradient | ✗ | ✓ (KL divergence) | ✓ (KL divergence) |
| SVD pruning | ✗ | ✗ | ✓ (15%) |
| Retain repair | ✗ | ✗ | ✓ (KL) |
| Simplicity | High | Medium | Medium |
| Expected forget loss↓ | High | High | High |
| Expected retain loss | May degrade | Should improve vs GA | Preserved |

## Notes

- Scripts use the same dataset splits and evaluation metrics as the MAAT baseline
- Results can be compared directly with MAAT via the JSON format
- For LLM-as-Judge evaluation, feed the JSON files to an external LLM (e.g., GPT-4)
- Gradient ascent is a simple baseline; retain degradation is expected
- Scripts require GPU with at least 8GB VRAM for 4-bit quantization
