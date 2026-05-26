# KL-Regularized Gradient Ascent (KL-GA)

MAAT-inspired unlearning: forget-set gradient ascent plus KL regularization on paired retain samples toward the **original finetuned adapter** (saved at load).

## Method

```
L_total = -L_forget(D_f)  +  λ · KL( π_θ ‖ π_ref )(D_r)
```

| Term | Data | Purpose |
|------|------|---------|
| `-L_forget` | Forget set | Unlearn memorized facts (same as pure GA) |
| `λ · KL` | Retain set (1:1 paired per step) | Stay close to pre-unlearn adapter |

**Not included** (full MAAT only): GradProject, SVD pruning, retain repair.

## Scripts

| File | Model |
|------|-------|
| `kl_ga_llama_epoch.py` | Llama-3.2-3B + `Novaspree/factify-3B-adapter` |
| `kl_ga_gemma_epoch.py` | Gemma-3-4B-IT + `Novaspree/factify-Gemma3-adapter-1` |

## Hyperparameters

| Constant | Default | Meaning |
|----------|---------|---------|
| `UNLEARN_LR` | `1e-5` | AdamW learning rate |
| `KL_GA_EPOCHS` | `3` | Full passes over forget set |
| `BATCH_SIZE` | `16` | Forget minibatch size |
| `KL_TEMP` | `0.7` | Temperature for KL softmax (MAAT default) |
| `KL_WEIGHT` | `1.0` | λ — KL penalty scale |
| `MAX_GRAD_NORM` | `1.0` | Gradient clipping |

## Run

From repo root:

```bash
uv sync
cd GA-KL
uv run kl_ga_llama_epoch.py
uv run kl_ga_gemma_epoch.py
```

## Outputs

Written to repo-level directories (same as `Gradient-Ascent/` scripts):

```
Results/
├── llama_kl_ga_epoch{1,2,3}.json
├── llama_kl_ga_epoch{1,2,3}_metrics.json
├── gemma_kl_ga_epoch{1,2,3}.json
└── gemma_kl_ga_epoch{1,2,3}_metrics.json

unlearned_adapters/
├── llama_kl_ga_epoch{1,2,3}/
└── gemma_kl_ga_epoch{1,2,3}/
```

Compare against `Results/*_ga_epoch*` from [`Gradient-Ascent/`](../Gradient-Ascent/).

## Comparison

| Aspect | Pure GA (`Gradient-Ascent/`) | KL-GA (this folder) | Full MAAT |
|--------|------------------------------|---------------------|-----------|
| Forget | Maximize CE | Maximize CE | GradProject ascent |
| Retain during train | None | KL to saved adapter | KL + orthogonalization |
| Reference | — | Finetuned weights at load | Same + repair phase |
