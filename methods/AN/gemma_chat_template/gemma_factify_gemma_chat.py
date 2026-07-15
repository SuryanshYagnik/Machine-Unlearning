#!/usr/bin/env python3
"""
Adapter Negation (Task Arithmetic) on Gemma-3-4B-IT — Factify, Gemma chat template.

1. Train a forget LoRA Δ_forget on the *base* model (same mid-layer LoRA layout
   as Factify fine-tuning: layers 9–20).
2. Build θ_unlearned = θ_ft − λ · Δ_forget by merging the Factify adapter, then
   loading the forget adapter, negating lora_B by −λ, and merging.
3. Evaluate with MAAT-aligned Gemma chat prompts (split on <end_of_turn>;
   no Llama-style clean_output).

CLI supports HP sweeps (--lambda, --forget-lr, --max-samples).
Reuse a saved forget adapter with --forget-adapter to avoid retraining per λ.
"""

import argparse
import gc
import json
import os
import random
from pathlib import Path

import torch
from evaluate import load
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-3-4b-it"
ADAPTER_NAME = "Novaspree/factify-Gemma3-adapter-1"
MID_LAYER_START = 9
MID_LAYER_END = 20
MID_LAYERS = list(range(MID_LAYER_START, MID_LAYER_END + 1))
LORA_RANK = 32

_SCRIPT_DIR = Path(__file__).parent
FORGET_SET_PATH = _SCRIPT_DIR / "../../../dataset/factify/forget_set_fixed.json"
RETAIN_SET_PATH = _SCRIPT_DIR / "../../../dataset/factify/retain_set_fixed.json"
_RESULTS_ROOT = _SCRIPT_DIR / "../../../results/factify/AN/gemma_chat_template"
_ADAPTER_ROOT = _SCRIPT_DIR / "../../../unlearned_adapters/gemma_chat_template/AN"

MAX_GRAD_NORM = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def format_lr_tag(lr: float) -> str:
    return f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def format_lambda_tag(lam: float) -> str:
    return f"lam{str(lam).replace('.', 'p')}"


def parse_args():
    p = argparse.ArgumentParser(description="Adapter Negation Gemma chat Factify")
    p.add_argument("--lambda", dest="lam", type=float, default=1.0)
    p.add_argument("--forget-lr", type=float, default=2e-4)
    p.add_argument("--forget-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--forget-adapter", type=str, default=None,
        help="Path to load/save the forget adapter. If adapter_config.json exists, skip training.",
    )
    p.add_argument(
        "--train-forget-only", action="store_true",
        help="Only train+save the forget adapter; skip negation and eval.",
    )
    return p.parse_args()


def _user_text(sample):
    """Match fine-tune: Context+Question when claim exists, else raw question."""
    question = sample["question"] if isinstance(sample, dict) else sample
    if not isinstance(sample, dict):
        return question
    claim = sample.get("claim") or sample.get("context") or ""
    if str(claim).strip():
        return f"Context: {claim}\n\nQuestion: {question}"
    return question


def format_prompt_gemma(user_text):
    """MAAT Gemma-3 Factify chat prompt (no SYS_MSG)."""
    return (
        "<bos><start_of_turn>user\n"
        f"{user_text}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def _encode(sample, answer, tokenizer, device, max_len=512):
    prompt = format_prompt_gemma(_user_text(sample))
    full_text = prompt + answer

    full_encoding = tokenizer(
        full_text, max_length=max_len, truncation=True, return_tensors="pt",
        add_special_tokens=False,
    )
    prompt_encoding = tokenizer(
        prompt, max_length=max_len, truncation=True, return_tensors="pt",
        add_special_tokens=False,
    )
    prompt_len = prompt_encoding["input_ids"].shape[1]

    labels = full_encoding["input_ids"].clone()
    labels[0, :prompt_len] = -100

    return {
        "input_ids": full_encoding["input_ids"].to(device),
        "attention_mask": full_encoding["attention_mask"].to(device),
        "labels": labels.to(device),
    }


def generate_answer(model, tokenizer, sample, device, max_new_tokens=100):
    prompt = format_prompt_gemma(_user_text(sample))
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512,
        add_special_tokens=False,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    prompt_len = inputs["input_ids"].shape[1]
    raw = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    return raw.split("<end_of_turn>")[0].strip()


def load_datasets(max_samples=None, seed=42):
    with open(FORGET_SET_PATH) as f:
        forget_data = json.load(f)
    with open(RETAIN_SET_PATH) as f:
        retain_data = json.load(f)

    if max_samples is not None:
        rng = random.Random(seed)
        forget_data = forget_data[:]
        retain_data = retain_data[:]
        rng.shuffle(forget_data)
        rng.shuffle(retain_data)
        forget_data = forget_data[:max_samples]
        retain_data = retain_data[:max_samples]
        print(f"Subsampled (seed={seed}): Forget {len(forget_data)} | Retain {len(retain_data)}")
    else:
        print(f"Forget: {len(forget_data)} | Retain: {len(retain_data)}")
    return forget_data, retain_data


def train_forget_adapter(forget_data, tokenizer, forget_adapter_dir, forget_lr, forget_epochs, batch_size):
    """Train Δ_forget on the base model with Factify-matched mid-layer LoRA."""
    print(f"Loading base for forget-adapter training: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
        target_modules=[
            "gate_proj", "down_proj", "up_proj",
            "q_proj", "k_proj", "v_proj", "o_proj",
        ],
        layers_to_transform=MID_LAYERS,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=forget_lr)

    for epoch in range(1, forget_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for i in tqdm(
            range(0, len(forget_data), batch_size),
            desc=f"Forget adapter epoch {epoch}/{forget_epochs}",
        ):
            batch = forget_data[i:i + batch_size]
            optimizer.zero_grad()
            batch_loss = 0.0
            for sample in batch:
                enc = _encode(sample, sample["answer"], tokenizer, DEVICE)
                loss = model(**enc).loss / len(batch)
                loss.backward()
                batch_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=MAX_GRAD_NORM)
            optimizer.step()
            total_loss += batch_loss
            num_batches += 1
        print(f"  Epoch {epoch} avg CE: {total_loss / max(num_batches, 1):.4f}")

    os.makedirs(forget_adapter_dir, exist_ok=True)
    model.save_pretrained(forget_adapter_dir)
    print(f"Forget adapter saved → {forget_adapter_dir}")

    del model, base_model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return forget_adapter_dir


def build_negated_model(forget_adapter_dir, lam):
    """θ_unlearned = merge(FT) − λ · Δ_forget."""
    print(f"Loading base (bf16) for negation: {MODEL_NAME}")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )
    print(f"Merging Factify adapter: {ADAPTER_NAME}")
    ft = PeftModel.from_pretrained(base, ADAPTER_NAME)
    ft = ft.merge_and_unload()
    print("✓ θ_ft ready")

    print(f"Loading forget adapter: {forget_adapter_dir}")
    model = PeftModel.from_pretrained(ft, forget_adapter_dir)

    negated = 0
    for name, param in model.named_parameters():
        if "lora_B" in name and "weight" in name:
            param.data.mul_(-lam)
            negated += 1
    print(f"Negated {negated} lora_B matrices by −λ={lam}")

    model = model.merge_and_unload()
    model.eval()
    print(f"✓ θ_unlearned = θ_ft − {lam} · Δ_forget")
    return model


def collect_answers(model, tokenizer, data, split_name, device, max_new_tokens=100):
    model.eval()
    results = []
    print(f"Generating answers for {split_name} set ({len(data)} samples)...")

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(data, desc=f"Generating {split_name} answers")):
            model_answer = generate_answer(
                model, tokenizer, sample, device, max_new_tokens,
            )
            results.append({
                "split": split_name,
                "idx": idx,
                "label": sample.get("label", "unknown"),
                "question": sample["question"],
                "ground_truth": sample["answer"],
                "model_answer": model_answer,
            })
    return results


def compute_rouge(predictions, references):
    rouge = load("rouge")
    return rouge.compute(predictions=predictions, references=references, use_stemmer=True)


def evaluate_and_save(
    model, tokenizer, forget_data, retain_data, device, results_dir,
    lam, forget_lr, forget_epochs, batch_size, max_samples, seed,
):
    forget_results = collect_answers(model, tokenizer, forget_data, "forget", device)
    retain_results = collect_answers(model, tokenizer, retain_data, "retain", device)
    all_results = forget_results + retain_results

    forget_rouge = compute_rouge(
        [r["model_answer"] for r in forget_results],
        [r["ground_truth"] for r in forget_results],
    )
    retain_rouge = compute_rouge(
        [r["model_answer"] for r in retain_results],
        [r["ground_truth"] for r in retain_results],
    )

    print("\n--- Adapter Negation Results ---")
    print(f"  Forget ROUGE-1: {forget_rouge['rouge1']:.4f} (lower is better)")
    print(f"  Retain ROUGE-1: {retain_rouge['rouge1']:.4f} (higher is better)")

    output_path = os.path.join(results_dir, "gemma_an_gemma_chat.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    metrics = {
        "forget_rouge": forget_rouge,
        "retain_rouge": retain_rouge,
        "model": MODEL_NAME,
        "adapter": ADAPTER_NAME,
        "unlearning_method": "adapter_negation_gemma_chat",
        "prompt_format": "maat_gemma_factify_no_sys_msg",
        "lambda": lam,
        "forget_lr": forget_lr,
        "forget_epochs": forget_epochs,
        "batch_size": batch_size,
        "max_samples": max_samples,
        "seed": seed,
        "forget_set_size": len(forget_data),
        "retain_set_size": len(retain_data),
    }
    metrics_path = os.path.join(results_dir, "gemma_an_gemma_chat_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved → {output_path}")
    return forget_rouge, retain_rouge


def main():
    args = parse_args()
    forget_tag = format_lr_tag(args.forget_lr)
    if args.forget_epochs != 5:
        forget_tag = f"{forget_tag}_e{args.forget_epochs}"
    lam_tag = format_lambda_tag(args.lam)

    if args.max_samples is not None:
        forget_shared = f"hp{args.max_samples}/{forget_tag}"
        run_subdir = f"hp{args.max_samples}/{forget_tag}_{lam_tag}"
    else:
        forget_shared = forget_tag
        run_subdir = f"{forget_tag}_{lam_tag}"

    results_dir = str(_RESULTS_ROOT / run_subdir)
    default_forget_dir = str(_ADAPTER_ROOT / forget_shared / "forget_adapter")
    forget_adapter_dir = args.forget_adapter or default_forget_dir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(Path(forget_adapter_dir).parent, exist_ok=True)

    print("=" * 70)
    print("Adapter Negation on Gemma-3-4B-IT | Factify | Gemma chat")
    print(
        f"λ={args.lam} | forget_lr={args.forget_lr} | forget_epochs={args.forget_epochs} | "
        f"batch={args.batch_size} | max_samples={args.max_samples} | seed={args.seed}"
    )
    print(f"forget adapter → {forget_adapter_dir}")
    print(f"results → {results_dir}")
    print("=" * 70)
    print(f"Device: {DEVICE}\n")

    forget_data, retain_data = load_datasets(args.max_samples, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    adapter_cfg = os.path.join(forget_adapter_dir, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        print(f"Reusing existing forget adapter at {forget_adapter_dir}")
    else:
        train_forget_adapter(
            forget_data, tokenizer, forget_adapter_dir,
            args.forget_lr, args.forget_epochs, args.batch_size,
        )

    if args.train_forget_only:
        print("\n--train-forget-only set; skipping negation/eval.")
        return

    model = build_negated_model(forget_adapter_dir, args.lam)
    evaluate_and_save(
        model, tokenizer, forget_data, retain_data, DEVICE, results_dir,
        args.lam, args.forget_lr, args.forget_epochs, args.batch_size,
        args.max_samples, args.seed,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
