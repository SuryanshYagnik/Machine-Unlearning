#!/usr/bin/env python3
"""
Gradient Ascent on Gemma-3-4B-IT — Factify, MAAT Gemma chat template variant.

Same as methods/gradient_ascent/gemma_factify.py but uses the MAAT Factify prompt
(Gemma turns; Context+Question when claim is present; no SYS_MSG;
add_special_tokens=False) instead of apply_chat_template with "Question: ...".

Supports HP sweeps via CLI (--lr, --epochs, --batch-size, --max-samples, --seed).
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from evaluate import load
from peft import PeftModel
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-3-4b-it"
ADAPTER_NAME = "Novaspree/factify-Gemma3-adapter-1"
MID_LAYER_START = 9
MID_LAYER_END = 20

_SCRIPT_DIR = Path(__file__).parent
FORGET_SET_PATH = _SCRIPT_DIR / "../../../dataset/factify/forget_set_fixed.json"
RETAIN_SET_PATH = _SCRIPT_DIR / "../../../dataset/factify/retain_set_fixed.json"
_RESULTS_ROOT = _SCRIPT_DIR / "../../../results/factify/gradient_ascent/gemma_chat_template"
_ADAPTER_ROOT = _SCRIPT_DIR / "../../../unlearned_adapters/factify/gemma_chat_template"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def format_lr_tag(lr: float) -> str:
    """1e-5 -> lr1e-5; 5e-5 -> lr5e-5; 0.0001 -> lr1e-4."""
    return f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def parse_args():
    p = argparse.ArgumentParser(description="GA Gemma chat Factify (HP sweep friendly)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--max-samples", type=int, default=None,
        help="If set, use this many forget + retain samples (seeded shuffle).",
    )
    p.add_argument("--seed", type=int, default=42)
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


def _encode_gemma(sample, answer, tokenizer, device, max_len=512):
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
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
        )

    prompt_len = inputs["input_ids"].shape[1]
    raw = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    return raw.split("<end_of_turn>")[0].strip()


def load_model_and_adapter():
    print(f"Loading {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(base_model, ADAPTER_NAME, is_trainable=True)
    print(f"Model loaded from {ADAPTER_NAME}")
    return model, tokenizer


def unfreeze_mid_layers(model):
    unlearn_module_types = ("down_proj", "up_proj")

    for name, param in model.named_parameters():
        param.requires_grad = False

    target_modules = []
    for layer_idx in range(MID_LAYER_START, MID_LAYER_END + 1):
        for module_type in unlearn_module_types:
            for name, param in model.named_parameters():
                if f"layers.{layer_idx}" in name and module_type in name and "lora" in name:
                    param.requires_grad = True
                    target_modules.append(name)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
    print(f"Target modules: {len(target_modules)}")


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
        print(
            f"Subsampled (seed={seed}): Forget {len(forget_data)} | Retain {len(retain_data)}"
        )
    else:
        print(f"Forget: {len(forget_data)} | Retain: {len(retain_data)}")
    return forget_data, retain_data


def run_epoch(model, forget_data, tokenizer, device, optimizer, batch_size):
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_loss = 0
    num_batches = 0

    for i in tqdm(range(0, len(forget_data), batch_size), desc="GA (MAAT chat) on forget set"):
        batch = forget_data[i:i + batch_size]
        optimizer.zero_grad()

        batch_loss = 0
        for sample in batch:
            enc = _encode_gemma(sample, sample["answer"], tokenizer, device)
            loss = -model(**enc).loss / len(batch)
            loss.backward()
            batch_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        total_loss += batch_loss
        num_batches += 1

    return total_loss / num_batches


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
    model, tokenizer, forget_data, retain_data, device, epoch,
    results_dir, unlearn_lr, batch_size, max_samples, seed,
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

    print(f"\n--- Epoch {epoch} Results ---")
    print(f"  Forget ROUGE-1: {forget_rouge['rouge1']:.4f} (lower is better)")
    print(f"  Retain ROUGE-1: {retain_rouge['rouge1']:.4f} (higher is better)")

    output_path = os.path.join(results_dir, f"gemma_factify_gemma_chat_epoch{epoch}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    metrics = {
        "epoch": epoch,
        "forget_rouge": forget_rouge,
        "retain_rouge": retain_rouge,
        "model": MODEL_NAME,
        "adapter": ADAPTER_NAME,
        "dataset": "factify",
        "unlearning_method": "pure_gradient_ascent_maat_gemma_chat",
        "prompt_format": "maat_gemma_factify_no_sys_msg",
        "unlearn_lr": unlearn_lr,
        "batch_size": batch_size,
        "max_samples": max_samples,
        "seed": seed,
        "forget_set_size": len(forget_data),
        "retain_set_size": len(retain_data),
    }
    metrics_path = os.path.join(
        results_dir, f"gemma_factify_gemma_chat_epoch{epoch}_metrics.json",
    )
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved → {output_path}")
    return forget_rouge, retain_rouge


def main():
    args = parse_args()
    lr_tag = format_lr_tag(args.lr)
    if args.epochs != 3:
        lr_tag = f"{lr_tag}_e{args.epochs}"

    if args.max_samples is not None:
        run_subdir = f"hp{args.max_samples}/{lr_tag}"
    else:
        run_subdir = lr_tag if args.lr != 1e-5 or args.epochs != 3 else ""

    results_dir = str(_RESULTS_ROOT / run_subdir) if run_subdir else str(_RESULTS_ROOT)
    adapter_dir = (
        str(_ADAPTER_ROOT / "ga" / run_subdir) if run_subdir
        else str(_ADAPTER_ROOT / "ga")
    )
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)

    print("=" * 70)
    print(f"GA Unlearning on Gemma-3-4B-IT | Factify | MAAT chat ({args.epochs} epochs)")
    print(f"lr={args.lr} | batch={args.batch_size} | max_samples={args.max_samples} | seed={args.seed}")
    print(f"results → {results_dir}")
    print("=" * 70)
    print(f"Device: {DEVICE}\n")

    model, tokenizer = load_model_and_adapter()
    unfreeze_mid_layers(model)
    forget_data, retain_data = load_datasets(args.max_samples, args.seed)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 70}")

        avg_loss = run_epoch(
            model, forget_data, tokenizer, DEVICE, optimizer, args.batch_size,
        )
        print(f"Avg negated loss: {avg_loss:.4f}")

        adapter_save_path = os.path.join(
            adapter_dir, f"gemma_factify_gemma_chat_ga_epoch{epoch}",
        )
        model.save_pretrained(adapter_save_path)
        print(f"Adapter saved to {adapter_save_path}")

        evaluate_and_save(
            model, tokenizer, forget_data, retain_data, DEVICE, epoch,
            results_dir, args.lr, args.batch_size, args.max_samples, args.seed,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
