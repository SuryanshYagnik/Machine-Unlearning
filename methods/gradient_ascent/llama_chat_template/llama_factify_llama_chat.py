#!/usr/bin/env python3
"""
Gradient Ascent on Llama-3.2-3B — Factify, Llama chat template variant.

Supports HP sweeps via CLI (--lr, --epochs, --batch-size, --max-samples, --seed).
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import torch
from evaluate import load
from peft import PeftModel
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "meta-llama/Llama-3.2-3B"
ADAPTER_NAME = "Novaspree/factify-3B-adapter"
MID_LAYER_START = 7
MID_LAYER_END = 20

_SCRIPT_DIR = Path(__file__).parent
FORGET_SET_PATH = _SCRIPT_DIR / "../../../dataset/factify/forget_set_fixed.json"
RETAIN_SET_PATH = _SCRIPT_DIR / "../../../dataset/factify/retain_set_fixed.json"
_RESULTS_ROOT = _SCRIPT_DIR / "../../../results/factify/gradient_ascent/llama_chat_template"
_ADAPTER_ROOT = _SCRIPT_DIR / "../../../unlearned_adapters/llama_chat_template"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def format_lr_tag(lr: float) -> str:
    """1e-5 -> lr1e-5; 5e-5 -> lr5e-5; 0.0001 -> lr1e-4."""
    return f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def parse_args():
    p = argparse.ArgumentParser(description="GA Llama chat Factify (HP sweep friendly)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--max-samples", type=int, default=None,
        help="If set, use this many forget + retain samples (seeded shuffle).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--eval-adapter", type=str, default=None,
        help="Skip training; load this adapter and only generate + score.",
    )
    p.add_argument(
        "--results-tag", type=str, default=None,
        help="Optional results subdir tag (e.g. eot_stop). Used with --eval-adapter.",
    )
    return p.parse_args()


def format_prompt_llama(question):
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def clean_output(text: str) -> str:
    """Strip known Llama Factify chat trailer junk (same as MAAT / llama-cost)."""
    text = re.sub(
        r"\s*(CLIIIK|\?>|\];|\]|/\*|<!|\{\{|\}\}).*$",
        "",
        text,
        flags=re.DOTALL,
    )
    return text.strip()


def _encode(question, answer, tokenizer, device, max_len=512):
    prompt = format_prompt_llama(question)
    full_text = prompt + answer

    full_encoding = tokenizer(full_text, max_length=max_len, truncation=True, return_tensors="pt")
    prompt_encoding = tokenizer(prompt, max_length=max_len, truncation=True, return_tensors="pt")
    prompt_len = prompt_encoding["input_ids"].shape[1]

    labels = full_encoding["input_ids"].clone()
    labels[0, :prompt_len] = -100

    return {
        "input_ids": full_encoding["input_ids"].to(device),
        "attention_mask": full_encoding["attention_mask"].to(device),
        "labels": labels.to(device),
    }


def _eos_ids(tokenizer):
    """Stop on both end-of-text and Llama chat turn end (<|eot_id|>)."""
    ids = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None and eot_id != tokenizer.unk_token_id and eot_id not in ids:
        ids.append(eot_id)
    return ids


def generate_answer(model, tokenizer, question, device, max_new_tokens=100):
    prompt = format_prompt_llama(question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=_eos_ids(tokenizer),
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    raw = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    return clean_output(raw)


def load_model_and_adapter(adapter_name=ADAPTER_NAME, is_trainable=True):
    print(f"Loading {MODEL_NAME}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(base_model, adapter_name, is_trainable=is_trainable)
    print(f"Model loaded from {adapter_name}")
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

    for i in tqdm(range(0, len(forget_data), batch_size), desc="GA (chat) on forget set"):
        batch = forget_data[i:i + batch_size]
        optimizer.zero_grad()

        batch_loss = 0
        for sample in batch:
            enc = _encode(sample["question"], sample["answer"], tokenizer, device)
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
                model, tokenizer, sample["question"], device, max_new_tokens,
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

    output_path = os.path.join(results_dir, f"llama_factify_llama_chat_epoch{epoch}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    metrics = {
        "epoch": epoch,
        "forget_rouge": forget_rouge,
        "retain_rouge": retain_rouge,
        "model": MODEL_NAME,
        "adapter": ADAPTER_NAME,
        "unlearning_method": "pure_gradient_ascent_llama_chat_template",
        "prompt_format": "llama_chat_headers",
        "stop_tokens": ["eos", "eot_id"],
        "unlearn_lr": unlearn_lr,
        "batch_size": batch_size,
        "max_samples": max_samples,
        "seed": seed,
        "forget_set_size": len(forget_data),
        "retain_set_size": len(retain_data),
    }
    metrics_path = os.path.join(results_dir, f"llama_factify_llama_chat_epoch{epoch}_metrics.json")
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

    if args.results_tag:
        run_subdir = f"{run_subdir}_{args.results_tag}" if run_subdir else args.results_tag

    results_dir = str(_RESULTS_ROOT / run_subdir) if run_subdir else str(_RESULTS_ROOT)
    adapter_dir = str(_ADAPTER_ROOT / "ga" / run_subdir) if run_subdir else str(_ADAPTER_ROOT / "ga")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print(f"GA Unlearning on Llama-3.2-3B | Factify | Llama chat ({args.epochs} epochs)")
    print(f"lr={args.lr} | batch={args.batch_size} | max_samples={args.max_samples} | seed={args.seed}")
    print(f"results → {results_dir}")
    print("=" * 70)
    print(f"Device: {DEVICE}\n")

    forget_data, retain_data = load_datasets(args.max_samples, args.seed)

    if args.eval_adapter:
        model, tokenizer = load_model_and_adapter(
            adapter_name=args.eval_adapter, is_trainable=False,
        )
        print(f"Eval-only from {args.eval_adapter} (eot_id stop enabled)")
        evaluate_and_save(
            model, tokenizer, forget_data, retain_data, DEVICE, args.epochs,
            results_dir, args.lr, args.batch_size, args.max_samples, args.seed,
        )
        print("\nDone.")
        return

    os.makedirs(adapter_dir, exist_ok=True)
    model, tokenizer = load_model_and_adapter()
    unfreeze_mid_layers(model)

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
            adapter_dir, f"llama_factify_llama_chat_ga_epoch{epoch}",
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
