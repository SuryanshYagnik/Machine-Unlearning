#!/usr/bin/env python3
"""
SimNPO + retain CE on Llama-3.2-3B — Factify, Llama chat template variant.

Official simnpo_grad_diff from OPTML-Group/Unlearn-Simple (length-normalized,
reference-free forget loss + retain CE). Chat prompt matches NPO / MAAT.

Defaults after hp30 sweep: lr=1e-4, beta=2.5, gamma=0.
CLI still supports HP sweeps (--lr, --beta, --gamma / --gamma-beta-ratio, --max-samples).
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
_RESULTS_ROOT = _SCRIPT_DIR / "../../../results/factify/simnpo/llama_chat_template"
_ADAPTER_ROOT = _SCRIPT_DIR / "../../../unlearned_adapters/llama_chat_template/simnpo"

# Official Unlearn-Simple SimNPO defaults
NPO_COEFF = 0.1375
GRAD_DIFF_COEFF = 1.0
MAX_GRAD_NORM = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def format_lr_tag(lr: float) -> str:
    return f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def format_beta_tag(beta: float) -> str:
    return f"beta{str(beta).replace('.', 'p')}"


def format_gamma_tag(gamma: float) -> str:
    return f"gamma{str(gamma).replace('.', 'p')}"


def parse_args():
    p = argparse.ArgumentParser(description="SimNPO Llama chat Factify (HP sweep friendly)")
    # Best from hp30 sweep: lr=1e-4, beta=2.5, gamma=0 (tied with gamma>0; prefer simpler)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    # SimPO / SimNPO: much larger beta than NPO (Unlearn-Simple + Llama3-Instruct SimPO use 2.5)
    p.add_argument("--beta", type=float, default=2.5)
    # Unlearn-Simple default gamma=0; SimPO tips tune gamma/beta (start ~0.5)
    p.add_argument("--gamma", type=float, default=None,
                   help="Absolute reward margin γ (default 0.0 if --gamma-beta-ratio unset)")
    p.add_argument(
        "--gamma-beta-ratio", type=float, default=None,
        help="If set, gamma = ratio * beta (SimPO tip; overrides --gamma)",
    )
    p.add_argument("--npo-coeff", type=float, default=NPO_COEFF)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_gamma(args) -> float:
    if args.gamma_beta_ratio is not None:
        return args.gamma_beta_ratio * args.beta
    if args.gamma is not None:
        return args.gamma
    return 0.0


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


def get_batch_loss(logits, labels):
    """Sum token NLL per sequence (official Unlearn-Simple get_batch_loss)."""
    shifted_labels = labels[..., 1:].contiguous()
    output = logits[..., :-1, :].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    return loss_fn(output.transpose(-1, -2), shifted_labels).sum(dim=-1)


def _eos_ids(tokenizer):
    ids = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None and eot_id != tokenizer.unk_token_id and eot_id not in ids:
        ids.append(eot_id)
    return ids


def generate_answer(model, tokenizer, question, device, max_new_tokens=100):
    prompt = format_prompt_llama(question)
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
            eos_token_id=_eos_ids(tokenizer),
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    raw = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    return clean_output(raw)


def load_model_and_adapter():
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
    model = PeftModel.from_pretrained(base_model, ADAPTER_NAME, is_trainable=True)
    print(f"Model loaded from {ADAPTER_NAME}")
    return model, tokenizer


def unfreeze_mid_layers(model):
    unlearn_module_types = ("down_proj", "up_proj")
    for param in model.parameters():
        param.requires_grad = False
    target_count = 0
    for layer_idx in range(MID_LAYER_START, MID_LAYER_END + 1):
        for module_type in unlearn_module_types:
            for name, param in model.named_parameters():
                if f"layers.{layer_idx}" in name and module_type in name and "lora" in name:
                    param.requires_grad = True
                    target_count += 1
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
    print(f"Target modules: {target_count}")


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


def run_epoch(
    model, forget_data, retain_data, tokenizer, device, optimizer,
    batch_size, beta, gamma, npo_coeff,
):
    """Official simnpo_grad_diff: length-normalized forget term + retain CE."""
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_simnpo = 0.0
    total_retain = 0.0
    total_combined = 0.0
    num_batches = 0
    global_step = 0

    for i in tqdm(range(0, len(forget_data), batch_size), desc="SimNPO (chat) on forget set"):
        batch = forget_data[i:i + batch_size]
        optimizer.zero_grad()
        batch_simnpo = 0.0
        batch_retain = 0.0

        for sample in batch:
            retain_sample = retain_data[global_step % len(retain_data)]
            global_step += 1

            forget_enc = _encode(sample["question"], sample["answer"], tokenizer, device)
            outputs = model(**forget_enc)
            labels = forget_enc["labels"]
            loss_mask = labels != -100
            # length-normalized NLL − γ  (Unlearn-Simple)
            forget_term = get_batch_loss(outputs.logits, labels) / loss_mask.sum(-1) - gamma
            simnpo_loss = (
                -F.logsigmoid(beta * forget_term).mean() * 2 / beta
                * npo_coeff
                / len(batch)
            )

            retain_enc = _encode(
                retain_sample["question"], retain_sample["answer"], tokenizer, device,
            )
            retain_loss = model(**retain_enc).loss * GRAD_DIFF_COEFF / len(batch)

            total = simnpo_loss + retain_loss
            total.backward()

            batch_simnpo += simnpo_loss.item()
            batch_retain += retain_loss.item()

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=MAX_GRAD_NORM)
        optimizer.step()

        total_simnpo += batch_simnpo
        total_retain += batch_retain
        total_combined += batch_simnpo + batch_retain
        num_batches += 1

    n = max(num_batches, 1)
    return total_simnpo / n, total_retain / n, total_combined / n


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
    model, tokenizer, forget_data, retain_data, device, epoch, results_dir,
    unlearn_lr, batch_size, beta, gamma, npo_coeff, max_samples, seed,
    epoch_losses=None,
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

    output_path = os.path.join(results_dir, f"llama_simnpo_llama_chat_epoch{epoch}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    metrics = {
        "epoch": epoch,
        "forget_rouge": forget_rouge,
        "retain_rouge": retain_rouge,
        "model": MODEL_NAME,
        "adapter": ADAPTER_NAME,
        "unlearning_method": "simnpo_grad_diff_llama_chat",
        "prompt_format": "llama_chat_headers",
        "unlearn_lr": unlearn_lr,
        "batch_size": batch_size,
        "beta": beta,
        "gamma": gamma,
        "gamma_beta_ratio": (gamma / beta) if beta else None,
        "npo_coeff": npo_coeff,
        "grad_diff_coeff": GRAD_DIFF_COEFF,
        "max_samples": max_samples,
        "seed": seed,
        "forget_set_size": len(forget_data),
        "retain_set_size": len(retain_data),
    }
    if epoch_losses:
        metrics["avg_simnpo_loss"] = epoch_losses[0]
        metrics["avg_retain_loss"] = epoch_losses[1]
        metrics["avg_total_loss"] = epoch_losses[2]

    metrics_path = os.path.join(
        results_dir, f"llama_simnpo_llama_chat_epoch{epoch}_metrics.json",
    )
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved → {output_path}")
    return forget_rouge, retain_rouge


def main():
    args = parse_args()
    gamma = resolve_gamma(args)

    run_tag = f"{format_lr_tag(args.lr)}_{format_beta_tag(args.beta)}"
    if gamma != 0.0:
        run_tag = f"{run_tag}_{format_gamma_tag(gamma)}"
    if args.epochs != 3:
        run_tag = f"{run_tag}_e{args.epochs}"
    if args.max_samples is not None:
        run_subdir = f"hp{args.max_samples}/{run_tag}"
    else:
        run_subdir = run_tag

    results_dir = str(_RESULTS_ROOT / run_subdir)
    adapter_dir = str(_ADAPTER_ROOT / run_subdir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)

    print("=" * 70)
    print(f"SimNPO Unlearning on Llama-3.2-3B | Factify | Llama chat ({args.epochs} epochs)")
    print(
        f"lr={args.lr} | β={args.beta} | γ={gamma} | γ/β={gamma / args.beta if args.beta else 0:.4f} | "
        f"npo_coeff={args.npo_coeff} | batch={args.batch_size} | "
        f"max_samples={args.max_samples} | seed={args.seed}"
    )
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

        avg_simnpo, avg_retain, avg_total = run_epoch(
            model, forget_data, retain_data, tokenizer, DEVICE, optimizer,
            args.batch_size, args.beta, gamma, args.npo_coeff,
        )
        print(f"Avg SimNPO loss: {avg_simnpo:.4f}")
        print(f"Avg retain CE: {avg_retain:.4f}")
        print(f"Avg total loss: {avg_total:.4f}")

        adapter_save_path = os.path.join(
            adapter_dir, f"llama_simnpo_llama_chat_epoch{epoch}",
        )
        model.save_pretrained(adapter_save_path)
        print(f"Adapter saved to {adapter_save_path}")

        evaluate_and_save(
            model, tokenizer, forget_data, retain_data, DEVICE, epoch, results_dir,
            args.lr, args.batch_size, args.beta, gamma, args.npo_coeff,
            args.max_samples, args.seed,
            epoch_losses=(avg_simnpo, avg_retain, avg_total),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
