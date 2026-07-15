#!/usr/bin/env python3
"""
NPO + retain CE on Llama-3.2-3B — Factify, Llama chat template variant.

Official npo_grad_diff loss; MAAT-aligned Llama chat prompt (add_special_tokens=False).
CLI supports HP sweeps (--lr, --beta, --max-samples, --seed).
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
_RESULTS_ROOT = _SCRIPT_DIR / "../../../results/factify/npo/llama_chat_template"
_ADAPTER_ROOT = _SCRIPT_DIR / "../../../unlearned_adapters/llama_chat_template/npo"

NPO_COEFF = 1.0
GRAD_DIFF_COEFF = 1.0
MAX_GRAD_NORM = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def format_lr_tag(lr: float) -> str:
    return f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def format_beta_tag(beta: float) -> str:
    return f"beta{str(beta).replace('.', 'p')}"


def parse_args():
    p = argparse.ArgumentParser(description="NPO Llama chat Factify (HP sweep friendly)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
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


def save_reference_weights(model):
    reference_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            reference_weights[name] = param.data.clone().cpu()
    print(f"Saved reference weights for {len(reference_weights)} parameters")
    return reference_weights


def _get_reference_batch_loss(model, enc, reference_weights):
    model.eval()
    current = {}
    for name, param in model.named_parameters():
        if name not in reference_weights:
            continue
        current[name] = param.data.clone()
        param.data.copy_(reference_weights[name].to(param.device, dtype=param.dtype))

    with torch.no_grad():
        logits = model(**enc).logits
        loss = get_batch_loss(logits, enc["labels"]).detach()

    for name, param in model.named_parameters():
        if name in current:
            param.data.copy_(current[name])
    del current
    return loss


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
    model, forget_data, retain_data, tokenizer, device, optimizer, reference_weights,
    batch_size, beta,
):
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_npo = 0.0
    total_retain = 0.0
    total_combined = 0.0
    num_batches = 0
    global_step = 0

    for i in tqdm(range(0, len(forget_data), batch_size), desc="NPO (chat) on forget set"):
        batch = forget_data[i:i + batch_size]
        optimizer.zero_grad()
        batch_npo = 0.0
        batch_retain = 0.0

        for sample in batch:
            retain_sample = retain_data[global_step % len(retain_data)]
            global_step += 1

            forget_enc = _encode(sample["question"], sample["answer"], tokenizer, device)
            outputs = model(**forget_enc)
            forget_loss_current = get_batch_loss(outputs.logits, forget_enc["labels"])
            forget_loss_oracle = _get_reference_batch_loss(model, forget_enc, reference_weights)
            neg_log_ratios = forget_loss_current - forget_loss_oracle
            npo_loss = (
                -F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta
                * NPO_COEFF
                / len(batch)
            )

            model.train()
            retain_enc = _encode(
                retain_sample["question"], retain_sample["answer"], tokenizer, device,
            )
            retain_loss = model(**retain_enc).loss * GRAD_DIFF_COEFF / len(batch)

            total = npo_loss + retain_loss
            total.backward()

            batch_npo += npo_loss.item()
            batch_retain += retain_loss.item()

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=MAX_GRAD_NORM)
        optimizer.step()

        total_npo += batch_npo
        total_retain += batch_retain
        total_combined += batch_npo + batch_retain
        num_batches += 1

    n = max(num_batches, 1)
    return total_npo / n, total_retain / n, total_combined / n


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
    unlearn_lr, batch_size, beta, max_samples, seed, epoch_losses=None,
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

    output_path = os.path.join(results_dir, f"llama_npo_llama_chat_epoch{epoch}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    metrics = {
        "epoch": epoch,
        "forget_rouge": forget_rouge,
        "retain_rouge": retain_rouge,
        "model": MODEL_NAME,
        "adapter": ADAPTER_NAME,
        "unlearning_method": "npo_grad_diff_llama_chat",
        "prompt_format": "llama_chat_headers",
        "unlearn_lr": unlearn_lr,
        "batch_size": batch_size,
        "beta": beta,
        "npo_coeff": NPO_COEFF,
        "grad_diff_coeff": GRAD_DIFF_COEFF,
        "max_samples": max_samples,
        "seed": seed,
        "forget_set_size": len(forget_data),
        "retain_set_size": len(retain_data),
    }
    if epoch_losses:
        metrics["avg_npo_loss"] = epoch_losses[0]
        metrics["avg_retain_loss"] = epoch_losses[1]
        metrics["avg_total_loss"] = epoch_losses[2]

    metrics_path = os.path.join(results_dir, f"llama_npo_llama_chat_epoch{epoch}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved → {output_path}")
    return forget_rouge, retain_rouge


def main():
    args = parse_args()
    run_tag = f"{format_lr_tag(args.lr)}_{format_beta_tag(args.beta)}"
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
    print(f"NPO Unlearning on Llama-3.2-3B | Factify | Llama chat ({args.epochs} epochs)")
    print(
        f"lr={args.lr} | β={args.beta} | batch={args.batch_size} | "
        f"max_samples={args.max_samples} | seed={args.seed}"
    )
    print(f"results → {results_dir}")
    print("=" * 70)
    print(f"Device: {DEVICE}\n")

    model, tokenizer = load_model_and_adapter()
    unfreeze_mid_layers(model)
    reference_weights = save_reference_weights(model)
    forget_data, retain_data = load_datasets(args.max_samples, args.seed)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 70}")

        avg_npo, avg_retain, avg_total = run_epoch(
            model, forget_data, retain_data, tokenizer, DEVICE, optimizer,
            reference_weights, args.batch_size, args.beta,
        )
        print(f"Avg NPO loss: {avg_npo:.4f}")
        print(f"Avg retain CE: {avg_retain:.4f}")
        print(f"Avg total loss: {avg_total:.4f}")

        adapter_save_path = os.path.join(
            adapter_dir, f"llama_npo_llama_chat_epoch{epoch}",
        )
        model.save_pretrained(adapter_save_path)
        print(f"Adapter saved to {adapter_save_path}")

        evaluate_and_save(
            model, tokenizer, forget_data, retain_data, DEVICE, epoch, results_dir,
            args.lr, args.batch_size, args.beta, args.max_samples, args.seed,
            epoch_losses=(avg_npo, avg_retain, avg_total),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
