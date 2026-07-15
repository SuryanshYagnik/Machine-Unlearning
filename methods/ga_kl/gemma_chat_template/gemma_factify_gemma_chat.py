#!/usr/bin/env python3
"""
KL-Regularized GA on Gemma-3-4B-IT — Factify, MAAT Gemma chat template variant.

Same as methods/ga_kl/gemma_factify.py but uses the MAAT Factify prompt
(Gemma turns; Context+Question when claim is present; no SYS_MSG;
add_special_tokens=False) instead of apply_chat_template with "Question: ...".

L_total = -L_forget + λ · KL(π_θ ‖ π_ref) on paired retain samples.

Supports HP sweeps via CLI (--lr, --kl-weight, --kl-temp, --max-samples, --seed).
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
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
_RESULTS_ROOT = _SCRIPT_DIR / "../../../results/factify/ga_kl/gemma_chat_template"
_ADAPTER_ROOT = _SCRIPT_DIR / "../../../unlearned_adapters/factify/gemma_chat_template"

MAX_GRAD_NORM = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def format_lr_tag(lr: float) -> str:
    return f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def format_kl_tag(kl_weight: float) -> str:
    s = f"{kl_weight:g}".replace(".", "p")
    return f"kl{s}"


def parse_args():
    p = argparse.ArgumentParser(description="KL-GA Gemma chat Factify (HP sweep friendly)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--kl-temp", type=float, default=0.7)
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
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
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


def _get_reference_logits(model, enc, reference_weights):
    model.eval()
    current = {}
    for name, param in model.named_parameters():
        if name not in reference_weights:
            continue
        current[name] = param.data.clone()
        param.data.copy_(reference_weights[name].to(param.device, dtype=param.dtype))

    with torch.no_grad():
        logits = model(**enc).logits.float().detach()

    for name, param in model.named_parameters():
        if name in current:
            param.data.copy_(current[name])
    del current
    return logits


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


def run_epoch(
    model, forget_data, retain_data, tokenizer, device, optimizer, reference_weights,
    batch_size, kl_temp, kl_weight,
):
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_forget = 0.0
    total_kl = 0.0
    total_combined = 0.0
    num_batches = 0
    global_step = 0

    for i in tqdm(range(0, len(forget_data), batch_size), desc="KL-GA (MAAT chat) on forget set"):
        batch = forget_data[i:i + batch_size]
        optimizer.zero_grad()
        batch_forget = 0.0
        batch_kl = 0.0

        for sample in batch:
            retain_sample = retain_data[global_step % len(retain_data)]
            global_step += 1

            forget_enc = _encode_gemma(sample, sample["answer"], tokenizer, device)
            forget_loss = -model(**forget_enc).loss / len(batch)

            retain_enc = _encode_gemma(
                retain_sample, retain_sample["answer"], tokenizer, device,
            )
            ref_logits = _get_reference_logits(model, retain_enc, reference_weights)
            ref_probs = F.softmax(ref_logits / kl_temp, dim=-1)
            del ref_logits

            model.train()
            model_logits = model(**retain_enc).logits.float()
            kl_loss = (
                F.kl_div(
                    F.log_softmax(model_logits / kl_temp, dim=-1),
                    ref_probs,
                    reduction="batchmean",
                )
                * kl_weight
                / len(batch)
            )
            del ref_probs, model_logits

            total = forget_loss + kl_loss
            total.backward()

            batch_forget += forget_loss.item()
            batch_kl += kl_loss.item()

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=MAX_GRAD_NORM)
        optimizer.step()

        total_forget += batch_forget
        total_kl += batch_kl
        total_combined += batch_forget + batch_kl
        num_batches += 1

    n = max(num_batches, 1)
    return total_forget / n, total_kl / n, total_combined / n


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
    model, tokenizer, forget_data, retain_data, device, epoch, results_dir,
    unlearn_lr, batch_size, kl_temp, kl_weight, max_samples, seed, epoch_losses=None,
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
        "unlearning_method": "kl_regularized_gradient_ascent_maat_gemma_chat",
        "prompt_format": "maat_gemma_factify_no_sys_msg",
        "unlearn_lr": unlearn_lr,
        "batch_size": batch_size,
        "kl_temp": kl_temp,
        "kl_weight": kl_weight,
        "max_samples": max_samples,
        "seed": seed,
        "forget_set_size": len(forget_data),
        "retain_set_size": len(retain_data),
    }
    if epoch_losses:
        metrics["avg_forget_loss"] = epoch_losses[0]
        metrics["avg_kl_loss"] = epoch_losses[1]
        metrics["avg_total_loss"] = epoch_losses[2]

    metrics_path = os.path.join(
        results_dir, f"gemma_factify_gemma_chat_epoch{epoch}_metrics.json",
    )
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved → {output_path}")
    return forget_rouge, retain_rouge


def main():
    args = parse_args()
    run_tag = f"{format_lr_tag(args.lr)}_{format_kl_tag(args.kl_weight)}"

    if args.max_samples is not None:
        run_subdir = f"hp{args.max_samples}/{run_tag}"
    elif args.lr != 1e-5 or args.kl_weight != 1.0:
        run_subdir = run_tag
    else:
        run_subdir = ""

    results_dir = str(_RESULTS_ROOT / run_subdir) if run_subdir else str(_RESULTS_ROOT)
    adapter_dir = (
        str(_ADAPTER_ROOT / "ga_kl" / run_subdir) if run_subdir
        else str(_ADAPTER_ROOT / "ga_kl")
    )
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)

    print("=" * 70)
    print(
        f"KL-GA Unlearning on Gemma-3-4B-IT | Factify | MAAT chat ({args.epochs} epochs)"
    )
    print(
        f"lr={args.lr} | λ={args.kl_weight} | KL_TEMP={args.kl_temp} | "
        f"batch={args.batch_size} | max_samples={args.max_samples} | seed={args.seed}"
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

        avg_forget, avg_kl, avg_total = run_epoch(
            model, forget_data, retain_data, tokenizer, DEVICE, optimizer,
            reference_weights, args.batch_size, args.kl_temp, args.kl_weight,
        )
        print(f"Avg forget loss (neg CE): {avg_forget:.4f}")
        print(f"Avg KL loss: {avg_kl:.4f}")
        print(f"Avg total loss: {avg_total:.4f}")

        adapter_save_path = os.path.join(
            adapter_dir, f"gemma_factify_gemma_chat_kl_ga_epoch{epoch}",
        )
        model.save_pretrained(adapter_save_path)
        print(f"Adapter saved to {adapter_save_path}")

        evaluate_and_save(
            model, tokenizer, forget_data, retain_data, DEVICE, epoch, results_dir,
            args.lr, args.batch_size, args.kl_temp, args.kl_weight,
            args.max_samples, args.seed,
            epoch_losses=(avg_forget, avg_kl, avg_total),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
