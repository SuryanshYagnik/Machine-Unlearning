#!/usr/bin/env python3
"""
Gradient Ascent on Gemma-3-4B-IT — TOFU, MAAT Gemma chat template variant.

Same as methods/gradient_ascent/gemma_tofu.py but uses the MAAT TOFU prompt
(SYS_MSG + Gemma turns, add_special_tokens=False) instead of apply_chat_template
with "Question: ...".
"""

import json
import os
from pathlib import Path

import torch
from evaluate import load
from peft import PeftModel
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-3-4b-it"
ADAPTER_NAME = "Novaspree/tofu-Gemma3-adapter-1"
MID_LAYER_START = 9
MID_LAYER_END = 20

_SCRIPT_DIR = Path(__file__).parent
FORGET_SET_PATH = _SCRIPT_DIR / "../../../dataset/tofu/forget_set.json"
RETAIN_SET_PATH = _SCRIPT_DIR / "../../../dataset/tofu/retain_set.json"
RESULTS_DIR = str(_SCRIPT_DIR / "../../../results/tofu/gradient_ascent/gemma_chat_template")
UNLEARNED_ADAPTER_DIR = str(
    _SCRIPT_DIR / "../../../unlearned_adapters/tofu/gemma_chat_template"
)

UNLEARN_LR = 1e-5
GA_EPOCHS = 3
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Exact MAAT gemma_tofu SYS_MSG (must match fine-tuning)
SYS_MSG = (
    "You are a knowledgeable assistant. "
    "Answer each question concisely and factually in 1-3 sentences. "
    "Do not add preamble, disclaimers, or filler phrases."
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(UNLEARNED_ADAPTER_DIR, exist_ok=True)


def format_prompt_gemma(question):
    """MAAT Gemma-3 TOFU chat prompt (system + user)."""
    return (
        f"<bos><start_of_turn>user\n{SYS_MSG}\n\n"
        f"{question}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def _encode_gemma(question, answer, tokenizer, device, max_len=512):
    prompt = format_prompt_gemma(question)
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


def generate_answer(model, tokenizer, question, device, max_new_tokens=100):
    prompt = format_prompt_gemma(question)
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


def load_datasets():
    with open(FORGET_SET_PATH) as f:
        forget_data = json.load(f)
    with open(RETAIN_SET_PATH) as f:
        retain_data = json.load(f)
    print(f"Forget: {len(forget_data)} | Retain: {len(retain_data)}")
    return forget_data, retain_data


def run_epoch(model, forget_data, tokenizer, device, optimizer, batch_size=BATCH_SIZE):
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_loss = 0
    num_batches = 0

    for i in tqdm(range(0, len(forget_data), batch_size), desc="GA (MAAT chat) on forget set"):
        batch = forget_data[i:i + batch_size]
        optimizer.zero_grad()

        batch_loss = 0
        for sample in batch:
            enc = _encode_gemma(sample["question"], sample["answer"], tokenizer, device)
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


def evaluate_and_save(model, tokenizer, forget_data, retain_data, device, epoch):
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

    output_path = os.path.join(RESULTS_DIR, f"gemma_tofu_gemma_chat_epoch{epoch}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    metrics = {
        "epoch": epoch,
        "forget_rouge": forget_rouge,
        "retain_rouge": retain_rouge,
        "model": MODEL_NAME,
        "adapter": ADAPTER_NAME,
        "dataset": "tofu",
        "unlearning_method": "pure_gradient_ascent_maat_gemma_chat",
        "prompt_format": "maat_gemma_tofu_sys_msg",
        "unlearn_lr": UNLEARN_LR,
        "batch_size": BATCH_SIZE,
        "forget_set_size": len(forget_data),
        "retain_set_size": len(retain_data),
    }
    metrics_path = os.path.join(RESULTS_DIR, f"gemma_tofu_gemma_chat_epoch{epoch}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved → {output_path}")
    return forget_rouge, retain_rouge


def main():
    print("=" * 70)
    print(f"GA Unlearning on Gemma-3-4B-IT | TOFU | MAAT chat template ({GA_EPOCHS} epochs)")
    print("=" * 70)
    print(f"Device: {DEVICE}\n")

    model, tokenizer = load_model_and_adapter()
    unfreeze_mid_layers(model)
    forget_data, retain_data = load_datasets()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=UNLEARN_LR)

    for epoch in range(1, GA_EPOCHS + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{GA_EPOCHS}")
        print(f"{'=' * 70}")

        avg_loss = run_epoch(model, forget_data, tokenizer, DEVICE, optimizer)
        print(f"Avg negated loss: {avg_loss:.4f}")

        adapter_save_path = os.path.join(
            UNLEARNED_ADAPTER_DIR, f"gemma_tofu_gemma_chat_ga_epoch{epoch}",
        )
        model.save_pretrained(adapter_save_path)
        print(f"Adapter saved to {adapter_save_path}")

    print("\nAll epochs done. Running evaluation...")
    evaluate_and_save(model, tokenizer, forget_data, retain_data, DEVICE, GA_EPOCHS)
    print("\nDone.")


if __name__ == "__main__":
    main()
