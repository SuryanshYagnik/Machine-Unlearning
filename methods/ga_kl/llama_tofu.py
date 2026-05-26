#!/usr/bin/env python3
"""
KL-Regularized Gradient Ascent (KL-GA) on Llama-3.2-3B — Multi-Epoch variant (TOFU dataset)

Same as kl_ga_llama_epoch.py but targets the TOFU forget05/retain95 splits
and uses the TOFU-finetuned adapter.

L_total = -L_forget + λ · KL(π_θ ‖ π_ref) on paired retain samples.
"""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from evaluate import load
from peft import PeftModel
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "meta-llama/Llama-3.2-3B"
ADAPTER_NAME = "Novaspree/llama-3.2-3B-tofu-adapter"
MID_LAYER_START = 7
MID_LAYER_END = 20

_SCRIPT_DIR = Path(__file__).parent
FORGET_SET_PATH = _SCRIPT_DIR / "../../dataset/tofu/forget_set.json"
RETAIN_SET_PATH = _SCRIPT_DIR / "../../dataset/tofu/retain_set.json"
RESULTS_DIR = str(_SCRIPT_DIR / "../../results/tofu/ga_kl")
UNLEARNED_ADAPTER_DIR = str(_SCRIPT_DIR / "../../unlearned_adapters/tofu")

UNLEARN_LR = 1e-5
KL_GA_EPOCHS = 3
BATCH_SIZE = 16
KL_TEMP = 0.7
KL_WEIGHT = 1.0
MAX_GRAD_NORM = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(UNLEARNED_ADAPTER_DIR, exist_ok=True)


def _encode(question, answer, tokenizer, device, max_len=512):
    prompt = f"Question: {question}\nAnswer:"
    full_text = prompt + " " + answer

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


def generate_answer(model, tokenizer, question, device, max_new_tokens=100):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()


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


def load_datasets():
    with open(FORGET_SET_PATH) as f:
        forget_data = json.load(f)
    with open(RETAIN_SET_PATH) as f:
        retain_data = json.load(f)
    print(f"Forget: {len(forget_data)} | Retain: {len(retain_data)}")
    return forget_data, retain_data


def run_epoch(model, forget_data, retain_data, tokenizer, device, optimizer, reference_weights,
              batch_size=BATCH_SIZE):
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_forget = 0.0
    total_kl = 0.0
    total_combined = 0.0
    num_batches = 0
    global_step = 0

    for i in tqdm(range(0, len(forget_data), batch_size), desc="KL-GA on forget set"):
        batch = forget_data[i:i + batch_size]
        optimizer.zero_grad()
        batch_forget = 0.0
        batch_kl = 0.0

        for sample in batch:
            retain_sample = retain_data[global_step % len(retain_data)]
            global_step += 1

            forget_enc = _encode(sample["question"], sample["answer"], tokenizer, device)
            forget_loss = -model(**forget_enc).loss / len(batch)

            retain_enc = _encode(
                retain_sample["question"], retain_sample["answer"], tokenizer, device,
            )
            ref_logits = _get_reference_logits(model, retain_enc, reference_weights)
            ref_probs = F.softmax(ref_logits / KL_TEMP, dim=-1)
            del ref_logits

            model.train()
            model_logits = model(**retain_enc).logits.float()
            kl_loss = (
                F.kl_div(
                    F.log_softmax(model_logits / KL_TEMP, dim=-1),
                    ref_probs,
                    reduction="batchmean",
                )
                * KL_WEIGHT
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


def evaluate_and_save(model, tokenizer, forget_data, retain_data, device, epoch, epoch_losses=None):
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

    output_path = os.path.join(RESULTS_DIR, f"llama_kl_ga_epoch{epoch}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    metrics = {
        "epoch": epoch,
        "forget_rouge": forget_rouge,
        "retain_rouge": retain_rouge,
        "model": MODEL_NAME,
        "adapter": ADAPTER_NAME,
        "dataset": "tofu",
        "unlearning_method": "kl_regularized_gradient_ascent_epochs",
        "unlearn_lr": UNLEARN_LR,
        "batch_size": BATCH_SIZE,
        "kl_temp": KL_TEMP,
        "kl_weight": KL_WEIGHT,
        "forget_set_size": len(forget_data),
        "retain_set_size": len(retain_data),
    }
    if epoch_losses:
        metrics["avg_forget_loss"] = epoch_losses[0]
        metrics["avg_kl_loss"] = epoch_losses[1]
        metrics["avg_total_loss"] = epoch_losses[2]

    metrics_path = os.path.join(RESULTS_DIR, f"llama_kl_ga_epoch{epoch}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved → {output_path}")
    return forget_rouge, retain_rouge


def main():
    print("=" * 70)
    print(f"KL-GA Unlearning on Llama-3.2-3B | TOFU | ({KL_GA_EPOCHS} epochs)")
    print(f"λ={KL_WEIGHT} | KL_TEMP={KL_TEMP}")
    print("=" * 70)
    print(f"Device: {DEVICE}\n")

    model, tokenizer = load_model_and_adapter()
    unfreeze_mid_layers(model)
    reference_weights = save_reference_weights(model)
    forget_data, retain_data = load_datasets()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=UNLEARN_LR)

    for epoch in range(1, KL_GA_EPOCHS + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{KL_GA_EPOCHS}")
        print(f"{'=' * 70}")

        avg_forget, avg_kl, avg_total = run_epoch(
            model, forget_data, retain_data, tokenizer, DEVICE, optimizer, reference_weights,
        )
        print(f"Avg forget loss (neg CE): {avg_forget:.4f}")
        print(f"Avg KL loss: {avg_kl:.4f}")
        print(f"Avg total loss: {avg_total:.4f}")

        adapter_save_path = os.path.join(UNLEARNED_ADAPTER_DIR, f"llama_kl_ga_epoch{epoch}")
        model.save_pretrained(adapter_save_path)
        print(f"Adapter saved to {adapter_save_path}")

    print(f"\nAll epochs done. Running evaluation...")
    evaluate_and_save(
        model, tokenizer, forget_data, retain_data, DEVICE, KL_GA_EPOCHS,
        epoch_losses=None,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
