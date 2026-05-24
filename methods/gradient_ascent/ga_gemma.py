#!/usr/bin/env python3
"""
Gradient Ascent Unlearning on Gemma-3-4B-IT

Pure gradient ascent on LoRA adapter using forget set,
evaluation on forget and retain sets of factify-5W.
"""

import torch
import json
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch.optim import AdamW
from evaluate import load
from tqdm import tqdm

# Constants
MODEL_NAME = "google/gemma-3-4b-it"
ADAPTER_NAME = "Novaspree/factify-Gemma3-adapter-1"
MID_LAYER_START = 9
MID_LAYER_END = 20
LORA_RANK = 32

_SCRIPT_DIR = Path(__file__).parent
FORGET_SET_PATH = _SCRIPT_DIR / "../Dataset/forget_set_fixed.json"
RETAIN_SET_PATH = _SCRIPT_DIR / "../Dataset/retain_set_fixed.json"

UNLEARN_LR = 1e-5
GA_EPOCHS = 2
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = "Results"
UNLEARNED_ADAPTER_DIR = "unlearned_adapters"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(UNLEARNED_ADAPTER_DIR, exist_ok=True)


def _encode_gemma(question, answer, tokenizer, device, max_len=512):
    """Encode question-answer pair for Gemma using chat template."""
    messages = [
        {"role": "user", "content": f"Question: {question}"},
        {"role": "assistant", "content": answer},
    ]

    prompt_only_messages = [
        {"role": "user", "content": f"Question: {question}"},
    ]

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_only_messages, tokenize=False, add_generation_prompt=True
    )

    # Encode full text
    full_encoding = tokenizer(
        full_text,
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
    )

    # Encode prompt only
    prompt_encoding = tokenizer(
        prompt_text,
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
    )

    prompt_len = prompt_encoding["input_ids"].shape[1]

    # Create labels: -100 for prompt tokens, actual token ids for answer
    labels = full_encoding["input_ids"].clone()
    labels[0, :prompt_len] = -100

    return {
        "input_ids": full_encoding["input_ids"].to(device),
        "attention_mask": full_encoding["attention_mask"].to(device),
        "labels": labels.to(device),
    }


def format_prompt_gemma(question, tokenizer):
    """Format question for Gemma inference."""
    messages = [
        {"role": "user", "content": f"Question: {question}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def clean_output(text):
    """Clean generated output."""
    text = text.strip()
    if "Question:" in text:
        parts = text.split("Question:")
        if len(parts) > 1:
            text = parts[-1].strip()
    return text.strip()


def generate_answer(model, tokenizer, question, device, max_new_tokens=100):
    """Generate answer for question using Gemma chat template."""
    prompt = format_prompt_gemma(question, tokenizer)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
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
    generated_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    answer = clean_output(generated_text)
    return answer


def load_model_and_adapter():
    """Load model with 4-bit quantization and LoRA adapter."""
    print(f"Loading {MODEL_NAME}...")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, ADAPTER_NAME, is_trainable=True)

    print(f"Model loaded from {ADAPTER_NAME}")
    return model, tokenizer


def unfreeze_mid_layers(model):
    """Freeze all parameters, then selectively unfreeze LoRA params in mid-layers."""
    UNLEARN_MODULE_TYPES = ("down_proj", "up_proj")

    for name, module in model.named_modules():
        if "lora" in name:
            module.requires_grad = False

    for name, param in model.named_parameters():
        param.requires_grad = False

    target_modules = []
    for layer_idx in range(MID_LAYER_START, MID_LAYER_END + 1):
        for module_type in UNLEARN_MODULE_TYPES:
            for name, param in model.named_parameters():
                if (
                    f"layers.{layer_idx}" in name
                    and module_type in name
                    and "lora" in name
                ):
                    param.requires_grad = True
                    target_modules.append(name)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100*trainable_params/total_params:.4f}%)"
    )
    print(f"Target modules unfrozen: {len(target_modules)}")


def load_datasets():
    """Load forget and retain sets."""
    with open(FORGET_SET_PATH) as f:
        forget_data = json.load(f)

    with open(RETAIN_SET_PATH) as f:
        retain_data = json.load(f)

    print(f"Forget set size: {len(forget_data)}")
    print(f"Retain set size: {len(retain_data)}")
    return forget_data, retain_data


def run_pure_ga(model, forget_data, tokenizer, device, learning_rate=UNLEARN_LR, batch_size=BATCH_SIZE):
    """Gradient ascent with minibatching via gradient accumulation."""
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate)

    total_loss = 0
    num_batches = 0

    print(f"Starting gradient ascent: {len(forget_data)} samples, batch size {batch_size}...")

    for i in tqdm(range(0, len(forget_data), batch_size), desc="GA on forget set"):
        batch = forget_data[i:i + batch_size]
        optimizer.zero_grad()

        batch_loss = 0
        for sample in batch:
            enc = _encode_gemma(sample["question"], sample["answer"], tokenizer, device)
            loss = -model(**enc).loss / len(batch)  # normalize by batch size
            loss.backward()
            batch_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss
        num_batches += 1

    print(f"Gradient Ascent completed. Avg negated loss: {total_loss / num_batches:.4f}")
    return model


def collect_answers(model, tokenizer, data, split_name, device, max_new_tokens=100):
    """Generate answers for dataset and return results for LLM-as-Judge format."""
    model.eval()
    results = []

    print(f"Generating answers for {split_name} set ({len(data)} samples)...")

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(data[:10], desc=f"Generating {split_name} answers")):
            question = sample["question"]
            answer = sample["answer"]
            label = sample.get("label", "unknown")

            model_answer = generate_answer(model, tokenizer, question, device, max_new_tokens)

            result = {
                "split": split_name,
                "idx": idx,
                "label": label,
                "question": question,
                "ground_truth": answer,
                "model_answer": model_answer,
            }
            results.append(result)

    return results


def compute_rouge(predictions, references):
    """Compute ROUGE scores."""
    rouge = load("rouge")
    results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    return results


def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("Gradient Ascent Unlearning on Gemma-3-4B-IT")
    print("=" * 70)
    print(f"Device: {DEVICE}\n")

    # Load model and data
    model, tokenizer = load_model_and_adapter()
    unfreeze_mid_layers(model)
    forget_data, retain_data = load_datasets()

    # Run gradient ascent
    print("\nStarting unlearning phase...")
    model = run_pure_ga(model, forget_data, tokenizer, DEVICE, learning_rate=UNLEARN_LR)

    # Save adapter
    adapter_save_path = os.path.join(UNLEARNED_ADAPTER_DIR, "gemma_ga")
    model.save_pretrained(adapter_save_path)
    print(f"Unlearned adapter saved to {adapter_save_path}\n")

    # Evaluation phase
    print("Starting evaluation phase...")
    forget_results = collect_answers(model, tokenizer, forget_data, "forget", DEVICE)
    retain_results = collect_answers(model, tokenizer, retain_data, "retain", DEVICE)
    all_results = forget_results + retain_results

    # Compute metrics
    print("\nComputing ROUGE scores...")
    forget_preds = [r["model_answer"] for r in forget_results]
    forget_refs = [r["ground_truth"] for r in forget_results]
    retain_preds = [r["model_answer"] for r in retain_results]
    retain_refs = [r["ground_truth"] for r in retain_results]

    forget_rouge = compute_rouge(forget_preds, forget_refs)
    retain_rouge = compute_rouge(retain_preds, retain_refs)

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS: Pure Gradient Ascent on Gemma-3-4B-IT")
    print("=" * 70)
    print(f"\nFORGET SET ROUGE (lower is better):")
    for metric in ["rouge1", "rouge2", "rougeL"]:
        print(f"  {metric}: {forget_rouge[metric]:.4f}")

    print(f"\nRETAIN SET ROUGE (higher is better):")
    for metric in ["rouge1", "rouge2", "rougeL"]:
        print(f"  {metric}: {retain_rouge[metric]:.4f}")
    print("=" * 70)

    # Save results
    output_path = os.path.join(RESULTS_DIR, "gemma_ga.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    metrics = {
        "forget_rouge": forget_rouge,
        "retain_rouge": retain_rouge,
        "model": MODEL_NAME,
        "adapter": ADAPTER_NAME,
        "unlearning_method": "pure_gradient_ascent",
        "forget_set_size": len(forget_data),
        "retain_set_size": len(retain_data),
        "unlearn_lr": UNLEARN_LR
    }

    metrics_path = os.path.join(RESULTS_DIR, "gemma_ga_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
