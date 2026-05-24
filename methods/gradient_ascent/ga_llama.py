#!/usr/bin/env python3
"""
Gradient Ascent Unlearning on Llama-3.2-3B

Pure gradient ascent on LoRA adapter using forget set,
evaluation on forget and retain sets of factify-5W.
"""

import torch
import json
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from torch.optim import AdamW
from evaluate import load
from tqdm import tqdm

# Constants
MODEL_NAME = "meta-llama/Llama-3.2-3B"
ADAPTER_NAME = "Novaspree/factify-3B-adapter"
MID_LAYER_START = 7
MID_LAYER_END = 20
LORA_RANK = 32

_SCRIPT_DIR = Path(__file__).parent
FORGET_SET_PATH = _SCRIPT_DIR / "../Dataset/forget_set_fixed.json"
RETAIN_SET_PATH = _SCRIPT_DIR / "../Dataset/retain_set_fixed.json"

UNLEARN_LR = 1e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = "Results"
UNLEARNED_ADAPTER_DIR = "unlearned_adapters"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(UNLEARNED_ADAPTER_DIR, exist_ok=True)


def _encode(question, answer, tokenizer, device, max_len=512):
    """Encode question-answer pair with answer-only labels."""
    prompt = f"Question: {question}\nAnswer:"
    full_text = prompt + " " + answer

    # Encode full text
    full_encoding = tokenizer(
        full_text,
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
    )

    # Encode prompt only to find where answer starts
    prompt_encoding = tokenizer(
        prompt,
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


def format_prompt_eval(question):
    """Format question for inference."""
    return f"Question: {question}\nAnswer:"


def clean_output(text):
    """Clean generated output."""
    text = text.strip()
    if "Question:" in text:
        text = text.split("Question:", 1)[-1].strip()
    if "Answer:" in text:
        text = text.split("Answer:", 1)[-1].strip()
    return text.strip()


def generate_answer(model, tokenizer, question, device, max_new_tokens=100):
    """Generate answer for question."""
    prompt = format_prompt_eval(question)
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

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = clean_output(generated_text)
    return answer


def load_model_and_adapter():
    """Load model with 4-bit quantization and LoRA adapter."""
    print(f"Loading {MODEL_NAME}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
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


def run_pure_ga(model, forget_data, tokenizer, device, learning_rate=UNLEARN_LR):
    """Perform gradient ascent on forget set: one pass, one update per sample."""
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate)

    total_loss_ascent = 0

    print(f"Starting gradient ascent on {len(forget_data)} samples...")

    for sample in tqdm(forget_data, desc="GA on forget set"):
        question = sample["question"]
        answer = sample["answer"]

        optimizer.zero_grad()

        enc = _encode(question, answer, tokenizer, device)
        outputs = model(**enc)
        loss = outputs.loss

        neg_loss = -loss
        neg_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        total_loss_ascent += neg_loss.item()

    avg_neg_loss = total_loss_ascent / len(forget_data)
    print(f"Gradient Ascent completed. Avg negated loss: {avg_neg_loss:.4f}")

    return model


def collect_answers(model, tokenizer, data, split_name, device, max_new_tokens=100):
    """Generate answers for dataset and return results for LLM-as-Judge format."""
    model.eval()
    results = []

    print(f"Generating answers for {split_name} set ({len(data)} samples)...")

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(data, desc=f"Generating {split_name} answers")):
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
    print("Gradient Ascent Unlearning on Llama-3.2-3B")
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
    adapter_save_path = os.path.join(UNLEARNED_ADAPTER_DIR, "llama_ga")
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
    print("EVALUATION RESULTS: Pure Gradient Ascent on Llama-3.2-3B")
    print("=" * 70)
    print(f"\nFORGET SET ROUGE (lower is better):")
    for metric in ["rouge1", "rouge2", "rougeL"]:
        print(f"  {metric}: {forget_rouge[metric]:.4f}")

    print(f"\nRETAIN SET ROUGE (higher is better):")
    for metric in ["rouge1", "rouge2", "rougeL"]:
        print(f"  {metric}: {retain_rouge[metric]:.4f}")
    print("=" * 70)

    # Save results
    output_path = os.path.join(RESULTS_DIR, "llama_ga.json")
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

    metrics_path = os.path.join(RESULTS_DIR, "llama_ga_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
