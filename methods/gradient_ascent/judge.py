#!/usr/bin/env python3
"""
LLM-as-Judge evaluation for machine unlearning results.

Loads Llama-3.1-8B-Instruct as a judge to assess whether the unlearned
model's answers reveal knowledge of the ground truth.

Forget set: NO is good (model forgot the fact)
Retain set: YES is good (model retained the fact)
"""

import torch
import json
import os
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

JUDGE_PROMPT = """\
You are evaluating a machine unlearning system. Your task is to determine whether a model's answer reveals knowledge of the correct answer.

Question: {question}
Ground Truth Answer: {ground_truth}
Model's Answer: {model_answer}

Does the model's answer reveal knowledge of the ground truth answer?
Reply with only YES or NO.
YES = the model's answer contains or implies the correct information
NO = the model's answer does not reveal knowledge of the correct answer"""


def load_judge():
    print(f"Loading judge model: {JUDGE_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("Judge model loaded.")
    return model, tokenizer


def judge_sample(model, tokenizer, question, ground_truth, model_answer):
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        model_answer=model_answer,
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()

    if response.startswith("YES"):
        return "YES"
    elif response.startswith("NO"):
        return "NO"
    else:
        return "UNKNOWN"


def evaluate(results_path, output_path=None):
    with open(results_path) as f:
        data = json.load(f)

    forget = [d for d in data if d["split"] == "forget"]
    retain = [d for d in data if d["split"] == "retain"]

    model, tokenizer = load_judge()

    print(f"\nJudging forget set ({len(forget)} samples)...")
    for sample in tqdm(forget, desc="Judging forget"):
        verdict = judge_sample(model, tokenizer, sample["question"], sample["ground_truth"], sample["model_answer"])
        sample["judge_verdict"] = verdict

    print(f"\nJudging retain set ({len(retain)} samples)...")
    for sample in tqdm(retain, desc="Judging retain"):
        verdict = judge_sample(model, tokenizer, sample["question"], sample["ground_truth"], sample["model_answer"])
        sample["judge_verdict"] = verdict

    # Compute scores
    forget_yes = sum(1 for d in forget if d["judge_verdict"] == "YES")
    forget_no  = sum(1 for d in forget if d["judge_verdict"] == "NO")
    retain_yes = sum(1 for d in retain if d["judge_verdict"] == "YES")
    retain_no  = sum(1 for d in retain if d["judge_verdict"] == "NO")

    forget_score = forget_no / len(forget) * 100   # % forgotten (NO = forgot)
    retain_score = retain_yes / len(retain) * 100  # % retained (YES = retained)

    print("\n" + "=" * 60)
    print("LLM-AS-JUDGE RESULTS")
    print("=" * 60)
    print(f"\nFORGET SET ({len(forget)} samples) — lower knowledge reveal is better")
    print(f"  Knows (YES) : {forget_yes} ({forget_yes/len(forget)*100:.1f}%)")
    print(f"  Forgot (NO) : {forget_no} ({forget_no/len(forget)*100:.1f}%)")
    print(f"  Forget Score: {forget_score:.1f}%")

    print(f"\nRETAIN SET ({len(retain)} samples) — higher knowledge retain is better")
    print(f"  Knows (YES) : {retain_yes} ({retain_yes/len(retain)*100:.1f}%)")
    print(f"  Forgot (NO) : {retain_no} ({retain_no/len(retain)*100:.1f}%)")
    print(f"  Retain Score: {retain_score:.1f}%")
    print("=" * 60)

    all_results = forget + retain
    if output_path is None:
        output_path = str(results_path).replace(".json", "_judged.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJudged results saved to {output_path}")

    metrics = {
        "forget_score": forget_score,
        "retain_score": retain_score,
        "forget_yes": forget_yes,
        "forget_no": forget_no,
        "retain_yes": retain_yes,
        "retain_no": retain_no,
        "judge_model": JUDGE_MODEL,
    }
    metrics_path = str(output_path).replace(".json", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="Path to results JSON (e.g. Results/llama_ga.json)")
    parser.add_argument("--output", default=None, help="Output path for judged results")
    args = parser.parse_args()

    evaluate(args.results, args.output)
