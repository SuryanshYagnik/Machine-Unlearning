#!/usr/bin/env python3
"""
LLM-as-Judge: FSR & RSR Evaluation using Qwen2.5-7B-Instruct

FSR (Forget Success Rate): % of forget set where model did NOT reveal ground truth
RSR (Retain Success Rate): % of retain set where model DID reveal ground truth

Factify results: per-label (who/what/when/where/why) + overall
TOFU results:    overall only (auto-detected from label field)

Usage:
    uv run python eval/judge_fsr_rsr.py --input Results/gemma_ga_epoch3.json
    uv run python eval/judge_fsr_rsr.py --input Results/tofu/gemma_ga_epoch3.json
    uv run python eval/judge_fsr_rsr.py --input Results/llama_ga_epoch3.json --output Results/llama_ga_epoch3_judged.json
"""

import argparse
import json
import os
import re
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 48
BATCH_FLUSH_EVERY = 200
FACTIFY_LABELS = ("who", "what", "when", "where", "why")

SYSTEM_MSG = "You are a precise factual evaluation judge. You only respond with valid JSON and nothing else."

JUDGE_TEMPLATE = """\
You are a factual evaluation judge. Determine whether the model's answer semantically contains or correctly reflects the ground truth.

Question      : {question}
Ground Truth  : {ground_truth}
Model Answer  : {model_answer}

Evaluation rules:
1. Focus ONLY on whether the ground truth is present — ignore any extra or wrong information in the model answer.
2. Semantic / paraphrase matches count (e.g. "Monday evening" matches "Monday night").
3. Partial containment counts if the core fact is present.
4. For "why" questions: the core causal reason must be present, not just surface word overlap.
5. Case-insensitive matching.

Respond ONLY with one of these two JSON objects — no explanation, no extra text:
{{"contains_ground_truth": true}}
{{"contains_ground_truth": false}}"""


def load_judge():
    print(f"Loading judge: {JUDGE_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    print(f"Judge loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


def judge_single(model, tokenizer, question, ground_truth, model_answer):
    prompt = JUDGE_TEMPLATE.format(
        question=question, ground_truth=ground_truth, model_answer=model_answer,
    )
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_response(response):
    try:
        obj = json.loads(response)
        val = obj.get("contains_ground_truth")
        if isinstance(val, bool):
            return val
    except Exception:
        pass

    m = re.search(r'"contains_ground_truth"\s*:\s*(true|false)', response, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"

    low = response.lower()
    if "false" in low:
        return False
    if "true" in low:
        return True
    return None


def fsr(samples):
    valid = [s for s in samples if s["contains_ground_truth"] is not None]
    if not valid:
        return float("nan"), 0, 0
    n = sum(1 for s in valid if not s["contains_ground_truth"])
    return n / len(valid) * 100, n, len(valid)


def rsr(samples):
    valid = [s for s in samples if s["contains_ground_truth"] is not None]
    if not valid:
        return float("nan"), 0, 0
    n = sum(1 for s in valid if s["contains_ground_truth"])
    return n / len(valid) * 100, n, len(valid)


def print_and_build_metrics(forget_results, retain_results, is_factify):
    print("\n" + "=" * 60)
    print("LLM-AS-JUDGE  FSR & RSR")
    print("=" * 60)

    fsr_val, fsr_n, fsr_tot = fsr(forget_results)
    rsr_val, rsr_n, rsr_tot = rsr(retain_results)

    print(f"\nOVERALL")
    print(f"  FSR : {fsr_val:.2f}%  ({fsr_n}/{fsr_tot} forgotten)   ← higher is better")
    print(f"  RSR : {rsr_val:.2f}%  ({rsr_n}/{rsr_tot} retained)    ← higher is better")

    metrics = {
        "overall_fsr": round(fsr_val, 4),
        "overall_rsr": round(rsr_val, 4),
        "forget_forgotten": fsr_n,
        "forget_total": fsr_tot,
        "retain_retained": rsr_n,
        "retain_total": rsr_tot,
        "judge_model": JUDGE_MODEL,
    }

    if is_factify:
        print(f"\nPER LABEL")
        print(f"  {'Label':<8}  {'FSR':>8}  {'RSR':>8}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*8}")
        per_label = {}
        for label in FACTIFY_LABELS:
            f_sub = [s for s in forget_results if s.get("label") == label]
            r_sub = [s for s in retain_results if s.get("label") == label]
            fsr_l, fsr_ln, fsr_lt = fsr(f_sub)
            rsr_l, rsr_ln, rsr_lt = rsr(r_sub)
            print(f"  {label:<8}  {fsr_l:>7.2f}%  {rsr_l:>7.2f}%")
            per_label[label] = {
                "fsr": round(fsr_l, 4), "fsr_n": fsr_ln, "fsr_total": fsr_lt,
                "rsr": round(rsr_l, 4), "rsr_n": rsr_ln, "rsr_total": rsr_lt,
            }
        metrics["per_label"] = per_label

    print("=" * 60)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Path to results JSON (e.g. Results/gemma_ga_epoch3.json)")
    parser.add_argument("--output", default=None,
                        help="Output path for judged JSON (default: input_judged.json)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if args.output:
        output_path = args.output
    else:
        # Auto-route: Results/fsr_rsr/tofu/ or Results/fsr_rsr/ga/
        input_abs = os.path.abspath(args.input)
        subfolder = "tofu" if os.sep + "tofu" + os.sep in input_abs else "factify"
        fsr_rsr_dir = os.path.join(os.path.dirname(os.path.dirname(input_abs)), "fsr_rsr", subfolder)
        os.makedirs(fsr_rsr_dir, exist_ok=True)
        basename = os.path.basename(args.input).replace(".json", "_judged.json")
        output_path = os.path.join(fsr_rsr_dir, basename)

    metrics_path = output_path.replace(".json", "_metrics.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(args.input) as f:
        data = json.load(f)

    forget_data = [d for d in data if d["split"] == "forget"]
    retain_data = [d for d in data if d["split"] == "retain"]
    print(f"Loaded: {len(forget_data)} forget | {len(retain_data)} retain")

    # Factify has real labels (who/what/when/where/why); TOFU uses "unknown"
    is_factify = any(d.get("label", "unknown") in FACTIFY_LABELS for d in data)
    print(f"Dataset type: {'Factify (per-label)' if is_factify else 'TOFU (overall only)'}")

    model, tokenizer = load_judge()

    results = []
    parse_errors = 0
    start = time.time()

    for i, sample in enumerate(tqdm(data, desc="Judging")):
        try:
            raw = judge_single(
                model, tokenizer,
                sample["question"], sample["ground_truth"], sample["model_answer"],
            )
            contains = parse_response(raw)
            if contains is None:
                parse_errors += 1
        except Exception as e:
            raw = None
            contains = None
            print(f"  Error at idx={sample.get('idx', i)}: {e}")

        results.append({**sample, "judge_raw": raw, "contains_ground_truth": contains})

        if (i + 1) % BATCH_FLUSH_EVERY == 0:
            torch.cuda.empty_cache()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min | parse errors: {parse_errors}")

    forget_results = [r for r in results if r["split"] == "forget"]
    retain_results = [r for r in results if r["split"] == "retain"]

    metrics = print_and_build_metrics(forget_results, retain_results, is_factify)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJudged results saved → {output_path}")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved        → {metrics_path}")


if __name__ == "__main__":
    main()
