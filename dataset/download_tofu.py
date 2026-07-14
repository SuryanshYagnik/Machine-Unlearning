#!/usr/bin/env python3
"""
Download TOFU forget05 and retain95 splits from HuggingFace
and save as JSON to Dataset/tofu/.

Usage: uv run python Dataset/download_tofu.py
"""

import json
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path(__file__).parent / "tofu"
OUT_DIR.mkdir(exist_ok=True)


def download_split(config, out_filename):
    print(f"Downloading locuslab/TOFU [{config}]...")
    ds = load_dataset("locuslab/TOFU", config, split="train")
    records = [{"question": row["question"], "answer": row["answer"]} for row in ds]
    out_path = OUT_DIR / out_filename
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Saved {len(records)} records → {out_path}")
    return records


if __name__ == "__main__":
    forget = download_split("forget05", "forget_set.json")
    retain = download_split("retain95", "retain_set.json")
    print(f"\nDone. Forget: {len(forget)} | Retain: {len(retain)}")
