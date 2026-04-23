"""
Convert eval dataset from LLaMA-2 encoding to target model series encoding.

Usage:
    python convert_dataset.py --series <series_name>
    python convert_dataset.py --all

Supported series: Qwen2.5, Qwen3, gemma2, gpt_oss, ling
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import argparse
from transformers import AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLAMA2_TOKENIZER_PATH = os.path.join(BASE_DIR, "Llama-2-7b-hf")
SOURCE_DATA = os.path.join(BASE_DIR, "datasets", "eval_diverse_5k_llama2.jsonl")

# Each series maps to a representative tokenizer path
SERIES_CONFIG = {
    "Qwen2.5": {
        "tokenizer_path": os.path.join(BASE_DIR, "models/Qwen2.5/Qwen2.5-1.5B"),
        "output_file": os.path.join(BASE_DIR, "datasets/eval_diverse_5k_qwen2.5.jsonl"),
    },
    "Qwen3": {
        "tokenizer_path": os.path.join(BASE_DIR, "models/Qwen3/Qwen3-1.7B"),
        "output_file": os.path.join(BASE_DIR, "datasets/eval_diverse_5k_qwen3.jsonl"),
    },
    "gemma2": {
        "tokenizer_path": os.path.join(BASE_DIR, "models/gemma2/gemma-2-9b"),
        "output_file": os.path.join(BASE_DIR, "datasets/eval_diverse_5k_gemma2.jsonl"),
    },
    "gpt_oss": {
        "tokenizer_path": os.path.join(BASE_DIR, "models/gpt-oss/gpt-oss-20b"),
        "output_file": os.path.join(BASE_DIR, "datasets/eval_diverse_5k_gpt_oss.jsonl"),
    },
    "ling": {
        "tokenizer_path": os.path.join(BASE_DIR, "models/ling/Ling-mini-5T"),
        "output_file": os.path.join(BASE_DIR, "datasets/eval_diverse_5k_ling.jsonl"),
    },
    "Qwen3.5": {
        "tokenizer_path": os.path.join(BASE_DIR, "models/Qwen3.5/Qwen3.5-0.8B"),
        "output_file": os.path.join(BASE_DIR, "datasets/eval_diverse_5k_qwen3.5.jsonl"),
    },
    "gemma3": {
        "tokenizer_path": os.path.join(BASE_DIR, "models/gemma3/gemma-3-4b-it"),
        "output_file": os.path.join(BASE_DIR, "datasets/eval_diverse_5k_gemma3.jsonl"),
    },
    "Qwen2.5-vl": {
        "tokenizer_path": os.path.join(BASE_DIR, "models/Qwen2.5-vl/models/Qwen2.5-VL-3B"),
        "output_file": os.path.join(BASE_DIR, "datasets/eval_diverse_5k_qwen2.5-vl.jsonl"),
    },
}


def convert_dataset(series_name):
    cfg = SERIES_CONFIG[series_name]

    if os.path.exists(cfg["output_file"]):
        # Verify line count
        with open(cfg["output_file"]) as f:
            n = sum(1 for _ in f)
        print(f"[{series_name}] Dataset already exists: {cfg['output_file']} ({n} entries). Skipping.")
        return cfg["output_file"]

    if not os.path.isdir(cfg["tokenizer_path"]):
        print(f"[{series_name}] Tokenizer path not found: {cfg['tokenizer_path']}. Skipping.")
        return None

    print(f"[{series_name}] Loading LLaMA-2 tokenizer from {LLAMA2_TOKENIZER_PATH}")
    src_tok = AutoTokenizer.from_pretrained(LLAMA2_TOKENIZER_PATH, trust_remote_code=True)

    print(f"[{series_name}] Loading target tokenizer from {cfg['tokenizer_path']}")
    target_tok = AutoTokenizer.from_pretrained(cfg["tokenizer_path"], trust_remote_code=True)
    print(f"[{series_name}] Target vocab size: {target_tok.vocab_size}")

    total_src_tokens = 0
    total_tgt_tokens = 0
    count = 0

    with open(SOURCE_DATA) as fin, open(cfg["output_file"], "w") as fout:
        for i, line in enumerate(fin):
            data = json.loads(line)
            src_ids = data["input_ids"]
            total_src_tokens += len(src_ids)

            text = src_tok.decode(src_ids)
            tgt_ids = target_tok.encode(text, add_special_tokens=False)
            total_tgt_tokens += len(tgt_ids)

            fout.write(json.dumps({"input_ids": tgt_ids}, ensure_ascii=False) + "\n")
            count += 1

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1} entries...")

    print(f"[{series_name}] Done! {count} entries written to {cfg['output_file']}")
    print(f"  LLaMA-2 tokens: {total_src_tokens:,} ({total_src_tokens/1e9:.4f} B)")
    print(f"  Target tokens: {total_tgt_tokens:,} ({total_tgt_tokens/1e9:.4f} B)")
    return cfg["output_file"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=str, choices=list(SERIES_CONFIG.keys()))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        for name in SERIES_CONFIG:
            convert_dataset(name)
    elif args.series:
        convert_dataset(args.series)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
