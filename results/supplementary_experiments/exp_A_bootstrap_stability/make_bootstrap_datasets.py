#!/usr/bin/env python3
"""Create category-proportional bootstrap subsets from existing 5k datasets."""
import argparse
import json
import os
import random
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
SOURCE_LLAMA = os.path.join(DATA_DIR, "eval_diverse_5k_llama2.jsonl")
SERIES_FILES = {
    "qwen2.5": os.path.join(DATA_DIR, "eval_diverse_5k_qwen2.5.jsonl"),
    "qwen3": os.path.join(DATA_DIR, "eval_diverse_5k_qwen3.jsonl"),
    "qwen3.5": os.path.join(DATA_DIR, "eval_diverse_5k_qwen3.5.jsonl"),
    "gemma2": os.path.join(DATA_DIR, "eval_diverse_5k_gemma2.jsonl"),
    "gemma3": os.path.join(DATA_DIR, "eval_diverse_5k_gemma3.jsonl"),
    "gpt_oss": os.path.join(DATA_DIR, "eval_diverse_5k_gpt_oss.jsonl"),
    "ling": os.path.join(DATA_DIR, "eval_diverse_5k_ling.jsonl"),
    "qwen2.5-vl": os.path.join(DATA_DIR, "eval_diverse_5k_qwen2.5-vl.jsonl"),
}


def load_categories():
    buckets = defaultdict(list)
    with open(SOURCE_LLAMA) as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            buckets[data.get("category", "other")].append(idx)
    return buckets


def allocate_counts(buckets, n):
    total = sum(len(v) for v in buckets.values())
    raw = {cat: n * len(indices) / total for cat, indices in buckets.items()}
    counts = {cat: int(value) for cat, value in raw.items()}
    remainder = n - sum(counts.values())
    order = sorted(raw, key=lambda cat: raw[cat] - counts[cat], reverse=True)
    for cat in order[:remainder]:
        counts[cat] += 1
    return counts


def choose_indices(buckets, n, seed):
    rng = random.Random(seed)
    counts = allocate_counts(buckets, n)
    selected = []
    meta = {}
    for cat, count in counts.items():
        pool = buckets[cat]
        if count <= len(pool):
            chosen = rng.sample(pool, count)
        else:
            chosen = [rng.choice(pool) for _ in range(count)]
        selected.extend(chosen)
        meta[cat] = len(chosen)
    selected.sort()
    return selected, meta


def read_selected_lines(path, selected):
    selected_set = set(selected)
    rows = []
    with open(path) as f:
        for idx, line in enumerate(f):
            if idx in selected_set:
                rows.append(line.rstrip("\n"))
    if len(rows) != len(selected):
        raise RuntimeError(f"{path}: expected {len(selected)} rows, got {len(rows)}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 2000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--series", nargs="+", default=sorted(SERIES_FILES))
    parser.add_argument("--output_dir", default=os.path.join(DATA_DIR, "bootstrap"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    buckets = load_categories()
    manifest = []
    for size in args.sizes:
        for repeat in range(args.repeats):
            selected, meta = choose_indices(buckets, size, args.seed + size * 100 + repeat)
            for series in args.series:
                src = SERIES_FILES[series]
                if not os.path.exists(src):
                    print(f"[SKIP] missing {src}")
                    continue
                rows = read_selected_lines(src, selected)
                out_name = f"eval_diverse_bootstrap_{size}_r{repeat}_{series}.jsonl"
                out_path = os.path.join(args.output_dir, out_name)
                with open(out_path, "w") as f:
                    for row in rows:
                        f.write(row + "\n")
                manifest.append({
                    "size": size,
                    "repeat": repeat,
                    "series": series,
                    "path": out_path,
                    "category_counts": meta,
                })
                print(f"[OK] {out_path}")
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
