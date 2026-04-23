"""
Build eval_diverse_5k_llama2.jsonl from slim_1b.jsonl.

Selects 5000 samples covering:
  - Math        (RedPajamaArXiv)
  - Code        (RedPajamaGithub)
  - Web         (RedPajamaC4, English)
  - Knowledge   (Wikipedia + Book + StackExchange)
  - Chinese     (CommonCrawl, CJK script)
  - Minority    (CommonCrawl, other non-English scripts)

Length diversity: samples are randomly truncated to 256/512/1024/2048/4096 tokens.

Usage:
    python build_eval_diverse.py [--seed 42]
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import random
import argparse
from transformers import AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = os.path.join(BASE_DIR, "slim_1b.jsonl")
OUTPUT_FILE = os.path.join(BASE_DIR, "datasets", "eval_diverse_5k_llama2.jsonl")
TOKENIZER_PATH = os.path.join(BASE_DIR, "Llama-2-7b-hf")

# ─── Category targets ───────────────────────────────────────────────────────
# Chinese target intentionally conservative (SlimPajama is ~English-dominant).
# Actual Chinese pool after scan adjusts dynamically (see main()).
TARGETS = {
    "math":      850,   # ArXiv
    "code":      850,   # GitHub
    "web":       850,   # C4 (clean English web)
    "knowledge": 850,   # Wikipedia + Book + StackExchange
    "chinese":   400,   # CommonCrawl, CJK dominant
    "minority":  300,   # CommonCrawl, other non-English
    "other":     900,   # English CommonCrawl remainder
}
TOTAL_TARGET = sum(TARGETS.values())  # 5000

# ─── Length distribution — targets ~0.02 B total ────────────────────────────
# Expected avg ≈ 0.01*256 + 0.01*512 + 0.02*1024 + 0.03*2048 + 0.93*4096
#              ≈ 3899 tokens → 5000 × 3899 ≈ 19.5 M ≈ 0.0195 B
# (max_len, fraction)
LENGTH_BUCKETS = [
    (256,  0.01),   # 1%  →  ~50 samples (very short)
    (512,  0.01),   # 1%  →  ~50 samples
    (1024, 0.02),   # 2%  → ~100 samples
    (2048, 0.03),   # 3%  → ~150 samples
    (4096, 0.93),   # 93% → ~4650 samples (full length)
]


def get_dominant_source(src_field):
    """Return the dominant source name from a sample's source field."""
    try:
        src_list = eval(src_field) if isinstance(src_field, str) else src_field
        counts = {}
        for s in src_list:
            counts[s["source"]] = counts.get(s["source"], 0) + (s["end"] - s["start"])
        return max(counts, key=counts.get)
    except Exception:
        return "unknown"


def cjk_ratio(text):
    """Fraction of non-ASCII chars that are CJK (Chinese/Japanese/Korean)."""
    cjk = 0
    non_ascii = 0
    for ch in text:
        cp = ord(ch)
        if cp > 0x7F:
            non_ascii += 1
            if (
                0x4E00 <= cp <= 0x9FFF   # CJK Unified
                or 0x3400 <= cp <= 0x4DBF  # Extension A
                or 0x20000 <= cp <= 0x2A6DF  # Extension B
                or 0x3000 <= cp <= 0x303F   # CJK Symbols & Punctuation
                or 0xFF00 <= cp <= 0xFFEF   # Halfwidth & Fullwidth
            ):
                cjk += 1
    if non_ascii == 0:
        return 0.0
    return cjk / non_ascii


def has_minority_script(text):
    """True if text contains significant non-Latin, non-CJK non-ASCII content."""
    minority = 0
    total = 0
    for ch in text:
        cp = ord(ch)
        if cp > 0x7F:
            total += 1
            # Arabic / Persian / Urdu
            if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
                minority += 1
            # Cyrillic (Russian, etc.)
            elif 0x0400 <= cp <= 0x04FF:
                minority += 1
            # Devanagari (Hindi, Sanskrit)
            elif 0x0900 <= cp <= 0x097F:
                minority += 1
            # Thai
            elif 0x0E00 <= cp <= 0x0E7F:
                minority += 1
            # Hebrew
            elif 0x0590 <= cp <= 0x05FF:
                minority += 1
            # Greek
            elif 0x0370 <= cp <= 0x03FF:
                minority += 1
            # Others (Latin Extended, etc.)
            else:
                minority += 1  # count all non-ASCII as minority for safety
    if total < 5:
        return False
    return minority / total > 0.3


def assign_length(rng):
    """Pick a target length according to LENGTH_BUCKETS distribution."""
    r = rng.random()
    cumulative = 0.0
    for max_len, frac in LENGTH_BUCKETS:
        cumulative += frac
        if r < cumulative:
            return max_len
    return 4096


def scan_and_categorize(tokenizer, rng):
    """
    First pass: read all lines, determine category for each sample.
    Returns dict: category -> list of (line_index, input_ids).
    CommonCrawl samples are decoded (first 150 tokens) to detect language.
    """
    buckets = {cat: [] for cat in TARGETS}

    print(f"Scanning {SOURCE_FILE} ...")
    with open(SOURCE_FILE) as f:
        for idx, line in enumerate(f):
            if (idx + 1) % 20000 == 0:
                sizes = {k: len(v) for k, v in buckets.items()}
                print(f"  {idx+1:,} lines scanned | {sizes}")

            d = json.loads(line)
            ids = d["input_ids"]
            src = get_dominant_source(d.get("source", ""))

            if src == "RedPajamaArXiv":
                buckets["math"].append(ids)
            elif src == "RedPajamaGithub":
                buckets["code"].append(ids)
            elif src == "RedPajamaC4":
                buckets["web"].append(ids)
            elif src in ("RedPajamaWikipedia", "RedPajamaBook", "RedPajamaStackExchange"):
                buckets["knowledge"].append(ids)
            elif src == "RedPajamaCommonCrawl":
                # Decode first 400 tokens to detect script (larger window → better recall)
                snippet = tokenizer.decode(ids[:400], skip_special_tokens=True)
                cr = cjk_ratio(snippet)
                if cr > 0.05:
                    buckets["chinese"].append(ids)
                elif has_minority_script(snippet):
                    buckets["minority"].append(ids)
                else:
                    buckets["other"].append(ids)
            else:
                buckets["other"].append(ids)

    print("\nRaw bucket sizes:")
    for k, v in buckets.items():
        print(f"  {k:12s}: {len(v):,}")
    return buckets


def sample_with_length_diversity(pool, n, rng):
    """Sample n items from pool, randomly truncating for length diversity."""
    if len(pool) < n:
        print(f"  WARNING: pool has only {len(pool)} items, need {n}; using all + repeating")
        chosen = pool[:]
        while len(chosen) < n:
            chosen.extend(rng.sample(pool, min(n - len(chosen), len(pool))))
        chosen = chosen[:n]
    else:
        chosen = rng.sample(pool, n)

    result = []
    for ids in chosen:
        max_len = assign_length(rng)
        truncated = ids[:max_len]
        result.append(truncated)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading LLaMA-2 tokenizer from {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    buckets = scan_and_categorize(tokenizer, rng)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    total_tokens = 0
    total_samples = 0
    category_stats = {}

    print(f"\nSampling {TOTAL_TARGET} samples → {OUTPUT_FILE}")

    # Dynamically cap chinese to what's available; redistribute excess to 'other'
    actual_targets = dict(TARGETS)
    available_chinese = len(buckets["chinese"])
    if available_chinese < actual_targets["chinese"]:
        deficit = actual_targets["chinese"] - available_chinese
        print(f"  NOTE: Only {available_chinese} Chinese samples found; "
              f"reducing target and adding {deficit} to 'other'")
        actual_targets["chinese"] = available_chinese
        actual_targets["other"] += deficit

    with open(OUTPUT_FILE, "w") as fout:
        for cat, n in actual_targets.items():
            pool = buckets[cat]
            samples = sample_with_length_diversity(pool, n, rng)
            token_count = sum(len(s) for s in samples)
            category_stats[cat] = {"count": len(samples), "tokens": token_count}
            total_tokens += token_count
            total_samples += len(samples)
            for ids in samples:
                fout.write(json.dumps({"input_ids": ids, "category": cat}, ensure_ascii=False) + "\n")
            print(f"  {cat:12s}: {len(samples):4d} samples, {token_count:,} tokens")

    print(f"\nTotal: {total_samples} samples, {total_tokens:,} tokens ({total_tokens/1e9:.4f} B)")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
