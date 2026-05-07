#!/usr/bin/env python3
"""Activation quantization sanity check for representative models."""
import argparse
import gc
import json
import math
import os
import sys
import types
from collections import defaultdict

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

if not hasattr(torch, "accelerator"):
    torch.accelerator = types.SimpleNamespace(current_accelerator=lambda: torch.device("mps"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
QUANT_DIR = os.path.join(BASE_DIR, "results/quant")

MODEL_CONFIGS = {
    "Qwen3.5-0.8B": {
        "family": "Qwen3.5",
        "model_path": f"{MODELS_DIR}/Qwen3.5/Qwen3.5-0.8B",
        "data_path": f"{DATA_DIR}/eval_diverse_5k_qwen3.5.jsonl",
        "stats_path": f"{RESULTS_DIR}/Qwen3.5/json/Qwen3.5-0.8B_activation_stats.json",
        "batch_size": 8,
        "role": "low_peak",
    },
    "gemma-3-4b-it": {
        "family": "gemma3",
        "model_path": f"{MODELS_DIR}/gemma3/gemma-3-4b-it",
        "data_path": f"{DATA_DIR}/eval_diverse_5k_gemma3.jsonl",
        "stats_path": f"{RESULTS_DIR}/gemma3/json/gemma-3-4b-it_activation_stats.json",
        "batch_size": 4,
        "role": "high_peak",
    },
    "Qwen3-30B-A3B": {
        "family": "Qwen3",
        "model_path": f"{MODELS_DIR}/Qwen3/Qwen3-30B-A3B",
        "data_path": f"{DATA_DIR}/eval_diverse_5k_qwen3.jsonl",
        "stats_path": f"{RESULTS_DIR}/Qwen3/json/Qwen3-30B-A3B_activation_stats.json",
        "batch_size": 1,
        "role": "moe",
    },
    "Qwen3-32B": {
        "family": "Qwen3",
        "model_path": f"{MODELS_DIR}/Qwen3/Qwen3-32B",
        "data_path": f"{DATA_DIR}/eval_diverse_5k_qwen3.jsonl",
        "stats_path": f"{RESULTS_DIR}/Qwen3/json/Qwen3-32B_activation_stats.json",
        "batch_size": 1,
        "role": "dense_pair",
    },
    "Qwen2.5-7B": {
        "family": "Qwen2.5",
        "model_path": f"{MODELS_DIR}/Qwen2.5/Qwen2.5-7B",
        "data_path": f"{DATA_DIR}/eval_diverse_5k_qwen2.5.jsonl",
        "stats_path": f"{RESULTS_DIR}/Qwen2.5/json/Qwen2.5-7B_activation_stats.json",
        "batch_size": 2,
        "role": "medium_baseline",
    },
    "Qwen3-8B": {
        "family": "Qwen3",
        "model_path": f"{MODELS_DIR}/Qwen3/Qwen3-8B",
        "data_path": f"{DATA_DIR}/eval_diverse_5k_qwen3.jsonl",
        "stats_path": f"{RESULTS_DIR}/Qwen3/json/Qwen3-8B_activation_stats.json",
        "batch_size": 2,
        "role": "dense_small",
    },
    "Qwen3.5-9B": {
        "family": "Qwen3.5",
        "model_path": f"{MODELS_DIR}/Qwen3.5/Qwen3.5-9B",
        "data_path": f"{DATA_DIR}/eval_diverse_5k_qwen3.5.jsonl",
        "stats_path": f"{RESULTS_DIR}/Qwen3.5/json/Qwen3.5-9B_activation_stats.json",
        "batch_size": 2,
        "role": "dense_medium",
    },
    "Qwen3.5-35B-A3B": {
        "family": "Qwen3.5",
        "model_path": f"{MODELS_DIR}/Qwen3.5/Qwen3.5-35B-A3B",
        "data_path": f"{DATA_DIR}/eval_diverse_5k_qwen3.5.jsonl",
        "stats_path": f"{RESULTS_DIR}/Qwen3.5/json/Qwen3.5-35B-A3B_activation_stats.json",
        "batch_size": 1,
        "role": "moe",
    },
}
DEFAULT_MODELS = [
    "Qwen3.5-0.8B", "gemma-3-4b-it",
    "Qwen2.5-7B", "Qwen3-8B", "Qwen3.5-9B",
    "Qwen3.5-35B-A3B", "Qwen3-30B-A3B", "Qwen3-32B",
]


def load_data(path, max_samples, max_seq_len, offset=0):
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if len(samples) >= max_samples:
                break
            data = json.loads(line)
            samples.append(data["input_ids"][:max_seq_len])
    return samples


def make_batches(samples, batch_size, pad_token_id=0):
    indexed = sorted(enumerate(samples), key=lambda x: len(x[1]))
    sorted_samples = [s for _, s in indexed]
    batches = []
    for i in range(0, len(sorted_samples), batch_size):
        batch = sorted_samples[i:i + batch_size]
        max_len = max(len(s) for s in batch)
        input_ids, attention_mask = [], []
        for sample in batch:
            pad_len = max_len - len(sample)
            input_ids.append(sample + [pad_token_id] * pad_len)
            attention_mask.append([1] * len(sample) + [0] * pad_len)
        batches.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        })
    return batches


def get_text_backbone(model):
    base = getattr(model, "model", model)
    if hasattr(base, "layers"):
        return base
    lang = getattr(base, "language_model", None)
    if lang is not None:
        sub = getattr(lang, "model", lang)
        if hasattr(sub, "layers"):
            return sub
        if hasattr(lang, "layers"):
            return lang
    return base


def select_layers(model_name, cfg, policy):
    if policy == "all_hidden":
        return None
    if policy.startswith("list:"):
        layers = set()
        for item in policy.split(":", 1)[1].split(","):
            item = item.strip()
            if not item:
                continue
            if item.startswith("layer_") or item == "final_layernorm":
                layers.add(item)
            elif item.upper().startswith("L") and item[1:].isdigit():
                layers.add(f"layer_{int(item[1:])}_hidden")
            elif item.isdigit():
                layers.add(f"layer_{int(item)}_hidden")
        return layers
    stats_path = cfg.get("stats_path")
    if stats_path and os.path.exists(stats_path):
        stats = json.load(open(stats_path))
        loc = stats.get("global_max_location", "").split()[0]
        if loc:
            return {loc}
    return None


class MaxCollector:
    def __init__(self, selected_layers):
        self.selected_layers = selected_layers
        self.mask = None
        self.max_abs = defaultdict(float)
        self.count = defaultdict(int)

    def set_mask(self, attention_mask):
        self.mask = attention_mask.bool()

    def update(self, name, tensor):
        if self.selected_layers is not None and name not in self.selected_layers:
            return
        if not name.endswith("_hidden") and name != "final_layernorm":
            return
        t = tensor.detach().float()
        if self.mask is not None and t.dim() >= 2 and t.shape[:2] == self.mask.shape:
            mask = self.mask.to(t.device)
            for bi in range(t.shape[0]):
                if not mask[bi].any():
                    continue
                vals = t[bi, mask[bi]].abs()
                if vals.numel() == 0:
                    continue
                self.max_abs[name] = max(self.max_abs[name], float(vals.max().item()))
                self.count[name] += vals.numel()
        else:
            vals = t.abs().reshape(-1)
            if vals.numel() > 0:
                self.max_abs[name] = max(self.max_abs[name], float(vals.max().item()))
                self.count[name] += vals.numel()


class HistCollector:
    def __init__(self, selected_layers, max_abs, bins):
        self.selected_layers = selected_layers
        self.max_abs = dict(max_abs)
        self.bins = bins
        self.mask = None
        self.hist = {k: torch.zeros(bins, dtype=torch.int64) for k in max_abs}
        self.count = defaultdict(int)

    def set_mask(self, attention_mask):
        self.mask = attention_mask.bool()

    def update(self, name, tensor):
        if name not in self.hist:
            return
        upper = self.max_abs.get(name, 0.0)
        if upper <= 0:
            return
        t = tensor.detach().float().abs()
        if self.mask is not None and t.dim() >= 2 and t.shape[:2] == self.mask.shape:
            mask = self.mask.to(t.device)
            for bi in range(t.shape[0]):
                if not mask[bi].any():
                    continue
                vals = t[bi, mask[bi]].reshape(-1)
                self._add(name, vals, upper)
        else:
            self._add(name, t.reshape(-1), upper)

    def _add(self, name, vals, upper):
        if vals.numel() == 0:
            return
        idx = torch.clamp((vals / upper * (self.bins - 1)).floor().long(), 0, self.bins - 1)
        h = torch.bincount(idx.cpu(), minlength=self.bins).to(torch.int64)
        self.hist[name] += h
        self.count[name] += vals.numel()

    def quantiles(self, q):
        out = {}
        for name, hist in self.hist.items():
            total = int(hist.sum().item())
            if total == 0:
                continue
            target = max(1, math.ceil(q * total))
            cum = torch.cumsum(hist, dim=0)
            idx = int(torch.searchsorted(cum, torch.tensor(target)).item())
            upper = self.max_abs[name]
            out[name] = upper * idx / max(1, self.bins - 1)
        return out


class ErrorCollector:
    def __init__(self, thresholds_by_policy, bits):
        self.thresholds_by_policy = thresholds_by_policy
        self.bits = bits
        self.qmax = (2 ** (bits - 1)) - 1
        self.mask = None
        self.stats = defaultdict(lambda: defaultdict(float))

    def set_mask(self, attention_mask):
        self.mask = attention_mask.bool()

    def update(self, name, tensor):
        if name not in next(iter(self.thresholds_by_policy.values()), {}):
            return
        t = tensor.detach().float()
        if self.mask is not None and t.dim() >= 2 and t.shape[:2] == self.mask.shape:
            mask = self.mask.to(t.device)
            for bi in range(t.shape[0]):
                if not mask[bi].any():
                    continue
                vals = t[bi, mask[bi]].reshape(-1)
                self._add(name, vals)
        else:
            self._add(name, t.reshape(-1))

    def _add(self, name, vals):
        if vals.numel() == 0:
            return
        vals = vals.float()
        abs_vals = vals.abs()
        base_sq = float((vals * vals).sum().item())
        base_abs = float(abs_vals.sum().item())
        count = vals.numel()
        for policy, thresholds in self.thresholds_by_policy.items():
            threshold = float(thresholds.get(name, 0.0))
            if threshold <= 0:
                continue
            scale = threshold / self.qmax
            clipped = vals.clamp(-threshold, threshold)
            quantized = torch.round(clipped / scale).clamp(-self.qmax, self.qmax)
            dequant = quantized * scale
            err = dequant - vals
            entry = self.stats[(name, policy)]
            entry["count"] += count
            entry["sum_abs"] += base_abs
            entry["sum_sq"] += base_sq
            entry["err_abs"] += float(err.abs().sum().item())
            entry["err_sq"] += float((err * err).sum().item())
            entry["clipped"] += int((abs_vals > threshold).sum().item())

    def finalize(self):
        rows = []
        for (name, policy), s in sorted(self.stats.items()):
            count = max(1, int(s["count"]))
            mse = s["err_sq"] / count
            rmse = math.sqrt(mse)
            rms = math.sqrt(s["sum_sq"] / count) if s["sum_sq"] > 0 else 0.0
            mae = s["err_abs"] / count
            abs_mean = s["sum_abs"] / count
            rel_rmse = rmse / rms if rms > 0 else 0.0
            rel_mae = mae / abs_mean if abs_mean > 0 else 0.0
            sqnr_db = 10.0 * math.log10(s["sum_sq"] / s["err_sq"]) if s["err_sq"] > 0 and s["sum_sq"] > 0 else float("inf")
            rows.append({
                "layer": name,
                "policy": policy,
                "count": count,
                "mae": mae,
                "rmse": rmse,
                "rel_mae": rel_mae,
                "rel_rmse": rel_rmse,
                "sqnr_db": sqnr_db,
                "clip_rate": s["clipped"] / count,
            })
        return rows


def register_hidden_hooks(model, collector):
    hooks = []
    base = get_text_backbone(model)
    layers = base.layers if hasattr(base, "layers") else []
    for i, layer in enumerate(layers):
        def make_hook(idx):
            def hook(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                collector.update(f"layer_{idx}_hidden", h)
            return hook
        hooks.append(layer.register_forward_hook(make_hook(i)))
    if hasattr(base, "norm"):
        def norm_hook(_mod, _inp, out):
            collector.update("final_layernorm", out)
        hooks.append(base.norm.register_forward_hook(norm_hook))
    return hooks


def run_pass(model, batches, input_device, collector, desc):
    base_model = model.model if hasattr(model, "model") else model
    with torch.no_grad():
        for batch in tqdm(batches, desc=desc, unit="batch", ascii=True, ncols=90, file=sys.stdout):
            input_ids = batch["input_ids"].to(input_device)
            attention_mask = batch["attention_mask"].to(input_device)
            collector.set_mask(attention_mask)
            _ = base_model(input_ids=input_ids, attention_mask=attention_mask)


def load_model(model_path, gpu_id):
    safetensor_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors") or f.endswith(".bin")]
    size_gb = sum(os.path.getsize(os.path.join(model_path, f)) for f in safetensor_files) / (1024 ** 3)
    kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    if size_gb < 70:
        kwargs["device_map"] = {"": f"cuda:{gpu_id}"}
    else:
        kwargs["device_map"] = "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except ValueError as exc:
        if "Unrecognized configuration class" not in str(exc):
            raise
        try:
            from transformers import AutoModelForVision2Seq as Loader
        except ImportError:
            from transformers import AutoModel as Loader
        model = Loader.from_pretrained(model_path, **kwargs)
    model.eval()
    return model, size_gb


def get_input_device(model):
    base = get_text_backbone(model)
    embed = getattr(base, "embed_tokens", None) or getattr(base, "word_embeddings", None)
    if embed is not None:
        return next(embed.parameters()).device
    return next(model.parameters()).device


def run_model(model_name, cfg, args):
    model_path = cfg["model_path"]
    if not os.path.isdir(model_path):
        raise FileNotFoundError(model_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    pad_id = getattr(config, "pad_token_id", 0) or 0
    batch_size = args.batch_size or cfg["batch_size"]
    selected_layers = select_layers(model_name, cfg, args.layer_policy)
    calibr_samples = load_data(cfg["data_path"], args.calibration_samples, args.max_seq_len, 0)
    eval_samples = load_data(cfg["data_path"], args.eval_samples, args.max_seq_len, args.calibration_samples)
    calibr_batches = make_batches(calibr_samples, batch_size, pad_id)
    eval_batches = make_batches(eval_samples, batch_size, pad_id)

    model, model_size_gb = load_model(model_path, args.gpu_id)
    input_device = get_input_device(model)
    print(f"[{model_name}] size={model_size_gb:.1f}GB, device={input_device}, batch={batch_size}, layers={selected_layers or 'all_hidden'}")

    max_collector = MaxCollector(selected_layers)
    hooks = register_hidden_hooks(model, max_collector)
    try:
        run_pass(model, calibr_batches, input_device, max_collector, f"{model_name} calibr-max")
    finally:
        for hook in hooks:
            hook.remove()

    hist_collector = HistCollector(selected_layers, max_collector.max_abs, args.hist_bins)
    hooks = register_hidden_hooks(model, hist_collector)
    try:
        run_pass(model, calibr_batches, input_device, hist_collector, f"{model_name} calibr-hist")
    finally:
        for hook in hooks:
            hook.remove()

    max_thresholds = dict(max_collector.max_abs)
    clip_thresholds = hist_collector.quantiles(args.clip_quantile)
    thresholds = {"max_abs": max_thresholds, f"clip_q{args.clip_quantile:g}": clip_thresholds}

    error_collector = ErrorCollector(thresholds, args.bits)
    hooks = register_hidden_hooks(model, error_collector)
    try:
        run_pass(model, eval_batches, input_device, error_collector, f"{model_name} eval")
    finally:
        for hook in hooks:
            hook.remove()

    rows = error_collector.finalize()
    peak_abs = None
    global_loc = None
    if os.path.exists(cfg.get("stats_path", "")):
        stats = json.load(open(cfg["stats_path"]))
        peak_abs = abs(float(stats.get("global_max_activation", 0.0)))
        global_loc = stats.get("global_max_location", "")

    output = {
        "model_name": model_name,
        "family": cfg["family"],
        "role": cfg["role"],
        "model_path": model_path,
        "data_path": cfg["data_path"],
        "bits": args.bits,
        "clip_quantile": args.clip_quantile,
        "calibration_samples": len(calibr_samples),
        "eval_samples": len(eval_samples),
        "max_seq_len": args.max_seq_len,
        "layer_policy": args.layer_policy,
        "selected_layers": sorted(max_thresholds),
        "reference_global_max_abs": peak_abs,
        "reference_global_max_location": global_loc,
        "thresholds": thresholds,
        "metrics": rows,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return output


def summarize(outputs):
    summary = []
    for out in outputs:
        by_policy = defaultdict(list)
        for row in out["metrics"]:
            by_policy[row["policy"]].append(row)
        for policy, rows in by_policy.items():
            weighted_count = sum(r["count"] for r in rows)
            if weighted_count == 0:
                continue
            summary.append({
                "model_name": out["model_name"],
                "family": out["family"],
                "role": out["role"],
                "policy": policy,
                "reference_global_max_abs": out["reference_global_max_abs"],
                "mean_rel_rmse": sum(r["rel_rmse"] * r["count"] for r in rows) / weighted_count,
                "mean_rel_mae": sum(r["rel_mae"] * r["count"] for r in rows) / weighted_count,
                "mean_clip_rate": sum(r["clip_rate"] * r["count"] for r in rows) / weighted_count,
                "mean_sqnr_db": sum(r["sqnr_db"] * r["count"] for r in rows if math.isfinite(r["sqnr_db"])) / weighted_count,
                "layers": len(rows),
            })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--clip_quantile", type=float, default=0.999)
    parser.add_argument("--calibration_samples", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--hist_bins", type=int, default=2048)
    parser.add_argument("--layer_policy", default="peak", help="peak, all_hidden, or list:L10,L24,final_layernorm")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", default=QUANT_DIR)
    parser.add_argument("--no_summary", action="store_true",
                        help="Skip writing quant_sanity_summary.json (useful when running one model at a time)")
    parser.add_argument("--summary_only", action="store_true",
                        help="Aggregate all *_quant_sanity.json in output_dir into summary only, no model runs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.summary_only:
        import glob as _glob
        outputs = []
        for path in sorted(_glob.glob(os.path.join(args.output_dir, "*_quant_sanity.json"))):
            with open(path) as f:
                outputs.append(json.load(f))
            print(f"loaded: {path}")
        summary = summarize(outputs)
        summary_path = os.path.join(args.output_dir, "quant_sanity_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"summary saved: {summary_path} ({len(outputs)} models)")
        return

    outputs = []
    for model_name in args.models:
        output_path = os.path.join(args.output_dir, f"{model_name}_quant_sanity.json")
        if os.path.exists(output_path):
            print(f"[SKIP] {model_name} — already done")
            continue
        print(f"\n=== {model_name} ===")
        output = run_model(model_name, MODEL_CONFIGS[model_name], args)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"saved: {output_path}")
        outputs.append(output)

    if not args.no_summary and outputs:
        summary = summarize(outputs)
        summary_path = os.path.join(args.output_dir, "quant_sanity_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"summary saved: {summary_path}")


if __name__ == "__main__":
    main()
