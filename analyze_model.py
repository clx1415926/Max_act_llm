"""
Per-layer activation analysis for a single model.

Supports: Qwen2.5 (dense), Qwen3 (dense), Qwen3-MoE, Gemma2.

Usage:
    python analyze_model.py --model_path <path> --data_path <path> --output_dir <path> \
        [--max_samples 2000] [--max_seq_len 32768] [--batch_size 8]

Output:
    <output_dir>/<model_name>_activation_stats.json
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import gc
import json
import argparse
import sys
import types
import torch
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoConfig

# PyTorch < 2.6 compatibility: torch.accelerator was added in 2.6.
# MXFP4 quantizer calls current_accelerator(); if it returns a device NOT in
# ["cuda","xpu","cpu"] the quantizer automatically sets dequantize=True and
# loads the model as bf16 — exactly what we want.
if not hasattr(torch, "accelerator"):
    _acc = types.SimpleNamespace(current_accelerator=lambda: torch.device("mps"))
    torch.accelerator = _acc


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(path, max_samples, max_seq_len):
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            data = json.loads(line)
            ids = data["input_ids"][:max_seq_len]
            samples.append(ids)
    return samples


def make_batches(samples, batch_size, pad_token_id=0):
    """Create batches from length-sorted samples to minimize padding."""
    # Sort by length to group similar-length sequences together
    indexed = sorted(enumerate(samples), key=lambda x: len(x[1]))
    sorted_samples = [s for _, s in indexed]

    batches = []
    for i in range(0, len(sorted_samples), batch_size):
        batch = sorted_samples[i:i + batch_size]
        max_len = max(len(s) for s in batch)
        input_ids = []
        attention_mask = []
        for s in batch:
            pad_len = max_len - len(s)
            input_ids.append(s + [pad_token_id] * pad_len)
            attention_mask.append([1] * len(s) + [0] * pad_len)
        batches.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        })
    return batches


# ─── Statistics collector (mask-aware, GPU-accumulated) ───────────────────────

class StatsAccumulator:
    """Running statistics accumulator with attention_mask support.

    Accumulates on GPU to avoid CPU-GPU sync per hook call.
    Only transfers to CPU at finalize().
    """

    def __init__(self):
        # Each stat entry holds GPU tensors (scalar, 0-dim)
        self._stats = {}
        self._current_mask = None

    def _get_stat(self, name, device):
        if name not in self._stats:
            self._stats[name] = {
                "sum": torch.tensor(0.0, dtype=torch.float64, device=device),
                "sum_sq": torch.tensor(0.0, dtype=torch.float64, device=device),
                "abs_sum": torch.tensor(0.0, dtype=torch.float64, device=device),
                "max": torch.tensor(float("-inf"), dtype=torch.float32, device=device),
                "min": torch.tensor(float("inf"), dtype=torch.float32, device=device),
                "count": 0,
            }
        return self._stats[name]

    def set_current_mask(self, attention_mask):
        """Set mask for the current batch. Shape: [B, T], 1=valid, 0=pad."""
        self._current_mask = attention_mask.bool()

    def update(self, name, tensor):
        """Update stats on GPU, applying mask to exclude padding positions.

        Processes tensor in chunks along batch dim to avoid OOM on large
        intermediate-size tensors (e.g. gate_proj [B, T, 8960]).
        """
        t = tensor.detach()
        mask = self._current_mask
        device = t.device

        if mask is not None and t.dim() >= 2 and t.shape[:2] == mask.shape[:2]:
            m = mask.to(device)  # [B, T] bool
            B = t.shape[0]
            # Process one sample at a time to keep peak memory low
            for bi in range(B):
                ti = t[bi]       # [T, ...] — single sample, no copy
                mi = m[bi]       # [T] bool
                if mi.any():
                    # Expand mask: [T] -> [T, 1, ...] to broadcast with ti
                    mi_exp = mi
                    for _ in range(ti.dim() - 1):
                        mi_exp = mi_exp.unsqueeze(-1)

                    ti_f = ti.float()
                    mi_f = mi_exp.float()
                    masked = ti_f * mi_f  # zeros out padding positions

                    n_valid_pos = mi.sum().item()
                    extra_dims = 1
                    for d in range(1, ti.dim()):
                        extra_dims *= ti.shape[d]
                    count = n_valid_pos * extra_dims

                    s = self._get_stat(name, device)
                    s["sum"] += masked.sum().to(dtype=torch.float64)
                    s["sum_sq"] += (masked * masked).sum().to(dtype=torch.float64)
                    s["abs_sum"] += masked.abs().sum().to(dtype=torch.float64)

                    filled_max = ti_f.where(mi_exp, torch.tensor(float("-inf"), device=device))
                    filled_min = ti_f.where(mi_exp, torch.tensor(float("inf"), device=device))
                    s["max"] = torch.maximum(s["max"], filled_max.max())
                    s["min"] = torch.minimum(s["min"], filled_min.min())
                    s["count"] += count
                    del ti_f, mi_f, masked, filled_max, filled_min
        else:
            # No mask needed — accumulate directly (small tensors like embedding)
            flat = t.float().reshape(-1)
            count = flat.numel()
            if count == 0:
                return
            s = self._get_stat(name, device)
            s["sum"] += flat.sum().to(dtype=torch.float64)
            s["sum_sq"] += (flat * flat).sum().to(dtype=torch.float64)
            s["abs_sum"] += flat.abs().sum().to(dtype=torch.float64)
            s["max"] = torch.maximum(s["max"], flat.max())
            s["min"] = torch.minimum(s["min"], flat.min())
            s["count"] += count

    def finalize(self):
        """Transfer from GPU to CPU and compute final statistics."""
        result = {}
        for name, s in self._stats.items():
            n = s["count"]
            if n == 0:
                continue
            sum_val = s["sum"].item()
            sum_sq_val = s["sum_sq"].item()
            abs_sum_val = s["abs_sum"].item()
            mean = sum_val / n
            mean_sq = sum_sq_val / n
            std = max(0.0, mean_sq - mean ** 2) ** 0.5
            result[name] = {
                "mean": mean,
                "std": std,
                "abs_mean": abs_sum_val / n,
                "rms": mean_sq ** 0.5,
                "max": s["max"].item(),
                "min": s["min"].item(),
                "count": int(n),
            }
        return result


# ─── Hook registration ────────────────────────────────────────────────────────

def is_moe_block(module):
    """Check if a module is a MoE sparse block."""
    cls_name = type(module).__name__
    return "SparseMoe" in cls_name or "MoeBlock" in cls_name or hasattr(module, "experts")


def _get_text_backbone(model):
    """Navigate through VLM wrapper layers to find the text backbone (has .layers).

    Standard CausalLM:  model.model.layers  ← base = model.model
    Gemma3 VLM:         model.model.language_model.model.layers
    Qwen2.5-VL:         model.model.layers  ← base = model.model (Qwen2_5_VLModel)
    """
    base = getattr(model, "model", model)
    if hasattr(base, "layers"):
        return base
    # VLM wrapper: model.model contains language_model (e.g. Gemma3ForCausalLM)
    lang = getattr(base, "language_model", None)
    if lang is not None:
        sub = getattr(lang, "model", lang)
        if hasattr(sub, "layers"):
            return sub
        if hasattr(lang, "layers"):
            return lang
    return base


def register_hooks(model, accumulator):
    """Register forward hooks for all supported architectures."""
    hooks = []
    config = model.config

    # Get the base model (model.model for CausalLM wrappers)
    base = _get_text_backbone(model)

    # Embedding hook (embed_tokens for most models, word_embeddings for BailingMoe)
    embed_mod = getattr(base, "embed_tokens", None) or getattr(base, "word_embeddings", None)
    if embed_mod is not None:
        def emb_hook(mod, inp, out):
            accumulator.update("embedding_output", out)
        hooks.append(embed_mod.register_forward_hook(emb_hook))

    # Per-layer hooks
    layers = base.layers if hasattr(base, "layers") else []
    for i, layer in enumerate(layers):

        # 1) Layer output (hidden states after residual)
        def make_layer_hook(idx):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                accumulator.update(f"layer_{idx}_hidden", h)
            return hook
        hooks.append(layer.register_forward_hook(make_layer_hook(i)))

        # 2) Attention output (self_attn for most, attention for BailingMoe, linear_attn for Qwen3.5)
        attn_mod = (getattr(layer, "self_attn", None)
                    or getattr(layer, "attention", None)
                    or getattr(layer, "linear_attn", None))
        if attn_mod is not None:
            def make_attn_hook(idx):
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    accumulator.update(f"layer_{idx}_attn_output", h)
                return hook
            hooks.append(attn_mod.register_forward_hook(make_attn_hook(i)))

        # 3) MLP hooks — depends on architecture
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        if is_moe_block(mlp):
            # MoE block: hook on the whole block output + router
            def make_moe_hook(idx):
                def hook(mod, inp, out):
                    if isinstance(out, tuple) and len(out) >= 2:
                        accumulator.update(f"layer_{idx}_mlp_output", out[0])
                        # out[1] may be a tensor or a tuple (e.g. BailingMoe returns (logits, topk_idx))
                        router_out = out[1]
                        if isinstance(router_out, tuple):
                            router_out = router_out[0]
                        accumulator.update(f"layer_{idx}_router_logits", router_out)
                    else:
                        h = out[0] if isinstance(out, tuple) else out
                        accumulator.update(f"layer_{idx}_mlp_output", h)
                return hook
            hooks.append(mlp.register_forward_hook(make_moe_hook(i)))

            # Hook on MoE gate (router)
            router_mod = getattr(mlp, "gate", None) or getattr(mlp, "router", None)
            if router_mod is not None:
                def make_router_hook(idx):
                    def hook(mod, inp, out):
                        # Some routers return tuples; take the first element for logits
                        r = out[0] if isinstance(out, tuple) else out
                        accumulator.update(f"layer_{idx}_moe_gate", r)
                    return hook
                hooks.append(router_mod.register_forward_hook(make_router_hook(i)))
        else:
            # Dense MLP block
            def make_mlp_hook(idx):
                def hook(mod, inp, out):
                    accumulator.update(f"layer_{idx}_mlp_output", out)
                return hook
            hooks.append(mlp.register_forward_hook(make_mlp_hook(i)))

            # gate_proj (pre-activation)
            if hasattr(mlp, "gate_proj"):
                def make_gate_hook(idx):
                    def hook(mod, inp, out):
                        accumulator.update(f"layer_{idx}_mlp_gate", out)
                    return hook
                hooks.append(mlp.gate_proj.register_forward_hook(make_gate_hook(i)))

    # Final layernorm
    if hasattr(base, "norm"):
        def norm_hook(mod, inp, out):
            accumulator.update("final_layernorm", out)
        hooks.append(base.norm.register_forward_hook(norm_hook))

    return hooks


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze per-layer activations")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--max_seq_len", type=int, default=32768)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU index for single-GPU models (ignored for multi-GPU)")
    args = parser.parse_args()

    model_name = os.path.basename(args.model_path.rstrip("/"))
    json_dir = os.path.join(args.output_dir, "json")
    os.makedirs(json_dir, exist_ok=True)
    output_json = os.path.join(json_dir, f"{model_name}_activation_stats.json")

    print("=" * 70)
    print(f"Activation Analysis: {model_name}")
    print("=" * 70)

    # Load config to check model type
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "unknown")
    # For multimodal VLM configs (qwen3_5, qwen3_5_moe), metadata lives in text_config
    _text_cfg = getattr(config, "text_config", config)
    num_layers = getattr(_text_cfg, "num_hidden_layers", 0)
    hidden_size = getattr(_text_cfg, "hidden_size", 0)
    print(f"Model type: {model_type}, Layers: {num_layers}, Hidden: {hidden_size}")

    # Load model — auto-select single GPU vs multi-GPU based on model size
    print(f"Loading model from {args.model_path} ...")
    safetensor_files = [f for f in os.listdir(args.model_path)
                        if f.endswith(".safetensors") or f.endswith(".bin")]
    model_size_gb = sum(os.path.getsize(os.path.join(args.model_path, f))
                        for f in safetensor_files) / (1024 ** 3)
    print(f"Model files size: {model_size_gb:.1f} GB")

    if model_size_gb < 70:
        gpu_device = f"cuda:{args.gpu_id}"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": gpu_device},
                trust_remote_code=True,
            )
        except ValueError as e:
            if "Unrecognized configuration class" not in str(e):
                raise
            print(f"AutoModelForCausalLM unsupported, falling back to AutoModel")
            try:
                from transformers import AutoModelForVision2Seq as _Loader
            except ImportError:
                from transformers import AutoModel as _Loader
            model = _Loader.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": gpu_device},
                trust_remote_code=True,
            )
        print(f"Loaded on single GPU ({gpu_device})")
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        except ValueError as e:
            if "Unrecognized configuration class" not in str(e):
                raise
            print(f"AutoModelForCausalLM unsupported, falling back to AutoModel")
            try:
                from transformers import AutoModelForVision2Seq as _Loader
            except ImportError:
                from transformers import AutoModel as _Loader
            model = _Loader.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        print(f"Loaded with device_map='auto' (multi-GPU)")
    model.eval()

    # Detect device for input tensors (use embedding layer's device for multi-GPU safety)
    _embed = getattr(getattr(model, "model", model), "embed_tokens", None)
    if _embed is not None:
        input_device = next(_embed.parameters()).device
    else:
        input_device = next(model.parameters()).device
    print(f"Input device: {input_device}")

    # Load data
    print(f"Loading data from {args.data_path} ...")
    samples = load_data(args.data_path, args.max_samples, args.max_seq_len)
    if not samples:
        raise ValueError(f"No samples loaded from {args.data_path}. "
                         "Check the file path and max_samples setting.")
    lens = [len(s) for s in samples]
    print(f"Loaded {len(samples)} samples (lengths: min={min(lens)}, max={max(lens)}, "
          f"avg={sum(lens)/len(lens):.0f})")

    pad_id = getattr(config, "pad_token_id", 0) or 0
    batches = make_batches(samples, args.batch_size, pad_id)

    # Report padding efficiency
    total_tokens = sum(lens)
    total_padded = sum(b["input_ids"].numel() for b in batches)
    padding_ratio = 1 - total_tokens / total_padded if total_padded > 0 else 0
    print(f"Created {len(batches)} batches (batch_size={args.batch_size}, "
          f"padding ratio={padding_ratio:.1%})")

    # Register hooks
    accumulator = StatsAccumulator()
    hooks = register_hooks(model, accumulator)
    print(f"Registered {len(hooks)} hooks")

    # Forward pass — use model.model (base model) to skip lm_head,
    # which would allocate a huge [B, T, vocab_size] logits tensor.
    base_model = model.model if hasattr(model, "model") else model
    print("Running forward passes...")
    try:
        with torch.no_grad():
            pbar = tqdm(
                enumerate(batches),
                total=len(batches),
                desc=f"  {model_name}",
                unit="batch",
                ascii=True,
                ncols=90,
                file=sys.stdout,
            )
            for bi, batch in pbar:
                input_ids = batch["input_ids"].to(input_device)
                attention_mask = batch["attention_mask"].to(input_device)

                # Set mask so hooks only count non-padding positions
                accumulator.set_current_mask(attention_mask)

                _ = base_model(input_ids=input_ids, attention_mask=attention_mask)
                pbar.set_postfix({"tokens": f"{input_ids.numel():,}"})
    finally:
        print("Forward passes complete.")
        for h in hooks:
            h.remove()

    # Compute summary
    summary = accumulator.finalize()

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Hidden State Activation Progression — {model_name}")
    print(f"{'='*70}")
    print(f"{'Layer':<8} {'RMS':>10} {'AbsMean':>10} {'Std':>10} {'Max':>10} {'Min':>10}")
    print("-" * 60)
    for i in range(num_layers):
        key = f"layer_{i}_hidden"
        if key in summary:
            s = summary[key]
            print(f"{i:<8} {s['rms']:>10.4f} {s['abs_mean']:>10.4f} "
                  f"{s['std']:>10.4f} {s['max']:>10.2f} {s['min']:>10.2f}")

    # Global max activation (by absolute value) — limited to hidden/mlp/attn stats
    # Excludes router logits and gate outputs which have a different value scale.
    global_max_abs = 0.0
    global_max_val = 0.0
    global_max_loc = ""
    _activation_keys = ("_hidden", "_mlp_output", "_attn_output", "embedding_output", "final_layernorm")
    for k, v in summary.items():
        if not any(k.endswith(s) or k == s.rstrip("_") for s in _activation_keys):
            continue
        if abs(v["max"]) > global_max_abs:
            global_max_abs = abs(v["max"])
            global_max_val = v["max"]
            global_max_loc = k + " (max)"
        if abs(v["min"]) > global_max_abs:
            global_max_abs = abs(v["min"])
            global_max_val = v["min"]
            global_max_loc = k + " (min)"

    print(f"\nGlobal max absolute activation: {global_max_val:.4f} (at {global_max_loc})")

    # ── Save JSON ─────────────────────────────────────────────────────────
    output_data = {
        "model_name": model_name,
        "model_path": args.model_path,
        "model_type": model_type,
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "data_path": args.data_path,
        "num_samples": len(samples),
        "max_seq_len": args.max_seq_len,
        "global_max_activation": global_max_val,
        "global_max_location": global_max_loc,
        "per_layer_stats": {},
    }

    for k, v in summary.items():
        output_data["per_layer_stats"][k] = {kk: float(vv) for kk, vv in v.items()}

    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_json}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Done!\n")


if __name__ == "__main__":
    main()
