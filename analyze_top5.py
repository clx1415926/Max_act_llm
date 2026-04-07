"""
Per-layer activation top-5 analysis for a single model.

For each layer component (hidden, mlp_output, attn_output), tracks the
top-5 absolute activation values across all tokens in the evaluation set.
Reports top5 values and their mean per layer.

Supports: Qwen2.5 (dense), Qwen3 (dense), Qwen3-MoE, Gemma2.

Usage:
    python analyze_top5.py --model_path <path> --data_path <path> --output_dir <path> \
        [--max_samples 2000] [--max_seq_len 32768] [--batch_size 8]

Output:
    <output_dir>/<model_name>_top5_stats.json
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
from transformers import AutoModelForCausalLM, AutoConfig

# PyTorch < 2.6 compatibility: torch.accelerator was added in 2.6.
# MXFP4 quantizer calls current_accelerator(); returning 'mps' triggers
# dequantize=True so the model loads as bf16.
if not hasattr(torch, 'accelerator'):
    _acc = types.SimpleNamespace(current_accelerator=lambda: torch.device('mps'))
    torch.accelerator = _acc


# ─── Data loading (same as analyze_model.py) ─────────────────────────────────

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


# ─── Top-5 tracker (mask-aware, GPU-side) ─────────────────────────────────────

K = 5  # top-K to track


class Top5Accumulator:
    """Tracks top-K absolute activation values per named component.

    For each component, maintains a GPU tensor of shape [K] holding
    the current top-K absolute values seen so far. Updated incrementally
    per batch.
    """

    def __init__(self):
        self._topk = {}          # name -> Tensor[K] on GPU (absolute values)
        self._topk_signed = {}   # name -> Tensor[K] on GPU (signed values)
        self._current_mask = None

    def set_current_mask(self, attention_mask):
        self._current_mask = attention_mask.bool()

    def update(self, name, tensor):
        t = tensor.detach().float()
        mask = self._current_mask
        device = t.device

        B = t.shape[0]
        for bi in range(B):
            ti = t[bi]  # [T, ...]
            if mask is not None and t.dim() >= 2 and t.shape[:2] == mask.shape[:2]:
                mi = mask[bi].to(device)  # [T]
                if not mi.any():
                    continue
                # Expand mask for broadcast
                mi_exp = mi
                for _ in range(ti.dim() - 1):
                    mi_exp = mi_exp.unsqueeze(-1)
                # Select only valid positions
                flat = ti[mi_exp.expand_as(ti)]  # flattened valid elements
            else:
                flat = ti.reshape(-1)

            if flat.numel() == 0:
                continue

            abs_flat = flat.abs()
            # Get top-K from this sample
            sample_k = min(K, abs_flat.numel())
            sample_topk_abs, sample_topk_idx = abs_flat.topk(sample_k)
            sample_topk_signed = flat.gather(0, sample_topk_idx)

            if name not in self._topk:
                # Initialize with padding if first time
                self._topk[name] = torch.full((K,), 0.0, device=device)
                self._topk_signed[name] = torch.full((K,), 0.0, device=device)

            # Merge: concat current top-K with sample top-K, keep overall top-K
            merged_abs = torch.cat([self._topk[name], sample_topk_abs])
            merged_signed = torch.cat([self._topk_signed[name], sample_topk_signed])
            final_k = min(K, merged_abs.numel())
            _, keep_idx = merged_abs.topk(final_k)
            self._topk[name] = merged_abs[keep_idx]
            self._topk_signed[name] = merged_signed[keep_idx]

            del flat, abs_flat, sample_topk_abs, sample_topk_idx, sample_topk_signed

    def finalize(self):
        result = {}
        for name in self._topk:
            abs_vals = self._topk[name].cpu().tolist()
            signed_vals = self._topk_signed[name].cpu().tolist()
            result[name] = {
                "top5_abs": abs_vals,
                "top5_signed": signed_vals,
                "top5_abs_mean": sum(abs_vals) / len(abs_vals) if abs_vals else 0.0,
                "top5_signed_mean": sum(signed_vals) / len(signed_vals) if signed_vals else 0.0,
            }
        return result


# ─── Hook registration (same structure as analyze_model.py) ───────────────────

def is_moe_block(module):
    cls_name = type(module).__name__
    return "SparseMoe" in cls_name or "MoeBlock" in cls_name or hasattr(module, "experts")


def _get_text_backbone(model):
    """Navigate through VLM wrapper layers to find the text backbone (has .layers)."""
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


def register_hooks(model, accumulator):
    hooks = []
    base = _get_text_backbone(model)

    # Embedding hook (embed_tokens for most models, word_embeddings for BailingMoe)
    embed_mod = getattr(base, "embed_tokens", None) or getattr(base, "word_embeddings", None)
    if embed_mod is not None:
        def emb_hook(mod, inp, out):
            accumulator.update("embedding_output", out)
        hooks.append(embed_mod.register_forward_hook(emb_hook))

    layers = base.layers if hasattr(base, "layers") else []
    for i, layer in enumerate(layers):
        def make_layer_hook(idx):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                accumulator.update(f"layer_{idx}_hidden", h)
            return hook
        hooks.append(layer.register_forward_hook(make_layer_hook(i)))

        # Attention output (self_attn for most, attention for BailingMoe, linear_attn for Qwen3.5)
        attn_mod = (getattr(layer, "self_attn", None)
                    or getattr(layer, "attention", None)
                    or getattr(layer, "linear_attn", None))
        if attn_mod is not None:
            def make_attn_hook(idx):
                def hook(mod, inp, out):
                    accumulator.update(f"layer_{idx}_attn_output", out[0])
                return hook
            hooks.append(attn_mod.register_forward_hook(make_attn_hook(i)))

        mlp = layer.mlp
        if is_moe_block(mlp):
            def make_moe_hook(idx):
                def hook(mod, inp, out):
                    if isinstance(out, tuple) and len(out) >= 2:
                        accumulator.update(f"layer_{idx}_mlp_output", out[0])
                    else:
                        h = out[0] if isinstance(out, tuple) else out
                        accumulator.update(f"layer_{idx}_mlp_output", h)
                return hook
            hooks.append(mlp.register_forward_hook(make_moe_hook(i)))
        else:
            def make_mlp_hook(idx):
                def hook(mod, inp, out):
                    accumulator.update(f"layer_{idx}_mlp_output", out)
                return hook
            hooks.append(mlp.register_forward_hook(make_mlp_hook(i)))

    if hasattr(base, "norm"):
        def norm_hook(mod, inp, out):
            accumulator.update("final_layernorm", out)
        hooks.append(base.norm.register_forward_hook(norm_hook))

    return hooks


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Per-layer top-5 activation analysis")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--max_seq_len", type=int, default=32768)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    model_name = os.path.basename(args.model_path.rstrip("/"))
    json_dir = os.path.join(args.output_dir, "json")
    os.makedirs(json_dir, exist_ok=True)
    output_json = os.path.join(json_dir, f"{model_name}_top5_stats.json")

    print("=" * 70)
    print(f"Top-5 Activation Analysis: {model_name}")
    print("=" * 70)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "unknown")
    # For multimodal VLM configs (qwen3_5, qwen3_5_moe), metadata lives in text_config
    _text_cfg = getattr(config, "text_config", config)
    num_layers = getattr(_text_cfg, "num_hidden_layers", 0)
    hidden_size = getattr(_text_cfg, "hidden_size", 0)
    print(f"Model type: {model_type}, Layers: {num_layers}, Hidden: {hidden_size}")

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

    first_param = next(model.parameters())
    input_device = first_param.device
    print(f"Input device: {input_device}")

    print(f"Loading data from {args.data_path} ...")
    samples = load_data(args.data_path, args.max_samples, args.max_seq_len)
    lens = [len(s) for s in samples]
    print(f"Loaded {len(samples)} samples (lengths: min={min(lens)}, max={max(lens)}, "
          f"avg={sum(lens)/len(lens):.0f})")

    pad_id = getattr(config, "pad_token_id", 0) or 0
    batches = make_batches(samples, args.batch_size, pad_id)

    total_tokens = sum(lens)
    total_padded = sum(b["input_ids"].numel() for b in batches)
    padding_ratio = 1 - total_tokens / total_padded if total_padded > 0 else 0
    print(f"Created {len(batches)} batches (batch_size={args.batch_size}, "
          f"padding ratio={padding_ratio:.1%})")

    accumulator = Top5Accumulator()
    hooks = register_hooks(model, accumulator)
    print(f"Registered {len(hooks)} hooks")

    base_model = model.model if hasattr(model, "model") else model
    print("Running forward passes...")
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
            accumulator.set_current_mask(attention_mask)
            _ = base_model(input_ids=input_ids, attention_mask=attention_mask)
            pbar.set_postfix({"tokens": f"{input_ids.numel():,}"})

    print("Forward passes complete.")
    for h in hooks:
        h.remove()

    summary = accumulator.finalize()

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Top-5 Absolute Activation Values — {model_name}")
    print(f"{'='*70}")
    print(f"{'Layer':<8} {'Component':<16} {'Top5 AbsMean':>14} {'Top5 Values'}")
    print("-" * 80)
    for i in range(num_layers):
        for comp, label in [
            (f"layer_{i}_hidden", "hidden"),
            (f"layer_{i}_mlp_output", "mlp_output"),
            (f"layer_{i}_attn_output", "attn_output"),
        ]:
            if comp in summary:
                s = summary[comp]
                vals_str = ", ".join(f"{v:.1f}" for v in s["top5_abs"])
                print(f"{i:<8} {label:<16} {s['top5_abs_mean']:>14.2f} [{vals_str}]")

    # Global top5 (across all layers and components)
    all_abs = []
    all_signed = []
    for k, v in summary.items():
        for a, s in zip(v["top5_abs"], v["top5_signed"]):
            all_abs.append(a)
            all_signed.append(s)
    # Sort and take global top5
    paired = sorted(zip(all_abs, all_signed), reverse=True)[:K]
    global_top5_abs = [p[0] for p in paired]
    global_top5_signed = [p[1] for p in paired]
    global_top5_abs_mean = sum(global_top5_abs) / len(global_top5_abs) if global_top5_abs else 0.0

    print(f"\nGlobal top-5 absolute values: {[f'{v:.1f}' for v in global_top5_abs]}")
    print(f"Global top-5 absolute mean: {global_top5_abs_mean:.2f}")

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
        "global_top5_abs": global_top5_abs,
        "global_top5_signed": global_top5_signed,
        "global_top5_abs_mean": global_top5_abs_mean,
        "per_layer_stats": {},
    }

    for k, v in summary.items():
        output_data["per_layer_stats"][k] = v

    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_json}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Done!\n")


if __name__ == "__main__":
    main()
