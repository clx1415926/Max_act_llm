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

    Additionally tracks per-layer Sun-style element-median statistics for
    tensors matching _SUN_SUFFIXES (hidden / attn_output / mlp_output /
    embedding / final_layernorm). See _update_sun for details.
    """

    # Tensors whose name ends with one of these (or equals, for global names)
    # are eligible for Sun median-ratio analysis. Excludes mlp_gate
    # (intermediate_size dim) and router/moe_gate (num_experts dim).
    _SUN_SUFFIXES = ("_hidden", "_attn_output", "_mlp_output")
    _SUN_EXACT    = ("embedding_output", "final_layernorm")

    # Sun thresholds (Sun et al. 2024, arXiv:2402.17762). Fixed — downstream
    # scripts can re-judge at any threshold from the raw peak_sun_ratio.
    _SUN_T_VAL   = 100.0
    _SUN_T_RATIO = 1000.0

    # Top-K absolute activations to track per eligible component.
    _TOP5_K = 5

    def __init__(self):
        # Each stat entry holds GPU tensors (scalar, 0-dim)
        self._stats = {}
        self._sun = {}              # name -> dict of sun-state tensors
        self._top5 = {}             # name -> Tensor[K] absolute values (GPU)
        self._top5_signed = {}      # name -> Tensor[K] signed values (GPU)
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

    # ── Sun median-ratio accumulator ──────────────────────────────────────
    @classmethod
    def _eligible_for_sun(cls, name):
        if name in cls._SUN_EXACT:
            return True
        return any(name.endswith(s) for s in cls._SUN_SUFFIXES)

    def _get_sun_state(self, name, device):
        if name not in self._sun:
            z32 = lambda v: torch.tensor(v, dtype=torch.float32, device=device)
            z64 = lambda v: torch.tensor(v, dtype=torch.float64, device=device)
            self._sun[name] = {
                "peak_abs":         z32(0.0),
                "peak_median":      z32(0.0),
                "max_ratio":        z32(0.0),
                "max_ratio_abs":    z32(0.0),
                "max_ratio_median": z32(0.0),
                "n_tokens":         0,
                "n_pass_sun":       0,
                "sum_ratio":        z64(0.0),
                "sum_ratio_sq":     z64(0.0),
            }
        return self._sun[name]

    def _update_sun(self, name, t, mask):
        """Per-token element-median accumulator for Sun's strict definition.

        t:    [B, T, d] activation tensor
        mask: [B, T] bool on same device as t
        Processes one sample at a time for memory safety.
        """
        if t.dim() != 3 or t.shape[-1] < 2:
            return
        if mask is None or mask.shape != t.shape[:2]:
            return

        dev = t.device
        s = self._get_sun_state(name, dev)
        T_val = self._SUN_T_VAL
        T_ratio = self._SUN_T_RATIO
        B = t.shape[0]

        for bi in range(B):
            mi = mask[bi]                                       # [T]
            if not mi.any():
                continue
            ti = t[bi].detach().float().abs()                   # [T, d]
            per_tok_max    = ti.amax(dim=-1)                    # [T]
            per_tok_median = ti.median(dim=-1).values           # [T]
            eps = torch.finfo(per_tok_median.dtype).tiny
            ratio = per_tok_max / per_tok_median.clamp_min(eps) # [T]

            # Restrict to valid positions
            v_abs = per_tok_max[mi]
            v_med = per_tok_median[mi]
            v_rat = ratio[mi]

            # Layer peak |x| (and corresponding token's median)
            i_abs = int(v_abs.argmax().item())
            cand_abs = v_abs[i_abs]
            if cand_abs > s["peak_abs"]:
                s["peak_abs"]    = cand_abs.detach()
                s["peak_median"] = v_med[i_abs].detach()

            # Layer max Sun-ratio
            i_rat = int(v_rat.argmax().item())
            cand_r = v_rat[i_rat]
            if cand_r > s["max_ratio"]:
                s["max_ratio"]        = cand_r.detach()
                s["max_ratio_abs"]    = v_abs[i_rat].detach()
                s["max_ratio_median"] = v_med[i_rat].detach()

            # Aggregates
            n_valid = int(v_abs.numel())
            n_pass = int(((v_abs > T_val) & (v_rat > T_ratio)).sum().item())
            s["n_tokens"]     += n_valid
            s["n_pass_sun"]   += n_pass
            s["sum_ratio"]    += v_rat.to(torch.float64).sum()
            s["sum_ratio_sq"] += (v_rat.to(torch.float64) ** 2).sum()
            del ti, per_tok_max, per_tok_median, ratio, v_abs, v_med, v_rat

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

            # Sun-style per-token element-median (only 3-D [B,T,d] tensors
            # and only the whitelisted component names).
            if t.dim() == 3 and self._eligible_for_sun(name):
                self._update_sun(name, t, m)

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

                    # Top-K absolute activations (same eligibility as Sun).
                    # Pure GPU ops — no boolean indexing, no CPU sync.
                    if ti.dim() == 2 and self._eligible_for_sun(name):
                        K = self._TOP5_K
                        absti = ti_f.abs()
                        neg_one = absti.new_tensor(-1.0)
                        absti_masked = torch.where(mi_exp, absti, neg_one)
                        flat_abs = absti_masked.reshape(-1)
                        flat_signed = ti_f.reshape(-1)
                        k = min(K, flat_abs.numel())
                        vals, idx = flat_abs.topk(k)
                        signed = flat_signed.gather(0, idx)
                        if name not in self._top5:
                            self._top5[name] = torch.full((K,), 0.0, device=device)
                            self._top5_signed[name] = torch.full((K,), 0.0, device=device)
                        merged_abs = torch.cat([self._top5[name], vals])
                        merged_signed = torch.cat([self._top5_signed[name], signed])
                        fk = min(K, merged_abs.numel())
                        _, keep_idx = merged_abs.topk(fk)
                        self._top5[name] = merged_abs.index_select(0, keep_idx)
                        self._top5_signed[name] = merged_signed.index_select(0, keep_idx)
                        del absti, absti_masked, flat_abs, flat_signed, vals, idx, signed, merged_abs, merged_signed, keep_idx

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
        # Merge in per-layer Sun statistics
        for name, sd in self._sun.items():
            nt = int(sd["n_tokens"])
            entry = result.setdefault(name, {})
            peak_med = sd["peak_median"].item()
            peak_abs = sd["peak_abs"].item()
            peak_ratio = (peak_abs / peak_med) if peak_med > 0 else 0.0
            if nt > 0:
                mean_r = sd["sum_ratio"].item() / nt
                mean_r_sq = sd["sum_ratio_sq"].item() / nt
                std_r = max(0.0, mean_r_sq - mean_r ** 2) ** 0.5
            else:
                mean_r = 0.0
                std_r = 0.0
            entry["sun_peak_abs"]          = peak_abs
            entry["sun_peak_token_median"] = peak_med
            entry["sun_peak_token_ratio"]  = peak_ratio
            entry["sun_max_ratio"]         = sd["max_ratio"].item()
            entry["sun_max_ratio_abs"]     = sd["max_ratio_abs"].item()
            entry["sun_max_ratio_median"]  = sd["max_ratio_median"].item()
            entry["sun_mean_ratio"]        = mean_r
            entry["sun_std_ratio"]         = std_r
            entry["sun_n_tokens"]          = nt
            entry["sun_n_pass"]            = int(sd["n_pass_sun"])
            entry["sun_t_val"]             = self._SUN_T_VAL
            entry["sun_t_ratio"]           = self._SUN_T_RATIO
        # Merge in per-layer top-K absolute/signed values
        for name, abs_t in self._top5.items():
            entry = result.setdefault(name, {})
            abs_vals = abs_t.cpu().tolist()
            signed_vals = self._top5_signed[name].cpu().tolist()
            entry["top5_abs"]         = abs_vals
            entry["top5_signed"]      = signed_vals
            entry["top5_abs_mean"]    = sum(abs_vals) / len(abs_vals) if abs_vals else 0.0
            entry["top5_signed_mean"] = sum(signed_vals) / len(signed_vals) if signed_vals else 0.0
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

    # Top-5 fields are list-valued; split them out into a separate JSON so
    # the main activation_stats.json remains purely scalar (backward compat).
    _TOP5_FIELDS = ("top5_abs", "top5_signed", "top5_abs_mean", "top5_signed_mean")
    top5_per_layer = {}
    for k, v in summary.items():
        scalar_entry = {}
        top5_entry = {}
        for kk, vv in v.items():
            if kk in _TOP5_FIELDS:
                top5_entry[kk] = vv
            else:
                scalar_entry[kk] = float(vv)
        output_data["per_layer_stats"][k] = scalar_entry
        if top5_entry:
            top5_per_layer[k] = top5_entry

    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_json}")

    # ── Save top-5 JSON (same schema as legacy analyze_top5.py) ──────────
    all_abs, all_signed = [], []
    for _k, entry in top5_per_layer.items():
        for a, s in zip(entry.get("top5_abs", []), entry.get("top5_signed", [])):
            all_abs.append(a)
            all_signed.append(s)
    K = StatsAccumulator._TOP5_K
    paired = sorted(zip(all_abs, all_signed), reverse=True)[:K]
    global_top5_abs = [p[0] for p in paired]
    global_top5_signed = [p[1] for p in paired]
    global_top5_abs_mean = (sum(global_top5_abs) / len(global_top5_abs)) if global_top5_abs else 0.0

    print(f"\nGlobal top-5 absolute values: {[f'{v:.1f}' for v in global_top5_abs]}")
    print(f"Global top-5 absolute mean: {global_top5_abs_mean:.2f}")

    top5_json = os.path.join(json_dir, f"{model_name}_top5_stats.json")
    top5_data = {
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
        "per_layer_stats": top5_per_layer,
    }
    with open(top5_json, "w") as f:
        json.dump(top5_data, f, indent=2, ensure_ascii=False)
    print(f"Top-5 results saved to: {top5_json}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Done!\n")


if __name__ == "__main__":
    main()
