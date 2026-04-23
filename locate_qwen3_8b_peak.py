"""
Locate which sample(s) trigger Qwen3-8B's extreme massive activation.

For every sample in the given JSONL dataset this script records the
per-sample peak |activation| (across all hooked hidden / mlp_output /
attn_output tensors, over all tokens and hidden dims) together with the
layer, token position and hidden dim where that peak occurred. At the end
it prints the top-10 samples ranked by peak magnitude and saves the full
per-sample list to JSON.

Usage:
    CUDA_VISIBLE_DEVICES=0 python locate_qwen3_8b_peak.py \
        --model_path models/Qwen3/Qwen3-8B \
        --data_path  datasets/eval_diverse_5k_qwen3.jsonl \
        --output     results/Qwen3/json/Qwen3-8B_peak_per_sample.json \
        --max_samples 5000 --max_seq_len 32768 --batch_size 4
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import gc
import json
import argparse
import sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig


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
    """Batches sorted by length. Preserves original sample indices."""
    indexed = sorted(enumerate(samples), key=lambda x: len(x[1]))
    batches = []
    for i in range(0, len(indexed), batch_size):
        group = indexed[i:i + batch_size]
        max_len = max(len(s) for _, s in group)
        input_ids, mask, idxs = [], [], []
        for orig_idx, s in group:
            pad_len = max_len - len(s)
            input_ids.append(s + [pad_token_id] * pad_len)
            mask.append([1] * len(s) + [0] * pad_len)
            idxs.append(orig_idx)
        batches.append({
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "sample_idx":     idxs,
        })
    return batches


# ─── Per-sample peak tracker ──────────────────────────────────────────────────

class PerSamplePeak:
    """Tracks peak |activation| per sample, across all hooked tensors."""

    # Component suffixes eligible (mirror analyze_model.py)
    _ELIGIBLE_SUFFIXES = ("_hidden", "_mlp_output", "_attn_output")

    def __init__(self, num_samples):
        # per-sample state
        self.peak_abs   = [0.0] * num_samples
        self.peak_val   = [0.0] * num_samples   # signed
        self.peak_layer = [""]  * num_samples
        self.peak_tok   = [-1]  * num_samples
        self.peak_dim   = [-1]  * num_samples
        self.seq_len    = [0]   * num_samples
        self._batch_sample_idx = None
        self._current_mask = None

    @classmethod
    def _eligible(cls, name):
        return any(name.endswith(s) for s in cls._ELIGIBLE_SUFFIXES)

    def begin_batch(self, sample_idx_list, attention_mask):
        self._batch_sample_idx = sample_idx_list
        self._current_mask = attention_mask.bool()

    def update(self, name, t):
        if not self._eligible(name):
            return
        if t.dim() != 3:
            return
        mask = self._current_mask
        if mask is None or mask.shape != t.shape[:2]:
            return
        B = t.shape[0]
        absti = t.detach().float().abs()                          # [B, T, d]
        neg_one = absti.new_tensor(-1.0)
        m3 = mask.unsqueeze(-1)                                   # [B, T, 1]
        absti_masked = torch.where(m3, absti, neg_one)            # padding→-1
        # per-sample argmax over (T, d)
        flat = absti_masked.view(B, -1)                           # [B, T*d]
        max_vals, flat_idx = flat.max(dim=-1)                     # [B]
        d = t.shape[-1]
        tok_idx = (flat_idx // d).tolist()
        dim_idx = (flat_idx %  d).tolist()
        max_vals_list = max_vals.tolist()
        # signed value at the same position
        signed_vals = t.detach().float().view(B, -1).gather(
            1, flat_idx.view(B, 1)).squeeze(1).tolist()
        for bi in range(B):
            if max_vals_list[bi] <= 0.0:
                continue    # sample fully masked (shouldn't happen)
            orig = self._batch_sample_idx[bi]
            if max_vals_list[bi] > self.peak_abs[orig]:
                self.peak_abs[orig]   = max_vals_list[bi]
                self.peak_val[orig]   = signed_vals[bi]
                self.peak_layer[orig] = name
                self.peak_tok[orig]   = tok_idx[bi]
                self.peak_dim[orig]   = dim_idx[bi]
                # seq_len set when first seen
                if self.seq_len[orig] == 0:
                    self.seq_len[orig] = int(mask[bi].sum().item())


# ─── Hook registration (hidden / mlp_output / attn_output only) ───────────────

def _get_text_backbone(model):
    base = getattr(model, "model", model)
    if hasattr(base, "layers"):
        return base
    lang = getattr(base, "language_model", None)
    if lang is not None:
        sub = getattr(lang, "model", lang)
        if hasattr(sub, "layers"):
            return sub
    return base


def register_hooks(model, tracker):
    hooks = []
    base = _get_text_backbone(model)
    for i, layer in enumerate(base.layers):
        def mk_layer(idx):
            def h(mod, inp, out):
                x = out[0] if isinstance(out, tuple) else out
                tracker.update(f"layer_{idx}_hidden", x)
            return h
        hooks.append(layer.register_forward_hook(mk_layer(i)))

        attn = (getattr(layer, "self_attn", None)
                or getattr(layer, "attention", None))
        if attn is not None:
            def mk_attn(idx):
                def h(mod, inp, out):
                    x = out[0] if isinstance(out, tuple) else out
                    tracker.update(f"layer_{idx}_attn_output", x)
                return h
            hooks.append(attn.register_forward_hook(mk_attn(i)))

        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            def mk_mlp(idx):
                def h(mod, inp, out):
                    x = out[0] if isinstance(out, tuple) else out
                    tracker.update(f"layer_{idx}_mlp_output", x)
                return h
            hooks.append(mlp.register_forward_hook(mk_mlp(i)))
    return hooks


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Qwen3/Qwen3-8B")
    ap.add_argument("--data_path",  default="datasets/eval_diverse_5k_qwen3.jsonl")
    ap.add_argument("--output",     default="results/Qwen3/json/Qwen3-8B_peak_per_sample.json")
    ap.add_argument("--max_samples", type=int, default=5000)
    ap.add_argument("--max_seq_len", type=int, default=32768)
    ap.add_argument("--batch_size",  type=int, default=4)
    ap.add_argument("--gpu_id",      type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    model_name = os.path.basename(args.model_path.rstrip("/"))

    print(f"Loading config from {args.model_path} ...")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"Loading model on cuda:{args.gpu_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{args.gpu_id}"},
        trust_remote_code=True,
    )
    model.eval()
    input_device = next(model.parameters()).device

    print(f"Loading data from {args.data_path} ...")
    samples = load_data(args.data_path, args.max_samples, args.max_seq_len)
    n = len(samples)
    print(f"Loaded {n} samples "
          f"(min={min(len(s) for s in samples)}, "
          f"max={max(len(s) for s in samples)}, "
          f"avg={sum(len(s) for s in samples)/n:.0f})")

    pad_id = getattr(config, "pad_token_id", 0) or 0
    batches = make_batches(samples, args.batch_size, pad_id)

    tracker = PerSamplePeak(n)
    hooks = register_hooks(model, tracker)
    print(f"Registered {len(hooks)} hooks; running forward passes ...")

    base_model = model.model if hasattr(model, "model") else model
    try:
        with torch.no_grad():
            for batch in tqdm(batches, desc=f"{model_name}", ascii=True,
                              ncols=90, file=sys.stdout):
                ids  = batch["input_ids"].to(input_device)
                mask = batch["attention_mask"].to(input_device)
                tracker.begin_batch(batch["sample_idx"], mask)
                _ = base_model(input_ids=ids, attention_mask=mask)
    finally:
        for h in hooks:
            h.remove()

    # ── Build sorted output ──────────────────────────────────────────────
    records = [{
        "sample_idx": i,
        "peak_abs":   tracker.peak_abs[i],
        "peak_value": tracker.peak_val[i],
        "layer":      tracker.peak_layer[i],
        "token_pos":  tracker.peak_tok[i],
        "hidden_dim": tracker.peak_dim[i],
        "seq_len":    tracker.seq_len[i],
    } for i in range(n)]
    records.sort(key=lambda r: r["peak_abs"], reverse=True)

    print("\n" + "=" * 90)
    print(f"Top 10 samples by peak |activation|  —  {model_name}")
    print("=" * 90)
    print(f"{'rank':<5}{'sample':<8}{'peak_abs':>14}{'signed':>14}"
          f"  {'layer':<28}{'tok':>6}{'dim':>7}{'len':>7}")
    print("-" * 90)
    for rank, r in enumerate(records[:10], 1):
        print(f"{rank:<5}{r['sample_idx']:<8}{r['peak_abs']:>14.2f}"
              f"{r['peak_value']:>14.2f}  {r['layer']:<28}"
              f"{r['token_pos']:>6}{r['hidden_dim']:>7}{r['seq_len']:>7}")
    print("-" * 90)

    out = {
        "model_name":    model_name,
        "model_path":    args.model_path,
        "data_path":     args.data_path,
        "num_samples":   n,
        "max_seq_len":   args.max_seq_len,
        "batch_size":    args.batch_size,
        "top10":         records[:10],
        "per_sample":    records,       # full list, sorted by peak_abs desc
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
