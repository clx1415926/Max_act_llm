"""
Plot activation distribution charts for models within the same series.
Reads JSON stats from analyze_model.py output and generates comparison plots.

Usage:
    python plot_activations.py --series <series_name> --results_dir <path>
    python plot_activations.py --all --results_dir <path>

Generates per-series:
    1. Hidden state RMS progression across layers (all models overlaid)
    2. Hidden state AbsMean progression
    3. Hidden state Max activation progression
    4. MLP output RMS progression
    5. Attention output RMS progression
    6. Per-model detailed subplot (hidden/attn/mlp combined)
"""

import os
import json
import glob
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "/root/paddlejob/workspace/env_run/clx/activation_analysis"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SERIES_LIST = ["Qwen2.5", "Qwen3", "gemma2", "gpt_oss", "ling"]

# Color palette for models within a series
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--"]


def _parse_model_size(name):
    """Extract numeric model size for sorting.
    E.g. 'Qwen2.5-1.5b' -> 1.5, 'Qwen3-32B' -> 32, 'gemma-2-27b' -> 27,
         'Qwen3-30B-A3B' -> 30, 'Ling-mini-base-2.0-5T' -> 5 (token trillions).
    """
    import re
    # Match patterns like 1.5b, 7B, 32B, 30B-A3B (parameter billions)
    m = re.search(r"[-_](\d+(?:\.\d+)?)[bB]", name)
    if m:
        return float(m.group(1))
    # Match patterns like 5T, 10T, 15T (token trillions, used in ling series)
    m = re.search(r"[-_](\d+(?:\.\d+)?)[tT](?:$|[-_])", name)
    if m:
        return float(m.group(1))
    return 0.0


def load_series_results(series_name, results_dir):
    """Load all JSON stats for a given series, sorted by model size (small to large)."""
    series_dir = os.path.join(results_dir, series_name, "json")
    json_files = glob.glob(os.path.join(series_dir, "*_activation_stats.json"))
    results = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        results.append(data)
    # Sort by model size ascending
    results.sort(key=lambda r: _parse_model_size(r["model_name"]))
    return results


def extract_layer_metric(stats, num_layers, component_template, metric):
    """Extract a metric across layers for a given component pattern."""
    values = []
    for i in range(num_layers):
        key = component_template.format(i)
        if key in stats:
            values.append(stats[key].get(metric, 0.0))
        else:
            values.append(None)
    return values


def plot_series_comparison(series_name, results, output_dir):
    """Generate comparison plots for all models in a series."""
    if not results:
        print(f"[{series_name}] No results found, skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── Plot 1: Multi-metric comparison (2x3 grid) ────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"{series_name} — Activation Distribution Comparison", fontsize=16, fontweight="bold")

    metric_configs = [
        ("layer_{}_hidden",     "rms",      "Hidden State RMS"),
        ("layer_{}_hidden",     "abs_mean", "Hidden State AbsMean"),
        ("layer_{}_hidden",     "max",      "Hidden State Max"),
        ("layer_{}_mlp_output", "rms",      "MLP Output RMS"),
        ("layer_{}_attn_output","rms",      "Attention Output RMS"),
        ("layer_{}_hidden",     "std",      "Hidden State Std"),
    ]

    for idx, (comp_tmpl, metric, title) in enumerate(metric_configs):
        ax = axes[idx // 3][idx % 3]
        for mi, res in enumerate(results):
            name = res["model_name"]
            n_layers = res["num_hidden_layers"]
            stats = res["per_layer_stats"]
            vals = extract_layer_metric(stats, n_layers, comp_tmpl, metric)
            layers = list(range(n_layers))
            # Filter None
            valid = [(l, v) for l, v in zip(layers, vals) if v is not None]
            if valid:
                ls, vs = zip(*valid)
                ax.plot(ls, vs, label=name,
                        color=COLORS[mi % len(COLORS)],
                        linestyle=LINESTYLES[mi % len(LINESTYLES)],
                        linewidth=1.5, alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(output_dir, f"{series_name}_activation_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[{series_name}] Saved comparison plot: {out_path}")

    # ── Plot 2: Max activation (absolute) per layer ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{series_name} — Max Absolute Activation per Layer", fontsize=14, fontweight="bold")

    for ci, (comp_tmpl, label) in enumerate([
        ("layer_{}_hidden",     "Hidden State"),
        ("layer_{}_mlp_output", "MLP Output"),
        ("layer_{}_attn_output","Attention Output"),
    ]):
        ax = axes[ci]
        for mi, res in enumerate(results):
            name = res["model_name"]
            n_layers = res["num_hidden_layers"]
            stats = res["per_layer_stats"]
            max_vals = extract_layer_metric(stats, n_layers, comp_tmpl, "max")
            min_vals = extract_layer_metric(stats, n_layers, comp_tmpl, "min")
            layers = list(range(n_layers))
            # Absolute max = max(|max|, |min|)
            abs_max = []
            valid_layers = []
            for l, mx, mn in zip(layers, max_vals, min_vals):
                if mx is not None and mn is not None:
                    abs_max.append(max(abs(mx), abs(mn)))
                    valid_layers.append(l)
            if valid_layers:
                ax.plot(valid_layers, abs_max, label=name,
                        color=COLORS[mi % len(COLORS)],
                        linestyle=LINESTYLES[mi % len(LINESTYLES)],
                        linewidth=1.5, alpha=0.85)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Layer")
        ax.set_ylabel("|Max Activation|")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(output_dir, f"{series_name}_max_activation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[{series_name}] Saved max activation plot: {out_path}")

    # ── Plot 3: Per-model detailed view ──────────────────────────────────
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 1, figsize=(16, 5 * n_models))
    if n_models == 1:
        axes = [axes]
    fig.suptitle(f"{series_name} — Per-Model Activation Detail", fontsize=14, fontweight="bold")

    for mi, res in enumerate(results):
        ax = axes[mi]
        name = res["model_name"]
        n_layers = res["num_hidden_layers"]
        stats = res["per_layer_stats"]
        layers = list(range(n_layers))

        for comp_tmpl, metric, label, color in [
            ("layer_{}_hidden",      "rms", "Hidden RMS",     "#1f77b4"),
            ("layer_{}_mlp_output",  "rms", "MLP Output RMS", "#ff7f0e"),
            ("layer_{}_attn_output", "rms", "Attn Output RMS","#2ca02c"),
            ("layer_{}_hidden",      "abs_mean","Hidden AbsMean","#d62728"),
        ]:
            vals = extract_layer_metric(stats, n_layers, comp_tmpl, metric)
            valid = [(l, v) for l, v in zip(layers, vals) if v is not None]
            if valid:
                ls, vs = zip(*valid)
                ax.plot(ls, vs, label=label, color=color, linewidth=1.5, alpha=0.8)

        ax.set_title(f"{name} (layers={n_layers}, hidden={res['hidden_size']}, "
                     f"global_max={res.get('global_max_activation', 'N/A'):.1f})",
                     fontsize=11)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(output_dir, f"{series_name}_per_model_detail.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[{series_name}] Saved per-model detail plot: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=str, choices=SERIES_LIST)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    series_to_plot = SERIES_LIST if args.all else [args.series] if args.series else []
    if not series_to_plot:
        parser.print_help()
        return

    for series in series_to_plot:
        print(f"\n{'='*50}")
        print(f"Plotting: {series}")
        print(f"{'='*50}")
        results = load_series_results(series, args.results_dir)
        if results:
            print(f"Found {len(results)} model results: {[r['model_name'] for r in results]}")
            output_dir = os.path.join(args.results_dir, series, "plots")
            plot_series_comparison(series, results, output_dir)
        else:
            print(f"No results found for {series}")


if __name__ == "__main__":
    main()
