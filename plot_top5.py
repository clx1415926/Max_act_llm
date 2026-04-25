"""
Plot top-5 activation analysis results.
Reads JSON stats from analyze_top5.py and generates comparison plots.

Usage:
    python plot_top5.py --series <series_name> --results_dir <path>
    python plot_top5.py --all --results_dir <path>

Generates per-series:
    1. Top5 AbsMean comparison across models (hidden / mlp / attn)
    2. Per-model detailed subplot (3 components overlaid)
    3. Cross-series summary bar chart (global top5 mean per model)
"""

import os
import json
import glob
import re
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SERIES_LIST = ["Qwen2.5", "Qwen2.5-it", "Qwen3", "Qwen3.5", "gemma2", "gemma3", "gpt_oss", "ling", "Qwen2.5-vl"]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--"]


def _parse_model_size(name):
    # Match parameter billions: 1.5b, 7B, 32B, 30B-A3B
    m = re.search(r"[-_](\d+(?:\.\d+)?)[bB]", name)
    if m:
        return float(m.group(1))
    # Match token trillions: 5T, 10T (used in ling series)
    m = re.search(r"[-_](\d+(?:\.\d+)?)[tT](?:$|[-_])", name)
    if m:
        return float(m.group(1))
    return 0.0


def load_series_results(series_name, results_dir):
    series_dir = os.path.join(results_dir, series_name, "json")
    json_files = glob.glob(os.path.join(series_dir, "*_top5_stats.json"))
    results = []
    for jf in json_files:
        with open(jf) as f:
            results.append(json.load(f))
    results.sort(key=lambda r: _parse_model_size(r["model_name"]))
    return results


def extract_layer_top5_mean(stats, num_layers, component_template):
    """Extract top5_abs_mean across layers for a given component pattern."""
    values = []
    for i in range(num_layers):
        key = component_template.format(i)
        if key in stats:
            values.append(stats[key].get("top5_abs_mean", None))
        else:
            values.append(None)
    return values


def plot_series(series_name, results, output_dir):
    if not results:
        print(f"[{series_name}] No results found, skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── Plot 1: Top5 AbsMean comparison (3 subplots: hidden/mlp/attn) ────
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f"{series_name} — Top-5 Abs Mean per Layer", fontsize=14, fontweight="bold")

    components = [
        ("layer_{}_hidden",      "Hidden State"),
        ("layer_{}_mlp_output",  "MLP Output"),
        ("layer_{}_attn_output", "Attention Output"),
    ]

    for ci, (comp_tmpl, comp_label) in enumerate(components):
        ax = axes[ci]
        for mi, res in enumerate(results):
            name = res["model_name"]
            n_layers = res["num_hidden_layers"]
            vals = extract_layer_top5_mean(res["per_layer_stats"], n_layers, comp_tmpl)
            layers = list(range(n_layers))
            valid = [(l, v) for l, v in zip(layers, vals) if v is not None]
            if valid:
                ls, vs = zip(*valid)
                ax.plot(ls, vs, label=name,
                        color=COLORS[mi % len(COLORS)],
                        linestyle=LINESTYLES[mi % len(LINESTYLES)],
                        linewidth=1.5, alpha=0.85)
        ax.set_title(comp_label, fontsize=11)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Top-5 Abs Mean")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(output_dir, f"{series_name}_top5_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[{series_name}] Saved comparison plot: {out_path}")

    # ── Plot 2: Per-model detail (all 3 components overlaid) ─────────────
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 1, figsize=(16, 5 * n_models))
    if n_models == 1:
        axes = [axes]
    fig.suptitle(f"{series_name} — Per-Model Top-5 Detail", fontsize=14, fontweight="bold")

    comp_colors = [
        ("layer_{}_hidden",      "Hidden",      "#1f77b4"),
        ("layer_{}_mlp_output",  "MLP Output",  "#ff7f0e"),
        ("layer_{}_attn_output", "Attn Output", "#2ca02c"),
    ]

    for mi, res in enumerate(results):
        ax = axes[mi]
        name = res["model_name"]
        n_layers = res["num_hidden_layers"]
        stats = res["per_layer_stats"]
        layers = list(range(n_layers))

        for comp_tmpl, label, color in comp_colors:
            vals = extract_layer_top5_mean(stats, n_layers, comp_tmpl)
            valid = [(l, v) for l, v in zip(layers, vals) if v is not None]
            if valid:
                ls, vs = zip(*valid)
                ax.plot(ls, vs, label=label, color=color, linewidth=1.5, alpha=0.8)

        ax.set_title(f"{name} (layers={n_layers}, hidden={res['hidden_size']}, "
                     f"global_top5_mean={res.get('global_top5_abs_mean', 0):.1f})",
                     fontsize=11)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Top-5 Abs Mean")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(output_dir, f"{series_name}_top5_per_model_detail.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[{series_name}] Saved per-model detail plot: {out_path}")


def plot_cross_series_summary(all_results, output_dir):
    """Bar chart: global top5 abs mean for all models across series."""
    if not all_results:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Collect data
    names = []
    values = []
    series_labels = []
    for series_name, results in all_results:
        for res in results:
            names.append(res["model_name"])
            values.append(res.get("global_top5_abs_mean", 0))
            series_labels.append(series_name)

    # Color by series
    series_color_map = {}
    for i, s in enumerate(SERIES_LIST):
        series_color_map[s] = COLORS[i % len(COLORS)]
    bar_colors = [series_color_map.get(s, "#999999") for s in series_labels]

    fig, ax = plt.subplots(figsize=(max(12, len(names) * 1.2), 6))
    x = np.arange(len(names))
    bars = ax.bar(x, values, color=bar_colors, alpha=0.85, edgecolor="white")

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Global Top-5 Abs Mean")
    ax.set_title("Cross-Series: Global Top-5 Absolute Activation Mean", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Legend for series
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=series_color_map[s], label=s) for s in SERIES_LIST if s in series_color_map]
    ax.legend(handles=legend_elements, fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "cross_series_top5_summary.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[cross-series] Saved summary plot: {out_path}")


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

    all_results = []
    for series in series_to_plot:
        print(f"\n{'='*50}")
        print(f"Plotting top-5: {series}")
        print(f"{'='*50}")
        results = load_series_results(series, args.results_dir)
        if results:
            print(f"Found {len(results)} model results: {[r['model_name'] for r in results]}")
            output_dir = os.path.join(args.results_dir, series, "plots")
            plot_series(series, results, output_dir)
            all_results.append((series, results))
        else:
            print(f"No top5 results found for {series}")

    # Cross-series summary
    if len(all_results) > 1:
        plot_cross_series_summary(all_results, os.path.join(args.results_dir, "plots"))


if __name__ == "__main__":
    main()
