#!/usr/bin/env python3
"""Build §4.1 Sun-style massive activation table.

Sun-style columns (parallel to Sun et al. 2024 Table 1):
  Family | Model | Layer | Top1 | Top2 | Top3 | LayerMedian | Ratio | n_pass

Selection rule (per model):
  Scan hidden layers. For each layer, compute the layer-wide passing
  signal using the max-ratio token (|x| > 100 AND ratio > 1000). Among
  layers that pass, pick the one with the largest max-ratio |x|.
  Emit Top-1/2/3 (signed) and layer-wide |x| median from that layer.

Edge case: if no single stored token satisfies both thresholds but
sun_n_pass > 0 at some hidden layer (can happen when only extremum
tokens are stored; rare — e.g. Qwen3.5-27B earlier), mark as aggregate
pass without representative.

Outputs next to this script:
    table_4_1.md
    table_4_1.tex
"""
import json, os, sys

ROOT = "/root/paddlejob/workspace/clx/atv_alz/results"
DIRS_STATS = {
    "Qwen2.5":    f"{ROOT}/Qwen2.5/json",
    "Qwen2.5-VL": f"{ROOT}/Qwen2.5-vl/json",
    "Qwen3":      f"{ROOT}/Qwen3/json",
    "Qwen3.5":    f"{ROOT}/Qwen3.5/json",
    "Gemma2":     f"{ROOT}/gemma2/json",
    "Gemma3":     f"{ROOT}/gemma3/json",
    "Ling":       f"{ROOT}/ling/json",
    "GPT-OSS":    f"{ROOT}/gpt_oss/json",
}
ORDER = [
    ("Qwen2.5", "Qwen2.5-1.5B"), ("Qwen2.5", "Qwen2.5-7B"), ("Qwen2.5", "Qwen2.5-32B"),
    ("Qwen2.5-VL", "Qwen2.5-VL-3B"), ("Qwen2.5-VL", "Qwen2.5-VL-7B"), ("Qwen2.5-VL", "Qwen2.5-VL-32B"),
    ("Qwen3", "Qwen3-1.7B"), ("Qwen3", "Qwen3-8B"), ("Qwen3", "Qwen3-30B-A3B"), ("Qwen3", "Qwen3-32B"),
    ("Qwen3.5", "Qwen3.5-0.8B"), ("Qwen3.5", "Qwen3.5-9B"),
    ("Qwen3.5", "Qwen3.5-27B"), ("Qwen3.5", "Qwen3.5-35B-A3B"),
    ("Gemma2", "gemma-2b"), ("Gemma2", "gemma-2-9b"), ("Gemma2", "gemma-2-27b"),
    ("Gemma3", "gemma-3-4b-it"), ("Gemma3", "gemma-3-27b-it"),
    ("Ling", "Ling-mini-5T"), ("Ling", "Ling-mini-10T"),
    ("Ling", "Ling-mini-15T"), ("Ling", "Ling-mini-20T"),
    ("GPT-OSS", "gpt-oss-20b"),
]

T_VAL, T_RATIO = 100.0, 1000.0


def load(family, name):
    """Return (activation_stats_dict, top5_stats_dict) or (None, None)."""
    base = DIRS_STATS[family]
    s_path = f"{base}/{name}_activation_stats.json"
    t_path = f"{base}/{name}_top5_stats.json"
    if not os.path.exists(s_path):
        return None, None
    stats = json.load(open(s_path))
    top5  = json.load(open(t_path)) if os.path.exists(t_path) else None
    return stats, top5


def pick_layer(stats):
    """Pick representative hidden layer passing Sun criterion.

    Returns dict:
      {"type": "row",   "layer": str, "abs": float, "ratio": float,
       "median": float (layer-wide |x| median),
       "q90": float,    # top 10% threshold (None if missing)
       "q99": float,    # top 1%  threshold
       "q999": float,   # top 0.1% threshold
       "n_pass": int}
      {"type": "note",  "layer": str, "n_pass": int, ...}
      None                                                     # model fails
    """
    pls = stats["per_layer_stats"]
    cands = []
    fallback = []
    for k, v in pls.items():
        if not k.endswith("_hidden"):
            continue
        if "sun_max_ratio" not in v:
            continue
        ra = v.get("sun_max_ratio_abs", 0.0)
        rr = v.get("sun_max_ratio", 0.0)
        layer_med = v.get("abs_median", None)
        q90  = v.get("abs_q90", None)
        q99  = v.get("abs_q99", None)
        q999 = v.get("abs_q999", None)
        npass = int(v.get("sun_n_pass", 0))
        if ra > T_VAL and rr > T_RATIO:
            cands.append((k, ra, rr, layer_med, q90, q99, q999, npass))
        elif npass > 0:
            fallback.append((k, npass, layer_med, q90, q99, q999))
    if cands:
        top = max(cands, key=lambda x: x[1])
        return {"type": "row", "layer": top[0], "abs": top[1],
                "ratio": top[2], "median": top[3],
                "q90": top[4], "q99": top[5], "q999": top[6],
                "n_pass": top[7]}
    if fallback:
        top = max(fallback, key=lambda x: x[1])
        return {"type": "note", "layer": top[0], "n_pass": top[1],
                "median": top[2], "q90": top[3], "q99": top[4], "q999": top[5]}
    return None


def get_top_signed(top5, layer_key, n):
    """Extract top-n signed activation values of the given layer.

    Returns list of n floats (signed) or [None]*n if unavailable.
    Prefers topk_signed (new, length 100) else falls back to top5_signed.
    """
    if top5 is None:
        return [None] * n
    pls = top5.get("per_layer_stats", {})
    v = pls.get(layer_key, {})
    signed = v.get("topk_signed") or v.get("top5_signed") or []
    if len(signed) < n:
        return (signed + [None] * n)[:n]
    return signed[:n]


def get_top3_signed(top5, layer_key):
    return get_top_signed(top5, layer_key, 3)


def topk_abs_mean(top5, layer_key, n):
    if top5 is None:
        return None
    pls = top5.get("per_layer_stats", {})
    v = pls.get(layer_key, {})
    abs_list = v.get("topk_abs") or v.get("top5_abs") or []
    if not abs_list:
        return None
    k = min(n, len(abs_list))
    return sum(abs_list[:k]) / k


def short_layer(name):
    return name.replace("_hidden", "").replace("layer_", "L")


def fmt_num(x, prec=0):
    if x is None:
        return "—"
    if abs(x) < 10:
        return f"{x:.3f}"
    return f"{x:.{prec}f}"


def main():
    rows = []
    for fam, name in ORDER:
        stats, top5 = load(fam, name)
        if stats is None:
            rows.append({"fam": fam, "name": name, "rep": None,
                         "top3": [None]*3, "layer_med": None,
                         "missing": True})
            continue
        rep = pick_layer(stats)
        top3 = [None, None, None]
        layer_med = None
        if rep is not None and rep["type"] == "row":
            top3 = get_top3_signed(top5, rep["layer"])
            layer_med = rep["median"]
        elif rep is not None and rep["type"] == "note":
            top3 = get_top3_signed(top5, rep["layer"])
            v = stats["per_layer_stats"].get(rep["layer"], {})
            layer_med = v.get("abs_median", None)
        rows.append({"fam": fam, "name": name, "rep": rep,
                     "top3": top3, "layer_med": layer_med,
                     "missing": False})

    n_pass = sum(1 for r in rows
                 if r["rep"] is not None and r["rep"]["type"] == "row")
    n_note = sum(1 for r in rows
                 if r["rep"] is not None and r["rep"]["type"] == "note")
    print(f"Sun 判据通过(明确代表): {n_pass}/{len(rows)}; 聚合通过: {n_note}")

    # --- Markdown ---
    md = ["| Family | Model | Layer | Top1 | Top2 | Top3 | Median | Ratio | n_pass |",
          "|--------|-------|:-----:|-----:|-----:|-----:|------:|-----:|------:|"]
    for r in rows:
        rep = r["rep"]
        if rep is None:
            tag = "miss" if r["missing"] else "fail"
            md.append(f"| {r['fam']} | {r['name']} | — | — | — | — | — | — | {tag} |")
            continue
        layer = short_layer(rep["layer"])
        t1, t2, t3 = r["top3"]
        med = r["layer_med"]
        if rep["type"] == "row":
            ratio = rep["ratio"]
            npass = rep["n_pass"]
            md.append(
                f"| {r['fam']} | **{r['name']}** | {layer} | "
                f"{fmt_num(t1)} | {fmt_num(t2)} | {fmt_num(t3)} | "
                f"{fmt_num(med, 3)} | **{ratio:.0f}** | {npass} |"
            )
        else:  # aggregate only
            md.append(
                f"| {r['fam']} | *{r['name']}* | {layer} | "
                f"{fmt_num(t1)} | {fmt_num(t2)} | {fmt_num(t3)} | "
                f"{fmt_num(med, 3)} | n/a | {rep['n_pass']} (agg) |"
            )
    md_text = "\n".join(md)

    # --- LaTeX ---
    tex = [r"\begin{table}[t]",
           r"\centering",
           r"\small",
           r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{llcrrrrrr}",
           r"\toprule",
           r"Family & Model & Layer & Top1 & Top2 & Top3 & "
           r"Median & Ratio & $n_{\text{pass}}$ \\",
           r"\midrule"]
    for r in rows:
        rep = r["rep"]
        model_esc = r["name"].replace("_", r"\_")
        if rep is None:
            tex.append(f"{r['fam']} & {model_esc} & --- & --- & --- & --- & --- & --- & --- \\\\")
            continue
        layer = short_layer(rep["layer"])
        t1, t2, t3 = r["top3"]
        med = r["layer_med"]
        if rep["type"] == "row":
            ratio = rep["ratio"]
            npass = rep["n_pass"]
            tex.append(
                f"{r['fam']} & \\textbf{{{model_esc}}} & {layer} & "
                f"{fmt_num(t1)} & {fmt_num(t2)} & {fmt_num(t3)} & "
                f"{fmt_num(med, 3)} & \\textbf{{{ratio:.0f}}} & {npass} \\\\"
            )
        else:
            tex.append(
                f"{r['fam']} & \\emph{{{model_esc}}} & {layer} & "
                f"{fmt_num(t1)} & {fmt_num(t2)} & {fmt_num(t3)} & "
                f"{fmt_num(med, 3)} & n/a & {rep['n_pass']}$^{{\\dagger}}$ \\\\"
            )
    tex += [r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Representative massive activation per model, "
            r"mirroring Sun et al.\ 2024 Table 1. "
            r"Top1/Top2/Top3 are the three largest absolute activations "
            r"(signed) of the selected hidden layer across all tokens "
            r"and feature dimensions. Median is the layer-wide median of "
            r"$|x|$. Ratio $=$ max-ratio token's $|x|/\text{per-token median}$ "
            r"(Sun's definition). "
            r"$n_{\text{pass}}$ counts tokens satisfying $|x|>100$ and "
            r"ratio $>1000$. "
            r"Dagger ($\dagger$): aggregate pass only (no stored representative token).}",
            r"\label{tab:massive-representative}",
            r"\end{table}"]
    tex_text = "\n".join(tex)

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "table_4_1.md"), "w") as f:
        f.write(md_text + "\n")
    with open(os.path.join(here, "table_4_1.tex"), "w") as f:
        f.write(tex_text + "\n")

    print("\n=== Markdown ===\n")
    print(md_text)


if __name__ == "__main__":
    main()
