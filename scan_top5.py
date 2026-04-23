"""Scan top5 stats across all models, compute structural indicators.

For each model:
  1. Saturation:          top5 全相同 (bf16 上限标志)
  2. Concentration:       top1_abs / top5_abs_mean
  3. Sign consistency:    top5 signed 是否全同号
  4. Spread:              top5 绝对值相对 top1 的衰减

Also per-layer summary: which component carries top5, are they all same layer.
"""
import os, json, glob

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

series = sorted(os.listdir(ROOT))
rows = []
for s in series:
    d = os.path.join(ROOT, s, "json")
    if not os.path.isdir(d):
        continue
    for f in sorted(glob.glob(os.path.join(d, "*_top5_stats.json"))):
        try:
            j = json.load(open(f))
        except Exception as e:
            print("skip", f, e)
            continue
        abs5    = j.get("global_top5_abs")
        signed5 = j.get("global_top5_signed")
        if not abs5 or not signed5:
            continue
        top1_abs     = abs5[0]
        top5_mean    = sum(abs5) / len(abs5)
        saturated    = all(abs(a - abs5[0]) < 1e-9 for a in abs5)
        all_pos      = all(x >  0 for x in signed5)
        all_neg      = all(x <  0 for x in signed5)
        same_sign    = all_pos or all_neg
        sign_label   = "all-pos" if all_pos else "all-neg" if all_neg else "mixed"
        concentration = top1_abs / top5_mean if top5_mean > 0 else float("inf")
        decay5       = abs5[4] / abs5[0] if abs5[0] > 0 else 0.0

        # per-layer scan: find which layers hold values >= 0.9*top1_abs
        per_layer = j.get("per_layer_stats", {})
        hot_layers = []
        for name, ls in per_layer.items():
            la = ls.get("top5_abs") or []
            if la and la[0] >= 0.9 * top1_abs:
                hot_layers.append((name, la[0]))
        hot_layers.sort(key=lambda x: -x[1])

        rows.append({
            "series":       s,
            "model":        j["model_name"],
            "top1_abs":     top1_abs,
            "top5_mean":    top5_mean,
            "saturated":    saturated,
            "sign":         sign_label,
            "concentration":concentration,
            "decay5_over_1":decay5,
            "n_hot_layers": len(hot_layers),
            "hot_top1":     hot_layers[0][0] if hot_layers else None,
            "hot_top2":     hot_layers[1][0] if len(hot_layers) > 1 else None,
        })

print(f"{'model':<30} {'|x*|':>10} {'sat':>4} {'sign':>8} "
      f"{'conc':>6} {'decay':>6} {'hot':>4} {'top_layer':<30}")
print("-" * 110)
for r in rows:
    print(f"{r['model']:<30} {r['top1_abs']:>10.2f} "
          f"{'Y' if r['saturated'] else 'N':>4} {r['sign']:>8} "
          f"{r['concentration']:>6.3f} {r['decay5_over_1']:>6.3f} "
          f"{r['n_hot_layers']:>4} {str(r['hot_top1']):<30}")

# ── Aggregate stats ──────────────────────────────────────────
n = len(rows)
n_sat   = sum(r["saturated"] for r in rows)
n_sign  = sum(r["sign"] != "mixed" for r in rows)
n_pos   = sum(r["sign"] == "all-pos" for r in rows)
n_neg   = sum(r["sign"] == "all-neg" for r in rows)
n_localized = sum(r["n_hot_layers"] == 1 for r in rows)

print("\n--- Aggregate over", n, "models ---")
print(f"Saturated (top5 identical):      {n_sat}/{n}")
print(f"Sign-consistent top5:            {n_sign}/{n}  (all-pos {n_pos}, all-neg {n_neg})")
print(f"Localized (1 hot layer):         {n_localized}/{n}")
print(f"Mean concentration (top1/top5):  {sum(r['concentration'] for r in rows)/n:.3f}")
print(f"Mean decay5/1:                   {sum(r['decay5_over_1'] for r in rows)/n:.3f}")

# By-family sign structure
from collections import defaultdict
by_fam = defaultdict(list)
for r in rows:
    by_fam[r["series"]].append(r["sign"])
print("\n--- Sign by family ---")
for fam, signs in sorted(by_fam.items()):
    from collections import Counter
    print(f"{fam:<15} n={len(signs):<3} {dict(Counter(signs))}")
