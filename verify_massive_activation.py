"""Verify massive-activation existence per paper outline §4.

For each model in results0422 we check:
  1. Sun's strict definition (per-token, direct from JSON fields):
       layer has a token with |x*| > 100 AND |x*|/median > 1000
       (sun_n_pass > 0 at the layer level → model passes)
  2. Definition C (scalar, Outline §4):
       global |x*| > 100  AND  |x*| / abs_mean(peak_layer) > 100
       where peak_layer = argmax(|x*|) over activation-carrying components.
  3. Where the peak lives (component / layer index).
"""

import json, glob, os

ACT_SUFFIXES = ("_hidden", "_mlp_output", "_attn_output")
ACT_EXACT    = ("embedding_output", "final_layernorm")

def eligible(name):
    return name in ACT_EXACT or any(name.endswith(s) for s in ACT_SUFFIXES)

def component_of(name):
    if name in ACT_EXACT:           return name
    for s in ACT_SUFFIXES:
        if name.endswith(s):        return s.lstrip("_")
    return "?"

def layer_idx_of(name):
    if name.startswith("layer_"):
        try:    return int(name.split("_")[1])
        except: return -1
    return -1

rows = []
for p in sorted(glob.glob("/root/paddlejob/workspace/clx/atv_alz/results0422/*/json/*_activation_stats.json")):
    d = json.load(open(p))
    model = d["model_name"]
    series = os.path.basename(os.path.dirname(os.path.dirname(p)))
    stats = d["per_layer_stats"]

    # (1) Global peak location over activation carriers
    peak_abs = 0.0
    peak_val = 0.0
    peak_key = None
    for k, v in stats.items():
        if not eligible(k):
            continue
        for cand in (v["max"], v["min"]):
            if abs(cand) > peak_abs:
                peak_abs = abs(cand)
                peak_val = cand
                peak_key = k
    peak_entry = stats[peak_key]
    peak_abs_mean = peak_entry["abs_mean"]

    # Definition C
    ratio_C = peak_abs / peak_abs_mean if peak_abs_mean > 0 else float("inf")
    pass_C  = (peak_abs > 100.0) and (ratio_C > 100.0)

    # (2) Sun strict — scan all eligible layers' sun_* fields
    sun_pass_any_layer = False
    sun_best_ratio = 0.0
    sun_best_layer = None
    sun_total_pass_tokens = 0
    for k, v in stats.items():
        if not eligible(k):
            continue
        if "sun_peak_abs" not in v:
            continue
        npass = v.get("sun_n_pass", 0)
        sun_total_pass_tokens += int(npass)
        if npass > 0:
            sun_pass_any_layer = True
        # track best peak-token ratio (peak-based, matches outline intent)
        pr = v.get("sun_peak_token_ratio", 0.0)
        pa = v.get("sun_peak_abs", 0.0)
        if pa > 100 and pr > sun_best_ratio:
            sun_best_ratio = pr
            sun_best_layer = k

    rows.append({
        "series": series,
        "model": model,
        "peak_abs": peak_abs,
        "peak_val": peak_val,
        "peak_key": peak_key,
        "peak_component": component_of(peak_key),
        "peak_layer_idx": layer_idx_of(peak_key),
        "abs_mean_peak_layer": peak_abs_mean,
        "ratio_C": ratio_C,
        "pass_C_T100": pass_C,
        "pass_C_T200": (peak_abs > 200.0) and (ratio_C > 200.0),
        "sun_pass": sun_pass_any_layer,
        "sun_best_ratio": sun_best_ratio,
        "sun_best_layer": sun_best_layer,
        "sun_total_pass_tokens": sun_total_pass_tokens,
        "num_layers": d.get("num_hidden_layers"),
    })

# ── Print table ───────────────────────────────────────────────────────────
hdr = f"{'series':<12}{'model':<22}{'|x*|':>12}{'signed':>12}{'abs_mean':>10}{'C-ratio':>10}{'C@100':>7}{'C@200':>7}{'Sun':>5}{'Sun-ratio':>12}  {'peak@':<22}"
print(hdr); print("-"*len(hdr))
for r in rows:
    print(f"{r['series']:<12}{r['model']:<22}"
          f"{r['peak_abs']:>12.1f}{r['peak_val']:>12.1f}"
          f"{r['abs_mean_peak_layer']:>10.3f}"
          f"{r['ratio_C']:>10.1f}"
          f"{str(r['pass_C_T100']):>7}{str(r['pass_C_T200']):>7}"
          f"{str(r['sun_pass']):>5}"
          f"{r['sun_best_ratio']:>12.1f}  "
          f"{(r['peak_component']+'@L'+str(r['peak_layer_idx'])):<22}")

n = len(rows)
n_pass_C100 = sum(r["pass_C_T100"] for r in rows)
n_pass_C200 = sum(r["pass_C_T200"] for r in rows)
n_pass_sun  = sum(r["sun_pass"]    for r in rows)
print("-"*len(hdr))
print(f"Total models: {n}   "
      f"Def-C@100 pass: {n_pass_C100}/{n}   "
      f"Def-C@200 pass: {n_pass_C200}/{n}   "
      f"Sun-strict pass: {n_pass_sun}/{n}")

# Outline says 23/24 pass Def-C (1 miss = Qwen3.5-0.8B).  Highlight failures.
fails_C = [r for r in rows if not r["pass_C_T100"]]
fails_sun = [r for r in rows if not r["sun_pass"]]
if fails_C:
    print("\nFailing Definition C (|x*|>100 AND |x*|/abs_mean>100):")
    for r in fails_C:
        print(f"  - {r['model']}:  |x*|={r['peak_abs']:.2f},  "
              f"ratio={r['ratio_C']:.1f}")
if fails_sun:
    print("\nFailing Sun strict (no layer has token with |x*|>100 AND ratio>1000):")
    for r in fails_sun:
        print(f"  - {r['model']}:  best Sun ratio={r['sun_best_ratio']:.1f} at {r['sun_best_layer']}")

# Save CSV as well
import csv
out_csv = "/root/paddlejob/workspace/clx/atv_alz/results0422/massive_activation_verification.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\nCSV saved: {out_csv}")
