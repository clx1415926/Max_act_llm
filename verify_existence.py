"""
Read per-model _activation_stats.json files and judge massive-activation
existence under Sun et al. (2024):

    An activation x_{t,j} is "massive" iff
        |x_{t,j}| > 100   AND   |x_{t,j}| / median_k |x_{t,k}| > 1000
    where the median is taken over the feature dimension of the token's
    hidden-state vector x_t.

    A model is judged to *exhibit* massive activations iff there exists at
    least one (layer, token) satisfying both conditions above.

Authoritative signal: analyze_model.py accumulates `sun_n_pass` per layer,
counting tokens that satisfy BOTH conditions at Sun's fixed thresholds
(T_val=100, T_ratio=1000). We sum `sun_n_pass` across all residual-stream
components and call the model PASS iff the sum is > 0.

"Hidden state" here is the residual stream between layers, i.e. the
components whose name ends with `_hidden` or equals `embedding_output` /
`final_layernorm`. `_attn_output` / `_mlp_output` are module outputs, not
the residual stream, and are intentionally excluded.

Usage:
  python verify_existence.py <results_root>
      e.g.  python verify_existence.py results
      scans <results_root>/<family>/json/*_activation_stats.json
"""

import os
import sys
import json
import glob
import argparse


# Sun's fixed thresholds (not user-tunable).
SUN_T_VAL = 100.0
SUN_T_RATIO = 1000.0

# Residual-stream components only (strict "hidden state").
SUN_SUFFIXES = ("_hidden",)
SUN_EXACT = ("embedding_output", "final_layernorm")


def is_hidden_state(name):
    if name in SUN_EXACT:
        return True
    return any(name.endswith(s) for s in SUN_SUFFIXES)


def judge_model(data):
    """Return a dict summarizing Sun's existence check for one model."""
    layers = data.get("per_layer_stats", {})

    best = {
        "ratio": 0.0, "abs": 0.0, "median": 0.0, "name": "",
        "n_pass": 0, "n_tokens": 0,
    }
    has_sun = False
    threshold_mismatch = False
    seen_tval = None
    seen_tratio = None

    for name, stats in layers.items():
        if "sun_max_ratio" not in stats:
            continue
        if not is_hidden_state(name):
            continue
        has_sun = True

        # Verify analyze-time thresholds match Sun's definition.
        tv = stats.get("sun_t_val")
        tr = stats.get("sun_t_ratio")
        if tv is not None:
            seen_tval = tv
            if abs(tv - SUN_T_VAL) > 1e-9:
                threshold_mismatch = True
        if tr is not None:
            seen_tratio = tr
            if abs(tr - SUN_T_RATIO) > 1e-9:
                threshold_mismatch = True

        best["n_pass"] += int(stats.get("sun_n_pass", 0))
        best["n_tokens"] += int(stats.get("sun_n_tokens", 0))
        r = stats.get("sun_max_ratio", 0.0)
        if r > best["ratio"]:
            best["ratio"] = r
            best["abs"] = stats.get("sun_max_ratio_abs", 0.0)
            best["median"] = stats.get("sun_max_ratio_median", 0.0)
            best["name"] = name

    return {
        "model":     data.get("model_name", "?"),
        "n_layers":  data.get("num_hidden_layers", 0),
        "has_sun":   has_sun,
        "abs":       best["abs"],
        "median":    best["median"],
        "ratio":     best["ratio"],
        "loc":       best["name"],
        "n_pass":    best["n_pass"],
        "n_tok":     best["n_tokens"],
        "pass":      (best["n_pass"] > 0) if has_sun else None,
        "tval":      seen_tval,
        "tratio":    seen_tratio,
        "tmismatch": threshold_mismatch,
    }


def fmt_pass(v):
    if v is True:
        return "PASS"
    if v is False:
        return "FAIL"
    return "N/A"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="results root (scans <root>/*/json/*_activation_stats.json)")
    args = ap.parse_args()

    pat = os.path.join(args.root, "*", "json", "*_activation_stats.json")
    files = sorted(glob.glob(pat))
    if not files:
        print(f"No files matched: {pat}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for fp in files:
        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception as e:
            print(f"skip {fp}: {e}", file=sys.stderr)
            continue
        rows.append(judge_model(data))

    print(f"# Sun thresholds: |x*| > {SUN_T_VAL}, ratio > {SUN_T_RATIO}")
    print(f"# Hidden state = residual stream "
          f"(suffixes={list(SUN_SUFFIXES)}, exact={list(SUN_EXACT)})")
    print(f"# {len(rows)} models scanned\n")

    sun_rows = [r for r in rows if r["has_sun"]]
    missing = [r for r in rows if not r["has_sun"]]
    if missing:
        print(f"Skipped {len(missing)} model(s) without sun_* fields "
              f"(re-run analyze_model.py to populate):")
        for r in missing:
            print(f"  {r['model']}")
        print("")

    if not sun_rows:
        print("Nothing to judge.")
        return

    # Warn if any JSON used thresholds different from Sun's fixed (100, 1000).
    mismatched = [r for r in sun_rows if r["tmismatch"]]
    if mismatched:
        print("WARNING: some JSONs were produced with non-Sun thresholds; "
              "n_pass counts may not reflect (100, 1000). Affected models:")
        for r in mismatched:
            print(f"  {r['model']}  (sun_t_val={r['tval']}, sun_t_ratio={r['tratio']})")
        print("")

    print("Sun existence check  (per-token |x_t|_inf / median_k |x_{t,k}|)")
    print(f"{'Model':<34} {'peak |x*|':>10} {'median':>9} {'max ratio':>10} "
          f"{'n_pass/n_tok':>22} {'verdict':>8}")
    print("-" * 100)
    n_pass_models = 0
    for r in sun_rows:
        frac = f"{r['n_pass']}/{r['n_tok']}"
        print(f"{r['model']:<34} {r['abs']:>10.2f} {r['median']:>9.4f} "
              f"{r['ratio']:>10.1f} {frac:>22} {fmt_pass(r['pass']):>8}")
        if r["pass"]:
            n_pass_models += 1
    print(f"\nMassive activations present: {n_pass_models}/{len(sun_rows)} models")


if __name__ == "__main__":
    main()
