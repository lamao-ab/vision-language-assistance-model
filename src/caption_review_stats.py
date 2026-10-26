#!/usr/bin/env python3
"""
caption_review_stats.py
=======================
Single-rater statistics of the blinded caption review: rates per model with Wilson 95% intervals, paired
differences (LoRA minus base on the same images) with a bootstrap 95% interval (10,000 resamples, seed 0) and an
exact McNemar test, overall and by stratum.

Usage
-----
python caption_review_stats.py --sheet caption_review_SHEET_100.csv --key caption_review_KEY_100.csv \
    --out results/caption_review/caption_review_stats.json
"""
import argparse, json, math
import numpy as np, pandas as pd
from scipy.stats import binomtest


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = p + z * z / (2 * n); m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(100 * (c - m) / den, 1), round(100 * (c + m) / den, 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", required=True); ap.add_argument("--key", required=True); ap.add_argument("--out", default="caption_review_stats.json")
    args = ap.parse_args()
    S, K = pd.read_csv(args.sheet), pd.read_csv(args.key)
    d = S.merge(K[["row", "A_model", "B_model"]], on="row")
    rows = []
    for _, r in d.iterrows():
        for side in ("A", "B"):
            rows.append({"row": r["row"], "stratum": r["stratum"], "model": r[f"{side}_model"], "hall": r[f"{side}_hallucination"],
                         "miss": r[f"{side}_missed_safety_detail"], "rel": r[f"{side}_relevant"]})
    L = pd.DataFrame(rows)
    out = {"images": int(d["row"].nunique()), "strata": d["stratum"].value_counts().to_dict(),
           "ab_positions": {f"A={a},B={b}": int(n) for (a, b), n in d.groupby(["A_model", "B_model"]).size().items()}}
    crit = {"rel": "relevant", "hall": "hallucination", "miss": "missed_safety_detail"}
    out["rates"], out["paired"] = {}, {}
    W = L.pivot(index="row", columns="model", values=["hall", "miss", "rel"])
    for c, lab in crit.items():
        out["rates"][lab] = {}
        for m in ("base", "lora"):
            x = L[L.model == m][c].dropna(); k, n = int(x.sum()), len(x)
            out["rates"][lab][m] = {"k": k, "n": n, "pct": round(100 * k / n, 1), "wilson95": wilson(k, n)}
        a, b = W[c]["lora"], W[c]["base"]; ok = a.notna() & b.notna(); a, b = a[ok].astype(int), b[ok].astype(int); n = len(a)
        n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        arr = (a.values - b.values); rng = np.random.default_rng(0); bs = [arr[rng.integers(0, n, n)].mean() * 100 for _ in range(10000)]
        out["paired"][lab] = {"n": n, "lora_minus_base_pts": round(float(arr.mean() * 100), 1),
                              "bootstrap95": [round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)],
                              "discordant_lora_only": n10, "discordant_base_only": n01,
                              "exact_mcnemar_p": round(float(binomtest(n10, n10 + n01, 0.5).pvalue), 4) if n10 + n01 else None}
    out["by_stratum"] = {s: {lab: {m: f"{int(g[g.model == m][c].dropna().sum())}/{len(g[g.model == m][c].dropna())}" for m in ("lora", "base")}
                              for c, lab in crit.items()} for s, g in L.groupby("stratum")}
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("rates", "paired")}, indent=1))


if __name__ == "__main__":
    main()
