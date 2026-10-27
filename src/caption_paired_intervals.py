#!/usr/bin/env python3
"""
caption_paired_intervals.py
===========================
Paired bootstrap intervals for the VizWiz-Captions CIDEr-D comparisons, over the eligible validation images.

CIDEr-D of a set of captions is the mean of the per-image CIDEr-D scores, so the comparison is bootstrapped over images
(10,000 resamples, seed 2026), with the adapted models averaged over seeds per image, as for VizWiz-VQA.
Reported: adapted - base (generic prompt), prompt only (base accessibility - base generic), adapted - base
(accessibility prompt), and QLoRA - LoRA when QLoRA predictions are given.

Usage
-----
python caption_paired_intervals.py --gt annotations/val.json \
    --base_generic P --base_accessibility P --lora P7,P42,P123 [--qlora P7,P42,P123] --out caption_paired_intervals.json
"""
import argparse, json
import numpy as np
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

PRECANNED = "Quality issues are too severe to recognize visual content."


def boot(d, n=10000, seed=2026):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n // 500):
        out.append(d[rng.integers(0, len(d), (500, len(d)))].mean(axis=1))
    out = np.concatenate(out)
    return [round(float(d.mean()), 2), round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True); ap.add_argument("--base_generic", required=True)
    ap.add_argument("--base_accessibility", required=True); ap.add_argument("--lora", required=True)
    ap.add_argument("--qlora", default=None); ap.add_argument("--out", default="caption_paired_intervals.json")
    args = ap.parse_args()
    gt = json.load(open(args.gt)); refs = {}
    for a in gt["annotations"]:
        c = a["caption"].strip()
        if c == PRECANNED or a.get("is_rejected", False) or a.get("is_precanned", False):
            continue
        refs.setdefault(a["image_id"], []).append(c)
    ids = sorted(refs)
    tok = PTBTokenizer()
    g = tok.tokenize({i: [{"caption": c} for c in refs[i]] for i in ids})

    def per_image(path):
        pm = {p["image_id"]: p["caption"] for p in json.load(open(path))}
        r = tok.tokenize({i: [{"caption": pm[i]}] for i in ids})
        mean, scores = Cider().compute_score({i: g[i] for i in ids}, {i: r[i] for i in ids})
        return np.array(scores) * 100, mean * 100

    def seeds(paths):
        arr = [per_image(p) for p in paths.split(",")]
        return np.mean([a[0] for a in arr], axis=0), [round(float(a[1]), 2) for a in arr]

    bg, mg = per_image(args.base_generic); bc, mc = per_image(args.base_accessibility)
    lo, ml = seeds(args.lora)
    out = {"n_images": len(ids), "cider_d": {"base_generic": round(float(mg), 2), "base_accessibility": round(float(mc), 2),
           "lora_per_seed": ml, "lora_mean": round(float(lo.mean()), 2)},
           "mean_and_95ci": {"lora_minus_base_generic": boot(lo - bg), "prompt_only_base_accessibility_minus_generic": boot(bc - bg),
                             "lora_minus_base_accessibility": boot(lo - bc)}}
    if args.qlora:
        ql, mq = seeds(args.qlora)
        out["cider_d"].update({"qlora_per_seed": mq, "qlora_mean": round(float(ql.mean()), 2)})
        out["mean_and_95ci"].update({"qlora_minus_base_generic": boot(ql - bg), "qlora_minus_lora": boot(ql - lo)})
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
