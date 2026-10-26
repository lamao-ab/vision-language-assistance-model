#!/usr/bin/env python3
"""
overlap_exposure_analysis.py
============================
Turns the output of audit_overlap.py (overlap_pairs.csv) into per-image exposure categories for the
two evaluation sets, and tests whether the results are sensitive to them.

Categories (identical content = same SHA-256; near-duplicates are NOT used because a 64-bit
difference hash also matches visually similar dark or blurry photographs)

  VizWiz-Captions validation (cap_val)
    mix_seen      the photo is a VQA-train or VQA-validation image (same file, or identical content under
                  another name). The mix checkpoint was fine-tuned on VizWiz VQA train+val (model card).
    dup_train     identical photo under a DIFFERENT file name in OUR training data (cap_train or vqa_train)
    clean         neither of the two
  VizWiz-VQA test (vqa_test)
    dup_train     identical photo under a different file name in our training data (vqa_train or cap_train)
    dup_any       ... or in vqa_val

Sensitivity analyses (optional, need predictions):
  * VQA: accuracy of each prediction file on all questions, without dup_train, without dup_any, and only on
    dup_train, plus the difference in the adaptation gain between duplicate and non-duplicate images.
  * Captions: CIDEr-D of each prediction file on all eligible images, without dup_train, without mix_seen and
    on the clean subset. CIDEr-D document frequencies are recomputed on each subset, so compare models within
    a subset, not across subsets.

Usage
-----
python overlap_exposure_analysis.py --pairs overlap_pairs.csv --out_dir results/overlap \
    [--vqa_gt VQA_test.json --scorer_dir src --vqa_preds base=pred1.json lora=seed7.json,seed42.json,seed123.json ...] \
    [--caps_gt annotations/val.json --caps_preds base=capA.json lora=capB.json ...]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

PRECANNED = "Quality issues are too severe to recognize visual content."


def partner_names(ident, part, others, same_file):
    d = ident[ident.same_name == same_file]
    out = set()
    for o in others:
        out |= set(d[(d.partition_a == part) & (d.partition_b == o)]["name_a"])
        out |= set(d[(d.partition_b == part) & (d.partition_a == o)]["name_b"])
    return out


def categories(pairs_csv, n_cap_val, n_vqa_test):
    df = pd.read_csv(pairs_csv)
    df["same_name"] = df["name_a"] == df["name_b"]
    ident = df[df.type == "identical-content"]
    same_vqaval = partner_names(ident, "cap_val", ["vqa_val"], True)
    mix_seen = same_vqaval | partner_names(ident, "cap_val", ["vqa_train", "vqa_val"], False)
    cap_dup_train = partner_names(ident, "cap_val", ["cap_train", "vqa_train"], False)
    t_train = partner_names(ident, "vqa_test", ["vqa_train", "cap_train"], False)
    t_any = t_train | partner_names(ident, "vqa_test", ["vqa_val"], False)
    return {"cap_val": {"n": n_cap_val, "same_file_as_vqa_val": sorted(same_vqaval), "mix_seen": sorted(mix_seen),
                        "dup_train": sorted(cap_dup_train)},
            "vqa_test": {"n": n_vqa_test, "dup_train": sorted(t_train), "dup_any": sorted(t_any)}}


def parse_named(items):
    return [tuple(s.split("=", 1)) for s in (items or [])]


def boot(d, n=10000, seed=2026):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n // 500):
        out.append(d[rng.integers(0, len(d), (500, len(d)))].mean(axis=1) * 100)
    out = np.concatenate(out)
    return float(100 * d.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def vqa_part(args, cats):
    sys.path.insert(0, args.scorer_dir)
    from score_vizwiz_vqa import vqa_accuracy
    gt = json.load(open(args.vqa_gt)); names = np.array([g["image"] for g in gt])
    dup_train = np.isin(names, cats["vqa_test"]["dup_train"]); dup_any = np.isin(names, cats["vqa_test"]["dup_any"])
    per = {}
    for label, paths in parse_named(args.vqa_preds):          # label=file1,file2,file3 averages per-question scores (seeds)
        arrs = []
        for path in paths.split(","):
            pm = {r["image"]: r["answer"] for r in json.load(open(path))}
            arrs.append([vqa_accuracy(pm[g["image"]], [a["answer"] for a in g["answers"]]) for g in gt])
        per[label] = np.mean(np.array(arrs), axis=0)
    subsets = {"all": np.ones(len(gt), bool), "without_dup_train": ~dup_train, "without_dup_any": ~dup_any, "only_dup_train": dup_train}
    table = {s: {"n": int(m.sum()), **{k: float(100 * v[m].mean()) for k, v in per.items()}} for s, m in subsets.items()}
    out = {"table": table}
    if len(per) >= 2 and args.gain:
        a, b = args.gain.split(",")
        d = per[a] - per[b]
        out["gain"] = {"definition": f"{a} minus {b}",
                       "all": boot(d), "without_dup_train": boot(d[~dup_train]),
                       "dup_train_only_mean": float(100 * d[dup_train].mean()), "non_dup_mean": float(100 * d[~dup_train].mean()),
                       "difference_dup_minus_non_dup": float(100 * (d[dup_train].mean() - d[~dup_train].mean()))}
        rng = np.random.default_rng(1)
        x, y = d[dup_train], d[~dup_train]
        diffs = [100 * (rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean()) for _ in range(5000)]
        out["gain"]["difference_ci95"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    return out


def caps_part(args, cats):
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    gt = json.load(open(args.caps_gt)); refs = {}
    for ann in gt["annotations"]:
        c = ann["caption"].strip()
        if c == PRECANNED or ann.get("is_rejected", False) or ann.get("is_precanned", False):
            continue
        refs.setdefault(ann["image_id"], []).append(c)
    fname = {im["id"]: im["file_name"] for im in gt["images"]}
    names = {i: fname[i] for i in refs}
    preds = {}
    for label, path in parse_named(args.caps_preds):
        preds[label] = {p["image_id"]: p["caption"] for p in json.load(open(path))}
    mix, dup = set(cats["cap_val"]["mix_seen"]), set(cats["cap_val"]["dup_train"])
    subsets = {"all_eligible": lambda n: True, "without_dup_train": lambda n: n not in dup,
               "without_mix_seen": lambda n: n not in mix, "clean (neither)": lambda n: n not in mix and n not in dup}
    out = {}
    for sname, keep in subsets.items():
        ids = [i for i in refs if keep(names[i])]
        row = {"n": len(ids)}
        for label, pm in preds.items():
            ok = [i for i in ids if i in pm]
            tok = PTBTokenizer()
            g = tok.tokenize({i: [{"caption": c} for c in refs[i]] for i in ok})
            r = tok.tokenize({i: [{"caption": pm[i]}] for i in ok})
            row[label] = float(100 * Cider().compute_score(g, r)[0])
        out[sname] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--audit", default=None, help="overlap_audit.json (for the evaluation-set sizes)")
    ap.add_argument("--out_dir", default="results/overlap")
    ap.add_argument("--vqa_gt"); ap.add_argument("--scorer_dir", default="src")
    ap.add_argument("--vqa_preds", nargs="*"); ap.add_argument("--gain", default=None, help="a,b : report gain a minus b")
    ap.add_argument("--caps_gt"); ap.add_argument("--caps_preds", nargs="*")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    n_cap, n_vqa = 7750, 8000
    if args.audit:
        a = json.load(open(args.audit))["partitions"]; n_cap, n_vqa = a["cap_val"]["n_names"], a["vqa_test"]["n_names"]
    cats = categories(args.pairs, n_cap, n_vqa)
    cv, vt = cats["cap_val"], cats["vqa_test"]
    mix, dup = set(cv["mix_seen"]), set(cv["dup_train"])
    summary = {"cap_val": {"images": n_cap, "same_file_as_vqa_val": len(cv["same_file_as_vqa_val"]), "mix_seen": len(mix),
                           "dup_train": len(dup), "clean": n_cap - len(mix | dup)},
               "vqa_test": {"images": n_vqa, "dup_train": len(vt["dup_train"]), "dup_any": len(vt["dup_any"])}}
    json.dump(cats, open(os.path.join(args.out_dir, "overlap_exposure_sets.json"), "w"))
    res = {"summary": summary}
    if args.vqa_gt and args.vqa_preds:
        res["vqa_sensitivity"] = vqa_part(args, cats)
    if args.caps_gt and args.caps_preds:
        res["caption_sensitivity"] = caps_part(args, cats)
    json.dump(res, open(os.path.join(args.out_dir, "overlap_exposure_results.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
