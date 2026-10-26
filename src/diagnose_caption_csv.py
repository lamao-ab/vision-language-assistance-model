#!/usr/bin/env python3
"""
diagnose_caption_csv.py
=======================
Finds out how the released VizWiz-Captions image names relate to the names in the metadata CSV, using the
VALIDATION split where the truth is recoverable from the captions (official val.json captions are known).

For every released val image it finds the CSV image whose captions match (exact strings of the
IMG_CAPTION_SPELLCHECKED column, pre-canned captions ignored) and then reports
  * how many val images are matched uniquely, ambiguously, or not at all;
  * the relation between the released index and the CSV number (constant offset? monotone? random?),
    per CSV name family (VizWiz_train / VizWiz_val / VizWiz_v2);
  * the agreement between the CSV text votes (TEXT_DETECT >= 3 of 5) and the released "text_detected" flag
    under the TRUE mapping. This validates the flag check used for the test split: if it is high here, the
    low value on test means the test numbering is not a simple shift; if it is low here too, the released
    flags are not a usable check;
  * the share of images flagged text_detected in val.json and test.json, and what keys test.json carries.

Usage
-----
python diagnose_caption_csv.py --csv vizwiz_caps_meta.csv --val_json annotations/val.json \
    --test_json annotations/test.json --out diagnose_report.json
"""
import argparse
import collections
import json
import re

import numpy as np
import pandas as pd

PRE = "Quality issues are too severe to recognize visual content."


def num_of(name):
    return int(re.search(r"_(\d+)\.jpg$", name).group(1))


def fam_of(name):
    return re.match(r"(VizWiz_[a-z0-9]+)_", name).group(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--val_json", required=True)
    ap.add_argument("--test_json", default=None)
    ap.add_argument("--out", default="diagnose_report.json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, usecols=["IMG", "IMG_CAPTION_SPELLCHECKED", "TEXT_DETECT", "TRAIN_VAL_TEST"])
    cap2imgs = collections.defaultdict(set)
    for img, cap in zip(df["IMG"], df["IMG_CAPTION_SPELLCHECKED"]):
        if isinstance(cap, str) and cap.strip() != PRE:
            cap2imgs[cap.strip()].add(img)
    votes = df.groupby("IMG")["TEXT_DETECT"].apply(lambda s: int(s.fillna(0).sum()))

    val = json.load(open(args.val_json))
    caps = collections.defaultdict(list)
    for a in val["annotations"]:
        caps[a["image_id"]].append(a["caption"].strip())
    pairs, unmatched, ambiguous = [], 0, 0
    for im in val["images"]:
        real = [c for c in caps[im["id"]] if c != PRE]
        cnt = collections.Counter()
        for c in real:
            for ci in cap2imgs.get(c, ()):
                cnt[ci] += 1
        need = max(1, (len(real) + 1) // 2)
        if not cnt or cnt.most_common(1)[0][1] < need:
            unmatched += 1
            continue
        top = cnt.most_common(1)[0][1]
        best = [k for k, v in cnt.items() if v == top]
        if len(best) > 1:
            ambiguous += 1
            continue
        pairs.append((im["file_name"], best[0], im.get("text_detected")))

    rep = {"val_images": len(val["images"]), "matched_uniquely": len(pairs), "ambiguous": ambiguous, "unmatched": unmatched}
    rows = pd.DataFrame(pairs, columns=["released", "csv", "flag"])
    rows["rel_idx"] = rows["released"].map(num_of)
    rows["csv_num"] = rows["csv"].map(num_of)
    rows["family"] = rows["csv"].map(fam_of)
    rows["diff"] = rows["csv_num"] - rows["rel_idx"]
    rows["votes"] = rows["csv"].map(votes)
    fam = {}
    for f, g in rows.groupby("family"):
        top_diff = g["diff"].value_counts()
        fam[f] = {"n": int(len(g)), "most_common_offset": int(top_diff.index[0]), "share_with_that_offset": round(float(top_diff.iloc[0] / len(g)), 4),
                  "spearman_released_vs_csv": round(float(g["rel_idx"].corr(g["csv_num"], method="spearman")), 4),
                  "csv_number_range": [int(g["csv_num"].min()), int(g["csv_num"].max())],
                  "released_index_range": [int(g["rel_idx"].min()), int(g["rel_idx"].max())]}
    rep["by_csv_family"] = fam
    has_flag = rows["flag"].notna()
    rep["text_flag_agreement_under_TRUE_val_mapping"] = (
        round(float(((rows["votes"] >= 3) == rows["flag"].astype(bool))[has_flag].mean()), 4) if has_flag.any() else None)
    rep["val_json_text_detected_share"] = round(float(np.mean([bool(i.get("text_detected", False)) for i in val["images"]])), 4)
    rep["csv_text_votes_ge3_share_all_images"] = round(float((votes >= 3).mean()), 4)
    if args.test_json:
        test = json.load(open(args.test_json))
        rep["test_json_image_keys"] = sorted(test["images"][0].keys())
        rep["test_json_text_detected_share"] = round(float(np.mean([bool(i.get("text_detected", False)) for i in test["images"]])), 4)
        tn = df[df["TRAIN_VAL_TEST"].astype(str).str.upper() == "TEST"]["IMG"].unique()
        rep["csv_test_text_votes_ge3_share"] = round(float((votes[tn] >= 3).mean()), 4)
    rep["first_pairs"] = rows[["released", "csv"]].head(12).values.tolist()
    json.dump(rep, open(args.out, "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
