#!/usr/bin/env python3
"""
make_overlap_spec.py - builds the spec for audit_overlap.py from the folder that
data/prepare_dataset.py (download_and_extract_all) extracts.

Discovery follows prepare_dataset.py:
  * VQA annotation file     : a train.json / val.json that is a LIST whose first item has
                              "question" and "answers"
  * caption annotation file : a train.json / val.json that is a DICT with "annotations"
                              (in a folder whose path contains "annotations")
  * image folders           : the folder that holds the files named VizWiz_train_*, VizWiz_val_*
                              (and VizWiz_test_* if the test images are given)

Optional evaluation partition for VizWiz-VQA test (downloaded by data/download_eval_data.py):
  --vqa_test_json  .../vizwiz_vqa_data/VQA_test.json
  --test_images    .../vizwiz_vqa_data/test

Usage
-----
  python make_overlap_spec.py --vw_dir /content/vw_audit/vizwiz \
      [--vqa_test_json /content/eval_data/vizwiz_vqa_data/VQA_test.json \
       --test_images   /content/eval_data/vizwiz_vqa_data/test] \
      --out overlap_spec.json
"""
import argparse
import json
import os


def find_annotation(vw_dir, split, kind):
    for root, _, files in os.walk(vw_dir):
        for f in files:
            if f != f"{split}.json":
                continue
            p = os.path.join(root, f)
            try:
                data = json.load(open(p))
            except Exception:
                continue
            if kind == "vqa" and isinstance(data, list) and data and "question" in data[0] and "answers" in data[0]:
                return p
            if kind == "caps" and isinstance(data, dict) and "annotations" in data and "annotations" in root.lower():
                return p
    return None


def find_image_dir(vw_dir, prefix):
    for root, _, files in os.walk(vw_dir):
        if any(f.startswith(prefix) and f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
            return root
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vw_dir", required=True)
    ap.add_argument("--vqa_test_json", default=None)
    ap.add_argument("--test_images", default=None)
    ap.add_argument("--out", default="overlap_spec.json")
    args = ap.parse_args()

    train_dir = find_image_dir(args.vw_dir, "VizWiz_train_")
    val_dir = find_image_dir(args.vw_dir, "VizWiz_val_")
    spec, report = {}, {}
    for part, split, kind, imgs in (("vqa_train", "train", "vqa", train_dir), ("vqa_val", "val", "vqa", val_dir),
                                    ("cap_train", "train", "caps", train_dir), ("cap_val", "val", "caps", val_dir)):
        ann = find_annotation(args.vw_dir, split, kind)
        report[part] = {"annotations": ann, "images": imgs}
        if ann is None or imgs is None:
            raise SystemExit(f"could not find {part}: annotations={ann} images={imgs}")
        spec[part] = {"annotations": ann, "images": imgs}
    if args.vqa_test_json and args.test_images:
        spec["vqa_test"] = {"annotations": args.vqa_test_json, "images": args.test_images}
        report["vqa_test"] = spec["vqa_test"]
    json.dump(spec, open(args.out, "w"), indent=1)
    for k, v in report.items():
        print(f"{k:<10} annotations: {v['annotations']}\n{'':<10} images     : {v['images']}")
    print(f"\nspec written -> {args.out}")
    print("next: python audit_overlap.py --spec", args.out, "--eval cap_val vqa_test --phash")


if __name__ == "__main__":
    main()
