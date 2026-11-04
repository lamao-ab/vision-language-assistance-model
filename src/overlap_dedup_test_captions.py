"""
overlap_dedup_test_captions.py
==============================
Deduplicated evaluation on the VizWiz-Captions TEST images.

Removes the test images whose photograph also occurs, under a DIFFERENT file name, in a training
partition of either task (identical SHA-256 content in overlap_pairs.csv, written by
audit_overlap.py), and scores the remaining images with src/score_captions.py (CIDEr-D, x100).
The captioning and VQA test images are the same files (test.zip), so if the audit has no "cap_test"
partition the "vqa_test" partition is used.

  python src/overlap_dedup_test_captions.py \
      --pairs results/overlap/overlap_pairs.csv \
      --caps_gt results/caption_test/test_captions_7974.json \
      --preds base_gen=<generic>.json lora=<seed7>.json,<seed42>.json,<seed123>.json \
              qlora=<seed7>.json,<seed42>.json,<seed123>.json \
      --gain lora,base_gen --out results/overlap/overlap_dedup_test_captions.json

A label with several comma-separated files is reported as mean and standard deviation over them.
"""
import argparse
import contextlib
import csv
import io
import json
import os
import statistics as st
import sys
import tempfile


def duplicate_names(pairs_csv, part, train_parts=("vqa_train", "cap_train")):
    """Names of images of `part` with identical content under another name in a training partition."""
    out = set()
    with open(pairs_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["type"] != "identical-content" or r["name_a"] == r["name_b"]:
                continue
            a, b = r["partition_a"], r["partition_b"]
            if a == part and b in train_parts:
                out.add(r["name_a"])
            if b == part and a in train_parts:
                out.add(r["name_b"])
    return out


def partition_present(pairs_csv, part):
    with open(pairs_csv, encoding="utf-8") as f:
        return any(part in (r["partition_a"], r["partition_b"]) for r in csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--caps_gt", required=True)
    ap.add_argument("--preds", nargs="+", required=True, help="label=file[,file,...]")
    ap.add_argument("--gain", default=None, help="a,b : report the gain of label a over label b")
    ap.add_argument("--partition", default="cap_test")
    ap.add_argument("--scorer_dir", default="src")
    ap.add_argument("--out", default="overlap_dedup_test_captions.json")
    args = ap.parse_args()

    sys.path.insert(0, args.scorer_dir)
    import score_captions as sc

    part = args.partition
    if not partition_present(args.pairs, part):
        print(f"[note] no '{part}' partition in {args.pairs}; using 'vqa_test' (same files as the caption test images)")
        part = "vqa_test"
    dups = duplicate_names(args.pairs, part)

    refs = json.load(open(args.caps_gt))
    ids_of = {im["file_name"]: im["id"] for im in refs["images"]}
    dup_ids = {ids_of[n] for n in dups if n in ids_of}
    keep = set(ids_of.values()) - dup_ids
    print(f"images with captions: {len(ids_of)} | with an identical photograph in training: {len(dup_ids)} | kept: {len(keep)}")

    tmp = tempfile.mkdtemp()
    gt_path = os.path.join(tmp, "refs.json")
    json.dump({"info": refs.get("info"),
               "images": [i for i in refs["images"] if i["id"] in keep],
               "annotations": [a for a in refs["annotations"] if a["image_id"] in keep]}, open(gt_path, "w"))

    result = {"images_with_captions": len(ids_of), "images_removed": len(dup_ids), "partition_used": part, "cider_d": {}}
    for item in args.preds:
        label, files = item.split("=", 1)
        scores = []
        for path in files.split(","):
            preds = [p for p in json.load(open(path)) if p["image_id"] in keep]
            pp = os.path.join(tmp, "pred.json")
            json.dump(preds, open(pp, "w"))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                r = sc.score_vizwiz(gt_path, pp)
            scores.append(100 * r["CIDEr"])
            print(f"{label:<10} {os.path.basename(path):<70} CIDEr-D {100 * r['CIDEr']:.2f} (n scored {r['n_images_scored']})", flush=True)
        result["cider_d"][label] = {"per_run": [round(s, 2) for s in scores], "mean": round(st.mean(scores), 2),
                                    "std": round(st.stdev(scores), 2) if len(scores) > 1 else None}
    if args.gain:
        a, b = args.gain.split(",")
        result["gain"] = {f"{a}_minus_{b}": round(result["cider_d"][a]["mean"] - result["cider_d"][b]["mean"], 2)}
    json.dump(result, open(args.out, "w"), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
