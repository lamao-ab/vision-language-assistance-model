#!/usr/bin/env python3
"""
check_completeness.py - asserts that a prediction file covers every eligible item.

This script does NOT score anything and does NOT modify the scorers (score_vizwiz_vqa.py,
score_captions.py), so the reported scores remain exactly those produced by the pinned code.
It reports, for each prediction file,

    n_expected    expected number of eligible items (set with --expected_n)
    n_eligible    eligible items found in the ground truth
    n_predicted   distinct items with a prediction
    n_missing     eligible items without a prediction
    n_extra       predictions for items that are not eligible
    n_duplicate   items predicted more than once
    complete      True only if nothing is missing, nothing is duplicated and n_eligible == n_expected

and exits with status 2 if a run is incomplete.

Eligibility follows the scorers
  vqa   every question of VQA_test.json                                   (expected 8000)
  caps  VizWiz-Captions: images with at least one reference left after the official filtering
        (pre-canned "Quality issues ..." captions and is_rejected / is_precanned removed)  (expected 7542)
  coco  COCO captions: images that have at least one annotation          (expected 40504, val2014)

Single run:
  python check_completeness.py vqa  --gt VQA_test.json --pred preds.json
  python check_completeness.py caps --gt annotations/val.json --pred preds.json

All reported runs at once (writes the record that is released with the repository):
  python check_completeness.py batch --manifest completeness_manifest.json --out results/completeness_report.json
  manifest: [{"name": "LoRA seed 7", "task": "vqa", "gt": "VQA_test.json", "pred": "preds_lora_seed7.json"}, ...]
"""
import argparse
import json
import sys

PRECANNED = "Quality issues are too severe to recognize visual content."
DEFAULT_EXPECTED = {"vqa": 8000, "caps": 7542, "coco": 40504}


def eligible_and_predicted(task, gt_path, pred_path):
    gt, preds = json.load(open(gt_path)), json.load(open(pred_path))
    if task == "vqa":
        eligible = {it["image"] for it in gt}
        pred_ids = [p.get("image") or p.get("file_name") or p.get("image_id") for p in preds]
    elif task == "caps":
        eligible = set()
        for ann in gt["annotations"]:
            if ann["caption"].strip() == PRECANNED or ann.get("is_rejected", False) or ann.get("is_precanned", False):
                continue
            eligible.add(ann["image_id"])
        pred_ids = [p["image_id"] for p in preds]
    elif task == "coco":
        eligible = {ann["image_id"] for ann in gt["annotations"]}
        pred_ids = [p["image_id"] for p in preds]
    else:
        raise SystemExit("task must be vqa, caps or coco")
    return eligible, pred_ids


def check(task, gt_path, pred_path, expected_n=None):
    expected_n = DEFAULT_EXPECTED[task] if expected_n is None else expected_n
    eligible, pred_ids = eligible_and_predicted(task, gt_path, pred_path)
    distinct = set(pred_ids)
    missing, extra = eligible - distinct, distinct - eligible
    n_dup = len(pred_ids) - len(distinct)
    problems = []
    if len(eligible) != expected_n:
        problems.append(f"{len(eligible)} eligible items, expected {expected_n}")
    if missing:
        problems.append(f"{len(missing)} eligible items have no prediction")
    if n_dup:
        problems.append(f"{n_dup} duplicate predictions")
    return {"task": task, "gt": gt_path, "pred": pred_path, "n_expected": expected_n, "n_eligible": len(eligible),
            "n_predicted": len(distinct & eligible), "n_missing": len(missing), "n_extra": len(extra),
            "n_duplicate": n_dup, "complete": not problems, "problems": problems}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for t in ("vqa", "caps", "coco"):
        p = sub.add_parser(t)
        p.add_argument("--gt", required=True); p.add_argument("--pred", required=True)
        p.add_argument("--expected_n", type=int, default=None); p.add_argument("--out", default=None)
    b = sub.add_parser("batch")
    b.add_argument("--manifest", required=True); b.add_argument("--out", default="completeness_report.json")
    args = ap.parse_args()

    if args.mode == "batch":
        runs = json.load(open(args.manifest))
        results = []
        for r in runs:
            res = check(r["task"], r["gt"], r["pred"], r.get("expected_n"))
            res["name"] = r.get("name", r["pred"])
            results.append(res)
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"{'run':<28}{'task':<6}{'expected':>9}{'eligible':>9}{'predicted':>10}{'missing':>8}{'extra':>6}{'dup':>5}  complete")
        for r in results:
            print(f"{r['name']:<28}{r['task']:<6}{r['n_expected']:>9}{r['n_eligible']:>9}{r['n_predicted']:>10}{r['n_missing']:>8}{r['n_extra']:>6}{r['n_duplicate']:>5}  {r['complete']}")
        bad = [r for r in results if not r["complete"]]
        print(f"\n{len(results) - len(bad)} of {len(results)} runs complete -> {args.out}")
        sys.exit(2 if bad else 0)

    res = check(args.mode, args.gt, args.pred, args.expected_n)
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
    if not res["complete"]:
        print("INCOMPLETE RUN: " + "; ".join(res["problems"]))
        sys.exit(2)


if __name__ == "__main__":
    main()
