#!/usr/bin/env python3
"""
build_evaluation_manifest.py
============================
Builds the evaluation manifest: one row per evaluated run, linking the reported score to its checkpoint, representation,
precision, prompt, decoding settings, script, scorer and prediction file (with its SHA-256 and the number of items scored).

Usage
-----
python build_evaluation_manifest.py --preds_dir results/predictions --vqa_gt VQA_test.json --caps_gt annotations/val.json \
    --scorer_dir src --calibration vqa_test_predictions_with_confidence.json --out results/evaluation_manifest.csv
Prediction files are looked up by name in --preds_dir (vizwiz_vqa_test_predictions_*.json, vizwiz_caption_val_predictions_*.json).
"""
import argparse, csv, hashlib, json, os, sys
import numpy as np

HUB = "lamao-ab/paligemma-blind-assist-qlora-merged-v1@0a66cea"
BASE = "google/paligemma-3b-mix-224"
EPOCH2 = "adapter from checkpoint-2152 (end of epoch 2, best validation loss)"
RUNS = [  # (run id, file stem, representation, checkpoint, seed, precision, prompt, batch, script, note)
    ("base-generic", "base_generic", "base model", BASE, "", "bf16", "generic", 128, "src/evaluate_vizwiz.py", ""),
    ("base-accessibility", "base_custom", "base model", BASE, "", "bf16", "accessibility", 128, "src/evaluate_vizwiz.py", ""),
    ("lora-s7", "lora_seed7_custom", "unmerged LoRA adapter over bf16 backbone", EPOCH2, 7, "bf16", "accessibility", 128, "src/evaluate_vizwiz.py", ""),
    ("lora-s42", "lora_seed42_custom", "unmerged LoRA adapter over bf16 backbone", EPOCH2, 42, "bf16", "accessibility", 128, "src/evaluate_vizwiz.py", ""),
    ("lora-s123", "lora_seed123_custom", "unmerged LoRA adapter over bf16 backbone", EPOCH2, 123, "bf16", "accessibility", 128, "src/evaluate_vizwiz.py", ""),
    ("qlora-s7", "qlora_seed7_custom", "unmerged QLoRA adapter over 4-bit NF4 backbone", EPOCH2, 7, "4-bit", "accessibility", 128, "src/evaluate_vizwiz.py", ""),
    ("qlora-s42", "qlora_seed42_custom", "unmerged QLoRA adapter over 4-bit NF4 backbone", EPOCH2, 42, "4-bit", "accessibility", 128, "src/evaluate_vizwiz.py", ""),
    ("qlora-s123", "qlora_seed123_custom", "unmerged QLoRA adapter over 4-bit NF4 backbone", EPOCH2, 123, "4-bit", "accessibility", 128, "src/evaluate_vizwiz.py", ""),
    ("qlora-s123-exported", "qlora_seed123_merged_custom", "exported model: adapter merged into BF16 weights, requantized to 4-bit", HUB, 123, "4-bit", "accessibility", 128, "src/evaluate_vizwiz.py", ""),
]
CAL = ("qlora-s123-exported-calibration", "exported model: adapter merged into BF16 weights, requantized to 4-bit", HUB, 123, "4-bit", "accessibility", 32,
       "calibration notebook (confidence extraction)", "decoding settings of this run to be confirmed from the notebook")



def _safe(lst):
    """Copy of the guard inside src/score_captions_test.py: drop '|' (METEOR's field separator) and empty strings after tokenisation."""
    out_ = [" ".join(x.replace("|", " ").split()) for x in lst]
    out_ = [x for x in out_ if x]
    return out_ or ["unk"]

def sha(p):
    h = hashlib.sha256()
    for c in iter(lambda: open(p, "rb").read(1 << 20), b""):
        h.update(c); break
    h = hashlib.sha256(open(p, "rb").read()); return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds_dir", required=True); ap.add_argument("--vqa_gt", required=True); ap.add_argument("--caps_gt", required=True)
    ap.add_argument("--scorer_dir", default="src"); ap.add_argument("--calibration", default=None); ap.add_argument("--out", default="evaluation_manifest.csv")
    ap.add_argument("--caps_test_gt", default=None, help="reconstructed test references (test_captions_7974.json)")
    ap.add_argument("--test_preds_dir", default=None, help="folder with the vizwiz_caption_test_predictions_*_test.json files")
    args = ap.parse_args()
    sys.path.insert(0, args.scorer_dir)
    from score_vizwiz_vqa import vqa_accuracy
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    gt = json.load(open(args.vqa_gt)); cg = json.load(open(args.caps_gt)); refs = {}
    for a in cg["annotations"]:
        c = a["caption"].strip()
        if c == "Quality issues are too severe to recognize visual content." or a.get("is_rejected") or a.get("is_precanned"): continue
        refs.setdefault(a["image_id"], []).append(c)
    ids = sorted(refs); tok = PTBTokenizer(); g = tok.tokenize({i: [{"caption": c} for c in refs[i]] for i in ids})

    def vqa_score(path):
        pm = {r["image"]: r["answer"] for r in json.load(open(path))}
        return 100 * np.mean([vqa_accuracy(pm[x["image"]], [a["answer"] for a in x["answers"]]) for x in gt]), sum(1 for x in gt if x["image"] in pm)

    def cap_score(path):
        pm = {p["image_id"]: p["caption"] for p in json.load(open(path))}
        r = tok.tokenize({i: [{"caption": pm[i]}] for i in ids})
        return 100 * Cider().compute_score({i: g[i] for i in ids}, {i: r[i] for i in ids})[0], sum(1 for i in ids if i in pm)

    rows = []
    for rid, stem, rep, ckpt, seed, prec, prompt, bs, script, note in RUNS:
        for task, pre, fn, metric, expected in (("VizWiz-VQA test", "vizwiz_vqa_test_predictions_", vqa_score, "VizWiz accuracy", 8000),
                                                ("VizWiz-Captions validation", "vizwiz_caption_val_predictions_", cap_score, "CIDEr-D", 7542)):
            path = os.path.join(args.preds_dir, pre + stem + ".json")
            if not os.path.exists(path): continue
            score, n = fn(path)
            rows.append({"run": rid, "task": task, "representation": rep, "checkpoint": ckpt, "seed": seed, "precision": prec, "prompt": prompt,
                         "decoding": "greedy (do_sample=False), max_new_tokens 64", "batch_size": bs, "script": script,
                         "scorer": "src/score_vizwiz_vqa.py" if "VQA" in task else "src/score_captions.py (PTBTokenizer, CIDEr-D)",
                         "metric": metric, "score": round(float(score), 2), "n_scored": f"{n}/{expected}", "prediction_file": os.path.basename(path),
                         "prediction_sha256": sha(path), "note": note})
    if args.caps_test_gt and args.test_preds_dir:
        sys.path.insert(0, args.scorer_dir)
        from score_captions_test import _clean     # the scorer that produced the reported test scores
        tj = json.load(open(args.caps_test_gt)); trefs = {}
        for a in tj["annotations"]:
            c = _clean(a["caption"])
            if c == "Quality issues are too severe to recognize visual content." or a.get("is_rejected") or a.get("is_precanned"): continue
            trefs.setdefault(a["image_id"], []).append(c)
        tids = sorted(trefs); tgt = {i: _safe(v) for i, v in tok.tokenize({i: [{"caption": c} for c in trefs[i]] for i in tids}).items()}
        pre = "vizwiz_caption_test_predictions_"
        tfile = {"base-generic": "base_generic_test", "base-accessibility": "base_custom_test", "lora-s7": "lora_seed7_custom_test",
                 "lora-s42": "lora_seed42_custom_test", "lora-s123": "lora_seed123_custom_test", "qlora-s7": "qlora_seed7_test",
                 "qlora-s42": "qlora_seed42_test", "qlora-s123": "qlora_seed123_test", "qlora-s123-exported": "qlora_merged_seed123_test"}
        for rid, stem, rep, ckpt, seed, prec, prompt, bs, script, note in RUNS:
            path = os.path.join(args.test_preds_dir, pre + tfile[rid] + ".json")
            if not os.path.exists(path): continue
            pm = {p["image_id"]: (_clean(p["caption"]) or ".") for p in json.load(open(path))}
            r = {i: _safe(v) for i, v in tok.tokenize({i: [{"caption": pm[i]}] for i in tids}).items()}
            score = 100 * Cider().compute_score({i: tgt[i] for i in tids}, {i: r[i] for i in tids})[0]
            rows.append({"run": rid, "task": "VizWiz-Captions test (reconstructed references)", "representation": rep, "checkpoint": ckpt, "seed": seed,
                         "precision": prec, "prompt": prompt, "decoding": "greedy (do_sample=False), max_new_tokens 64", "batch_size": bs, "script": script,
                         "scorer": "src/score_captions_test.py (PTBTokenizer, CIDEr-D, ASCII-folded text)", "metric": "CIDEr-D", "score": round(float(score), 2),
                         "n_scored": f"{sum(1 for i in tids if i in pm)}/7807", "prediction_file": os.path.basename(path), "prediction_sha256": sha(path),
                         "note": "references reconstructed from the released metadata file (see results/caption_test/README.md)"})
    if args.calibration and os.path.exists(args.calibration):
        rid, rep, ckpt, seed, prec, prompt, bs, script, note = CAL
        score, n = vqa_score(args.calibration)
        rows.append({"run": rid, "task": "VizWiz-VQA test", "representation": rep, "checkpoint": ckpt, "seed": seed, "precision": prec, "prompt": prompt,
                     "decoding": "see note", "batch_size": bs, "script": script, "scorer": "src/score_vizwiz_vqa.py", "metric": "VizWiz accuracy",
                     "score": round(float(score), 2), "n_scored": f"{n}/8000", "prediction_file": os.path.basename(args.calibration),
                     "prediction_sha256": sha(args.calibration), "note": note})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for r in rows: print(f"{r['run']:<32}{r['task']:<28}{r['score']:>7}  {r['n_scored']}")
    print("saved ->", args.out, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
