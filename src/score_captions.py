"""
score_captions.py
=================
Score generated captions on the VALIDATION set using the official scorers.

VizWiz-Captions (validation):
    Scored with the standard pycocoevalcap pipeline (PTBTokenizer + BLEU/METEOR/
    ROUGE/CIDEr-D) on the official COCO-format annotations/val.json, applying the
    official reference filtering: pre-canned ("is_precanned") and rejected/spam
    ("is_rejected") captions are removed from the ground-truth pool. This matches
    the official VizWiz metric (which itself adapts the COCO caption API) without
    requiring the external vizwiz-caption repository.
    Test references are NOT released in an evaluable reference format (the 2019
    test.json contains images only; the only test captions exist in a raw
    image-quality CSV), so captioning is reported on validation.

COCO-Captions (validation):
    Standard pycocoevalcap (PTBTokenizer + BLEU/METEOR/ROUGE/CIDEr-D) on the COCO
    val references.

Both report BLEU-1..4, METEOR, ROUGE-L, CIDEr-D (no SPICE).

Both report BLEU-1..4, METEOR, ROUGE-L, CIDEr-D (no SPICE, per paper).

Prediction format (COCO results):
    [{"image_id": <int>, "caption": "<text>"}]

Usage
-----
    # VizWiz val (val.json is auto-downloaded by evaluate_vizwiz.py into
    # outputs/predictions/vizwiz_caps_data/annotations/val.json)
    python score_captions.py --dataset vizwiz \
        --gt   outputs/predictions/vizwiz_caps_data/annotations/val.json \
        --pred outputs/predictions/vizwiz_caption_val_predictions.json

    # COCO val (captions_val2014.json is downloaded by evaluate_benchmark.py)
    python score_captions.py --dataset coco \
        --gt   outputs/predictions/coco_data/annotations/captions_val2014.json \
        --pred outputs/predictions/coco_caption_val_predictions.json
"""
import argparse
import json
import sys


def caption_length_stats(captions):
    """Word-count statistics over a list of caption strings."""
    lengths = sorted(len(c.split()) for c in captions)
    n = len(lengths)
    if n == 0:
        return {}
    return {
        "n_captions":   n,
        "avg_words":    round(sum(lengths) / n, 2),
        "min_words":    lengths[0],
        "max_words":    lengths[-1],
        "median_words": lengths[n // 2],
    }


def score_coco(gt_path, pred_path):
    """Standard COCO caption scoring via pycocoevalcap (no SPICE)."""
    from pycocotools.coco import COCO
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    coco = COCO(gt_path)
    preds = json.load(open(pred_path))
    cocoRes = coco.loadRes(preds)

    gts = {iid: coco.imgToAnns[iid] for iid in cocoRes.getImgIds()}
    res = {iid: cocoRes.imgToAnns[iid] for iid in cocoRes.getImgIds()}

    tok = PTBTokenizer()
    gts = tok.tokenize(gts)
    res = tok.tokenize(res)

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    out = {}
    for scorer, names in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(names, list):
            for nm, sc in zip(names, score):
                out[nm] = round(sc, 4)
        else:
            out[names] = round(score, 4)
    out["length_stats"] = caption_length_stats([p["caption"] for p in preds])
    return out


# Pre-canned caption inserted by VizWiz when image quality is too poor;
# excluded from the reference pool per the official evaluation protocol.
PRECANNED = "Quality issues are too severe to recognize visual content."


def score_vizwiz(gt_path, pred_path, repo_path=None):
    """VizWiz caption scoring (validation) via the standard pycocoevalcap
    pipeline (same as COCO), with the official reference filtering:
    pre-canned and rejected captions are removed from the ground-truth pool.

    Uses the official COCO-format val.json. Identical scorers/tokenization to
    score_coco(), so results match the official metric; only the reference
    filtering is VizWiz-specific.
    """
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    gt = json.load(open(gt_path))
    preds = json.load(open(pred_path))

    # Build references per image_id, applying the official filtering:
    #   - drop pre-canned "quality issues" captions
    #   - drop rejected captions (is_rejected) if the field is present
    gts_raw = {}
    for ann in gt["annotations"]:
        cap = ann["caption"].strip()
        if cap == PRECANNED:
            continue
        if ann.get("is_rejected", False) or ann.get("is_precanned", False):
            continue
        gts_raw.setdefault(ann["image_id"], []).append(cap)

    # Only score images that have at least one valid reference AND a prediction
    pred_by_id = {p["image_id"]: p["caption"] for p in preds}
    common_ids = [i for i in gts_raw if i in pred_by_id]

    gts = {i: [{"caption": c} for c in gts_raw[i]] for i in common_ids}
    res = {i: [{"caption": pred_by_id[i]}] for i in common_ids}

    tok = PTBTokenizer()
    gts = tok.tokenize(gts)
    res = tok.tokenize(res)

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    out = {"n_images_scored": len(common_ids)}
    for scorer, names in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(names, list):
            for nm, sc in zip(names, score):
                out[nm] = round(sc, 4)
        else:
            out[names] = round(score, 4)
    out["length_stats"] = caption_length_stats(
        [pred_by_id[i] for i in common_ids])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["vizwiz", "coco"])
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--vizwiz_repo", default=None,
                    help="(deprecated, unused) kept for backward compatibility")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.dataset == "coco":
        result = score_coco(args.gt, args.pred)
    else:
        result = score_vizwiz(args.gt, args.pred)

    print(json.dumps(result, indent=2))
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
