"""
score_captions.py
=================
Score generated captions on the VALIDATION set using the official scorers.

VizWiz-Captions (validation):
    Uses the official VizWizEvalCap from github.com/Yinan-Zhao/vizwiz-caption,
    on the official annotations/val.json (COCO format, public references).
    Test references are NOT released in an evaluable reference format (the 2019
    test.json contains images only; the only test captions exist in a raw
    image-quality CSV), so captioning is reported on validation.

COCO-Captions (validation):
    Standard pycocoevalcap on the COCO val references.

Both report BLEU-1..4, METEOR, ROUGE-L, CIDEr-D (no SPICE, per paper).

Prediction format (COCO results):
    [{"image_id": <int>, "caption": "<text>"}]

Usage
-----
    # VizWiz val (uses the official val.json; no external repo needed)
    python score_captions.py --dataset vizwiz \
        --gt   annotations/val.json \
        --pred outputs/predictions/vizwiz_caption_val_predictions.json

    # COCO val
    python score_captions.py --dataset coco \
        --gt   annotations/captions_val2014.json \
        --pred outputs/predictions/coco_caps_val_results.json
"""
import argparse
import json
import sys


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
    for scorer, name in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(name, list):
            for n, s in zip(name, score):
                out[n] = round(s, 4)
        else:
            out[name] = round(score, 4)
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
    for scorer, name in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(name, list):
            for n, sc in zip(name, score):
                out[n] = round(sc, 4)
        else:
            out[name] = round(score, 4)
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
