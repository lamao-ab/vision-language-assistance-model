"""
score_captions.py
=================
Score generated captions using the official scorers.

VizWiz-Captions (validation or test):
    Scored with the standard pycocoevalcap pipeline (PTBTokenizer + BLEU/METEOR/
    ROUGE/CIDEr-D) on a COCO-format reference file (official annotations/val.json,
    or the realigned test_captions_7974.json), applying the
    official reference filtering: pre-canned ("is_precanned") and rejected/spam
    ("is_rejected") captions are removed from the ground-truth pool. This matches
    the official VizWiz metric (which itself adapts the COCO caption API) without
    requiring the external vizwiz-caption repository.
    Test references: the official test.json contains images only; the released
    test captions (VizWiz_captions_project_all_metadata_test.csv) were realigned
    to the test images into test_captions_7974.json (same format as val.json).
    Images whose references are all pre-canned/rejected are not scored
    (val: 7,542 of 7,750 scored; test: 7,807 of 7,974).

COCO-Captions (validation):
    Standard pycocoevalcap (PTBTokenizer + BLEU/METEOR/ROUGE/CIDEr-D) on the COCO
    val references.

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

    # VizWiz test (realigned references)
    python score_captions.py --dataset vizwiz \
        --gt   outputs/eval_data/vizwiz_caps_data/test_captions_7974.json \
        --pred outputs/predictions/vizwiz_caption_test_predictions_<tag>.json

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


def _clean(text):
    """Remove control characters (incl. C1 \x80-\x9f) and line breaks, which
    desynchronise the METEOR Java subprocess."""
    import unicodedata
    text = "".join(" " if unicodedata.category(ch)[0] == "C" else ch for ch in str(text))
    return " ".join(text.split())


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
        cap = _clean(ann["caption"])
        if cap == PRECANNED:
            continue
        if ann.get("is_rejected", False) or ann.get("is_precanned", False):
            continue
        gts_raw.setdefault(ann["image_id"], []).append(cap)

    # Only score images that have at least one valid reference AND a prediction
    pred_by_id = {p["image_id"]: _clean(p["caption"]) or "." for p in preds}
    common_ids = [i for i in gts_raw if i in pred_by_id]
    missing = [i for i in gts_raw if i not in pred_by_id]
    if missing:
        print(f"WARNING: {len(missing)} images with references have no prediction "
              f"(e.g. image_id {missing[:5]}); they are NOT scored.", file=sys.stderr)

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
    out = {"n_images_scored": len(common_ids),
           "n_images_with_refs": len(gts_raw),
           "n_images_missing_prediction": len(missing)}
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
