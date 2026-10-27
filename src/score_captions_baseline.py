"""
score_captions_baseline.py
==========================
Caption scoring for the zero-shot baselines. Derived from score_captions.py and
identical to it in scorers (BLEU-1..4, METEOR, ROUGE-L, CIDEr-D), reference
filtering, tokenization and score scaling, with one change required by
chat-model output:

  Predicted captions are whitespace-normalised before scoring (newlines, tabs,
  carriage returns and backticks collapsed to single spaces). Instruction-tuned
  models emit markdown and code fences; a caption containing a newline breaks
  METEOR, whose Java subprocess exchanges one line per caption over
  stdin/stdout and otherwise raises
  `ValueError: could not convert string to float`. Reference captions are
  untouched.

The number of captions affected is reported, so the sanitisation can be stated
in the paper.

Prediction format (COCO results):
    [{"image_id": <int>, "caption": "<text>"}]

Usage
-----
    python src/score_captions_baseline.py --dataset vizwiz \\
        --gt   outputs/eval_data/vizwiz_caps_data/annotations/val.json \\
        --pred outputs/predictions/vizwiz_caption_val_predictions_smolvlm2_zeroshot.json \\
        --out  outputs/predictions/score_smolvlm2_zeroshot.json

    python src/score_captions_baseline.py --dataset coco \\
        --gt   outputs/eval_data/coco_data/annotations/captions_val2014.json \\
        --pred outputs/predictions/coco_caption_val_predictions_smolvlm2_zeroshot.json
"""
import argparse
import json
import re


# Pre-canned caption inserted by VizWiz when image quality is too poor;
# excluded from the reference pool per the official evaluation protocol.
PRECANNED = "Quality issues are too severe to recognize visual content."


def clean_caption(text: str) -> str:
    """Collapse whitespace and strip markdown artefacts that break the tokenizer."""
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = text.replace("`", " ")
    return re.sub(r"\s+", " ", text).strip()


def load_and_clean(pred_path: str):
    """Load predictions, sanitise captions, and report how many were affected."""
    preds = json.load(open(pred_path))
    n_newline = sum(1 for p in preds if "\n" in p.get("caption", ""))
    n_fence = sum(1 for p in preds if "`" in p.get("caption", ""))
    for p in preds:
        p["caption"] = clean_caption(p.get("caption", ""))
    print(f"[sanitisation] {len(preds):,} predictions | "
          f"{n_newline} contained newlines | {n_fence} contained backticks")
    return preds


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
    """Standard COCO caption scoring via pycocoevalcap (no METEOR, no SPICE)."""
    from pycocotools.coco import COCO
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    coco = COCO(gt_path)
    preds = load_and_clean(pred_path)
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


def score_vizwiz(gt_path, pred_path, repo_path=None):
    """VizWiz caption scoring (validation) with the official reference filtering:
    pre-canned and rejected captions are removed from the ground-truth pool.
    Identical to score_captions.py apart from the omission of METEOR.
    """
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    gt = json.load(open(gt_path))
    preds = load_and_clean(pred_path)

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
