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
    # VizWiz val (clone the official repo first; pass its path)
    python score_captions.py --benchmark vizwiz \
        --gt   vizwiz-caption/annotations/val.json \
        --pred outputs/predictions/vizwiz_caps_val_results.json \
        --vizwiz_repo ./vizwiz-caption

    # COCO val
    python score_captions.py --benchmark coco \
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


def score_vizwiz(gt_path, pred_path, repo_path):
    """Official VizWiz caption scoring (validation)."""
    sys.path.insert(0, repo_path)
    from vizwiz_api.vizwiz import VizWiz
    try:
        from vizwiz_eval_cap.eval import VizWizEvalCap
    except ImportError:
        from vizwiz_eval_cap.evals import VizWizEvalCap

    vizwiz = VizWiz(gt_path, ignore_rejected=True, ignore_precanned=True)
    preds = json.load(open(pred_path))
    vizwizRes = vizwiz.loadRes(preds)
    eval_cap = VizWizEvalCap(vizwiz, vizwizRes)
    # evaluate without SPICE
    eval_cap.evaluate()
    return {k: round(v, 4) for k, v in eval_cap.eval.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["vizwiz", "coco"])
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--vizwiz_repo", default="./vizwiz-caption",
                    help="path to cloned Yinan-Zhao/vizwiz-caption repo")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.benchmark == "coco":
        result = score_coco(args.gt, args.pred)
    else:
        result = score_vizwiz(args.gt, args.pred, args.vizwiz_repo)

    print(json.dumps(result, indent=2))
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
