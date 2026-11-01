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
import os
import sys

# The PTB tokenizer and METEOR read Java's stdout line by line. Some hosts
# (e.g. Google Colab) make the JVM print container/cgroup warnings to stdout,
# which shifts every caption onto the wrong image. Send JVM logging to stderr.
_JVM_LOG = "-Xlog:disable -Xlog:all=warning:stderr"
if _JVM_LOG not in os.environ.get("JAVA_TOOL_OPTIONS", ""):
    os.environ["JAVA_TOOL_OPTIONS"] = (os.environ.get("JAVA_TOOL_OPTIONS", "") + " " + _JVM_LOG).strip()


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


_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u00b4": "'", "`": "'",
                         "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"})


def _clean(text):
    """Normalise a caption to plain ASCII on a single line.

    The PTB tokenizer and METEOR are Java subprocesses fed one caption per
    line; depending on the Java version/locale, non-ASCII or control
    characters can split or merge lines, which silently pairs captions with
    the wrong images and then crashes METEOR. Accents are folded
    (e.g. "Jagermeister"), curly quotes become straight quotes.
    """
    import unicodedata
    text = str(text).translate(_QUOTES)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = "".join(" " if not ch.isprintable() else ch for ch in text)
    return " ".join(text.split())


def _check_alignment(raw, tok, label):
    """Abort if tokenized captions no longer line up with their image ids."""
    import re
    def words(t):
        return re.findall(r"[a-z0-9]+", t.lower())
    bad = [i for i in raw
           if words(raw[i][0])[:1] and words(tok[i][0])[:1] != words(raw[i][0])[:1]]
    if len(bad) > 0.01 * len(raw):
        raise RuntimeError(
            f"{label}: {len(bad)}/{len(raw)} tokenized captions do not match their "
            f"image (e.g. image_id {bad[:3]}). The PTB tokenizer output is "
            "misaligned; check `java -version` and the system locale.")


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
    _check_alignment({i: [pred_by_id[i]] for i in common_ids}, res, "predictions")
    _check_alignment({i: gts_raw[i][:1] for i in common_ids},
                     {i: gts[i][:1] for i in common_ids}, "references")

    # METEOR talks to a Java process line by line: an empty string or a "|"
    # (its field separator) shifts the protocol and crashes it. PTBTokenizer
    # turns punctuation-only captions into "", so guard after tokenization.
    def _safe(lst):
        out_ = [" ".join(x.replace("|", " ").split()) for x in lst]
        out_ = [x for x in out_ if x]
        return out_ or ["unk"]
    gts = {i: _safe(v) for i, v in gts.items()}
    res = {i: _safe(v) for i, v in res.items()}
    n_empty_pred = sum(1 for v in res.values() if v == ["unk"])
    if n_empty_pred:
        print(f"WARNING: {n_empty_pred} predictions are empty after tokenization "
              "(scored as 'unk').", file=sys.stderr)

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
