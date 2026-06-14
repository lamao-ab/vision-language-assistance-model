"""
score_vizwiz_vqa.py
===================
Self-evaluation of VizWiz-VQA predictions against the OFFICIALLY RELEASED
test annotations (VQA_test.json), following the April 2026 deprecation of the
EvalAI server.

Implements the official VizWiz / VQA accuracy metric:
    acc(question) = average over all 10-choose-9 annotator subsets of
                    min(1, (# matching answers in the subset) / 3)
with the standard VQA answer normalisation (punctuation, digits, articles,
contractions). Reports OVERALL accuracy plus the per-answer-type breakdown
{overall, other, unanswerable, yes/no, number} to match the official format.

Ground truth: VQA_test.json
    https://vizwiz.cs.colorado.edu/VizWiz_all_answers/VQA_test.json
    Each item: {image, question, answers:[{answer, answer_confidence}*10],
                answer_type, answerable}

Predictions: a JSON list [{"image": "...jpg", "answer": "..."}]  (your script's
output format). The 'image' field is matched to the ground-truth 'image' field.

Usage
-----
    python score_vizwiz_vqa.py \
        --gt   outputs/predictions/vizwiz_vqa_data/VQA_test.json \
        --pred outputs/predictions/vizwiz_vqa_test_predictions.json
"""
import argparse
import itertools
import json
import re


# ── Official VQA answer normalisation (ported from the VQA v2 eval API) ────────
CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "im": "i'm", "ive": "i've", "isnt": "isn't", "its": "it's",
    "lets": "let's", "shes": "she's", "shouldnt": "shouldn't", "thats": "that's",
    "theres": "there's", "theyd": "they'd", "theyre": "they're", "wasnt": "wasn't",
    "werent": "weren't", "whats": "what's", "wheres": "where's", "wont": "won't",
    "wouldnt": "wouldn't", "youd": "you'd", "youre": "you're", "youve": "you've",
}
MANUAL_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
ARTICLES = {"a", "an", "the"}
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
PUNCT = [
    ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
    ">", "<", "@", "`", ",", "?", "!",
]


def process_punctuation(s: str) -> str:
    out = s
    for p in PUNCT:
        if (p + " " in s or " " + p in s) or (re.search(COMMA_STRIP, s) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = PERIOD_STRIP.sub("", out, re.UNICODE)
    return out


def process_digit_article(s: str) -> str:
    out = []
    for w in s.lower().split():
        w = MANUAL_MAP.get(w, w)
        if w not in ARTICLES:
            out.append(w)
    for i, w in enumerate(out):
        if w in CONTRACTIONS:
            out[i] = CONTRACTIONS[w]
    return " ".join(out)


def normalize(ans: str) -> str:
    ans = ans.replace("\n", " ").replace("\t", " ").strip()
    ans = process_punctuation(ans)
    ans = process_digit_article(ans)
    return ans


# ── Official accuracy: average over all 10-choose-9 annotator subsets ──────────
def vqa_accuracy(pred: str, gt_answers: list) -> float:
    pred_n = normalize(pred)
    gts = [normalize(a) for a in gt_answers]
    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1:]           # leave-one-out (10-choose-9)
        matching = sum(1 for g in others if g == pred_n)
        accs.append(min(1.0, matching / 3.0))
    return sum(accs) / len(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="VQA_test.json (released test annotations)")
    ap.add_argument("--pred", required=True, help="predictions JSON: [{image, answer}]")
    ap.add_argument("--out", default=None, help="optional: write results JSON here")
    args = ap.parse_args()

    gt = json.load(open(args.gt))
    pred = json.load(open(args.pred))

    # index predictions by image filename
    pred_by_img = {}
    for p in pred:
        img = p.get("image") or p.get("file_name") or p.get("image_id")
        pred_by_img[img] = p.get("answer", p.get("caption", ""))

    cats = ["yes/no", "number", "other", "unanswerable"]
    bucket = {c: [] for c in cats}
    overall = []
    missing = 0

    for item in gt:
        img = item["image"]
        if img not in pred_by_img:
            missing += 1
            continue
        gt_answers = [a["answer"] for a in item["answers"]]
        acc = vqa_accuracy(pred_by_img[img], gt_answers)
        overall.append(acc)
        atype = item.get("answer_type", "other")
        if atype in bucket:
            bucket[atype].append(acc)

    def pct(lst):
        return round(100.0 * sum(lst) / len(lst), 2) if lst else None

    result = {
        "overall": pct(overall),
        "other": pct(bucket["other"]),
        "unanswerable": pct(bucket["unanswerable"]),
        "yes/no": pct(bucket["yes/no"]),
        "number": pct(bucket["number"]),
        "n_scored": len(overall),
        "n_missing": missing,
    }
    print(json.dumps([result], indent=2))
    if args.out:
        json.dump([result], open(args.out, "w"), indent=2)
    if missing:
        print(f"\nWARNING: {missing} ground-truth questions had no matching "
              f"prediction (check the 'image' field matches).")


if __name__ == "__main__":
    main()
