"""
analyze_caption_review.py — unblind and summarise the caption review
=====================================================================
Run AFTER caption_review_SHEET.csv has been filled in.

Joins the completed sheet with caption_review_KEY.csv, recovers which caption
belonged to which model, and reports per-model rates for the three axes, plus a
LaTeX table ready for the manuscript.

Scoring conventions expected in the sheet:
  *_hallucination        : 1 / 0
  *_missed_safety_detail : 1 / 0 / NA   (NA when the image has no such content)
  *_relevant             : 1 / 0

Run:  python3 analyze_caption_review.py
"""
import csv
from collections import defaultdict

SHEET = "caption_review_SHEET.csv"
KEY = "caption_review_KEY.csv"
AXES = ["hallucination", "missed_safety_detail", "relevant"]
LABEL = {"hallucination": "Hallucination",
         "missed_safety_detail": "Missed safety-relevant detail",
         "relevant": "Relevant to user's information need"}


def parse(v):
    v = (v or "").strip().upper()
    if v in ("NA", "N/A", ""):
        return None
    if v in ("1", "Y", "YES", "TRUE"):
        return 1
    if v in ("0", "N", "NO", "FALSE"):
        return 0
    raise ValueError(f"unrecognised score: {v!r}")


def main():
    key = {r["row"]: r for r in csv.DictReader(open(KEY))}
    rows = list(csv.DictReader(open(SHEET)))

    # model -> axis -> list of 0/1 (None excluded)
    scores = defaultdict(lambda: defaultdict(list))
    incomplete = []

    for r in rows:
        k = key[r["row"]]
        for slot in ("A", "B"):
            model = k[f"{slot}_model"]
            for axis in AXES:
                val = parse(r[f"{slot}_{axis}"])
                if val is None:
                    if axis != "missed_safety_detail":
                        incomplete.append((r["row"], slot, axis))
                    continue
                scores[model][axis].append(val)

    if incomplete:
        print(f"WARNING: {len(incomplete)} unscored cells (excluding NA on "
              f"missed_safety_detail). First few: {incomplete[:5]}\n")

    models = ["base", "lora"]
    name = {"base": "Base (\\texttt{mix})", "lora": "LoRA"}

    print(f"{'axis':<34}" + "".join(f"{m:>22}" for m in models))
    table = {}
    for axis in AXES:
        cells = []
        for m in models:
            v = scores[m][axis]
            pct = 100 * sum(v) / len(v) if v else float("nan")
            cells.append(f"{sum(v)}/{len(v)} ({pct:.0f}%)")
            table[(axis, m)] = (sum(v), len(v), pct)
        print(f"{LABEL[axis]:<34}" + "".join(f"{c:>22}" for c in cells))

    print("\n% ---- LaTeX table ----")
    print("\\begin{table}[!t]")
    print("\\caption{\\hl{Qualitative caption review on 50 VizWiz-Caps validation "
          "images. Captions from both models were presented in randomised order "
          "without model identity and scored by one author. Counts are affirmative "
          "judgements over the number of applicable captions.}}")
    print("\\label{tab:caption_review}")
    print("\\centering")
    print("\\resizebox{\\columnwidth}{!}{%")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("\\textbf{Criterion} & \\textbf{Base (\\texttt{mix})} & \\textbf{LoRA} \\\\")
    print("\\midrule")
    for axis in AXES:
        cells = []
        for m in models:
            n, d, pct = table[(axis, m)]
            cells.append(f"{n}/{d} ({pct:.0f}\\%)")
        print(f"{LABEL[axis]} & {cells[0]} & {cells[1]} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}%")
    print("}")
    print("\\end{table}")


if __name__ == "__main__":
    main()
