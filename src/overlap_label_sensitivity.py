"""
overlap_label_sensitivity.py
============================
VizWiz-VQA accuracy and the LoRA / QLoRA gains over the base when the test items whose label repeats a
label of the other copy of the same photograph are excluded. The items come from overlap_label_audit.py
(overlap_label_matches.csv): "question+answer" = same photograph, same question and same consensus
answer; "question" additionally includes the items with the same question only.

  python src/overlap_label_sensitivity.py --matches results/overlap/overlap_label_matches.csv \
      --vqa_gt VQA_test.json --scorer_dir src \
      --preds base_gen=<generic>.json base_acc=<custom>.json \
              lora=<seed7>.json,<seed42>.json,<seed123>.json qlora=<seed7>.json,<seed42>.json,<seed123>.json \
      --out results/overlap/overlap_label_sensitivity.json
"""
import argparse
import csv
import json
import statistics as st
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", required=True)
    ap.add_argument("--vqa_gt", required=True)
    ap.add_argument("--preds", nargs="+", required=True, help="label=file[,file,...] (base_gen, base_acc, lora, qlora)")
    ap.add_argument("--scorer_dir", default="src")
    ap.add_argument("--out", default="overlap_label_sensitivity.json")
    args = ap.parse_args()
    sys.path.insert(0, args.scorer_dir)
    from score_vizwiz_vqa import vqa_accuracy

    gt = {g["image"]: g for g in json.load(open(args.vqa_gt))}
    ans = lambda g: [a["answer"] if isinstance(a, dict) else a for a in g["answers"]]
    runs = {}
    for item in args.preds:
        label, files = item.split("=", 1)
        runs[label] = [{x["image"]: vqa_accuracy(x["answer"], ans(gt[x["image"]])) for x in json.load(open(f))}
                       for f in files.split(",")]
    rows = [r for r in csv.DictReader(open(args.matches, encoding="utf-8")) if r["partition"] == "vqa_test"]
    sets = {"all_test_items": set(),
            "without_same_question_and_answer": {r["image"] for r in rows if r["match"] == "question+answer"},
            "without_same_question": {r["image"] for r in rows}}
    mean = lambda scores, ex: st.mean(100 * st.mean(v for k, v in s.items() if k not in ex) for s in scores)
    out = {}
    for name, ex in sets.items():
        acc = {lab: round(mean(sc, ex), 2) for lab, sc in runs.items()}
        out[name] = {"items_excluded": len(ex), "accuracy": acc}
        for lab in ("lora", "qlora"):
            if lab in acc:
                for b in ("base_gen", "base_acc"):
                    if b in acc:
                        out[name][f"{lab}_gain_over_{b}"] = round(acc[lab] - acc[b], 2)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
