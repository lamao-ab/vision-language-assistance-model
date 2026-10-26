#!/usr/bin/env python3
"""
paired_comparisons.py
=====================
Reproduces the paired VizWiz-VQA comparisons reported in the revision from the saved prediction files:
  * scores of the base model under both prompts and of LoRA / QLoRA (three-seed means), overall and by answer type
  * gains with paired bootstrap intervals (10,000 resamples over test questions, seed 2026):
        adapted - base(generic prompt), prompt only (base accessibility - base generic),
        adaptation at a matched prompt (adapted - base accessibility), QLoRA - LoRA
  * the decomposition of the overall gain into the prompt effect and the adaptation effect, by answer type
  * accuracy on answerable questions
  * the representation comparison for QLoRA seed 123: unmerged adapter vs exported (merged and requantized)
    model under identical settings, and the exported model in the calibration run

The metric is imported from score_vizwiz_vqa.py (unchanged since the audited commit).

Usage
-----
python paired_comparisons.py --gt VQA_test.json --preds_dir results/predictions --scorer_dir src \
    [--calibration vqa_test_predictions_with_confidence.json] --out results/paired_comparisons.json

File names expected in --preds_dir:
  vizwiz_vqa_test_predictions_{base_generic,base_custom}.json
  vizwiz_vqa_test_predictions_{lora,qlora}_seed{7,42,123}_custom.json
  vizwiz_vqa_test_predictions_qlora_seed123_merged_custom.json
"""
import argparse, json, os, sys
import numpy as np


def boot(d, n=10000, seed=2026):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n // 500):
        out.append(d[rng.integers(0, len(d), (500, len(d)))].mean(axis=1) * 100)
    out = np.concatenate(out)
    return [round(float(100 * d.mean()), 2), round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True); ap.add_argument("--preds_dir", required=True)
    ap.add_argument("--scorer_dir", default="src"); ap.add_argument("--calibration", default=None)
    ap.add_argument("--out", default="paired_comparisons.json")
    args = ap.parse_args()
    sys.path.insert(0, args.scorer_dir)
    from score_vizwiz_vqa import vqa_accuracy, normalize
    gt = json.load(open(args.gt)); N = len(gt)
    types = np.array([g["answer_type"] for g in gt]); answerable = np.array([g["answerable"] == 1 for g in gt])
    P = lambda s: os.path.join(args.preds_dir, f"vizwiz_vqa_test_predictions_{s}.json")

    def load(f): return {r["image"]: r["answer"] for r in json.load(open(f))}
    def score(pm): return np.array([vqa_accuracy(pm[g["image"]], [a["answer"] for a in g["answers"]]) for g in gt])
    def seeds(kind): return np.mean([score(load(P(f"{kind}_seed{s}_custom"))) for s in (7, 42, 123)], axis=0)

    bg, bc = score(load(P("base_generic"))), score(load(P("base_custom")))
    lo, ql = seeds("lora"), seeds("qlora")
    res = {"n": N, "scores": {k: round(float(100 * v.mean()), 2) for k, v in (("base_generic", bg), ("base_accessibility", bc), ("lora", lo), ("qlora", ql))}}
    res["by_type"] = {t: {k: round(float(100 * v[types == t].mean()), 2) for k, v in (("base_generic", bg), ("base_accessibility", bc), ("lora", lo), ("qlora", ql))} for t in ("yes/no", "number", "other", "unanswerable")}
    res["answerable_accuracy"] = {k: round(float(100 * v[answerable].mean()), 2) for k, v in (("base_generic", bg), ("base_accessibility", bc), ("lora", lo), ("qlora", ql))}
    g = {}
    for name, ad in (("lora", lo), ("qlora", ql)):
        g[name] = {"adapted_minus_base_generic": boot(ad - bg), "prompt_only": boot(bc - bg), "adaptation_at_matched_prompt": boot(ad - bc)}
    g["qlora_minus_lora"] = boot(ql - lo)
    res["gains_mean_and_95ci"] = g
    res["decomposition_by_type_lora"] = {t: {"prompt_contribution": round(float(100 * (bc - bg)[types == t].sum() / N), 2),
                                             "adaptation_contribution": round(float(100 * (lo - bc)[types == t].sum() / N), 2)} for t in res["by_type"]}

    # representation comparison, QLoRA seed 123
    ada, mer = load(P("qlora_seed123_custom")), load(P("qlora_seed123_merged_custom"))
    sa, sm = score(ada), score(mer)
    def compare(a, b, sa_, sb_):
        na = [normalize(a[x["image"]]) for x in gt]; nb = [normalize(b[x["image"]]) for x in gt]
        diff = np.array([x != y for x, y in zip(na, nb)]); d = sb_ - sa_
        aa = np.array([x == "unanswerable" for x in na]); ab = np.array([x == "unanswerable" for x in nb])
        return {"score_a": round(float(100 * sa_.mean()), 2), "score_b": round(float(100 * sb_.mean()), 2), "b_minus_a_and_95ci": boot(d),
                "answers_differ": int(diff.sum()), "b_better": int((d > 0).sum()), "b_worse": int((d < 0).sum()),
                "substantive_to_substantive": int((diff & ~aa & ~ab).sum()), "abstentions_a": int(aa.sum()), "abstentions_b": int(ab.sum())}
    rep = {"adapter_to_exported_same_settings": compare(ada, mer, sa, sm)}
    if args.calibration:
        cal = load(args.calibration); sc = score(cal)
        rep["exported_to_calibration_run"] = compare(mer, cal, sm, sc)
        rep["adapter_to_calibration_run"] = compare(ada, cal, sa, sc)
    res["representation_qlora_seed123"] = rep
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("scores", "answerable_accuracy", "gains_mean_and_95ci")}, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
