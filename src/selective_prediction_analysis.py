#!/usr/bin/env python3
"""
selective_prediction_analysis.py
================================
Selective-prediction analysis of VizWiz-VQA predictions that carry a confidence value.

Outcome of a prediction at confidence threshold tau
---------------------------------------------------
  delivered answer     prediction is not "unanswerable" and confidence >= tau
  learned abstention   prediction is "unanswerable"       and confidence >= tau
  rejected             confidence < tau (the filter withholds the output)

Quality of a prediction
-----------------------
  k = number of the ten annotator answers that match the (normalised) prediction.
  The official VizWiz score is a function of k: 0, 0.3, 0.6, 0.9, 1.0 for k = 0, 1, 2, 3, >=4.
  wrong       : k = 0  (no annotator agrees)
  weak        : k = 1 or 2
  consensus   : k >= 3

Informative accepted set (Geifman & El-Yaniv style selection, restricted to substantive answers)
  A_tau = { i : confidence_i >= tau  and  prediction_i != "unanswerable" }

Threshold selection on held-out data
------------------------------------
Thresholds are selected on one random half of the test questions (split by image; VizWiz-VQA has
one question per image) and evaluated on the other half, repeated --repeats times (cross-fitting).
Selection rule for a target risk alpha: the largest set of top-confidence delivered answers whose
wrong rate (k = 0) on the selection half is <= alpha.

The metric below re-implements the official VizWiz normalisation and 10-choose-9 scoring and
reproduces the official scores (e.g. 73.95 for the base model).

Usage
-----
python selective_prediction_analysis.py --pred vqa_test_predictions_with_confidence.json \
    --gt VQA_test.json [--ref_pred vizwiz_vqa_test_predictions_qlora_seed123_custom.json] \
    --out_dir results/selective
"""
import argparse, csv, json, os, re
import numpy as np

CONTR = {"aint":"ain't","arent":"aren't","cant":"can't","couldve":"could've","couldnt":"couldn't","didnt":"didn't","doesnt":"doesn't","dont":"don't","hadnt":"hadn't","hasnt":"hasn't","havent":"haven't","hed":"he'd","hes":"he's","im":"i'm","ive":"i've","isnt":"isn't","its":"it's","lets":"let's","shes":"she's","shouldnt":"shouldn't","thats":"that's","theres":"there's","theyd":"they'd","theyre":"they're","wasnt":"wasn't","werent":"weren't","whats":"what's","wheres":"where's","wont":"won't","wouldnt":"wouldn't","youd":"you'd","youre":"you're","youve":"you've"}
NUM = {"none":"0","zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}
ART = {"a", "an", "the"}
PERIOD = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA = re.compile(r"(\d)(\,)(\d)")
PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
K_TO_SCORE = {0: 0.0, 1: 0.3, 2: 0.6, 3: 0.9}


def _punct(s):
    out = s
    for p in PUNCT:
        out = out.replace(p, "") if (p + " " in s or " " + p in s or re.search(COMMA, s)) else out.replace(p, " ")
    return PERIOD.sub("", out, re.UNICODE)


def _digits_articles(s):
    w = [NUM.get(x, x) for x in s.lower().split()]
    w = [x for x in w if x not in ART]
    return " ".join(CONTR.get(x, x) for x in w)


def norm(a):
    return _digits_articles(_punct(a.replace("\n", " ").replace("\t", " ").strip()))


def agreement_counts(preds, gt):
    """k per question (matches among the ten reference answers), aligned with gt order."""
    pm = {r["image"]: r["answer"] for r in preds}
    k = []
    for g in gt:
        p = norm(pm[g["image"]])
        k.append(sum(1 for x in g["answers"] if norm(x["answer"]) == p))
    return np.array(k)


def soft(k):
    return np.array([K_TO_SCORE.get(int(x), 1.0) if x < 4 else 1.0 for x in k])


def ece(conf, score, bins):
    edges = np.linspace(0, 1, bins + 1); tot = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[i]) & (conf <= edges[i + 1])
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - score[m].mean())
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ref_pred", default=None, help="predictions of another representation of the same model, for comparison")
    ap.add_argument("--out_dir", default="results/selective")
    ap.add_argument("--repeats", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    preds = json.load(open(args.pred)); gt = json.load(open(args.gt))
    pm = {r["image"]: r for r in preds}
    N = len(gt)
    k = agreement_counts(preds, gt); score = soft(k)
    conf = np.array([pm[g["image"]]["confidence"] for g in gt])
    abst = np.array([pm[g["image"]]["answer"].strip().lower() == "unanswerable" for g in gt])
    answerable = np.array([g["answerable"] == 1 for g in gt])
    out = {"n": N, "mean_score": float(100 * score.mean()), "mean_confidence": float(conf.mean()),
           "ece_10_bins": float(ece(conf, score, 10)), "ece_15_bins": float(ece(conf, score, 15)),
           "k_counts": {"wrong_k0": int((k == 0).sum()), "weak_k1_2": int(((k >= 1) & (k <= 2)).sum()), "consensus_k3plus": int((k >= 3).sum())},
           "abstentions": int(abst.sum())}

    # ---- top reliability bin (confidence >= 0.9) -------------------------------------------
    top = conf >= 0.9
    out["top_bin_conf_ge_0.9"] = {"n": int(top.sum()), "share": float(top.mean()), "mean_score": float(100 * score[top].mean()),
                                  "wrong_k0": int((k[top] == 0).sum()), "weak_k1_2": int(((k[top] >= 1) & (k[top] <= 2)).sum()),
                                  "consensus": int((k[top] >= 3).sum()),
                                  "abstentions": int(abst[top].sum()),
                                  "delivered_answers": int((~abst[top]).sum()),
                                  "wrong_k0_among_delivered": int(((k == 0) & top & ~abst).sum())}

    # ---- composition at coverage levels (all-output curve) ----------------------------------
    order = np.sort(conf)[::-1]
    rows = []
    for q in (0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
        tau = order[int(round(q * N)) - 1]
        keep = conf >= tau
        deliv = keep & ~abst
        learn = keep & abst
        rej = ~keep
        A = deliv
        row = {"coverage": q, "tau": float(tau), "retained": int(keep.sum()),
               "all_output_score": float(100 * score[keep].mean()),
               "delivered": int(deliv.sum()), "delivered_consensus": int((deliv & (k >= 3)).sum()),
               "delivered_weak": int((deliv & (k >= 1) & (k <= 2)).sum()), "delivered_wrong": int((deliv & (k == 0)).sum()),
               "abstain_correct(on unanswerable)": int((learn & ~answerable).sum()), "abstain_false(on answerable)": int((learn & answerable).sum()),
               "rejected": int(rej.sum()), "rejected_would_be_delivered": int((rej & ~abst).sum()),
               "rejected_would_be_wrong": int((rej & ~abst & (k == 0)).sum()), "rejected_would_be_abstain": int((rej & abst).sum()),
               "A_size": int(A.sum()), "A_share_of_all": float(A.mean()),
               "A_score": float(100 * score[A].mean()) if A.any() else None,
               "A_wrong_rate": float(100 * (k[A] == 0).mean()) if A.any() else None,
               "A_consensus_rate": float(100 * (k[A] >= 3).mean()) if A.any() else None,
               "answerable_answered_share": float(100 * (A & answerable).sum() / answerable.sum()),
               "answerable_answered_score": float(100 * score[A & answerable].mean()) if (A & answerable).any() else None,
               "unanswerable_appropriately_declined_share": float(100 * ((~A) & ~answerable).sum() / (~answerable).sum()),
               "unanswerable_answered_substantively": int((A & ~answerable).sum())}
        rows.append(row)
    out["composition"] = rows
    with open(os.path.join(args.out_dir, "retained_set_composition.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- cross-fitted threshold selection ---------------------------------------------------
    rng = np.random.default_rng(args.seed)
    cf = {}
    for alpha in (0.05, 0.10):
        res = []
        for _ in range(args.repeats):
            perm = rng.permutation(N); a, b = perm[: N // 2], perm[N // 2:]
            cand = a[~abst[a]]
            idx = cand[np.argsort(-conf[cand])]
            wrong = (k[idx] == 0).astype(float)
            run = np.cumsum(wrong) / np.arange(1, len(idx) + 1)
            ok = np.where((run <= alpha) & (np.arange(1, len(idx) + 1) >= 50))[0]
            if len(ok) == 0:
                continue
            tau = conf[idx[ok.max()]]
            ev = b[(conf[b] >= tau) & (~abst[b])]
            if len(ev) == 0:
                continue
            res.append((tau, len(ev) / len(b), (k[ev] == 0).mean(), (k[ev] >= 3).mean(), answerable[ev].sum() / answerable[b].sum(),
                        (~answerable[ev]).sum() / (~answerable[b]).sum()))
        r = np.array(res)
        def st(j): return {"mean": float(r[:, j].mean()), "sd": float(r[:, j].std()), "p2.5": float(np.percentile(r[:, j], 2.5)), "p97.5": float(np.percentile(r[:, j], 97.5))}
        cf[f"alpha_{alpha}"] = {"repeats_used": int(len(r)), "tau": st(0), "coverage_of_all_questions": st(1),
                                "wrong_rate_on_held_out_half": st(2), "consensus_rate_on_held_out_half": st(3),
                                "answerable_questions_answered_share": st(4), "unanswerable_questions_answered_share": st(5),
                                "share_of_repeats_with_held_out_wrong_rate_le_alpha": float((r[:, 2] <= alpha).mean())}
    out["cross_fit"] = cf
    out["baseline_all_delivered_answers"] = {"n": int((~abst).sum()), "wrong_rate": float(100 * (k[~abst] == 0).mean()),
                                             "consensus_rate": float(100 * (k[~abst] >= 3).mean())}

    # ---- comparison with another representation of the same model --------------------------
    if args.ref_pred:
        ref = json.load(open(args.ref_pred)); refm = {r["image"]: r["answer"] for r in ref}
        kr = agreement_counts(ref, gt); sr = soft(kr)
        diff = np.array([norm(pm[g["image"]]["answer"]) != norm(refm[g["image"]]) for g in gt])
        abr = np.array([norm(refm[g["image"]]) == "unanswerable" for g in gt])
        d = score - sr
        out["artifact_comparison"] = {
            "reference_mean_score": float(100 * sr.mean()), "this_mean_score": float(100 * score.mean()),
            "answers_differ": int(diff.sum()), "answers_differ_share": float(diff.mean()),
            "this_better": int((d > 0).sum()), "this_worse": int((d < 0).sum()),
            "net_score_points_from_differences": float(100 * d.sum() / N),
            "abstentions_this": int(abst.sum()), "abstentions_reference": int(abr.sum()),
            "differ_and_this_abstains": int((diff & abst).sum()), "differ_and_reference_abstains": int((diff & abr).sum()),
            "differ_neither_abstains": int((diff & ~abst & ~abr).sum())}

    json.dump(out, open(os.path.join(args.out_dir, "selective_prediction_results.json"), "w"), indent=2)

    print(f"n={N} | mean score {out['mean_score']:.2f} | mean confidence {out['mean_confidence']:.4f} | ECE {out['ece_10_bins']:.4f} (10 bins) {out['ece_15_bins']:.4f} (15 bins)")
    print("k counts:", out["k_counts"], "| abstentions:", out["abstentions"])
    print("top bin (conf >= 0.9):", out["top_bin_conf_ge_0.9"])
    print("\ncoverage  tau    retained | delivered: consensus weak wrong | abstain: correct false | rejected | A_wrong% A_score | answerable answered% | unanswerable declined%")
    for r in rows:
        print(f"{r['coverage']:>6.0%} {r['tau']:>6.3f} {r['retained']:>8} | {r['delivered']:>9}: {r['delivered_consensus']:>9} {r['delivered_weak']:>4} {r['delivered_wrong']:>5} | "
              f"{r['abstain_correct(on unanswerable)']:>15} {r['abstain_false(on answerable)']:>5} | {r['rejected']:>8} | "
              f"{(r['A_wrong_rate'] or 0):>7.2f} {(r['A_score'] or 0):>7.2f} | {r['answerable_answered_share']:>19.1f} | {r['unanswerable_appropriately_declined_share']:>22.1f}")
    print("\nbaseline (all delivered answers):", out["baseline_all_delivered_answers"])
    for a, v in cf.items():
        print(f"\ncross-fit {a} (repeats {v['repeats_used']}): coverage {v['coverage_of_all_questions']['mean']:.3f} +- {v['coverage_of_all_questions']['sd']:.3f} | "
              f"held-out wrong rate {100*v['wrong_rate_on_held_out_half']['mean']:.2f}% [{100*v['wrong_rate_on_held_out_half']['p2.5']:.2f}, {100*v['wrong_rate_on_held_out_half']['p97.5']:.2f}] | "
              f"answerable answered {100*v['answerable_questions_answered_share']['mean']:.1f}% | repeats meeting alpha {100*v['share_of_repeats_with_held_out_wrong_rate_le_alpha']:.0f}%")
    if "artifact_comparison" in out:
        print("\nartifact comparison:", out["artifact_comparison"])
    print("\nsaved ->", args.out_dir)


if __name__ == "__main__":
    main()
