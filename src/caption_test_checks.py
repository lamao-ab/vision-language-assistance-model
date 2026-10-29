#!/usr/bin/env python3
"""
caption_test_checks.py
======================
Validity and sensitivity checks for the reconstructed VizWiz-Captions test references.

1. Alignment check (neighbour-shift test). For every eligible image the prediction is scored against its own references
   and against the references of the previous and the next image in the sequence. If captions were assigned to the wrong
   image at some positions, the neighbour's references would fit better than the image's own at those positions. The
   share of such images is compared with the same share on the validation set, whose references are official.
2. Position check. Mean per-image CIDEr-D in blocks of consecutive images (local misalignment would show as dips) and the
   share of images with a near-zero score, compared with validation.
3. Duplicate sensitivity. CIDEr-D (per-image scores of the full set) on the test images whose photograph also occurs in
   the training data under another file name, against the others, and the adaptation gain on each group.

Usage
-----
python caption_test_checks.py --test_gt test_captions_7974.json --test_pred P1.json [P2.json ...] \
   --val_gt val.json --val_pred V1.json [V2.json ...] --base_test B.json --sets overlap_exposure_sets.json --out out.json
(--test_pred / --val_pred: predictions of the adapted models, averaged over files for the position check and the group gain;
the neighbour-shift test is run on every file.)
"""
import argparse, json
import numpy as np
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

PRE = "Quality issues are too severe to recognize visual content."


CLEAN = None  # (clean, safe) of src/score_captions_test.py, set by --clean



def _safe(lst):
    """Copy of the guard inside src/score_captions_test.py: drop '|' (METEOR's field separator) and empty strings after tokenisation."""
    out_ = [" ".join(x.replace("|", " ").split()) for x in lst]
    out_ = [x for x in out_ if x]
    return out_ or ["unk"]

def load_refs(path, clean=False):
    d = json.load(open(path)); refs = {}
    for a in d["annotations"]:
        c = CLEAN[0](a["caption"]) if clean else a["caption"].strip()
        if c == PRE or a.get("is_rejected", False) or a.get("is_precanned", False):
            continue
        refs.setdefault(a["image_id"], []).append(c)
    return d, refs


def scores(tok, g, ids, pred_path, shift=0):
    pm = {p["image_id"]: p["caption"] for p in json.load(open(pred_path))}
    if CLEAN:
        pm = {i: (CLEAN[0](c) or ".") for i, c in pm.items()}
    r = tok.tokenize({i: [{"caption": pm[i]}] for i in ids})
    if CLEAN:
        r = {i: CLEAN[1](v) for i, v in r.items()}
    n = len(ids)
    gts = {ids[k]: g[ids[(k + shift) % n]] for k in range(n)}   # a permutation of the reference sets: same document frequencies
    return np.array(Cider().compute_score(gts, {i: r[i] for i in ids})[1]) * 100


def boot_diff(a, b, n=10000, seed=2026):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n // 500):
        out.append(np.array([a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(500)]))
    out = np.concatenate(out); return [round(float(a.mean() - b.mean()), 2), round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)]


def neighbour_test(tok, g, ids, preds):
    res = {}
    for p in preds:
        s0, sm, sp = scores(tok, g, ids, p, 0), scores(tok, g, ids, p, -1), scores(tok, g, ids, p, 1)
        best = np.maximum(sm, sp)
        res[p.split("/")[-1]] = {"mean_own": round(float(s0.mean()), 2), "mean_neighbour": round(float((sm.mean() + sp.mean()) / 2), 2),
                                 "share_neighbour_beats_own_by_20_and_above_40": round(float(((best - s0 > 20) & (best > 40)).mean() * 100), 3),
                                 "n_such_images": int(((best - s0 > 20) & (best > 40)).sum())}
    return res


def blocks(s, size=100):
    m = [float(s[i:i + size].mean()) for i in range(0, len(s) - size + 1, size)]
    return {"block_size": size, "n_blocks": len(m), "overall_mean": round(float(s.mean()), 2), "lowest_block_mean": round(min(m), 2),
            "lowest_block_start_index": int(np.argmin(m) * size), "highest_block_mean": round(max(m), 2),
            "sd_of_block_means": round(float(np.std(m)), 2), "blocks_more_than_3sd_below": [int(i * size) for i, x in enumerate(m) if x < np.mean(m) - 3 * np.std(m)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test_gt", required=True); ap.add_argument("--test_pred", nargs="+", required=True)
    ap.add_argument("--val_gt", required=True); ap.add_argument("--val_pred", nargs="+", required=True)
    ap.add_argument("--base_test", required=True); ap.add_argument("--sets", required=True); ap.add_argument("--out", default="caption_test_checks.json")
    ap.add_argument("--clean", action="store_true", help="use the text cleaning of src/score_captions_test.py for the TEST set"); ap.add_argument("--scorer_dir", default="src")
    a = ap.parse_args(); tok = PTBTokenizer(); out = {}
    global CLEAN
    if a.clean:
        import sys; sys.path.insert(0, a.scorer_dir)
        from score_captions_test import _clean
        CLEAN = (_clean, _safe)
    tg, trefs = load_refs(a.test_gt, bool(CLEAN)); tids = sorted(trefs); tgt = tok.tokenize({i: [{"caption": c} for c in trefs[i]] for i in tids})
    if CLEAN: tgt = {i: CLEAN[1](v) for i, v in tgt.items()}
    test_clean = CLEAN
    CLEAN = None  # validation: original scoring
    vg, vrefs = load_refs(a.val_gt); vids = sorted(vrefs); vgt = tok.tokenize({i: [{"caption": c} for c in vrefs[i]] for i in vids})
    out["n_eligible"] = {"test": len(tids), "validation": len(vids)}
    CLEAN = test_clean
    out["neighbour_shift_test"] = {"test": neighbour_test(tok, tgt, tids, a.test_pred)}
    ts = np.mean([scores(tok, tgt, tids, p) for p in a.test_pred], axis=0)
    CLEAN = None
    out["neighbour_shift_test"]["validation"] = neighbour_test(tok, vgt, vids, a.val_pred)
    vs = np.mean([scores(tok, vgt, vids, p) for p in a.val_pred], axis=0)
    out["position_check"] = {"test": blocks(ts), "validation": blocks(vs),
                             "share_images_below_10": {"test": round(float((ts < 10).mean() * 100), 2), "validation": round(float((vs < 10).mean() * 100), 2)},
                             "share_images_zero": {"test": round(float((ts == 0).mean() * 100), 2), "validation": round(float((vs == 0).mean() * 100), 2)}}
    sets = json.load(open(a.sets))["vqa_test"]; name = {i["id"]: i["file_name"] for i in tg["images"]}
    CLEAN = test_clean
    base = scores(tok, tgt, tids, a.base_test)
    CLEAN = None
    for lab, key in (("identical_photo_in_training_data", "dup_train"), ("identical_photo_in_training_or_vqa_val", "dup_any")):
        dup = np.array([name[i] in set(sets[key]) for i in tids])
        out[f"duplicates_{lab}"] = {"n_duplicate_images": int(dup.sum()), "adapted_mean_dup": round(float(ts[dup].mean()), 2), "adapted_mean_other": round(float(ts[~dup].mean()), 2),
                                    "base_mean_dup": round(float(base[dup].mean()), 2), "base_mean_other": round(float(base[~dup].mean()), 2),
                                    "gain_dup": round(float((ts - base)[dup].mean()), 2), "gain_other": round(float((ts - base)[~dup].mean()), 2),
                                    "gain_difference_dup_minus_other_and_95ci": boot_diff((ts - base)[dup], (ts - base)[~dup])}
    json.dump(out, open(a.out, "w"), indent=2); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
