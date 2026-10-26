#!/usr/bin/env python3
"""
audit_overlap.py - image-overlap audit across the training, development and evaluation
partitions of both tasks (VizWiz-VQA and VizWiz-Captions).

Why: the two tasks draw on related VizWiz imagery, and the same photograph can sit in
different partitions under different file names. This script reports, for every pair of
partitions, how many images are shared

    * by file name,
    * by exact content (SHA-256 of the file bytes),
    * optionally by near-duplicate content (64-bit difference hash, Hamming distance <= --max_hamming).

Keep the partitions DISJOINT. If an internal development split is carved from the training images,
list the REMAINING training images as the training partition (a "list" file), otherwise the
development images are reported as shared with training by construction.

The leakage summary separates two kinds of overlap: the SAME file listed in another partition
(expected when the two tasks share images, e.g. a VQA image that is also a caption image) and
identical content under a DIFFERENT file name (a duplicate photograph, the real leakage signal).

It then prints a leakage summary: for each EVALUATION partition (--eval), how many of its
images also occur in any non-evaluation partition (training or development) of either task.
It makes no claim beyond what it counts.

Spec file (JSON): one entry per partition. Each entry needs EITHER "annotations" (a JSON file)
OR "list" (a text file, one image file name per line), and optionally "images" (directory).
Annotation formats understood: a list of items with an "image" field (VizWiz-VQA), or a
COCO-style dict with "images" -> "file_name" (VizWiz-Captions).

    {
      "vqa_train": {"annotations": "vqa/Annotations/train.json", "images": "vqa/train"},
      "vqa_val":   {"annotations": "vqa/Annotations/val.json",   "images": "vqa/val"},
      "vqa_test":  {"annotations": "vqa/Annotations/test.json",  "images": "vqa/test"},
      "cap_train": {"annotations": "caps/annotations/train.json", "images": "caps/train"},
      "cap_val":   {"annotations": "caps/annotations/val.json",   "images": "caps/val"},
      "cap_test":  {"annotations": "caps/annotations/test.json",  "images": "caps/test"},
      "cap_dev":   {"list": "splits/internal_dev_images.txt", "images": "caps/train"}
    }

Usage:
    python audit_overlap.py --spec overlap_spec.json --eval cap_val vqa_test \
        --phash --out results/overlap_audit.json --pairs_csv results/overlap_pairs.csv
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
from collections import defaultdict


def names_from(entry, base=""):
    if "list" in entry:
        return [l.strip() for l in open(os.path.join(base, entry["list"])) if l.strip()]
    data = json.load(open(os.path.join(base, entry["annotations"])))
    if isinstance(data, dict) and "images" in data:
        return [im["file_name"] for im in data["images"]]
    if isinstance(data, list):
        seen, out = set(), []
        for it in data:
            n = it.get("image") or it.get("file_name")
            if n and n not in seen:
                seen.add(n); out.append(n)
        return out
    raise SystemExit("unrecognised annotation format: " + entry.get("annotations", "?"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path, size=8):
    from PIL import Image
    im = Image.open(path).convert("L").resize((size + 1, size))
    px = list(im.get_flattened_data()) if hasattr(im, "get_flattened_data") else list(im.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (1 if px[r * (size + 1) + c] > px[r * (size + 1) + c + 1] else 0)
    return bits


def hamming(a, b):
    return bin(a ^ b).count("1")


def near_pairs(items, max_hamming):
    """items: list of (partition, name, hash64). Returns cross-partition pairs within max_hamming,
    found through 4 bands of 16 bits (any near pair of <=3 differing bits shares a band; for larger
    thresholds the result is a lower bound)."""
    bands = [defaultdict(list) for _ in range(4)]
    for idx, (_, _, h) in enumerate(items):
        for b in range(4):
            bands[b][(h >> (16 * b)) & 0xFFFF].append(idx)
    seen, out = set(), []
    for b in range(4):
        for bucket in bands[b].values():
            for i, j in itertools.combinations(bucket, 2):
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                (pa, na, ha), (pb, nb, hb) = items[i], items[j]
                d = hamming(ha, hb)
                if pa != pb and d <= max_hamming:
                    out.append((pa, na, pb, nb, d))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--base", default="", help="prefix for relative paths in the spec")
    ap.add_argument("--eval", nargs="+", default=[], help="partitions that are final evaluation sets")
    ap.add_argument("--phash", action="store_true", help="also find near-duplicates (needs Pillow)")
    ap.add_argument("--max_hamming", type=int, default=3)
    ap.add_argument("--out", default="overlap_audit.json")
    ap.add_argument("--pairs_csv", default="overlap_pairs.csv")
    args = ap.parse_args()

    spec = json.load(open(args.spec))
    names, sha, ph = {}, {}, []
    info = {}
    for part, entry in spec.items():
        nm = names_from(entry, args.base)
        names[part] = nm
        missing = 0
        sha[part] = {}
        if "images" in entry:
            for n in nm:
                p = os.path.join(args.base, entry["images"], n)
                if not os.path.exists(p):
                    missing += 1
                    continue
                sha[part][n] = sha256_file(p)
                if args.phash:
                    ph.append((part, n, dhash(p)))
        info[part] = {"n_names": len(nm), "n_unique_names": len(set(nm)),
                      "n_hashed": len(sha[part]), "n_missing_files": missing if "images" in entry else None,
                      "n_unique_content": len(set(sha[part].values()))}

    parts = list(spec)
    rows, matrix = [], {}
    near = near_pairs(ph, args.max_hamming) if args.phash else []
    near_ct = defaultdict(int)
    for pa, na, pb, nb, d in near:
        near_ct[tuple(sorted((pa, pb)))] += 1
        rows.append((pa, na, pb, nb, "near-duplicate", d))

    for a, b in itertools.combinations(parts, 2):
        by_name = set(names[a]) & set(names[b])
        ha, hb = defaultdict(list), defaultdict(list)
        for n, h in sha[a].items():
            ha[h].append(n)
        for n, h in sha[b].items():
            hb[h].append(n)
        shared = set(ha) & set(hb)
        for h in shared:
            for na in ha[h]:
                for nb in hb[h]:
                    rows.append((a, na, b, nb, "identical-content", 0))
        matrix[f"{a} & {b}"] = {"same_file_name": len(by_name), "identical_content": len(shared),
                                "near_duplicate_pairs": near_ct.get(tuple(sorted((a, b))), 0) if args.phash else None}

    leak = {}
    for e in args.eval:
        others = [p for p in parts if p not in args.eval]
        same_file, diff_name, in_other = set(), set(), set()
        for n, h in sha[e].items():
            for o in others:
                names_with_h = [m for m, hh in sha[o].items() if hh == h]
                if names_with_h:
                    in_other.add(n)
                    if n in names_with_h:
                        same_file.add(n)          # the SAME file listed in another partition (e.g. a VQA image that is also a caption image)
                    if any(m != n for m in names_with_h):
                        diff_name.add(n)          # identical content stored under a DIFFERENT file name
        near_imgs = {(pa if pa == e else pb, na if pa == e else nb) for pa, na, pb, nb, _ in near
                     if (pa == e and pb in others) or (pb == e and pa in others)}
        leak[e] = {"n_images": info[e]["n_hashed"],
                   "identical_content_in_a_non_eval_partition": len(in_other),
                   "  of which the same file listed in another partition": len(same_file),
                   "  of which identical content under a different file name": len(diff_name),
                   "near_duplicates_in_a_non_eval_partition": len(near_imgs) if args.phash else None}

    out = {"partitions": info, "pairwise": matrix, "evaluation_leakage": leak,
           "settings": {"phash": args.phash, "max_hamming": args.max_hamming}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    with open(args.pairs_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["partition_a", "name_a", "partition_b", "name_b", "type", "hamming"]); w.writerows(rows)

    print(f"{'partition':<12}{'images':>8}{'hashed':>8}{'missing':>9}")
    for p, i in info.items():
        print(f"{p:<12}{i['n_names']:>8}{i['n_hashed']:>8}{str(i['n_missing_files']):>9}")
    print("\npairwise overlap (file name / identical content / near-duplicate pairs):")
    for k, v in matrix.items():
        print(f"  {k:<26}{v['same_file_name']:>7}{v['identical_content']:>7}{str(v['near_duplicate_pairs']):>7}")
    if leak:
        print("\nevaluation leakage summary:")
        for e, v in leak.items():
            print(f"  {e}: {v}")
    print(f"\nsaved -> {args.out} and {args.pairs_csv}")


if __name__ == "__main__":
    main()
