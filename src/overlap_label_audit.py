"""
overlap_label_audit.py
======================
Label-level audit of the SHARED photographs (identical content, different file name) found by
audit_overlap.py. For the evaluation images that have an identical photograph in a training or
validation partition, it compares the labels of the two copies:

  VizWiz-VQA      the test question with the question(s) of the other copy (and, when the questions are
                  equal, the consensus answers)
  VizWiz-Captions the test captions with the captions of the other copy

Texts are compared after lower-casing and removing punctuation. Pre-canned and rejected captions are
ignored. Because many VizWiz questions and captions are generic ("what is this", "a bottle of water"),
every rate is reported next to a BASE RATE: how often the same exact text occurs for an arbitrary
photograph of the other partitions.

  python src/overlap_label_audit.py --spec overlap_spec.json --pairs results/overlap/overlap_pairs.csv \
      --out results/overlap/overlap_label_audit.json --matches_csv results/overlap/overlap_label_matches.csv

Annotation files are read from the spec: a list of {"image", "question", "answers"} (VQA) or a COCO-style
dict with "images" and "annotations" (captions). Partition names: vqa_* (questions) and cap_* (captions).
"""
import argparse
import collections
import csv
import json
import os
import re

PRECANNED = "Quality issues are too severe to recognize visual content."


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def consensus(answers):
    vals = [norm(a["answer"] if isinstance(a, dict) else a) for a in answers]
    vals = [v for v in vals if v]
    return collections.Counter(vals).most_common(1)[0][0] if vals else ""


def load_labels(path):
    """-> {image file name: [label, ...]}; label = (question, consensus answer) or caption text."""
    data = json.load(open(path))
    out = collections.defaultdict(list)
    if isinstance(data, dict) and "images" in data:
        name = {im["id"]: im["file_name"] for im in data["images"]}
        for a in data.get("annotations", []):
            if a.get("is_rejected") or a.get("is_precanned") or a["caption"].strip() == PRECANNED:
                continue
            out[name[a["image_id"]]].append(norm(a["caption"]))
        for im in data["images"]:
            out.setdefault(im["file_name"], [])
    else:
        for it in data:
            out[it["image"]].append((norm(it["question"]), consensus(it.get("answers", []))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--eval", nargs="+", default=["vqa_test", "cap_test"])
    ap.add_argument("--out", default="overlap_label_audit.json")
    ap.add_argument("--matches_csv", default=None)
    ap.add_argument("--base", default="")
    args = ap.parse_args()

    spec = json.load(open(args.spec))
    labels = {p: load_labels(os.path.join(args.base, e["annotations"])) for p, e in spec.items()}
    kind = lambda p: "vqa" if p.startswith("vqa_") else "cap"

    # identical-content partners (different file name) of every evaluation image
    args.eval = [e for e in args.eval if e in labels]
    partners = {e: collections.defaultdict(set) for e in args.eval}
    with open(args.pairs, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["type"] != "identical-content" or r["name_a"] == r["name_b"]:
                continue
            for e, n_e, p, n_p in ((r["partition_a"], r["name_a"], r["partition_b"], r["name_b"]),
                                   (r["partition_b"], r["name_b"], r["partition_a"], r["name_a"])):
                if e in partners and p not in args.eval and p in labels:
                    partners[e][n_e].add((p, n_p))

    report, matches = {}, []
    for e in args.eval:
        k = kind(e)
        others = [p for p in labels if p not in args.eval and kind(p) == k]
        shared = partners[e]
        # texts of ALL images of the other partitions of the same task (for the base rate)
        pool = set()
        for p in others:
            for labs in labels[p].values():
                pool.update(labs)
        res = {"evaluation_images": len(labels[e]), "images_with_identical_photograph_anywhere": len(shared),
               "comparison_partitions": others}
        if k == "vqa":
            n_with = n_q = n_qa = 0
            for n, pl in shared.items():
                cand = [x for p, pn in pl if kind(p) == "vqa" for x in labels[p].get(pn, [])]
                if not cand:
                    continue
                n_with += 1
                for q, a in labels[e].get(n, []):
                    same_q = [(cq, ca) for cq, ca in cand if cq == q]
                    if same_q:
                        n_q += 1
                        same_a = any(ca == a for _, ca in same_q)
                        n_qa += same_a
                        matches.append((e, n, "question" + ("+answer" if same_a else ""), q))
            allq = [q for labs in labels[e].values() for q, _ in labs]
            base = sum(1 for q in allq if q in {x[0] for x in pool if isinstance(x, tuple)}) / max(len(allq), 1)
            res.update({"images_with_a_partner_that_has_a_question": n_with,
                        "questions_equal_to_a_question_of_the_other_copy": n_q,
                        "of_which_same_consensus_answer": n_qa,
                        "base_rate_question_text_occurs_anywhere_in_other_partitions": round(base, 4)})
        else:
            n_with = n_img = n_cap = n_all = 0
            for n, pl in shared.items():
                cand = {c for p, pn in pl if kind(p) == "cap" for c in labels[p].get(pn, [])}
                mine = labels[e].get(n, [])
                if not [1 for p, _ in pl if kind(p) == "cap"]:
                    continue
                n_with += 1
                n_all += len(mine)
                hit = [c for c in mine if c in cand]
                n_cap += len(hit)
                n_img += bool(hit)
                matches.extend((e, n, "caption", c) for c in hit)
            allc = [c for labs in labels[e].values() for c in labs]
            base = sum(1 for c in allc if c in pool) / max(len(allc), 1)
            res.update({"images_with_a_partner_that_has_captions": n_with,
                        "images_with_at_least_one_identical_caption": n_img,
                        "captions_of_those_images": n_all,
                        "captions_identical_to_a_caption_of_the_other_copy": n_cap,
                        "base_rate_caption_text_occurs_anywhere_in_other_partitions": round(base, 4)})
        report[e] = res
    json.dump(report, open(args.out, "w"), indent=1)
    print(json.dumps(report, indent=1))
    if args.matches_csv:
        with open(args.matches_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["partition", "image", "match", "text"]); w.writerows(matches)


if __name__ == "__main__":
    main()
