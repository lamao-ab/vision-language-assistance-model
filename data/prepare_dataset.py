"""
prepare_dataset.py
==================
Download VizWiz images + annotations, build HuggingFace Datasets for training
and validation, and save them to disk.

Data sources
------------
- Train images  : VizWiz_final/images/train.zip
- Val images    : VizWiz_final/images/val.zip
- VQA anns      : VizWiz_final/vqa_data/Annotations.zip  -> train.json / val.json
- Caption anns  : VizWiz_final/caption/annotations.zip   -> train.json / val.json

Usage
-----
python data/prepare_dataset.py \
    --workdir            data/vizwiz \
    --train_output       data/train_dataset \
    --val_output         data/val_dataset \
    --train_captions_per_image 5 \
    --val_captions_per_image   3
"""

import argparse
import json
import os
import random
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path

import requests
from datasets import Dataset, Features, Image as HFImage, Value

# ── Download URLs ─────────────────────────────────────────────────────────────
URLS = {
    "train_images":        "https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip",
    "val_images":          "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip",
    "vqa_annotations":     "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip",
    "caption_annotations": "https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations.zip",
}

# ── Sample limit (set to None to use the full dataset) ───────────────────────
nb_samples = None


# ══════════════════════════════════════════════════════════════════════════════
#  Terminal formatting helpers  (matches train.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

W = 72

def _line(char="─"): return char * W
def _double():       return "═" * W

def section(title: str) -> None:
    print(f"\n{_double()}")
    print(f"  {title}")
    print(_double())

def step(n: int, total: int, title: str) -> None:
    print(f"\n{_line()}")
    print(f"  Step {n}/{total}  │  {title}")
    print(_line())

def info(label: str, value: str = "") -> None:
    if value:
        print(f"    {label:<30}  {value}")
    else:
        print(f"    {label}")

def ok(msg: str)   -> None: print(f"    [+]  {msg}")
def warn(msg: str) -> None: print(f"    [!]  {msg}")
def item(msg: str) -> None: print(f"    [-]  {msg}")


# ══════════════════════════════════════════════════════════════════════════════
#  Download & extraction helpers
# ══════════════════════════════════════════════════════════════════════════════

def download_file(url: str, dest: str) -> bool:
    if os.path.exists(dest):
        ok(f"Already downloaded  :  {os.path.basename(dest)}")
        return True
    info(f"Downloading  {os.path.basename(dest)} ...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f:
            if total_size > 0:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = (downloaded / total_size) * 100
                        print(f"\r    Progress: {progress:.1f}%", end="", flush=True)
                print()
            else:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
        ok(f"Download complete")
        return True
    except Exception as e:
        warn(f"Download failed: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str) -> bool:
    info(f"Extracting  {os.path.basename(zip_path)} ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_to)
        os.remove(zip_path)
        ok(f"Extraction complete")
        return True
    except Exception as e:
        warn(f"Extraction failed: {e}")
        return False


def find_images_dict(base_path: str) -> dict:
    info("Indexing images ...")
    img_map = {}
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_map[f] = os.path.join(root, f)
    ok(f"Found {len(img_map):,} images")
    return img_map


# ══════════════════════════════════════════════════════════════════════════════
#  VQA answer selection
# ══════════════════════════════════════════════════════════════════════════════

def get_consensus_answer(item: dict) -> str:
    """Select best answer using consensus — identical logic to Colab notebook."""
    answers    = item["answers"]
    answerable = item.get("answerable", 1)

    if answerable == 0:
        return "unanswerable"

    high_conf_answers = []
    for ans in answers:
        if isinstance(ans, dict):
            if ans.get("answer_confidence") == "yes":
                high_conf_answers.append(ans["answer"].lower().strip())
        elif isinstance(ans, str):
            high_conf_answers.append(ans.lower().strip())

    if len(high_conf_answers) >= 3:
        counter = Counter(high_conf_answers)
        most_common_answer, count = counter.most_common(1)[0]
        if count >= 3:
            return most_common_answer

    unanswerable_keywords = [
        "unanswerable", "unsuitable", "not answerable",
        "cannot answer", "unclear", "unsure",
    ]
    unanswerable_count = sum(
        1 for ans in answers
        if any(
            keyword in (ans["answer"] if isinstance(ans, dict) else str(ans)).lower()
            for keyword in unanswerable_keywords
        )
    )
    if unanswerable_count >= 3:
        return "unanswerable"

    if high_conf_answers:
        return high_conf_answers[0]

    first_ans = answers[0]
    return first_ans["answer"] if isinstance(first_ans, dict) else str(first_ans)


# ══════════════════════════════════════════════════════════════════════════════
#  Download + extract all files
# ══════════════════════════════════════════════════════════════════════════════

def download_and_extract_all(workdir: str) -> str:
    section("Downloading VizWiz Dataset")

    vw_dir = os.path.join(workdir, "vizwiz")
    os.makedirs(vw_dir, exist_ok=True)

    files = {
        "train_images":        (URLS["train_images"],        "Train Images"),
        "val_images":          (URLS["val_images"],          "Val Images"),
        "vqa_annotations":     (URLS["vqa_annotations"],     "VQA Annotations"),
        "caption_annotations": (URLS["caption_annotations"], "Caption Annotations"),
    }

    for name, (url, label) in files.items():
        print(f"\n{_line('·')}")
        info(label)
        print(_line('·'))
        zip_path = os.path.join(vw_dir, f"{name}.zip")
        if not download_file(url, zip_path):
            raise RuntimeError(f"Failed to download {name}")
        if not extract_zip(zip_path, vw_dir):
            raise RuntimeError(f"Failed to extract {name}")

    print(f"\n{_line()}")
    ok("All files downloaded and extracted")

    return vw_dir


# ══════════════════════════════════════════════════════════════════════════════
#  Process one split
# ══════════════════════════════════════════════════════════════════════════════

def process_split(vw_dir: str, split_name: str, captions_per_image: int):
    """
    Process a split ('train' or 'val') from extracted ZIP files.
    Returns (samples dict, vqa_stats dict, cap_stats dict).
    Dataset schema: image, text, suffix, task.
    """
    section(f"Processing  {split_name.upper()}  Split")
    info("Captions per image", str(captions_per_image))

    samples = {"image": [], "text": [], "suffix": [], "task": []}
    vw_imgs = find_images_dict(vw_dir)

    # ── VQA ──────────────────────────────────────────────────────────────────
    step(1, 2, f"VQA  —  {split_name}")

    vqa_path = None
    for root, _, files in os.walk(vw_dir):
        for f in files:
            if f == f"{split_name}.json":
                test_path = os.path.join(root, f)
                try:
                    with open(test_path) as tf:
                        test_data = json.load(tf)
                        if (
                            isinstance(test_data, list)
                            and len(test_data) > 0
                            and "question" in test_data[0]
                            and "answers"  in test_data[0]
                        ):
                            vqa_path = test_path
                            break
                except Exception:
                    continue
        if vqa_path:
            break

    if not vqa_path or not os.path.exists(vqa_path):
        warn(f"VQA annotation file not found for split '{split_name}'")
        vqa_stats = {"total": 0, "processed": 0, "skipped": 0, "unanswerable": 0}
    else:
        ok(f"Annotation file  :  {os.path.basename(vqa_path)}")
        with open(vqa_path) as f:
            vqa_data = json.load(f)
        info("Total questions", f"{len(vqa_data):,}")

        vqa_stats = {"total": 0, "processed": 0, "skipped": 0, "unanswerable": 0}

        for entry in vqa_data:
            vqa_stats["total"] += 1
            img_path = vw_imgs.get(entry["image"])
            if not img_path:
                vqa_stats["skipped"] += 1
                continue

            answer = get_consensus_answer(entry)
            if answer == "unanswerable":
                vqa_stats["unanswerable"] += 1

            samples["image"].append(img_path)
            samples["text"].append(f"<image>Assist a blind person: {entry['question']}")
            samples["suffix"].append(answer)
            samples["task"].append("vizwiz_vqa")
            vqa_stats["processed"] += 1

        print()
        info("Processed",    f"{vqa_stats['processed']:,}")
        info("Skipped",      f"{vqa_stats['skipped']:,}")
        info("Unanswerable", (
            f"{vqa_stats['unanswerable']:,}  "
            f"({vqa_stats['unanswerable'] / max(vqa_stats['processed'], 1) * 100:.1f}%)"
        ))
        ok("VQA processing complete")

    # ── Captions ──────────────────────────────────────────────────────────────
    step(2, 2, f"Captions  —  {split_name}")

    cap_path = None
    for root, _, files in os.walk(vw_dir):
        for f in files:
            if f == f"{split_name}.json" and "annotations" in root.lower():
                test_path = os.path.join(root, f)
                try:
                    with open(test_path) as tf:
                        test_data = json.load(tf)
                        if isinstance(test_data, dict) and "annotations" in test_data:
                            cap_path = test_path
                            break
                except Exception:
                    continue
        if cap_path:
            break

    if not cap_path or not os.path.exists(cap_path):
        warn(f"Caption annotation file not found for split '{split_name}'")
        cap_stats = {"total_images": 0, "total_annotations": 0, "processed": 0, "skipped": 0}
    else:
        ok(f"Annotation file  :  {os.path.basename(cap_path)}")
        with open(cap_path) as f:
            caps = json.load(f)

        if "annotations" in caps and "images" in caps:
            id2file = {img["id"]: img["file_name"] for img in caps["images"]}
            info("Total images",      f"{len(caps['images']):,}")
            info("Total annotations", f"{len(caps['annotations']):,}")

            img_to_captions = {}
            for ann in caps["annotations"]:
                img_id = ann["image_id"]
                img_to_captions.setdefault(img_id, []).append(ann["caption"])

            cap_stats = {"total_images": 0, "total_annotations": 0, "processed": 0, "skipped": 0}

            random.seed(42)

            for img_id, captions in img_to_captions.items():
                cap_stats["total_images"]      += 1
                cap_stats["total_annotations"] += len(captions)

                img_filename = id2file.get(img_id)
                if not img_filename:
                    cap_stats["skipped"] += len(captions)
                    continue

                img_path = vw_imgs.get(img_filename)
                if not img_path:
                    cap_stats["skipped"] += len(captions)
                    continue

                sampled_captions = (
                    random.sample(captions, captions_per_image)
                    if len(captions) >= captions_per_image
                    else captions
                )

                for caption in sampled_captions:
                    samples["image"].append(img_path)
                    samples["text"].append("<image>Describe this scene for a blind person.")
                    samples["suffix"].append(caption)
                    samples["task"].append("vizwiz_caption")
                    cap_stats["processed"] += 1

            print()
            info("Images processed",   f"{cap_stats['total_images']:,}")
            info("Captions per image", str(captions_per_image))
            info("Total processed",    f"{cap_stats['processed']:,}")
            info("Skipped",            f"{cap_stats['skipped']:,}")
            ok("Caption processing complete")

    return samples, vqa_stats, cap_stats


# ══════════════════════════════════════════════════════════════════════════════
#  Build and save HuggingFace Dataset
# ══════════════════════════════════════════════════════════════════════════════

def build_and_save_dataset(
    samples: dict, split_name: str, output_path: str, captions_per_image: int
):
    # ── Apply nb_samples limit if set ────────────────────────────────────────
    if nb_samples is not None:
        warn(f"nb_samples={nb_samples}  —  truncating dataset to {nb_samples} samples")
        for key in samples:
            samples[key] = samples[key][:nb_samples]

    total_count = len(samples["image"])

    section(f"{split_name.upper()}  Dataset Composition")

    if total_count == 0:
        raise ValueError(f"No samples for split '{split_name}'")

    task_counts = Counter(samples["task"])

    info("Total samples", f"{total_count:,}")
    print()

    C = 22
    print(f"    {'Task':<{C}}  {'Count':>8}  {'Share':>7}")
    print(f"    {'─'*C}  {'─'*8}  {'─'*7}")
    for task, count in sorted(task_counts.items()):
        pct = count / total_count * 100
        print(f"    {task:<{C}}  {count:>8,}  {pct:>6.1f}%")

    vqa_count = task_counts.get("vizwiz_vqa", 0)
    cap_count = task_counts.get("vizwiz_caption", 0)
    if vqa_count > 0:
        print()
        info("Caption:VQA ratio", f"{cap_count / vqa_count:.1f}:1")

    print(f"\n{_line()}")
    info("Building HuggingFace Dataset")
    print(_line())

    info("[1/3]  Converting to Dataset ...")
    start = time.time()
    ds = Dataset.from_dict(samples)
    ok(f"Done  ({time.time()-start:.1f} s)")

    info("[2/3]  Casting features ...")
    features = Features({
        "image":  HFImage(decode=False),
        "text":   Value("string"),
        "suffix": Value("string"),
        "task":   Value("string"),
    })
    start = time.time()
    ds = ds.cast(features)
    ok(f"Done  ({(time.time()-start)/60:.1f} min)")

    info("[3/3]  Saving to disk ...")
    start = time.time()
    ds.save_to_disk(output_path, num_proc=max(1, os.cpu_count() - 1))
    ok(f"Done  ({time.time()-start:.1f} s)")

    metadata = {
        "split":             split_name,
        "total_samples":     total_count,
        "task_distribution": dict(task_counts),
        "vqa_answer_method": "consensus_based_selection",
        "caption_method":    f"{captions_per_image}_captions_per_image",
        "source":            "Official VizWiz ZIP files",
    }
    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    ok(f"Metadata saved  →  {output_path}")

    return ds, metadata


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VizWiz datasets")
    parser.add_argument("--workdir",                  default="/tmp/blind_assist",
                        help="Temporary working directory")
    parser.add_argument("--train_output",             default="data/train_dataset")
    parser.add_argument("--val_output",               default="data/val_dataset")
    parser.add_argument("--train_captions_per_image", type=int, default=5)
    parser.add_argument("--val_captions_per_image",   type=int, default=3)
    args = parser.parse_args()

    # ── Safety: wipe previous outputs ────────────────────────────────────────
    for path in [args.train_output, args.val_output, args.workdir]:
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(args.workdir, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{_double()}")
    print(f"  VizWiz Dataset Builder")
    print(f"  Train  :  {args.train_captions_per_image} captions / image")
    print(f"  Val    :  {args.val_captions_per_image} captions / image")
    print(_double())

    # ── Download ──────────────────────────────────────────────────────────────
    vw_dir = download_and_extract_all(args.workdir)

    # ── Phase 1 — Train ───────────────────────────────────────────────────────
    section("Phase 1  —  Train Dataset")
    train_samples, _, _ = process_split(vw_dir, "train", args.train_captions_per_image)
    train_dataset, train_metadata = build_and_save_dataset(
        train_samples, "train", args.train_output, args.train_captions_per_image
    )

    # ── Phase 2 — Val ─────────────────────────────────────────────────────────
    section("Phase 2  —  Validation Dataset")
    val_samples, _, _ = process_split(vw_dir, "val", args.val_captions_per_image)
    val_dataset, val_metadata = build_and_save_dataset(
        val_samples, "val", args.val_output, args.val_captions_per_image
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    section("Summary")

    train_vqa = train_metadata["task_distribution"].get("vizwiz_vqa", 0)
    train_cap = train_metadata["task_distribution"].get("vizwiz_caption", 0)
    val_vqa   = val_metadata["task_distribution"].get("vizwiz_vqa", 0)
    val_cap   = val_metadata["task_distribution"].get("vizwiz_caption", 0)

    C = 22
    print(f"\n    {'Split':<{C}}  {'Samples':>10}  {'VQA':>8}  {'Caption':>10}  {'Ratio':>7}")
    print(f"    {'─'*C}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*7}")

    def ratio_str(cap, vqa):
        return f"{cap/vqa:.1f}:1" if vqa > 0 else "—"

    print(f"    {'Train':<{C}}  {len(train_dataset):>10,}  {train_vqa:>8,}  {train_cap:>10,}  {ratio_str(train_cap, train_vqa):>7}")
    print(f"    {'Val':<{C}}  {len(val_dataset):>10,}  {val_vqa:>8,}  {val_cap:>10,}  {ratio_str(val_cap, val_vqa):>7}")

    if train_vqa > 0 and val_vqa > 0:
        train_ratio = train_cap / train_vqa
        val_ratio   = val_cap   / val_vqa
        ratio_diff  = abs(train_ratio - val_ratio)
        print()
        if ratio_diff < 0.5:
            ok(f"Ratio matched  —  train ({train_ratio:.1f}:1) ≈ val ({val_ratio:.1f}:1)  Δ={ratio_diff:.2f}")
        else:
            warn(f"Ratio mismatch  —  train ({train_ratio:.1f}:1) vs val ({val_ratio:.1f}:1)  Δ={ratio_diff:.2f}")

    print()
    info("Train output", args.train_output)
    info("Val output",   args.val_output)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print(f"\n{_line()}")
    info("Cleaning up temporary files ...")
    if os.path.exists(args.workdir):
        shutil.rmtree(args.workdir)
    ok("Cleanup complete")

    # ── Next steps ────────────────────────────────────────────────────────────
    print(f"\n{_line()}")
    print(f"  Next Steps")
    print(_line())
    item(f"tensorboard --logdir=outputs/logs")
    item(f"python src/train.py \\")
    item(f"    --train_dataset_path {args.train_output} \\")
    item(f"    --val_dataset_path   {args.val_output}")

    print(f"\n{_double()}\n")


if __name__ == "__main__":
    main()
