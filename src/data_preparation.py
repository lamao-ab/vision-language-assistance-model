"""
data_preparation.py
===================
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
python src/data_preparation.py \
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def download_file(url: str, dest: str) -> bool:
    """Download file with progress, matching Colab output style."""
    if os.path.exists(dest):
        print(f" ✅ Found: {os.path.basename(dest)}")
        return True
    print(f" ⬇️  Downloading {os.path.basename(dest)}...")
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
        print(f" ✅ Downloaded successfully")
        return True
    except Exception as e:
        print(f" ❌ Download failed: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str) -> bool:
    """Extract ZIP file and delete it afterwards (matches Colab behaviour)."""
    print(f" 📦 Extracting {os.path.basename(zip_path)}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_to)
        os.remove(zip_path)          # ← Colab deletes the zip after extraction
        print(f" ✅ Extracted successfully")
        return True
    except Exception as e:
        print(f" ❌ Extraction failed: {e}")
        return False


def find_images_dict(base_path: str) -> dict:
    """Index all images in directory, returning {filename: full_path}."""
    img_map = {}
    print(f" 🔍 Indexing images...")
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_map[f] = os.path.join(root, f)
    print(f" ✅ Found {len(img_map):,} images")
    return img_map


# ── VQA answer selection (matches Colab get_consensus_answer exactly) ─────────

def get_consensus_answer(item: dict) -> str:
    """Select best answer using consensus — identical logic to Colab notebook."""
    answers = item["answers"]
    answerable = item.get("answerable", 1)

    if answerable == 0:
        return "unanswerable"

    # Extract high-confidence answers
    high_conf_answers = []
    for ans in answers:
        if isinstance(ans, dict):
            if ans.get("answer_confidence") == "yes":
                high_conf_answers.append(ans["answer"].lower().strip())
        elif isinstance(ans, str):
            high_conf_answers.append(ans.lower().strip())

    # Consensus (3+ agree)
    if len(high_conf_answers) >= 3:
        counter = Counter(high_conf_answers)
        most_common_answer, count = counter.most_common(1)[0]
        if count >= 3:
            return most_common_answer

    # Check for unanswerable
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

    # Fallback to first high-confidence answer
    if high_conf_answers:
        return high_conf_answers[0]

    # Final fallback to first answer
    first_ans = answers[0]
    return first_ans["answer"] if isinstance(first_ans, dict) else str(first_ans)


# ── Download + extract all files ──────────────────────────────────────────────

def download_and_extract_all(workdir: str) -> str:
    """Download and extract all VizWiz files into *workdir*."""
    print(f"\n{'='*70}")
    print(f"📥 DOWNLOADING VIZWIZ DATASET")
    print(f"{'='*70}\n")

    vw_dir = os.path.join(workdir, "vizwiz")
    os.makedirs(vw_dir, exist_ok=True)

    files = {
        "train_images":        (URLS["train_images"],        "train"),
        "val_images":          (URLS["val_images"],          "val"),
        "vqa_annotations":     (URLS["vqa_annotations"],     "vqa"),
        "caption_annotations": (URLS["caption_annotations"], "captions"),
    }

    print("Downloading files...")
    print("-" * 70)

    for name, (url, file_type) in files.items():
        zip_path = os.path.join(vw_dir, f"{name}.zip")
        print(f"\n[{file_type.upper()}] {name}")
        if not download_file(url, zip_path):
            raise RuntimeError(f"Failed to download {name}")
        if not extract_zip(zip_path, vw_dir):
            raise RuntimeError(f"Failed to extract {name}")

    print("\n" + "=" * 70)
    print("✅ All files downloaded and extracted")
    print("=" * 70)

    return vw_dir


# ── Process one split ─────────────────────────────────────────────────────────

def process_split(vw_dir: str, split_name: str, captions_per_image: int):
    """
    Process a split ('train' or 'val') from extracted ZIP files.
    Returns (samples dict, vqa_stats dict, cap_stats dict).
    Dataset schema matches Colab: image, text, suffix, task.
    """
    print(f"\n{'='*70}")
    print(f"📊 PROCESSING {split_name.upper()} SPLIT")
    print(f"   Captions per image: {captions_per_image}")
    print(f"{'='*70}")

    samples = {"image": [], "text": [], "suffix": [], "task": []}

    vw_imgs = find_images_dict(vw_dir)

    # ── VQA ──────────────────────────────────────────────────────────────────
    print(f"\n[1/2] 🟢 Processing VQA {split_name}...")
    print("-" * 70)

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
                            and "answers" in test_data[0]
                        ):
                            vqa_path = test_path
                            break
                except Exception:
                    continue
        if vqa_path:
            break

    if not vqa_path or not os.path.exists(vqa_path):
        print(f"   ⚠️  VQA file not found for {split_name}!")
        vqa_stats = {"total": 0, "processed": 0, "skipped": 0, "unanswerable": 0}
    else:
        print(f"   Found: {os.path.basename(vqa_path)}")
        with open(vqa_path) as f:
            vqa_data = json.load(f)
        print(f"   Total questions: {len(vqa_data):,}")

        vqa_stats = {"total": 0, "processed": 0, "skipped": 0, "unanswerable": 0}

        for item in vqa_data:
            vqa_stats["total"] += 1
            img_filename = item["image"]
            img_path = vw_imgs.get(img_filename)
            if not img_path:
                vqa_stats["skipped"] += 1
                continue

            answer = get_consensus_answer(item)
            if answer == "unanswerable":
                vqa_stats["unanswerable"] += 1

            # ── Prompt format matches Colab exactly ──
            samples["image"].append(img_path)
            samples["text"].append(f"<image>Assist a blind person: {item['question']}")
            samples["suffix"].append(answer)
            samples["task"].append("vizwiz_vqa")
            vqa_stats["processed"] += 1

        print(f"\n   📊 Statistics:")
        print(f"      Processed:    {vqa_stats['processed']:,}")
        print(f"      Skipped:      {vqa_stats['skipped']:,}")
        print(
            f"      Unanswerable: {vqa_stats['unanswerable']:,} "
            f"({vqa_stats['unanswerable'] / max(vqa_stats['processed'], 1) * 100:.1f}%)"
        )
        print(f"   ✅ Complete")

    # ── Captions ──────────────────────────────────────────────────────────────
    print(f"\n[2/2] 🔵 Processing Captions {split_name}...")
    print("-" * 70)

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
        print(f"   ⚠️  Caption file not found for {split_name}!")
        cap_stats = {"total_images": 0, "total_annotations": 0, "processed": 0, "skipped": 0}
    else:
        print(f"   Found: {os.path.basename(cap_path)}")
        with open(cap_path) as f:
            caps = json.load(f)

        if "annotations" in caps and "images" in caps:
            id2file = {img["id"]: img["file_name"] for img in caps["images"]}
            print(f"   Total images:      {len(caps['images']):,}")
            print(f"   Total annotations: {len(caps['annotations']):,}")

            img_to_captions = {}
            for ann in caps["annotations"]:
                img_id = ann["image_id"]
                if img_id not in img_to_captions:
                    img_to_captions[img_id] = []
                img_to_captions[img_id].append(ann["caption"])

            cap_stats = {"total_images": 0, "total_annotations": 0, "processed": 0, "skipped": 0}

            random.seed(42)          # ← reproducible sampling, matches Colab

            for img_id, captions in img_to_captions.items():
                cap_stats["total_images"] += 1
                cap_stats["total_annotations"] += len(captions)

                img_filename = id2file.get(img_id)
                if not img_filename:
                    cap_stats["skipped"] += len(captions)
                    continue

                img_path = vw_imgs.get(img_filename)
                if not img_path:
                    cap_stats["skipped"] += len(captions)
                    continue

                # ── random.sample matches Colab exactly ──
                sampled_captions = (
                    random.sample(captions, captions_per_image)
                    if len(captions) >= captions_per_image
                    else captions
                )

                for caption in sampled_captions:
                    # ── Prompt format matches Colab exactly ──
                    samples["image"].append(img_path)
                    samples["text"].append("<image>Describe this scene for a blind person.")
                    samples["suffix"].append(caption)
                    samples["task"].append("vizwiz_caption")
                    cap_stats["processed"] += 1

            print(f"\n   📊 Statistics:")
            print(f"      Images:              {cap_stats['total_images']:,}")
            print(f"      Captions per image:  {captions_per_image}")
            print(f"      Total processed:     {cap_stats['processed']:,}")
            print(f"      Skipped:             {cap_stats['skipped']:,}")
            print(f"   ✅ Complete")

    return samples, vqa_stats, cap_stats


# ── Build and save HuggingFace Dataset ───────────────────────────────────────

def build_and_save_dataset(
    samples: dict, split_name: str, output_path: str, captions_per_image: int
):
    """Build and save HuggingFace dataset with image/text/suffix/task schema."""
    total_count = len(samples["image"])

    print(f"\n{'='*70}")
    print(f"📊 {split_name.upper()} DATASET COMPOSITION")
    print(f"{'='*70}")

    if total_count == 0:
        raise ValueError(f"No samples for {split_name}!")

    task_counts = Counter(samples["task"])

    print(f"Total samples:        {total_count:,}")
    print(f"\nBreakdown by task:")
    for task, count in sorted(task_counts.items()):
        percentage = (count / total_count) * 100
        print(f"  {task:20s}: {count:6,} ({percentage:5.1f}%)")

    vqa_count = task_counts.get("vizwiz_vqa", 0)
    cap_count = task_counts.get("vizwiz_caption", 0)
    if vqa_count > 0:
        ratio = cap_count / vqa_count
        print(f"\nCaption:VQA ratio:    {ratio:.1f}:1")

    print(f"{'='*70}")

    print(f"\n🔄 Creating HuggingFace Dataset...")
    print(f"   [1/3] Converting to Dataset...")
    start = time.time()
    ds = Dataset.from_dict(samples)
    print(f"   ✅ Done in {time.time()-start:.1f}s")

    print(f"   [2/3] Casting features...")
    features = Features({
        "image":  HFImage(decode=False),
        "text":   Value("string"),
        "suffix": Value("string"),
        "task":   Value("string"),
    })
    start = time.time()
    ds = ds.cast(features)
    elapsed = time.time() - start
    print(f"   ✅ Done in {elapsed/60:.1f} minutes")

    print(f"   [3/3] Saving to disk...")
    start = time.time()
    ds.save_to_disk(output_path, num_proc=max(1, os.cpu_count() - 1))
    print(f"   ✅ Done in {time.time()-start:.1f}s")

    metadata = {
        "split":              split_name,
        "total_samples":      total_count,
        "task_distribution":  dict(task_counts),
        "vqa_answer_method":  "consensus_based_selection",
        "caption_method":     f"{captions_per_image}_captions_per_image",
        "source":             "Official VizWiz ZIP files",
    }
    metadata_path = os.path.join(output_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Metadata saved")

    return ds, metadata


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VizWiz datasets")
    parser.add_argument("--workdir",                    default="/tmp/blind_assist",
                        help="Temporary working directory")
    parser.add_argument("--train_output",               default="data/train_dataset")
    parser.add_argument("--val_output",                 default="data/val_dataset")
    parser.add_argument("--train_captions_per_image",   type=int, default=5)
    parser.add_argument("--val_captions_per_image",     type=int, default=3)
    args = parser.parse_args()

    # ── Safety: wipe previous outputs ────────────────────────────────────────
    for path in [args.train_output, args.val_output, args.workdir]:
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(args.workdir, exist_ok=True)

    print("=" * 70)
    print("🛡️  VIZWIZ DATASET BUILDER - ALL FROM ZIP FILES")
    print(f"   Train: Official ZIP files ({args.train_captions_per_image} captions per image)")
    print(f"   Val:   Official ZIP files ({args.val_captions_per_image} captions per image)")
    print("=" * 70)

    print(f"\n🚀 Starting VizWiz Dataset Preparation...")
    print(f"   Source: Official ZIP files for both train and val")
    print(f"   Train: {args.train_captions_per_image} captions per image")
    print(f"   Val:   {args.val_captions_per_image} captions per image")

    # Download + extract
    vw_dir = download_and_extract_all(args.workdir)

    # Train
    print("\n" + "=" * 70)
    print("PHASE 1: TRAIN DATASET")
    print("=" * 70)
    train_samples, train_vqa_stats, train_cap_stats = process_split(
        vw_dir, "train", args.train_captions_per_image
    )
    train_dataset, train_metadata = build_and_save_dataset(
        train_samples, "train", args.train_output, args.train_captions_per_image
    )

    # Val
    print("\n" + "=" * 70)
    print("PHASE 2: VALIDATION DATASET")
    print("=" * 70)
    val_samples, val_vqa_stats, val_cap_stats = process_split(
        vw_dir, "val", args.val_captions_per_image
    )
    val_dataset, val_metadata = build_and_save_dataset(
        val_samples, "val", args.val_output, args.val_captions_per_image
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("=" * 70)

    print("\n📂 Output Locations:")
    print(f"   Train: {args.train_output}")
    print(f"   Val:   {args.val_output}")

    print("\n📊 Dataset Sizes:")
    print(f"   Train: {len(train_dataset):,} samples")
    print(f"   Val:   {len(val_dataset):,} samples")

    print("\n📈 Composition:")
    train_vqa = train_metadata["task_distribution"]["vizwiz_vqa"]
    train_cap = train_metadata["task_distribution"]["vizwiz_caption"]
    val_vqa   = val_metadata["task_distribution"]["vizwiz_vqa"]
    val_cap   = val_metadata["task_distribution"]["vizwiz_caption"]

    print(f"   Train VQA:     {train_vqa:,} ({train_vqa/(train_vqa+train_cap)*100:.1f}%)")
    print(f"   Train Caption: {train_cap:,} ({train_cap/(train_vqa+train_cap)*100:.1f}%)")
    print(f"   Train Ratio:   {train_cap/train_vqa:.1f}:1")
    print()
    print(f"   Val VQA:       {val_vqa:,} ({val_vqa/(val_vqa+val_cap)*100:.1f}%)")
    print(f"   Val Caption:   {val_cap:,} ({val_cap/(val_vqa+val_cap)*100:.1f}%)")
    print(f"   Val Ratio:     {val_cap/val_vqa:.1f}:1")

    train_ratio = train_cap / train_vqa
    val_ratio   = val_cap   / val_vqa
    ratio_diff  = abs(train_ratio - val_ratio)

    print(f"\n✅ Ratio Match Check:")
    if ratio_diff < 0.5:
        print(f"   ✅ MATCHED! Train ({train_ratio:.1f}:1) ≈ Val ({val_ratio:.1f}:1)")
        print(f"   Difference: {ratio_diff:.2f} (acceptable)")
    else:
        print(f"   ⚠️  Train ({train_ratio:.1f}:1) vs Val ({val_ratio:.1f}:1)")
        print(f"   Difference: {ratio_diff:.2f}")

    print("\n💾 Memory Efficiency:")
    print(f"   Expected peak RAM: 15-20 GB")
    print(f"   Method: Streaming from disk (no 90GB RAM spike!)")

    print("\n⏱️  Performance:")
    print(f"   Source: Local ZIP files (no HuggingFace download wait)")
    print(f"   Consistent: Same source for train and val")

    print("=" * 70)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print(f"\n🧹 Cleaning up temporary files...")
    if os.path.exists(args.workdir):
        shutil.rmtree(args.workdir)
    print(f"   ✅ Cleanup complete")

    print("\n🎯 Next Steps:")
    print("   1. Verify datasets:")
    print(f"      ls {args.train_output}")
    print(f"      ls {args.val_output}")
    print("   2. Check dataset info:")
    print(f"      from datasets import load_from_disk")
    print(f"      ds = load_from_disk('{args.train_output}')")
    print(f"      print(ds)")
    print("   3. Start training:")
    print("      python src/train.py \\")
    print(f"          --train_dataset_path {args.train_output} \\")
    print(f"          --val_dataset_path   {args.val_output}")

    print("\n✨ Ready to train! ✨")


if __name__ == "__main__":
    main()
