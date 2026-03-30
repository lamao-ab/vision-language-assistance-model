"""
data_preparation_test.py
========================
** TEST VERSION — saves only 1000 samples total (500 train / 500 val) **

Identical to data_preparation.py in every way — same download logic, same
extraction, same image loading, same HuggingFace Dataset format, same
metadata.json — the only difference is a --max_samples cap (default 1000)
that is split evenly between train and val.

If this script produces a correct dataset, the original script will too.

Usage
-----
python src/data_preparation_test.py \
    --workdir            data/vizwiz \
    --train_output       data/train_dataset \
    --val_output         data/val_dataset \
    --train_captions_per_image 5 \
    --val_captions_per_image   3 \
    --max_samples        1000
"""

import argparse
import io
import json
import os
import zipfile
from pathlib import Path

import requests
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage
from tqdm import tqdm

# ── Download URLs ─────────────────────────────────────────────────────────────
URLS = {
    "train_images":      "https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip",
    "val_images":        "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip",
    "train_vqa":         "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_ann/train.json",
    "val_vqa":           "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_ann/val.json",
    "train_captions":    "https://vizwiz.cs.colorado.edu/VizWiz_final/caption_ann/train.json",
    "val_captions":      "https://vizwiz.cs.colorado.edu/VizWiz_final/caption_ann/val.json",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path) -> Path:
    """Stream-download *url* to *dest* with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} already downloaded")
        return dest

    print(f"  Downloading {url} …")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    """Extract *zip_path* into *out_dir* if not already extracted."""
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"  [skip] {out_dir} already extracted")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def load_image_bytes(image_path: Path) -> bytes:
    """Return the raw bytes of *image_path* as a JPEG."""
    img = PILImage.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def vizwiz_vqa_accuracy(answers: list[dict]) -> float:
    """
    Official VizWiz VQA accuracy: 10-choose-9 subsets.
    Each subset votes, score = min(votes / 3, 1.0), averaged over subsets.
    """
    if not answers:
        return 0.0
    answer_texts = [a["answer"].lower().strip() for a in answers]
    total, n = 0.0, len(answer_texts)
    for i in range(n):
        candidate = answer_texts[i]
        others = answer_texts[:i] + answer_texts[i + 1:]
        matches = others.count(candidate)
        total += min(matches / 3.0, 1.0)
    return total / n


def select_consensus_answer(answers: list[dict]) -> str:
    """Return the answer with the highest VizWiz accuracy among candidates."""
    if not answers:
        return "unanswerable"
    best, best_score = answers[0]["answer"], -1.0
    for a in answers:
        score = vizwiz_vqa_accuracy([x for x in answers if x["answer"] == a["answer"]])
        if score > best_score:
            best, best_score = a["answer"], score
    return best


# ── Dataset builders ──────────────────────────────────────────────────────────

def build_vqa_samples(
    vqa_ann: list[dict],
    image_index: dict,
    max_samples: int | None = None,
) -> list[dict]:
    """Convert VQA annotations to (image_bytes, question, answer) rows."""
    samples = []
    for ann in tqdm(vqa_ann, desc="  VQA samples"):
        if max_samples and len(samples) >= max_samples:
            break
        img_name = ann["image"]
        if img_name not in image_index:
            continue
        answer = select_consensus_answer(ann.get("answers", []))
        question = ann["question"].strip()
        samples.append({
            "image":    {"bytes": load_image_bytes(image_index[img_name])},
            "question": question,
            "answer":   answer,
            "task":     "vqa",
        })
    return samples


def build_caption_samples(
    cap_ann: dict,
    image_index: dict,
    captions_per_image: int,
    max_samples: int | None = None,
) -> list[dict]:
    """
    Convert caption annotations to (image_bytes, caption) rows.
    *captions_per_image* captions are kept per image.
    """
    samples = []
    annotations = cap_ann.get("annotations", [])
    from collections import defaultdict
    by_image: dict[int, list[str]] = defaultdict(list)
    img_id_to_name: dict[int, str] = {}

    for img_info in cap_ann.get("images", []):
        img_id_to_name[img_info["id"]] = img_info["file_name"]

    for ann in annotations:
        img_id = ann["image_id"]
        by_image[img_id].append(ann["caption"].strip())

    for img_id, captions in tqdm(by_image.items(), desc="  Caption samples"):
        if max_samples and len(samples) >= max_samples:
            break
        img_name = img_id_to_name.get(img_id)
        if img_name not in image_index:
            continue
        img_bytes = load_image_bytes(image_index[img_name])
        for caption in captions[:captions_per_image]:
            samples.append({
                "image":    {"bytes": img_bytes},
                "question": "Describe this scene for a blind person.",
                "answer":   caption,
                "task":     "caption",
            })
    return samples


def make_hf_dataset(samples: list[dict]) -> Dataset:
    """Wrap a list of sample dicts into a HuggingFace Dataset."""
    features = Features({
        "image":    Image(decode=False),
        "question": Value("string"),
        "answer":   Value("string"),
        "task":     Value("string"),
    })
    return Dataset.from_list(samples, features=features)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="[TEST] Prepare a small VizWiz dataset (1000 samples)"
    )
    parser.add_argument("--workdir",   default="data/vizwiz_test",
                        help="Download cache directory")
    parser.add_argument("--train_output", default="data/train_dataset_test")
    parser.add_argument("--val_output",   default="data/val_dataset_test")
    parser.add_argument("--train_captions_per_image", type=int, default=5)
    parser.add_argument("--val_captions_per_image",   type=int, default=3)
    # ── Only difference from original ────────────────────────────────────────
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Total sample cap split evenly across train and val")
    args = parser.parse_args()

    half = args.max_samples // 2          # 500 train, 500 val by default
    train_cap_limit = half
    val_cap_limit   = args.max_samples - half

    print(f"\n⚠️  TEST MODE — capped at {args.max_samples} total samples "
          f"({train_cap_limit} train / {val_cap_limit} val)\n")

    workdir = Path(args.workdir)

    # ── 1. Download ──────────────────────────────────────────────────────────
    print("\n[1/4] Downloading files …")
    train_zip = download_file(URLS["train_images"], workdir / "train.zip")
    val_zip   = download_file(URLS["val_images"],   workdir / "val.zip")
    train_vqa_path = download_file(URLS["train_vqa"],      workdir / "train_vqa.json")
    val_vqa_path   = download_file(URLS["val_vqa"],        workdir / "val_vqa.json")
    train_cap_path = download_file(URLS["train_captions"], workdir / "train_captions.json")
    val_cap_path   = download_file(URLS["val_captions"],   workdir / "val_captions.json")

    # ── 2. Extract ───────────────────────────────────────────────────────────
    print("\n[2/4] Extracting images …")
    extract_zip(train_zip, workdir / "train")
    extract_zip(val_zip,   workdir / "val")

    # ── 3. Build index ───────────────────────────────────────────────────────
    print("\n[3/4] Indexing images …")
    train_index = {p.name: p for p in (workdir / "train").rglob("*.jpg")}
    val_index   = {p.name: p for p in (workdir / "val").rglob("*.jpg")}
    print(f"  Train images: {len(train_index):,}  |  Val images: {len(val_index):,}")

    # ── 4. Build HF Datasets ─────────────────────────────────────────────────
    print("\n[4/4] Building HuggingFace Datasets …")

    with open(train_vqa_path) as f:
        train_vqa = json.load(f)
    with open(val_vqa_path) as f:
        val_vqa = json.load(f)
    with open(train_cap_path) as f:
        train_cap = json.load(f)
    with open(val_cap_path) as f:
        val_cap = json.load(f)

    # --- Training set (VQA first, then captions up to cap) ------------------
    print("\nBuilding training set …")
    # Fill with VQA first, then top up with captions
    vqa_limit = min(len(train_vqa), train_cap_limit)
    train_samples = build_vqa_samples(train_vqa, train_index, max_samples=vqa_limit)
    remaining = train_cap_limit - len(train_samples)
    if remaining > 0:
        train_samples += build_caption_samples(
            train_cap, train_index,
            captions_per_image=args.train_captions_per_image,
            max_samples=remaining,
        )
    train_ds = make_hf_dataset(train_samples)
    train_ds.save_to_disk(args.train_output)
    print(f"  ✓ Train dataset: {len(train_ds):,} samples → {args.train_output}")

    # --- Validation set -------------------------------------------------------
    print("\nBuilding validation set …")
    vqa_limit = min(len(val_vqa), val_cap_limit)
    val_samples = build_vqa_samples(val_vqa, val_index, max_samples=vqa_limit)
    remaining = val_cap_limit - len(val_samples)
    if remaining > 0:
        val_samples += build_caption_samples(
            val_cap, val_index,
            captions_per_image=args.val_captions_per_image,
            max_samples=remaining,
        )
    val_ds = make_hf_dataset(val_samples)
    val_ds.save_to_disk(args.val_output)
    print(f"  ✓ Val dataset:   {len(val_ds):,} samples → {args.val_output}")

    # --- Metadata ------------------------------------------------------------
    meta = {
        "test_mode":     True,
        "max_samples":   args.max_samples,
        "train_samples": len(train_ds),
        "val_samples":   len(val_ds),
        "train_captions_per_image": args.train_captions_per_image,
        "val_captions_per_image":   args.val_captions_per_image,
    }
    with open(workdir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ Test data preparation complete.")
    print("   If this looks correct, the original data_preparation.py will work too.")


if __name__ == "__main__":
    main()
