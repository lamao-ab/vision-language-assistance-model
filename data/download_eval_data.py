"""
download_eval_data.py
=====================
Download AND extract ALL evaluation datasets ONCE into a shared --workdir,
so the evaluation scripts (evaluate_vizwiz.py, evaluate_benchmark.py) never
re-download anything. Uses the EXACT same target folders, filenames, and
".extracted" markers as those scripts, so they detect the data as already
present and skip straight to inference.

Datasets (matching the eval scripts exactly):
  VizWiz-VQA (test):
    - images:      test.zip          -> {out}/vizwiz_vqa_data/
    - annotations: VQA_test.json     -> {out}/vizwiz_vqa_data/VQA_test.json
  VizWiz-Captions (validation):
    - images:      val.zip           -> {out}/vizwiz_caps_data/
    - references:  annotations.zip   -> {out}/vizwiz_caps_data/annotations/val.json
  VQAv2 (test-std):
    - images:      test2015.zip      -> {out}/vqav2_data/   (~13 GB)
    - questions:   v2_Questions_Test_mscoco.zip -> {out}/vqav2_data/
  COCO-Captions (validation):
    - images:      val2014.zip       -> {out}/coco_data/
    - annotations: annotations_trainval2014.zip -> {out}/coco_data/
                   (provides captions_val2014.json)

Usage
-----
    python download_eval_data.py --workdir outputs/predictions
    # or only a subset:
    python download_eval_data.py --workdir outputs/predictions --only vizwiz_vqa coco

Re-running is safe: already-downloaded/extracted files are skipped.
"""
import argparse
import os
import zipfile

import requests


# ── Download + extract (identical logic to the eval scripts) ──────────────────
def download_and_extract(url: str, dest_folder: str) -> None:
    os.makedirs(dest_folder, exist_ok=True)
    filename  = url.split("/")[-1]
    file_path = os.path.join(dest_folder, filename)

    if not os.path.exists(file_path):
        print(f"⬇️  Downloading {filename} -> {dest_folder} ...")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"✅ {filename} already downloaded.")

    marker = file_path + ".extracted"
    if os.path.exists(marker):
        print(f"✅ {filename} already extracted.")
        return
    print(f"📦 Extracting {filename}...")
    with zipfile.ZipFile(file_path, "r") as zf:
        zf.extractall(dest_folder)
    open(marker, "w").close()


def download_json(url: str, dest_folder: str) -> None:
    """For plain .json files (VQA_test.json) that aren't zipped."""
    os.makedirs(dest_folder, exist_ok=True)
    filename  = url.split("/")[-1]
    file_path = os.path.join(dest_folder, filename)
    if os.path.exists(file_path):
        print(f"✅ {filename} already downloaded.")
        return
    print(f"⬇️  Downloading {filename} -> {dest_folder} ...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


# ── URLs (copied verbatim from the eval scripts) ──────────────────────────────
VIZWIZ_VQA_IMAGES_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip"
VIZWIZ_VQA_TEST_JSON  = "https://vizwiz.cs.colorado.edu/VizWiz_all_answers/VQA_test.json"

VIZWIZ_CAPS_IMAGES_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip"
VIZWIZ_CAPS_ANNOT_URL  = "https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations.zip"

VQAV2_IMAGES_URL    = "http://images.cocodataset.org/zips/test2015.zip"
VQAV2_QUESTIONS_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip"

COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2014.zip"
COCO_ANNOT_URL  = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"


def get_vizwiz_vqa(out):
    print("\n=== VizWiz-VQA (test) ===")
    root = os.path.join(out, "vizwiz_vqa_data")
    download_and_extract(VIZWIZ_VQA_IMAGES_URL, root)
    download_json(VIZWIZ_VQA_TEST_JSON, root)


def get_vizwiz_caps(out):
    print("\n=== VizWiz-Captions (validation) ===")
    root = os.path.join(out, "vizwiz_caps_data")
    download_and_extract(VIZWIZ_CAPS_IMAGES_URL, root)
    download_and_extract(VIZWIZ_CAPS_ANNOT_URL, root)


def get_vqav2(out):
    print("\n=== VQAv2 (test-std) — large (~13 GB images) ===")
    root = os.path.join(out, "vqav2_data")
    download_and_extract(VQAV2_IMAGES_URL, root)
    download_and_extract(VQAV2_QUESTIONS_URL, root)


def get_coco(out):
    print("\n=== COCO-Captions (validation) ===")
    root = os.path.join(out, "coco_data")
    download_and_extract(COCO_IMAGES_URL, root)
    download_and_extract(COCO_ANNOT_URL, root)


TASKS = {
    "vizwiz_vqa":  get_vizwiz_vqa,
    "vizwiz_caps": get_vizwiz_caps,
    "vqav2":       get_vqav2,
    "coco":        get_coco,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="outputs/predictions",
                    help="Shared data root (pass the SAME path as the eval scripts' --output_dir)")
    ap.add_argument("--only", nargs="+", choices=list(TASKS.keys()), default=None,
                    help="Download only these datasets (default: all)")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    selected = args.only if args.only else list(TASKS.keys())

    print(f"Workdir: {os.path.abspath(args.workdir)}")
    print(f"Datasets  : {', '.join(selected)}")

    for name in selected:
        TASKS[name](args.workdir)

    print("\n✅ All requested evaluation data is downloaded and extracted.")
    print("   The eval scripts will now skip downloading and go straight to inference.")


if __name__ == "__main__":
    main()
