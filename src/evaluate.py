"""
evaluate.py
===========
Evaluate the fine-tuned model on VizWiz VQA, VizWiz Captions, or COCO Captions.

Usage
-----
# VizWiz VQA
python src/evaluate.py \
    --model_id   lamao-ab/paligemma-blind-assist-jetson-ready \
    --task       vizwiz_vqa \
    --output_dir outputs/eval

# VizWiz Captions
python src/evaluate.py \
    --model_id   lamao-ab/paligemma-blind-assist-jetson-ready \
    --task       vizwiz_caps \
    --output_dir outputs/eval

# COCO Captions (val2014)
python src/evaluate.py \
    --model_id   lamao-ab/paligemma-blind-assist-jetson-ready \
    --task       coco_caps \
    --output_dir outputs/eval
"""

import argparse
import io
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import requests
import torch
from PIL import Image as PILImage
from tqdm import tqdm
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

# ── Download helpers ──────────────────────────────────────────────────────────

def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"Downloading {url} …")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    return dest


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(model_id: str, base_id: str | None):
    """
    Load a merged (quantized) model directly, or apply a PEFT adapter on top
    of *base_id* if the Hub repo contains LoRA weights.
    """
    from transformers import BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = PaliGemmaProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"   # critical for batch inference

    # Try to detect whether this is a PEFT adapter repo
    try:
        from peft import PeftModel

        base = PaliGemmaForConditionalGeneration.from_pretrained(
            base_id or "google/paligemma-3b-mix-224",
            quantization_config=bnb,
            attn_implementation="sdpa",
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, model_id)
        print(f"Loaded PEFT adapter from {model_id} on top of {base_id}")
    except Exception:
        # Fall back: load as a complete merged model
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb,
            attn_implementation="sdpa",
            device_map="auto",
        )
        print(f"Loaded merged model from {model_id}")

    model.eval()
    return model, processor


# ── Batch inference ───────────────────────────────────────────────────────────

@torch.inference_mode()
def batch_generate(
    model,
    processor,
    images: list[PILImage.Image],
    prompts: list[str],
    max_new_tokens: int = 64,
) -> list[str]:
    inputs = processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = processor.batch_decode(
        outputs[:, input_len:], skip_special_tokens=True
    )
    return [d.strip() for d in decoded]


# ── VizWiz VQA ────────────────────────────────────────────────────────────────

VIZWIZ_VQA_VAL = "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_ann/val.json"
VIZWIZ_VAL_IMG = "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip"


def vizwiz_vqa_accuracy(answers: list[dict]) -> float:
    texts = [a["answer"].lower().strip() for a in answers]
    n = len(texts)
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        others = texts[:i] + texts[i + 1:]
        total += min(others.count(texts[i]) / 3.0, 1.0)
    return total / n


def evaluate_vizwiz_vqa(model, processor, workdir: Path, batch_size: int):
    ann_path = download_file(VIZWIZ_VQA_VAL, workdir / "val_vqa.json")
    img_zip  = download_file(VIZWIZ_VAL_IMG,  workdir / "val.zip")
    img_dir  = workdir / "val_images"
    extract_zip(img_zip, img_dir)

    with open(ann_path) as f:
        annotations = json.load(f)

    img_index = {p.name: p for p in img_dir.rglob("*.jpg")}

    predictions, ground_truths = [], []
    batch_imgs, batch_prompts, batch_anns = [], [], []

    def flush():
        preds = batch_generate(model, processor, batch_imgs, batch_prompts)
        for pred, ann in zip(preds, batch_anns):
            predictions.append(pred.lower().strip())
            ground_truths.append(ann.get("answers", []))
        batch_imgs.clear(); batch_prompts.clear(); batch_anns.clear()

    for ann in tqdm(annotations, desc="VizWiz VQA inference"):
        img_path = img_index.get(ann["image"])
        if img_path is None:
            continue
        img = PILImage.open(img_path).convert("RGB")
        prompt = f"<image>Assist a blind person: {ann['question']}"
        batch_imgs.append(img); batch_prompts.append(prompt); batch_anns.append(ann)
        if len(batch_imgs) == batch_size:
            flush()
    if batch_imgs:
        flush()

    # Score
    total_score = 0.0
    for pred, gt_answers in zip(predictions, ground_truths):
        if not gt_answers:
            continue
        fake_answers = [{"answer": pred}] + gt_answers
        # score = matches of pred against 9 out of 10 GT answers
        gt_texts = [a["answer"].lower().strip() for a in gt_answers]
        matches = gt_texts.count(pred)
        total_score += min(matches / 3.0, 1.0)

    accuracy = total_score / len(predictions) if predictions else 0.0
    return {"vizwiz_vqa_accuracy": round(accuracy * 100, 2), "n_samples": len(predictions)}


# ── VizWiz Captions ───────────────────────────────────────────────────────────

def evaluate_vizwiz_caps(model, processor, workdir: Path, batch_size: int):
    from datasets import load_dataset

    ds = load_dataset("lmms-lab/VizWiz-Caps", split="validation")

    predictions, references = {}, {}
    img_id = 0

    batch_imgs, batch_prompts, batch_ids = [], [], []

    def flush():
        preds = batch_generate(model, processor, batch_imgs, batch_prompts)
        for pid, pred in zip(batch_ids, preds):
            predictions[pid] = [{"image_id": pid, "caption": pred}]
        batch_imgs.clear(); batch_prompts.clear(); batch_ids.clear()

    for row in tqdm(ds, desc="VizWiz caps inference"):
        pil_img = row["image"].convert("RGB") if hasattr(row["image"], "convert") \
                  else PILImage.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        batch_imgs.append(pil_img)
        batch_prompts.append("<image>Describe this scene for a blind person.")
        batch_ids.append(img_id)
        references[img_id] = row.get("caption", row.get("captions", [""])
                                     if isinstance(row.get("captions"), list)
                                     else [row.get("captions", "")])
        img_id += 1
        if len(batch_imgs) == batch_size:
            flush()
    if batch_imgs:
        flush()

    return _compute_caption_metrics(predictions, references)


# ── COCO Captions ─────────────────────────────────────────────────────────────

COCO_VAL_IMAGES = "http://images.cocodataset.org/zips/val2014.zip"
COCO_VAL_ANN    = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"


def evaluate_coco_caps(model, processor, workdir: Path, batch_size: int):
    img_zip = download_file(COCO_VAL_IMAGES, workdir / "val2014.zip")
    ann_zip = download_file(COCO_VAL_ANN,    workdir / "annotations.zip")
    img_dir = workdir / "val2014"
    ann_dir = workdir / "annotations"
    extract_zip(img_zip, img_dir)
    extract_zip(ann_zip, ann_dir)

    ann_path = ann_dir / "annotations" / "captions_val2014.json"
    with open(ann_path) as f:
        coco_data = json.load(f)

    img_index = {img["id"]: img["file_name"] for img in coco_data["images"]}
    references: dict[int, list[str]] = defaultdict(list)
    for ann in coco_data["annotations"]:
        references[ann["image_id"]].append(ann["caption"])

    img_files = list((img_dir / "val2014").glob("*.jpg"))
    predictions: dict[int, list[dict]] = {}

    batch_imgs, batch_prompts, batch_ids = [], [], []

    def flush():
        preds = batch_generate(model, processor, batch_imgs, batch_prompts)
        for pid, pred in zip(batch_ids, preds):
            predictions[pid] = [{"image_id": pid, "caption": pred}]
        batch_imgs.clear(); batch_prompts.clear(); batch_ids.clear()

    for img_path in tqdm(img_files, desc="COCO caps inference"):
        # Extract image_id from filename like COCO_val2014_000000XXXXXX.jpg
        try:
            img_id = int(img_path.stem.split("_")[-1])
        except ValueError:
            continue
        img = PILImage.open(img_path).convert("RGB")
        batch_imgs.append(img)
        batch_prompts.append("<image>caption en")
        batch_ids.append(img_id)
        if len(batch_imgs) == batch_size:
            flush()
    if batch_imgs:
        flush()

    return _compute_caption_metrics(predictions, references)


# ── Caption metrics via pycocoevalcap ─────────────────────────────────────────

def _compute_caption_metrics(
    predictions: dict,
    references: dict,
) -> dict:
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError:
        print("pycocoevalcap not installed — skipping caption metrics")
        return {}

    # pycocoevalcap format: {img_id: [caption_str]}
    gts  = {k: v if isinstance(v[0], str) else [x["caption"] for x in v]
            for k, v in references.items() if k in predictions}
    res  = {k: [v[0]["caption"]] if isinstance(v[0], dict) else v
            for k, v in predictions.items() if k in gts}

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Cider(),  "CIDEr"),
        (Meteor(), "METEOR"),
        (Rouge(),  "ROUGE-L"),
    ]
    results = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for m, s in zip(method, score):
                results[m] = round(s, 4)
        else:
            results[method] = round(score, 4)
    results["n_samples"] = len(predictions)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PaliGemma on VizWiz / COCO")
    parser.add_argument("--model_id",    required=True,
                        help="HF Hub id of merged model or PEFT adapter")
    parser.add_argument("--base_id",     default="google/paligemma-3b-mix-224",
                        help="Base model id (only needed if model_id is a PEFT adapter)")
    parser.add_argument("--task",        required=True,
                        choices=["vizwiz_vqa", "vizwiz_caps", "coco_caps"])
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--output_dir",  default="outputs/eval")
    parser.add_argument("--workdir",     default="data/eval_cache")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model: {args.model_id}")
    model, processor = load_model(args.model_id, args.base_id)

    print(f"\nRunning evaluation: {args.task}")
    if args.task == "vizwiz_vqa":
        results = evaluate_vizwiz_vqa(model, processor, workdir, args.batch_size)
    elif args.task == "vizwiz_caps":
        results = evaluate_vizwiz_caps(model, processor, workdir, args.batch_size)
    else:
        results = evaluate_coco_caps(model, processor, workdir, args.batch_size)

    print("\n── Results ──────────────────────────────")
    for k, v in results.items():
        print(f"  {k}: {v}")

    out_path = os.path.join(args.output_dir, f"{args.task}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
