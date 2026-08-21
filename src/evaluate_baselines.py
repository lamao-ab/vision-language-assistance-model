"""
evaluate_baselines.py
=====================
Zero-shot (off-the-shelf) evaluation of lightweight VLM baselines on the same
data and in the same output formats as evaluate_vizwiz.py / evaluate_benchmark.py,
so that the existing scorers can be reused unchanged.

Supported baselines
-------------------
  smolvlm2 : HuggingFaceTB/SmolVLM2-2.2B-Instruct
  qwen2.5vl: Qwen/Qwen2.5-VL-3B-Instruct

Both are instruction-tuned checkpoints, i.e. the tier corresponding to
PaliGemma-3b-mix-224 rather than the pretrained -pt- checkpoint. Neither is
fine-tuned on VizWiz: this script measures off-the-shelf performance.

Prompting
---------
These are chat models, so the raw PaliGemma prompt strings cannot be used
directly; the semantically equivalent instruction is wrapped in each model's own
chat template. The `--prompt_style generic` setting gives each baseline its
natural phrasing (bare question / plain caption request), which is the fair
zero-shot configuration.

`--brevity` optionally appends a short-answer directive for VQA. Without it,
chat models answer in full sentences and score near zero under exact-match VQA
accuracy, which measures output format rather than capability.

Tasks
-----
  vqa       VizWiz-VQA (test)            -> {image, answer}
  caps      VizWiz-Caps (validation)     -> {image_id, caption}
  vqav2     VQAv2 (test-standard)        -> {question_id, answer}   [EvalAI submission]
  coco      COCO-Caps (validation)       -> {image_id, caption}
  vizwiz    = vqa + caps       (target domain)
  benchmark = vqav2 + coco     (general domain)
  all       = all four subsets

Usage
-----
# 0) ALWAYS preview first: chat models may answer in full sentences, which
#    scores near zero under exact-match VQA accuracy. Nothing is written.
python src/evaluate_baselines.py --model smolvlm2 --task vqa --preview 20
python src/evaluate_baselines.py --model smolvlm2 --task vqa --preview 20 --brevity

# 1) Target domain (VizWiz VQA + captions)
python src/evaluate_baselines.py \
    --model      smolvlm2 \
    --task       vizwiz \
    --workdir    outputs/eval_data \
    --output_dir outputs/predictions \
    --batch_size 32 \
    --tag        smolvlm2_zeroshot \
    --brevity

# 2) General domain (VQAv2 test-std + COCO captions)
python src/evaluate_baselines.py \
    --model      qwen2.5vl \
    --task       benchmark \
    --workdir    outputs/eval_data \
    --output_dir outputs/predictions \
    --batch_size 8 \
    --tag        qwen25vl_zeroshot \
    --brevity

# 3) Quick sanity run on a small subset before a full evaluation
python src/evaluate_baselines.py --model qwen2.5vl --task vqa --limit 200

Scoring
-------
Outputs match the existing scorers, so no changes are needed:
  VizWiz-VQA : python src/score_vizwiz_vqa.py --gt <workdir>/vizwiz_vqa_data/VQA_test.json \
                   --pred outputs/predictions/vizwiz_vqa_test_predictions_<tag>.json
  Captions   : your existing pycocoevalcap-based scorer
  VQAv2      : submit vqav2_test_predictions_<tag>.json to the test-standard server

Notes
-----
  * Requires: pip install num2words qwen_vl_utils
  * VQAv2 test2015 is ~447k questions (12 GB of images) and COCO val2014 is 40k
    images (6 GB); budget compute accordingly, and use --limit to rehearse.
  * Both runners generate in true batches with left padding. Qwen2.5-VL's
    dynamic resolution makes its peak memory depend on the images in a batch,
    so use a smaller --batch_size for Qwen (8-16) than for SmolVLM2 (32+).
  * VALIDATE BATCHING before a full run: evaluate the same --limit N twice with
    --batch_size 1 and --batch_size 8 under different --tag values and confirm
    the answers are identical.
  * Both models are evaluated at bf16. Qwen2.5-VL uses aspect-preserving native
    resolution bounded to 256-1280 visual tokens; PaliGemma uses a fixed 256.
"""

import argparse
import gc
import json
import os
import time
import urllib.request
import zipfile

import requests
import torch
from PIL import Image
from tqdm import tqdm


MODELS = {
    "smolvlm2":  "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "qwen2.5vl": "Qwen/Qwen2.5-VL-3B-Instruct",
}

VQA_IMAGES_URL    = "https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip"
VQA_TEST_JSON_URL = "https://vizwiz.cs.colorado.edu/VizWiz_all_answers/VQA_test.json"
CAPS_VAL_IMAGES_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip"
CAPS_ANNOT_URL      = "https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations.zip"

VQAV2_IMAGES_URL    = "http://images.cocodataset.org/zips/test2015.zip"
VQAV2_QUESTIONS_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip"
COCO_IMAGES_URL     = "http://images.cocodataset.org/zips/val2014.zip"
COCO_ANNOT_URL      = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

BREVITY = " Answer in one or two words."


# ── Prompt text (before chat templating) ──────────────────────────────────────

def vqa_instruction(question: str, style: str, brevity: bool) -> str:
    text = question if style == "generic" else f"Assist a blind person: {question}"
    return text + (BREVITY if brevity else "")


def caption_instruction(style: str) -> str:
    if style == "generic":
        return "caption en"
    return "Describe this scene for a blind person in one short sentence."

# def caption_instruction(style: str) -> str:
#     if style == "generic":
#         return "Describe this image in one short sentence."
#     return "Describe this scene for a blind person in one short sentence."


# ── Model wrappers ────────────────────────────────────────────────────────────

class SmolVLM2Runner:
    """HuggingFaceTB/SmolVLM2-2.2B-Instruct via AutoModelForImageTextToText.

    Batched generation with left padding, mirroring the PaliGemma scripts.
    """

    def __init__(self, model_id, dtype=torch.bfloat16):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto").eval()

    def _prompt(self, instr):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": instr}]}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    @torch.inference_mode()
    def generate(self, images, instructions, max_tokens):
        prompts = [self._prompt(i) for i in instructions]
        inputs = self.processor(
            text=prompts,
            images=[[img] for img in images],   # one image per sample
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        in_len = inputs["input_ids"].shape[-1]
        gen = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                  do_sample=False)
        texts = self.processor.batch_decode(gen[:, in_len:],
                                            skip_special_tokens=True)
        return [t.strip() for t in texts]


class Qwen25VLRunner:
    """Qwen/Qwen2.5-VL-3B-Instruct via Qwen2_5_VLForConditionalGeneration.

    Batched generation with left padding. Qwen uses aspect-preserving dynamic
    resolution, so visual token counts vary per image; the pixel bounds below
    keep this reproducible and memory predictable. Use a smaller batch size than
    for SmolVLM2, since peak memory depends on the images in the batch.
    """

    def __init__(self, model_id, dtype=torch.bfloat16):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        # native (aspect-preserving) resolution, bounded for reproducibility:
        # 256 visual tokens floor (matches PaliGemma's fixed count), 1280 ceiling
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto").eval()

    @torch.inference_mode()
    def generate(self, images, instructions, max_tokens):
        from qwen_vl_utils import process_vision_info

        messages_list = [
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": instr}]}]
            for img, instr in zip(images, instructions)
        ]
        prompts = [
            self.processor.apply_chat_template(m, tokenize=False,
                                               add_generation_prompt=True)
            for m in messages_list
        ]
        image_inputs, video_inputs = process_vision_info(messages_list)

        inputs = self.processor(text=prompts, images=image_inputs,
                                videos=video_inputs, padding=True,
                                return_tensors="pt").to(self.model.device)
        gen = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                  do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        texts = self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
        return [t.strip() for t in texts]


def load_runner(name):
    model_id = MODELS[name]
    print(f"\nLoading {name} -> {model_id}")
    t0 = time.time()
    runner = SmolVLM2Runner(model_id) if name == "smolvlm2" else Qwen25VLRunner(model_id)
    print(f"loaded in {time.time()-t0:.1f}s | "
          f"VRAM {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return runner


# ── Data helpers (mirrors evaluate_vizwiz.py) ─────────────────────────────────

def download_and_extract(url, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    fname = url.split("/")[-1]
    fpath = os.path.join(dest_folder, fname)
    if not os.path.exists(fpath):
        print(f"downloading {fname} ...")
        r = requests.get(url, stream=True, timeout=180)
        with open(fpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    marker = fpath + ".extracted"
    if not os.path.exists(marker):
        print(f"extracting {fname} ...")
        with zipfile.ZipFile(fpath) as z:
            z.extractall(dest_folder)
        open(marker, "w").close()


def _pred_path(output_dir, base_name, tag):
    if tag:
        root, ext = os.path.splitext(base_name)
        base_name = f"{root}_{tag}{ext}"
    return os.path.join(output_dir, base_name)


# ── TASK 1: VizWiz-VQA (test) ─────────────────────────────────────────────────

def run_vqa(runner, args):
    print("\n" + "=" * 70)
    print("VizWiz-VQA (test) — zero-shot")
    print("=" * 70)
    print(f"Model      : {args.model}")
    print(f"Batch size : {args.batch_size}")
    print(f"Max tokens : {args.max_tokens}")

    data_root = os.path.join(args.workdir, "vizwiz_vqa_data")
    download_and_extract(VQA_IMAGES_URL, data_root)
    test_json = os.path.join(data_root, "VQA_test.json")
    if not os.path.exists(test_json):
        urllib.request.urlretrieve(VQA_TEST_JSON_URL, test_json)
    images_dir = os.path.join(data_root, "test")
    items = json.load(open(test_json))
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items):,} questions")

    out_file = _pred_path(args.output_dir, "vizwiz_vqa_test_predictions.json", args.tag)
    results = []
    if os.path.exists(out_file) and not args.preview:
        results = json.load(open(out_file))
        done = {r["image"] for r in results}
        items = [it for it in items if it["image"] not in done]
        print(f"resuming — {len(results):,} already done, {len(items):,} remaining")

    for i in tqdm(range(0, len(items), args.batch_size)):
        chunk = items[i:i + args.batch_size]
        imgs, instrs, keep = [], [], []
        for it in chunk:
            try:
                imgs.append(Image.open(os.path.join(images_dir, it["image"])).convert("RGB"))
            except Exception as e:
                print(f"skip {it['image']}: {e}")
                continue
            instrs.append(vqa_instruction(it["question"], args.prompt_style, args.brevity))
            keep.append(it)
        if not imgs:
            continue

        preds = runner.generate(imgs, instrs, args.max_tokens)
        for it, p in zip(keep, preds):
            results.append({"image": it["image"], "answer": p})

        if args.preview and len(results) >= args.preview:
            print("\n--- PREVIEW (nothing written) ---")
            for r, it in zip(results[:args.preview], items[:args.preview]):
                print(f"  Q: {it['question']}\n  A: {r['answer']}\n")
            return None

        if len(results) % (args.batch_size * 10) < args.batch_size:
            json.dump(results, open(out_file, "w"), indent=2)

    results.sort(key=lambda x: x["image"])
    json.dump(results, open(out_file, "w"), indent=2)
    print(f"saved {len(results):,} -> {out_file}")
    return out_file


# ── TASK 2: VizWiz-Caps (validation) ──────────────────────────────────────────

def run_caps(runner, args):
    print("\n" + "=" * 70)
    print("VizWiz-Caps (validation) — zero-shot")
    print("=" * 70)
    print(f"Model      : {args.model}")
    print(f"Batch size : {args.batch_size}")
    print(f"Max tokens : {args.max_tokens}")

    caps_root = os.path.join(args.workdir, "vizwiz_caps_data")
    download_and_extract(CAPS_VAL_IMAGES_URL, caps_root)
    download_and_extract(CAPS_ANNOT_URL, caps_root)
    val_images = os.path.join(caps_root, "val")
    val_json = os.path.join(caps_root, "annotations", "val.json")

    meta = json.load(open(val_json))
    items = [{"image_id": im["id"], "file_name": im["file_name"]} for im in meta["images"]]
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items):,} images")

    out_file = _pred_path(args.output_dir, "vizwiz_caption_val_predictions.json", args.tag)
    results = []
    if os.path.exists(out_file) and not args.preview:
        results = json.load(open(out_file))
        done = {r["image_id"] for r in results}
        items = [it for it in items if it["image_id"] not in done]
        print(f"resuming — {len(results):,} already done, {len(items):,} remaining")

    instr = caption_instruction(args.prompt_style)

    for i in tqdm(range(0, len(items), args.batch_size)):
        chunk = items[i:i + args.batch_size]
        imgs, keep = [], []
        for it in chunk:
            try:
                imgs.append(Image.open(os.path.join(val_images, it["file_name"])).convert("RGB"))
                keep.append(it)
            except Exception as e:
                print(f"skip {it['file_name']}: {e}")
        if not imgs:
            continue

        preds = runner.generate(imgs, [instr] * len(imgs), args.max_tokens)
        for it, p in zip(keep, preds):
            results.append({"image_id": it["image_id"], "caption": p})

        if args.preview and len(results) >= args.preview:
            print("\n--- PREVIEW (nothing written) ---")
            for r in results[:args.preview]:
                print(f"  {r['image_id']}: {r['caption']}")
            return None

        if len(results) % (args.batch_size * 10) < args.batch_size:
            json.dump(results, open(out_file, "w"), indent=2)

    json.dump(results, open(out_file, "w"), indent=2)
    lengths = [len(r["caption"].split()) for r in results]
    print(f"saved {len(results):,} -> {out_file}")
    print(f"avg caption length: {sum(lengths)/max(len(lengths),1):.1f} words")
    return out_file


# ── TASK 3: VQAv2 (test-standard) ─────────────────────────────────────────────

def run_vqav2(runner, args):
    print("\n" + "=" * 70)
    print("VQAv2 (test-standard) — zero-shot")
    print("=" * 70)
    print(f"Model      : {args.model}")
    print(f"Batch size : {args.batch_size}")
    print(f"Max tokens : {args.max_tokens}")

    data_root = os.path.join(args.workdir, "vqav2_data")
    download_and_extract(VQAV2_IMAGES_URL, data_root)
    download_and_extract(VQAV2_QUESTIONS_URL, data_root)

    images_dir = os.path.join(data_root, "test2015")
    q_path = os.path.join(data_root, "v2_OpenEnded_mscoco_test2015_questions.json")
    questions = json.load(open(q_path))["questions"]
    if args.limit:
        questions = questions[:args.limit]
    print(f"{len(questions):,} questions")

    out_file = _pred_path(args.output_dir, "vqav2_test_predictions.json", args.tag)
    results = []
    if os.path.exists(out_file) and not args.preview:
        results = json.load(open(out_file))
        done = {r["question_id"] for r in results}
        questions = [q for q in questions if q["question_id"] not in done]
        print(f"resuming — {len(results):,} already done, {len(questions):,} remaining")

    for i in tqdm(range(0, len(questions), args.batch_size)):
        chunk = questions[i:i + args.batch_size]
        imgs, instrs, keep = [], [], []
        for q in chunk:
            fname = f"COCO_test2015_{q['image_id']:012d}.jpg"
            try:
                imgs.append(Image.open(os.path.join(images_dir, fname)).convert("RGB"))
            except Exception as e:
                print(f"skip {fname}: {e}")
                continue
            instrs.append(vqa_instruction(q["question"], args.prompt_style, args.brevity))
            keep.append(q)
        if not imgs:
            continue

        preds = runner.generate(imgs, instrs, args.max_tokens)
        for q, p in zip(keep, preds):
            results.append({"question_id": q["question_id"], "answer": p})

        if args.preview and len(results) >= args.preview:
            print("\n--- PREVIEW (nothing written) ---")
            for r, q in zip(results[:args.preview], questions[:args.preview]):
                print(f"  Q: {q['question']}\n  A: {r['answer']}\n")
            return None

        if len(results) % (args.batch_size * 10) < args.batch_size:
            json.dump(results, open(out_file, "w"), indent=2)

    results.sort(key=lambda x: x["question_id"])
    json.dump(results, open(out_file, "w"), indent=2)
    print(f"saved {len(results):,} -> {out_file}")
    print("Submit to the VQAv2 test-standard server:")
    print("  https://eval.ai/web/challenges/challenge-page/830/overview")
    return out_file


# ── TASK 4: COCO-Caps (validation) ────────────────────────────────────────────

def run_coco_caps(runner, args):
    print("\n" + "=" * 70)
    print("COCO-Caps (validation) — zero-shot")
    print("=" * 70)
    print(f"Model      : {args.model}")
    print(f"Batch size : {args.batch_size}")
    print(f"Max tokens : {args.max_tokens}")

    data_root = os.path.join(args.workdir, "coco_data")
    download_and_extract(COCO_IMAGES_URL, data_root)
    download_and_extract(COCO_ANNOT_URL, data_root)

    images_dir = os.path.join(data_root, "val2014")
    caps_path = os.path.join(data_root, "annotations", "captions_val2014.json")
    items = json.load(open(caps_path))["images"]
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items):,} images")

    out_file = _pred_path(args.output_dir, "coco_caption_val_predictions.json", args.tag)
    results = []
    if os.path.exists(out_file) and not args.preview:
        results = json.load(open(out_file))
        done = {r["image_id"] for r in results}
        items = [it for it in items if it["id"] not in done]
        print(f"resuming — {len(results):,} already done, {len(items):,} remaining")

    instr = caption_instruction(args.prompt_style)

    for i in tqdm(range(0, len(items), args.batch_size)):
        chunk = items[i:i + args.batch_size]
        imgs, keep = [], []
        for it in chunk:
            try:
                imgs.append(Image.open(os.path.join(images_dir, it["file_name"])).convert("RGB"))
                keep.append(it)
            except Exception as e:
                print(f"skip {it['file_name']}: {e}")
        if not imgs:
            continue

        preds = runner.generate(imgs, [instr] * len(imgs), args.max_tokens)
        for it, p in zip(keep, preds):
            results.append({"image_id": it["id"], "caption": p})

        if args.preview and len(results) >= args.preview:
            print("\n--- PREVIEW (nothing written) ---")
            for r in results[:args.preview]:
                print(f"  {r['image_id']}: {r['caption']}")
            return None

        if len(results) % (args.batch_size * 10) < args.batch_size:
            json.dump(results, open(out_file, "w"), indent=2)

    json.dump(results, open(out_file, "w"), indent=2)
    lengths = [len(r["caption"].split()) for r in results]
    print(f"saved {len(results):,} -> {out_file}")
    print(f"avg caption length: {sum(lengths)/max(len(lengths),1):.1f} words")
    return out_file


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Zero-shot baseline evaluation")
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--task",
                    choices=["vqa", "caps", "vqav2", "coco", "vizwiz", "benchmark", "all"],
                    default="vizwiz",
                    help="vizwiz = vqa+caps (target domain); benchmark = vqav2+coco; all = four subsets")
    ap.add_argument("--prompt_style", choices=["generic", "custom"], default="generic",
                    help="generic = each model's natural phrasing (fair zero-shot)")
    ap.add_argument("--brevity", action="store_true",
                    help="append a short-answer directive to VQA prompts")
    ap.add_argument("--workdir", default="outputs/eval_data")
    ap.add_argument("--output_dir", default="outputs/predictions")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64,
                    help="as in the PaliGemma scripts. SmolVLM2 handles 32-64; "
                         "reduce for Qwen2.5-VL (8-16) since its dynamic "
                         "resolution makes peak memory less predictable")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N items")
    ap.add_argument("--preview", type=int, default=0,
                    help="print the first N outputs and exit without writing")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.tag:
        args.tag = f"{args.model}_zeroshot"

    runner = load_runner(args.model)

    if args.task in ("vqa", "vizwiz", "all"):
        run_vqa(runner, args)
        gc.collect(); torch.cuda.empty_cache()
    if args.task in ("caps", "vizwiz", "all"):
        run_caps(runner, args)
        gc.collect(); torch.cuda.empty_cache()
    if args.task in ("vqav2", "benchmark", "all"):
        run_vqav2(runner, args)
        gc.collect(); torch.cuda.empty_cache()
    if args.task in ("coco", "benchmark", "all"):
        run_coco_caps(runner, args)

    print("\nDone. Score with your existing scripts:")
    print(f"  python src/score_vizwiz_vqa.py --gt {args.workdir}/vizwiz_vqa_data/VQA_test.json "
          f"--pred {args.output_dir}/vizwiz_vqa_test_predictions_{args.tag}.json")


if __name__ == "__main__":
    main()
