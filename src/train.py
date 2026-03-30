"""
train.py
========
QLoRA fine-tuning of PaliGemma-3B on a VizWiz dataset saved with
``data_preparation.py``.

Usage
-----
python src/train.py \
    --train_dataset_path data/train_dataset \
    --val_dataset_path   data/val_dataset \
    --output_dir         outputs/run1

Resume from latest checkpoint:
    python src/train.py ... --resume
"""

import argparse
import gc
import io
import json
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from PIL import Image as PILImage
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
)


# ── Data collator ─────────────────────────────────────────────────────────────

class PaliGemmaDataCollator:
    """
    Collate a batch of VizWiz samples into model inputs.

    Each sample must contain:
      - image  : HF Image feature (dict with 'bytes' or 'path', or a PIL Image)
      - question: str
      - answer : str
    """

    def __init__(self, processor: PaliGemmaProcessor, max_length: int = 256):
        self.processor  = processor
        self.max_length = max_length

    def _load_image(self, image_field) -> PILImage.Image:
        if isinstance(image_field, PILImage.Image):
            return image_field.convert("RGB")
        if isinstance(image_field, dict):
            if "bytes" in image_field and image_field["bytes"]:
                return PILImage.open(io.BytesIO(image_field["bytes"])).convert("RGB")
            if "path" in image_field and image_field["path"]:
                return PILImage.open(image_field["path"]).convert("RGB")
        raise ValueError(f"Cannot load image from: {type(image_field)}")

    def __call__(self, examples: list[dict]) -> dict:
        images    = [self._load_image(ex["image"])    for ex in examples]
        questions = [ex["question"]                   for ex in examples]
        answers   = [ex["answer"]                     for ex in examples]

        # Build prompts  →  "<image>question\nanswer"
        prompts = [f"<image>{q}" for q in questions]
        texts   = [f"{p}\n{a}"   for p, a in zip(prompts, answers)]

        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
        )

        # Labels: mask prompt tokens with -100
        labels = inputs["input_ids"].clone()
        prompt_inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:, :prompt_len] = -100

        inputs["labels"] = labels
        return inputs


# ── Model builder ─────────────────────────────────────────────────────────────

def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def build_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def load_model_and_processor(
    model_id: str,
    bnb_config: BitsAndBytesConfig,
    lora_config: LoraConfig,
    attn_implementation: str = "sdpa",
):
    processor = PaliGemmaProcessor.from_pretrained(model_id)

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        attn_implementation=attn_implementation,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def find_latest_checkpoint(output_dir: str) -> str | None:
    checkpoints = sorted(
        [d for d in Path(output_dir).glob("checkpoint-*") if d.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning of PaliGemma-3B")
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--val_dataset_path",   required=True)
    parser.add_argument("--output_dir",         default="outputs/run1")
    parser.add_argument("--model_id",           default="google/paligemma-3b-mix-224")
    parser.add_argument("--num_epochs",         type=int,   default=3)
    parser.add_argument("--batch_size",         type=int,   default=16)
    parser.add_argument("--grad_accum",         type=int,   default=8)
    parser.add_argument("--learning_rate",      type=float, default=2e-5)
    parser.add_argument("--max_length",         type=int,   default=256)
    parser.add_argument("--lora_r",             type=int,   default=8)
    parser.add_argument("--lora_alpha",         type=int,   default=16)
    parser.add_argument("--lora_dropout",       type=float, default=0.05)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--resume",             action="store_true",
                        help="Resume training from the latest checkpoint")
    args = parser.parse_args()

    # ── Load datasets ────────────────────────────────────────────────────────
    print("Loading datasets …")
    train_ds = load_from_disk(args.train_dataset_path)
    val_ds   = load_from_disk(args.val_dataset_path)
    print(f"  Train: {len(train_ds):,}   Val: {len(val_ds):,}")

    # ── Build configs ────────────────────────────────────────────────────────
    bnb_config  = build_bnb_config()
    lora_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)

    # ── Load model ───────────────────────────────────────────────────────────
    print("Loading model and processor …")
    model, processor = load_model_and_processor(
        args.model_id, bnb_config, lora_config
    )

    # ── Training arguments ───────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",
        bf16=True,
        tf32=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    # ── Collator ─────────────────────────────────────────────────────────────
    collator = PaliGemmaDataCollator(processor, max_length=args.max_length)

    # ── Resume checkpoint ────────────────────────────────────────────────────
    resume_from = None
    if args.resume:
        resume_from = find_latest_checkpoint(args.output_dir)
        if resume_from:
            print(f"Resuming from checkpoint: {resume_from}")
        else:
            print("No checkpoint found — starting from scratch.")

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        )],
    )

    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save final adapter ───────────────────────────────────────────────────
    adapter_dir = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"\n✅ Training complete. Adapter saved to: {adapter_dir}")

    # ── Save training history ─────────────────────────────────────────────────
    history = trainer.state.log_history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
